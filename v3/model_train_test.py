import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import os
import numpy as np
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans  # For automated color extraction
import imageio
from PIL import Image, ImageFilter
import cv2

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set image size for Flowers (we use 64x64 images)
img_size = 64

# Update transforms for training and testing
transform_train = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])
batch_size = 64  # Adjust based on available GPU memory

# For Flowers102 the dataset has 102 classes.
# We will later extract the class names from the dataset.
# (For visualization purposes we will only show a subset.)
class_names = None

# Define color prototypes (RGB) for 10 color categories
COLOR_CATEGORIES = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 128, 0]),
    "blue": np.array([0, 0, 255]),
    "yellow": np.array([255, 255, 0]),
    "orange": np.array([255, 165, 0]),
    "purple": np.array([128, 0, 128]),
    "pink": np.array([255, 192, 203]),
    "brown": np.array([165, 42, 42]),
    "white": np.array([255, 255, 255]),
    "black": np.array([0, 0, 0])
}

COLOR_MAPPING = {"red": 0, "green": 1, "blue": 2, "yellow": 3,
                 "orange": 4, "purple": 5, "pink": 6, "brown": 7,
                 "white": 8, "black": 9}


def rgb_to_hsv(r, g, b):
    """
    Convert r,g,b in [0..1] to (h,s,v) with:
      h in [0..360)
      s,v in [0..1]
    """
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn

    # Hue
    if diff < 1e-6:
        h = 0.0
    elif mx == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:  # mx == b
        h = (60 * ((r - g) / diff) + 240) % 360

    # Value
    v = mx

    # Saturation
    if mx < 1e-6:
        s = 0.0
    else:
        s = diff / mx

    return h, s, v


def hsv_to_color_name(h, s, v):
    """
    Rule-based classification from HSV to a color name (string).
    Skips "green" and "black" (assuming we don't want to label flowers as green/black).
    If no rule matches, we fallback to nearest color (excluding green/black).
    """
    # 1) Check a few "special" categories first
    # We'll skip labeling as black or green to avoid background or extremely dark results.

    # White: high value, low saturation
    if v > 0.85 and s < 0.2:
        return "white"

    # Brown: hue in [10..40], moderate saturation, moderate value
    #        e.g. s up to ~0.6, v up to ~0.6
    if 10 <= h <= 40 and s <= 0.6 and v <= 0.6:
        return "brown"

    # Pink: hue around [300..345], or 0..15 if it's not too saturated
    #       We can also interpret pink as red with a high value, moderate saturation
    #       We'll do a direct approach for pink.
    if (300 <= h < 360) or (0 <= h < 20):
        # Distinguish pink vs red by brightness or saturation
        # e.g. pink if v>0.5 and s<0.8, etc.
        if v > 0.6 and s < 0.8:
            return "pink"
        else:
            return "red"

    # Red: hue near 0 or > 340, if not covered by pink
    if (h < 20 or h > 340) and s > 0.2 and v > 0.2:
        return "red"

    # Orange: hue in [20..45], if not covered by brown
    if 20 <= h < 45 and s > 0.3 and v > 0.3:
        return "orange"

    # Yellow: hue in [45..65], s>0.3, v>0.3
    if 45 <= h < 65 and s > 0.3 and v > 0.3:
        return "yellow"

    # Green: (we skip final classification as green, but if you do want it, define it here)
    # if 65 <= h < 170 and s > 0.2 and v > 0.2:
    #     return "green"

    # Blue: hue in [170..250], s>0.2, v>0.2
    if 170 <= h < 250 and s > 0.2 and v > 0.2:
        return "blue"

    # Purple: hue in [250..290], s>0.2, v>0.2
    if 250 <= h < 310 and s > 0.2 and v > 0.2:
        return "purple"

    # If we get here, we do a fallback to the nearest color (skipping green & black).
    return None


def fallback_nearest_color(r255, g255, b255):
    """
    Fallback to the nearest color in COLOR_CATEGORIES, but skip "green" and "black".
    """
    best_color = None
    best_dist = 1e9
    for color_name, rgb_val in COLOR_CATEGORIES.items():
        if color_name in ["green", "black"]:
            continue
        dist = np.linalg.norm(np.array([r255, g255, b255]) - rgb_val.astype(np.float32))
        if dist < best_dist:
            best_dist = dist
            best_color = color_name
    return best_color


def extract_color_category(image, k=5):
    """
    K-means + HSV-based color classification.
    We skip labeling as "green" or "black" to reduce background/dark confusion.
    """
    try:
        # 1) Convert to NumPy and blur
        if hasattr(image, 'convert'):  # PIL
            image = image.convert('RGB')
            image = image.filter(ImageFilter.GaussianBlur(radius=1))
            img_np = np.array(image)
        elif hasattr(image, 'numpy') and hasattr(image, 'permute'):  # PyTorch tensor
            if image.ndim == 3 and image.shape[0] <= 3:
                img_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                img_np = image.cpu().numpy()
        elif isinstance(image, np.ndarray):
            img_np = image.copy()
        else:
            raise TypeError("Unsupported image type")

        # 2) Handle grayscale/alpha
        if img_np.ndim == 2:
            img_np = np.stack([img_np, img_np, img_np], axis=2)
        if img_np.shape[2] == 4:
            img_np = img_np[..., :3]

        # 3) Normalize [0..1]
        if img_np.max() > 1.0:
            pixels = img_np.reshape(-1, 3).astype(np.float32) / 255.0
        else:
            pixels = img_np.reshape(-1, 3).astype(np.float32)

        # 4) Filter out extremely dark/bright or unsaturated
        brightness = pixels.mean(axis=1)
        max_c = pixels.max(axis=1)
        min_c = pixels.min(axis=1)
        saturation = (max_c - min_c) / np.maximum(max_c, 1e-6)

        # Example thresholds
        # Keep: 0.15 < brightness < 0.95, saturation > 0.1
        mask_bright = (brightness > 0.15) & (brightness < 0.95)
        mask_sat = saturation > 0.1
        combined_mask = mask_bright & mask_sat
        if np.sum(combined_mask) < 50:
            # if too few remain, skip saturation mask
            combined_mask = mask_bright
        filtered_pixels = pixels[combined_mask]
        if len(filtered_pixels) < 10:
            return "unknown", -1

        # 5) K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(filtered_pixels)
        centers = kmeans.cluster_centers_
        counts = np.bincount(kmeans.labels_)

        # 6) Weighted selection (bigger cluster => more weight)
        #    also boost more saturated clusters
        cluster_weights = []
        for i, c in enumerate(centers):
            size_weight = counts[i]
            r, g, b = c
            c_max = max(r, g, b)
            c_min = min(r, g, b)
            c_sat = (c_max - c_min) / (c_max + 1e-6)
            # cluster weight = size * (1 + saturation bonus)
            cluster_weight = size_weight * (1.0 + 1.5 * c_sat)
            cluster_weights.append(cluster_weight)

        # 7) Sort clusters by descending weight
        sorted_idx = np.argsort(cluster_weights)[::-1]

        # 8) Try each cluster in descending order, do HSV classification
        fallback_choice = None
        for idx in sorted_idx:
            r, g, b = centers[idx]
            h, s, v = rgb_to_hsv(r, g, b)
            color_name = hsv_to_color_name(h, s, v)
            if color_name is not None:
                # We have a rule-based color
                return color_name, COLOR_MAPPING[color_name]
            else:
                # fallback if none rule matched
                # but only store first cluster as fallback
                if fallback_choice is None:
                    fallback_choice = idx

        # If all clusters fail direct rules, fallback to nearest color for top cluster
        if fallback_choice is not None:
            r, g, b = centers[fallback_choice]
            r255, g255, b255 = (r * 255, g * 255, b * 255)
            fallback_color = fallback_nearest_color(r255, g255, b255)
            return fallback_color, COLOR_MAPPING[fallback_color]

        # Otherwise unknown
        return "unknown", -1

    except Exception as e:
        print(f"Error in color extraction: {e}")
        return "unknown", -1

def create_flower_color_visualization(flowers, num_samples=20, save_path='flower_color_visualization.png'):
    """
    Create a visualization of flower samples and their color labels.
    Handles two cases:
    1. Standard Flowers102 dataset (returns image, label)
    2. Flowers102WithColor dataset (returns image, flower_label, color_label)
    """
    if flowers is None:
        print("Cannot create flower color visualization. Dataset is not loaded.")
        return

    # Make sure our imports are available
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torchvision import transforms

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    indices = np.random.choice(len(flowers), num_samples, replace=False)

    # Define reverse mapping from color index to name
    reverse_color_mapping = {idx: name for name, idx in COLOR_MAPPING.items()}

    for i, idx in enumerate(indices):
        if i >= len(axes):
            break

        # Get the item from dataset
        item = flowers[idx]

        # Handle different types of datasets
        if len(item) == 2:  # Standard Flowers102 returns (image, label)
            img, flower_label = item

            # Extract color directly from the image
            if isinstance(img, torch.Tensor):
                img_pil = transforms.ToPILImage()(img)
            else:
                img_pil = img

            # Use our simplified color extraction function
            try:
                color_name, color_idx = extract_color_category(img_pil)
            except Exception as e:
                print(f"Error extracting color from image: {e}")
                color_name = "unknown"
                color_idx = -1

        else:  # Flowers102WithColor returns (image, flower_label, color_label)
            img, flower_label, color_idx = item

            # Get color name from index using our reverse mapping
            if color_idx in reverse_color_mapping:
                color_name = reverse_color_mapping[color_idx]
            else:
                # If we're getting an index not in our mapping, extract color from image
                if isinstance(img, torch.Tensor):
                    img_pil = transforms.ToPILImage()(img)
                else:
                    img_pil = img

                try:
                    color_name, _ = extract_color_category(img_pil)
                except Exception as e:
                    print(f"Error extracting color from image: {e}")
                    color_name = "unknown"

                    # Display the image
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()
            axes[i].imshow(img_np)
        else:
            axes[i].imshow(img)

        # Get corresponding RGB value for the color
        if color_name in COLOR_CATEGORIES:
            color_rgb = COLOR_CATEGORIES[color_name] / 255.0
        else:
            # Fallback to gray if color is unknown
            color_rgb = np.array([0.5, 0.5, 0.5])

        class_name = f"Flower class {flower_label}"
        axes[i].set_title(f"{class_name}\nColor: {color_name}", fontsize=10)
        axes[i].axis('off')

        # Add color square
        axes[i].add_patch(plt.Rectangle((5, 5), 10, 10, color=color_rgb, alpha=0.8))

    plt.tight_layout()
    plt.suptitle("Flower samples and their auto-extracted color labels", fontsize=16, y=1.02)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Visualization saved to {save_path}")
    return save_path

class Flowers102WithColor(torch.utils.data.Dataset):
    """
    A wrapper for the Flowers102 dataset so that each sample returns:
    (image, flower_type label, color label as an integer)
    The color label is automatically extracted using k-means clustering.
    """
    def __init__(self, root, split, transform, precompute_color=True):
        self.flowers = datasets.Flowers102(root=root, split=split, download=True, transform=transform)
        self.precompute_color = precompute_color
        if precompute_color:
            self.color_labels = []
            print("Precomputing color labels for the dataset...")
            for i in tqdm(range(len(self.flowers)), desc="Computing Colors"):
                image, _ = self.flowers[i]  # Get PIL image (before transform)
                # extract_color_category returns (color_name, color_index)
                color_label = extract_color_category(image)
                # Save only the integer index
                self.color_labels.append(color_label[1])
        else:
            self.color_labels = None

    def __getitem__(self, index):
        image, flower_label = self.flowers[index]
        if self.color_labels is not None:
            # Return only the integer index
            color_label = self.color_labels[index]
        else:
            # Compute on the fly and take only the index
            color_label = extract_color_category(image)
            if isinstance(color_label, (list, tuple)):
                color_label = color_label[1]
        return image, flower_label, color_label

    def __len__(self):
        return len(self.flowers)

# =============================================================================
# ORIGINAL MODEL COMPONENTS (mostly unchanged)
# =============================================================================

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=False),
            Swish(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        return x * attention

class CenterLoss(nn.Module):
    def __init__(self, num_classes=102, feat_dim=256, min_distance=1.0, repulsion_strength=1.0):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.min_distance = min_distance
        self.repulsion_strength = repulsion_strength
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        with torch.no_grad():
            if num_classes == 2:
                self.centers[0] = -torch.ones(feat_dim) / math.sqrt(feat_dim)
                self.centers[1] = torch.ones(feat_dim) / math.sqrt(feat_dim)
            else:
                for i in range(num_classes):
                    self.centers[i] = torch.randn(feat_dim)
                    self.centers[i] = self.centers[i] / torch.norm(self.centers[i]) * 2.0
    def compute_pairwise_distances(self, x, y):
        n = x.size(0)
        m = y.size(0)
        x_norm = (x ** 2).sum(1).view(n, 1)
        y_norm = (y ** 2).sum(1).view(1, m)
        distmat = x_norm + y_norm - 2.0 * torch.mm(x, y.transpose(0, 1))
        distmat = torch.clamp(distmat, min=1e-12)
        distmat = torch.sqrt(distmat)
        return distmat
    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = self.compute_pairwise_distances(x, self.centers)
        classes = torch.arange(self.num_classes).to(labels.device)
        labels_expanded = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expanded.eq(classes.expand(batch_size, self.num_classes))
        attraction_dist = distmat * mask.float()
        attraction_loss = attraction_dist.sum() / batch_size
        center_distances = self.compute_pairwise_distances(self.centers, self.centers)
        diff_mask = 1.0 - torch.eye(self.num_classes, device=x.device)
        repulsion_loss = torch.clamp(self.min_distance - center_distances, min=0.0)
        repulsion_loss = (repulsion_loss * diff_mask).sum() / (self.num_classes * (self.num_classes - 1) + 1e-6)
        intra_class_variance = 0.0
        for c in range(self.num_classes):
            class_mask = (labels == c)
            if torch.sum(class_mask) > 1:
                class_samples = x[class_mask]
                class_center = torch.mean(class_samples, dim=0)
                variance = torch.mean(torch.sum((class_samples - class_center) ** 2, dim=1))
                intra_class_variance += variance
        if self.num_classes > 0:
            intra_class_variance = intra_class_variance / self.num_classes
        total_loss = attraction_loss + self.repulsion_strength * repulsion_loss - 0.1 * intra_class_variance
        with torch.no_grad():
            self.avg_center_dist = torch.sum(center_distances * diff_mask) / (self.num_classes * (self.num_classes - 1) + 1e-6)
            self.avg_sample_dist = torch.mean(distmat)
            self.center_attraction = attraction_loss.item()
            self.center_repulsion = repulsion_loss.item()
            self.intra_variance = intra_class_variance.item() if isinstance(intra_class_variance, torch.Tensor) else intra_class_variance
        return total_loss

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ln1 = LayerNorm2d(channels)
        self.swish = Swish()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ln2 = LayerNorm2d(channels)
        self.ca = CALayer(channels)
        self.sa = SpatialAttention()
    def forward(self, x):
        residual = x
        out = self.swish(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out = self.ca(out)
        out = self.sa(out)
        out += residual
        out = self.swish(out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            LayerNorm2d(64),
            Swish()
        )
        self.skip_features = []
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            LayerNorm2d(128),
            Swish()
        )
        self.res1 = ResidualBlock(128)
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            LayerNorm2d(256),
            Swish()
        )
        self.res2 = ResidualBlock(256)
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            LayerNorm2d(512),
            Swish()
        )
        self.res3 = ResidualBlock(512)
        self.fc_mu = nn.Sequential(
            nn.Linear(512 * 8 * 8, 512),
            nn.LayerNorm(512),
            Swish(),
            nn.Linear(512, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(512 * 8 * 8, 512),
            nn.LayerNorm(512),
            Swish(),
            nn.Linear(512, latent_dim)
        )
    def forward(self, x):
        self.skip_features = []
        x = self.initial_conv(x)
        self.skip_features.append(x)
        x = self.down1(x)
        x = self.res1(x)
        self.skip_features.append(x)
        x = self.down2(x)
        x = self.res2(x)
        self.skip_features.append(x)
        x = self.down3(x)
        x = self.res3(x)
        self.skip_features.append(x)
        x_flat = x.view(x.size(0), -1)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            Swish(),
            nn.Linear(512, 512 * 8 * 8),
            nn.LayerNorm(512 * 8 * 8),
            Swish()
        )
        self.res3 = ResidualBlock(512)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            Swish()
        )
        self.res2 = ResidualBlock(256)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            Swish()
        )
        self.res1 = ResidualBlock(128)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            Swish()
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            Swish(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, z, encoder_features=None):
        x = self.fc(z)
        x = x.view(-1, 512, 8, 8)
        x = self.res3(x)
        x = self.up3(x)
        x = self.res2(x)
        x = self.up2(x)
        x = self.res1(x)
        x = self.up1(x)
        x = self.final_conv(x)
        return x

def euclidean_distance_loss(x, y, reduction='mean'):
    squared_diff = (x - y) ** 2
    squared_dist = squared_diff.view(x.size(0), -1).sum(dim=1)
    euclidean_dist = torch.sqrt(squared_dist + 1e-8)
    if reduction == 'mean':
        return euclidean_dist.mean()
    elif reduction == 'sum':
        return euclidean_dist.sum()
    else:
        return euclidean_dist

class SimpleAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256, num_classes=102):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            Swish(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            Swish(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        self.register_buffer('class_centers', torch.zeros(num_classes, latent_dim))
        self.register_buffer('center_counts', torch.zeros(num_classes))
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-2.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z
    def encode_with_params(self, x):
        mu, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, min=-2.0, max=10.0)
        return mu, logvar
    def decode(self, z):
        encoder_features = getattr(self, 'stored_encoder_features', None)
        return self.decoder(z, encoder_features)
    def classify(self, z):
        return self.classifier(z)
    def compute_center_loss(self, z, labels):
        centers_batch = self.class_centers[labels]
        squared_diff = (z - centers_batch) ** 2
        squared_dist = squared_diff.sum(dim=1)
        center_loss = torch.sqrt(squared_dist + 1e-8).mean()
        return center_loss
    def update_centers(self, z, labels, momentum=0.9):
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                class_samples = z[mask]
                class_mean = class_samples.mean(0)
                old_center = self.class_centers[label]
                new_center = momentum * old_center + (1 - momentum) * class_mean
                self.class_centers[label] = new_center
    def kl_divergence(self, mu, logvar):
        mu = torch.clamp(mu, min=-10.0, max=10.0)
        logvar = torch.clamp(logvar, min=-2.0, max=10.0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.clamp(kl_loss, min=0.0, max=100.0).mean()
        mu_reg = 1e-4 * torch.sum(mu.pow(2))
        return kl_loss + mu_reg
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        self.stored_encoder_features = self.encoder.skip_features
        x_recon = self.decoder(z, self.encoder.skip_features)
        return x_recon, mu, logvar, z

# =============================================================================
# NEW MULTI-CONDITION EMBEDDING & CONDITIONAL UNet (for Diffusion)
# =============================================================================

class MultiConditionEmbedding(nn.Module):
    def __init__(self, num_flower_types=102, num_colors=10, n_channels=256):
        super().__init__()
        self.flower_emb = nn.Embedding(num_flower_types, n_channels)
        self.color_emb = nn.Embedding(num_colors, n_channels)
        self.fc = nn.Linear(n_channels * 2, n_channels)
    def forward(self, flower_label, color_label):
        emb_flower = self.flower_emb(flower_label)
        emb_color = self.color_emb(color_label)
        combined = torch.cat((emb_flower, emb_color), dim=-1)
        return self.fc(combined)

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels=256):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(n_channels, n_channels * 2)
        self.act = Swish()
        self.lin2 = nn.Linear(n_channels * 2, n_channels)
    def forward(self, t):
        half_dim = self.n_channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        if emb.shape[1] < self.n_channels:
            padding = torch.zeros(emb.shape[0], self.n_channels - emb.shape[1], device=emb.device)
            emb = torch.cat([emb, padding], dim=1)
        return self.lin2(self.act(self.lin1(emb)))

class ConditionalUNet(nn.Module):
    def __init__(self, latent_dim=256, hidden_dims=[256, 512, 1024, 512, 256],
                 time_emb_dim=256, num_classes=102, num_colors=10, dropout_rate=0.3):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim
        self.time_emb = TimeEmbedding(n_channels=time_emb_dim)
        # Multi-condition embedding combines flower type and color
        self.multi_cond_emb = MultiConditionEmbedding(num_flower_types=num_classes, num_colors=num_colors, n_channels=time_emb_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dims[0])
        # Projection layers for time embeddings per stage.
        self.time_projections = nn.ModuleList([nn.Linear(time_emb_dim, dim) for dim in hidden_dims])
        # NEW: Projection layers for condition embeddings per stage.
        self.cond_projections = nn.ModuleList([nn.Linear(time_emb_dim, dim) for dim in hidden_dims])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=dropout_rate)
            for dim in hidden_dims
        ])
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            residual_block = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i]),
                nn.LayerNorm(hidden_dims[i]),
                nn.Dropout(dropout_rate),
                Swish()
            )
            layer_norm = nn.LayerNorm(hidden_dims[i])
            proj = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.layers.append(nn.ModuleList([residual_block, layer_norm, proj]))
        self.final_time_proj = nn.Linear(time_emb_dim, hidden_dims[-1])
        self.final_class_proj = nn.Linear(time_emb_dim, hidden_dims[-1])
        self.final_norm = nn.LayerNorm(hidden_dims[-1])
        self.final = nn.Linear(hidden_dims[-1], latent_dim)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    def forward(self, x, t, flower_label, color_label):
        residual = x
        t_emb_base = self.time_emb(t)  # shape: (batch, time_emb_dim)
        cond_emb = self.multi_cond_emb(flower_label, color_label)  # shape: (batch, time_emb_dim)
        h = self.latent_proj(x)  # shape: (batch, hidden_dims[0])
        for i, (block, layer_norm, downsample) in enumerate(self.layers):
            t_emb = self.time_projections[i](t_emb_base)  # shape: (batch, hidden_dims[i])
            cond_emb_proj = self.cond_projections[i](cond_emb)  # shape: (batch, hidden_dims[i])
            h = h + t_emb + cond_emb_proj
            h_residual = h
            h = block(h)
            h = h + h_residual
            h_norm = layer_norm(h)
            h_attn, _ = self.attention_layers[i](h_norm.unsqueeze(0), h_norm.unsqueeze(0), h_norm.unsqueeze(0))
            h = h + h_attn.squeeze(0)
            h = downsample(h)
        t_emb_final = self.final_time_proj(t_emb_base)
        cond_final = self.final_class_proj(cond_emb)
        h = h + t_emb_final + cond_final
        h = self.final_norm(h)
        out = self.final(h)
        return out + torch.sigmoid(self.residual_weight) * self.final(residual)

# =============================================================================
# CONDITIONAL DIFFUSION MODEL (UPDATED TO USE TWO CONDITIONS)
# =============================================================================

class ConditionalDenoiseDiffusion():
    def __init__(self, eps_model, n_steps=1000, device=None):
        super().__init__()
        self.eps_model = eps_model
        self.device = device
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps
    def p_sample(self, xt, t, flower_label, color_label):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=xt.device)
        eps_theta = self.eps_model(xt, t, flower_label, color_label)
        alpha_t = self.alpha[t].reshape(-1, 1).to(xt.device)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1).to(xt.device)
        mean = (xt - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_theta) / torch.sqrt(alpha_t)
        var = self.beta[t].reshape(-1, 1)
        if t[0] > 0:
            noise = torch.randn_like(xt)
            return mean + torch.sqrt(var) * noise
        else:
            return mean
    def sample(self, shape, device, flower_label, color_label):
        x = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(self.n_steps)), desc="Sampling"):
            x = self.p_sample(x, t, flower_label, color_label)
        return x
    def loss(self, x0, flower_label, color_label):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        eps = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps)
        eps_theta = self.eps_model(xt, t, flower_label, color_label)
        return euclidean_distance_loss(eps, eps_theta)

# =============================================================================
# VISUALIZATION FUNCTIONS (mostly unchanged)
# =============================================================================

def generate_samples_grid(autoencoder, diffusion, n_per_class=5, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    device = next(autoencoder.parameters()).device
    autoencoder.eval()
    diffusion.eps_model.eval()
    n_classes_vis = min(10, len(class_names))
    fig, axes = plt.subplots(n_classes_vis, n_per_class + 1, figsize=((n_per_class + 1) * 2, n_classes_vis * 2))
    fig.suptitle(f'Flowers102 Samples Generated by VAE-Diffusion Model', fontsize=16, y=0.98)
    for i in range(n_classes_vis):
        axes[i, 0].text(0.5, 0.5, class_names[i],
                        fontsize=10, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))
        axes[i, 0].axis('off')
        # For visualization we use the flower label and a default color label (e.g. 0)
        class_tensor = torch.tensor([i] * n_per_class, device=device)
        color_tensor = torch.tensor([0] * n_per_class, device=device)
        latent_shape = (n_per_class, autoencoder.latent_dim)
        samples = diffusion.sample(latent_shape, device, class_tensor, color_tensor)
        with torch.no_grad():
            decoded = autoencoder.decode(samples)
        for j in range(n_per_class):
            img = decoded[j].cpu().permute(1, 2, 0).numpy()
            axes[i, j + 1].imshow(img)
            axes[i, j + 1].axis('off')
            if i == 0:
                axes[i, j + 1].set_title(f'Sample {j + 1}', fontsize=9)
    description = (
        "This visualization shows images generated by the conditional diffusion model using a VAE decoder on the Oxford 102 Flowers dataset.\n"
        "Only a subset of classes (first 10) are visualized."
    )
    plt.figtext(0.5, 0.01, description, ha='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    save_path = f"{save_dir}/vae_samples_grid_subset.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    autoencoder.train()
    diffusion.eps_model.train()
    print(f"Generated sample grid for a subset of classes")
    return save_path

def visualize_denoising_steps(autoencoder, diffusion, class_idx, save_path=None):
    device = next(autoencoder.parameters()).device
    autoencoder.eval()
    diffusion.eps_model.eval()
    print(f"Generating latent space projection for class {class_names[class_idx]}...")
    flowers_test = datasets.Flowers102(root="./data", split='test', download=True, transform=transform_test)
    test_loader = DataLoader(flowers_test, batch_size=500, shuffle=False)
    all_latents = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            mu, logvar = autoencoder.encode_with_params(images)
            all_latents.append(mu.detach().cpu().numpy())
            all_labels.append(labels.numpy())
    all_latents = np.vstack(all_latents)
    all_labels = np.concatenate(all_labels)
    print("Computing PCA projection...")
    pca = PCA(n_components=2, random_state=42)
    latents_2d = pca.fit_transform(all_latents)
    n_samples = 5
    steps_to_show = 8
    step_size = diffusion.n_steps // steps_to_show
    timesteps = list(range(0, diffusion.n_steps, step_size))[::-1]
    # For denoising visualization, use the flower label and a default color (0)
    class_tensor = torch.tensor([class_idx] * n_samples, device=device)
    color_tensor = torch.tensor([0] * n_samples, device=device)
    x = torch.randn((n_samples, autoencoder.latent_dim), device=device)
    samples_per_step = []
    path_latents = []
    with torch.no_grad():
        for t in timesteps:
            current_x = x.clone()
            for time_step in range(t, -1, -1):
                current_x = diffusion.p_sample(current_x, torch.tensor([time_step], device=device), class_tensor, color_tensor)
            path_latents.append(current_x[0:1].detach().cpu().numpy())
            decoded = autoencoder.decode(current_x)
            samples_per_step.append(decoded.cpu())
        path_latents.append(current_x[0:1].detach().cpu().numpy())
    path_latents = np.vstack(path_latents)
    path_2d = pca.transform(path_latents)
    fig = plt.figure(figsize=(16, 16))
    gs = plt.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.3)
    ax_denoising = fig.add_subplot(gs[0])
    grid_rows = n_samples
    grid_cols = len(timesteps)
    ax_denoising.set_title(f"VAE-Diffusion Denoising Process for {class_names[class_idx]}", fontsize=16, pad=10)
    ax_denoising.set_xticks([])
    ax_denoising.set_yticks([])
    gs_denoising = gs[0].subgridspec(grid_rows, grid_cols, wspace=0.1, hspace=0.1)
    for i in range(n_samples):
        for j, t in enumerate(timesteps):
            ax = fig.add_subplot(gs_denoising[i, j])
            img = samples_per_step[j][i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            if i == 0:
                ax.set_title(f't={t}', fontsize=9)
            if j == 0:
                ax.set_ylabel(f"Sample {i + 1}", fontsize=9)
            if i == 0:
                for spine in ax.spines.values():
                    spine.set_color('red')
                    spine.set_linewidth(2)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.figtext(0.02, 0.65, "Path Tracked →", fontsize=12, color='red',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))
    ax_latent = fig.add_subplot(gs[1])
    for i in range(min(10, len(class_names))):
        mask = all_labels == i
        alpha = 0.3 if i != class_idx else 0.8
        size = 20 if i != class_idx else 40
        ax_latent.scatter(
            latents_2d[mask, 0],
            latents_2d[mask, 1],
            label=class_names[i],
            alpha=alpha,
            s=size
        )
    ax_latent.plot(
        path_2d[:, 0],
        path_2d[:, 1],
        'r-o',
        linewidth=2.5,
        markersize=8,
        label=f"Diffusion Path",
        zorder=10
    )
    for i in range(len(path_2d) - 1):
        ax_latent.annotate(
            "",
            xy=(path_2d[i + 1, 0], path_2d[i + 1, 1]),
            xytext=(path_2d[i, 0], path_2d[i, 1]),
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5)
        )
    for i, t in enumerate(timesteps):
        ax_latent.annotate(
            f"t={t}",
            xy=(path_2d[i, 0], path_2d[i, 1]),
            xytext=(path_2d[i, 0] + 2, path_2d[i, 1] + 2),
            fontsize=8,
            color='darkred'
        )
    ax_latent.scatter(path_2d[0, 0], path_2d[0, 1], c='black', s=100, marker='x', label="Start (Noise)", zorder=11)
    ax_latent.scatter(path_2d[-1, 0], path_2d[-1, 1], c='green', s=100, marker='*', label="End (Generated)", zorder=11)
    target_mask = all_labels == class_idx
    target_center = np.mean(latents_2d[target_mask], axis=0)
    ax_latent.scatter(target_center[0], target_center[1], c='green', s=300, marker='*',
                      edgecolor='black', alpha=0.7, zorder=9)
    ax_latent.annotate(
        f"TARGET: {class_names[class_idx]}",
        xy=(target_center[0], target_center[1]),
        xytext=(target_center[0] + 5, target_center[1] + 5),
        fontsize=14,
        fontweight='bold',
        color='darkgreen',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    )
    ax_latent.set_title(f"VAE-Diffusion Path in Latent Space for {class_names[class_idx]}", fontsize=16)
    ax_latent.legend(fontsize=10, loc='best')
    ax_latent.grid(True, linestyle='--', alpha=0.7)
    plt.figtext(
        0.5, 0.01,
        "This visualization shows the denoising process (top) and the corresponding path in latent space (bottom).\n"
        "The first row (highlighted in red) corresponds to the latent space path.",
        ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.1)
    if save_path is None:
        save_path = f"./results/denoising_path_{class_names[class_idx]}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"VAE Denoising visualization for {class_names[class_idx]} saved to {save_path}")
    autoencoder.train()
    diffusion.eps_model.train()
    return save_path

def visualize_reconstructions(autoencoder, epoch, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    device = next(autoencoder.parameters()).device
    flowers_test = datasets.Flowers102(root="./data", split='test', download=True, transform=transform_test)
    test_loader = DataLoader(flowers_test, batch_size=8, shuffle=True)
    test_images, test_labels = next(iter(test_loader))
    test_images = test_images.to(device)
    autoencoder.eval()
    with torch.no_grad():
        mu, logvar = autoencoder.encode_with_params(test_images)
        z = autoencoder.reparameterize(mu, logvar)
        reconstructed = autoencoder.decode(z)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        img = test_images[i].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        label_text = class_names[test_labels[i]] if class_names is not None else str(test_labels[i].item())
        axes[0, i].set_title(f'Original: {label_text}')
        axes[0, i].axis('off')
        recon_img = reconstructed[i].cpu().permute(1, 2, 0).numpy()
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title('Reconstruction')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/vae_reconstruction_epoch_{epoch}.png")
    plt.close()
    autoencoder.train()

def visualize_latent_space(autoencoder, epoch, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    device = next(autoencoder.parameters()).device
    flowers_test = datasets.Flowers102(root="./data", split='test', download=True, transform=transform_test)
    test_loader = DataLoader(flowers_test, batch_size=500, shuffle=False)
    autoencoder.eval()
    all_latents = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            mu, logvar = autoencoder.encode_with_params(images)
            all_latents.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())
    all_latents = np.vstack(all_latents)
    all_labels = np.concatenate(all_labels)
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=1000)
        latents_2d = tsne.fit_transform(all_latents)
        plt.figure(figsize=(10, 8))
        for i in range(min(10, len(class_names))):
            mask = all_labels == i
            plt.scatter(latents_2d[mask, 0], latents_2d[mask, 1], label=class_names[i], alpha=0.6)
        plt.title(f"t-SNE Visualization of VAE Latent Space (Epoch {epoch})")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/vae_latent_space_epoch_{epoch}.png")
        plt.close()
    except Exception as e:
        print(f"t-SNE visualization error: {e}")
    autoencoder.train()

def generate_class_samples(autoencoder, diffusion, target_class, num_samples=5, save_path=None):
    device = next(autoencoder.parameters()).device
    autoencoder.eval()
    diffusion.eps_model.eval()
    if isinstance(target_class, str):
        if target_class in class_names:
            target_class = class_names.index(target_class)
        else:
            raise ValueError(f"Invalid class name: {target_class}. Must be one of {class_names}")
    # For generation, we now also require a target color. Here we use a default (e.g. 0).
    class_tensor = torch.tensor([target_class] * num_samples, device=device)
    color_tensor = torch.tensor([0] * num_samples, device=device)
    latent_shape = (num_samples, autoencoder.latent_dim)
    with torch.no_grad():
        latent_samples = diffusion.sample(latent_shape, device, class_tensor, color_tensor)
        samples = autoencoder.decode(latent_samples)
    if save_path:
        plt.figure(figsize=(num_samples * 2, 3))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            img = samples[i].cpu().permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"{class_names[target_class]}")
        plt.suptitle(f"Generated {class_names[target_class]} Samples")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    return samples


def generate_class_color_samples(autoencoder, diffusion, target_class, target_color, num_samples=5, save_path=None):
    """
    Generate samples conditioned on both target class and target color.

    Parameters:
      autoencoder: the trained autoencoder model.
      diffusion: the conditional diffusion model.
      target_class: the target class label (either as a string or an integer index).
      target_color: the target color (either as a string, e.g. "red", or an integer index).
      num_samples: number of samples to generate.
      save_path: if provided, the path to save the generated image grid.

    Returns:
      samples: the generated images.
    """
    device = next(autoencoder.parameters()).device
    autoencoder.eval()
    diffusion.eps_model.eval()

    # Process target_class: if string, convert to index using global class_names.
    if isinstance(target_class, str):
        if target_class in class_names:
            target_class_idx = class_names.index(target_class)
        else:
            raise ValueError(f"Invalid class name: {target_class}. Must be one of {class_names}")
    else:
        target_class_idx = target_class

    # Process target_color: if string, convert using COLOR_MAPPING.
    if isinstance(target_color, str):
        if target_color in COLOR_MAPPING:
            target_color_idx = COLOR_MAPPING[target_color]
        else:
            raise ValueError(f"Invalid color: {target_color}. Must be one of {list(COLOR_MAPPING.keys())}")
    else:
        target_color_idx = target_color

    # Create tensors for class and color
    class_tensor = torch.tensor([target_class_idx] * num_samples, device=device)
    color_tensor = torch.tensor([target_color_idx] * num_samples, device=device)
    latent_shape = (num_samples, autoencoder.latent_dim)

    with torch.no_grad():
        latent_samples = diffusion.sample(latent_shape, device, class_tensor, color_tensor)
        samples = autoencoder.decode(latent_samples)

    if save_path:
        plt.figure(figsize=(num_samples * 2, 3))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            img = samples[i].cpu().permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.axis('off')
            # Display both the class and color (for example: "Rose - Red")
            plt.title(f"{class_names[target_class_idx]}\n{target_color}", fontsize=9)
        plt.suptitle(f"Generated {class_names[target_class_idx]} Samples in {target_color} color")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    return samples

def create_diffusion_animation(autoencoder, diffusion, class_idx, num_frames=50, seed=42,
                               save_path=None, temp_dir=None, fps=10, reverse=False):
    device = next(autoencoder.parameters()).device
    autoencoder.eval()
    diffusion.eps_model.eval()
    if isinstance(class_idx, str):
        if class_idx in class_names:
            class_idx = class_names.index(class_idx)
        else:
            raise ValueError(f"Invalid class name: {class_idx}. Must be one of {class_names}")
    if temp_dir is None:
        temp_dir = os.path.join('./temp_frames', f'class_{class_idx}_{seed}')
    os.makedirs('./temp_frames', exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    if save_path is None:
        save_dir = './results'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'diffusion_animation_{class_names[class_idx]}.gif')
    torch.manual_seed(seed)
    np.random.seed(seed)
    class_tensor = torch.tensor([class_idx], device=device)
    color_tensor = torch.tensor([0], device=device)  # default color for animation
    total_steps = diffusion.n_steps
    if num_frames >= total_steps:
        timesteps = list(range(total_steps))
    else:
        step_size = total_steps // num_frames
        timesteps = list(range(0, total_steps, step_size))
        if timesteps[-1] != total_steps - 1:
            timesteps.append(total_steps - 1)
    if reverse:
        timesteps = sorted(timesteps, reverse=True)
    else:
        timesteps = sorted(timesteps)
        backward_timesteps = sorted(timesteps[1:-1], reverse=True)
        timesteps = timesteps + backward_timesteps
    print(f"Creating diffusion animation for class '{class_names[class_idx]}'...")
    frame_paths = []
    with torch.no_grad():
        print("Generating initial clean image...")
        x = torch.randn((1, autoencoder.latent_dim), device=device)
        for time_step in tqdm(range(total_steps - 1, -1, -1), desc="Denoising"):
            x = diffusion.p_sample(x, torch.tensor([time_step], device=device), class_tensor, color_tensor)
        clean_x = x.clone()
        print("Generating animation frames...")
        for i, t in enumerate(tqdm(timesteps)):
            current_x = clean_x.clone()
            if t > 0:
                torch.manual_seed(seed)
                eps = torch.randn_like(current_x)
                alpha_bar_t = diffusion.alpha_bar[t].reshape(-1, 1)
                current_x = torch.sqrt(alpha_bar_t) * current_x + torch.sqrt(1 - alpha_bar_t) * eps
            decoded = autoencoder.decode(current_x)
            img = decoded[0].cpu().permute(1, 2, 0).numpy()
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(img)
            ax.axis('off')
            progress = (t / total_steps) * 100
            title = f'Class: {class_names[class_idx]} (t={t}, {progress:.1f}% noise)'
            ax.set_title(title)
            frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
            plt.savefig(frame_path, bbox_inches='tight')
            plt.close(fig)
            frame_paths.append(frame_path)
    print(f"Creating GIF animation at {fps} fps...")
    with imageio.get_writer(save_path, mode='I', fps=fps, loop=0) as writer:
        for frame_path in frame_paths:
            image_frame = imageio.imread(frame_path)
            writer.append_data(image_frame)
    print("Cleaning up temporary files...")
    for frame_path in frame_paths:
        os.remove(frame_path)
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass
    print(f"Animation saved to {save_path}")
    return save_path

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:16].to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.feature_extractor = vgg
        self.criterion = euclidean_distance_loss
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))
    def forward(self, x, y):
        device = next(self.parameters()).device
        x = x.to(device)
        y = y.to(device)
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        x_features = self.feature_extractor(x)
        y_features = self.feature_extractor(y)
        return self.criterion(x_features, y_features)

class Discriminator64(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x).view(-1)

# =============================================================================
# TRAINING FUNCTIONS (UPDATED TO UNPACK FLOWER & COLOR LABELS)
# =============================================================================

def train_autoencoder(autoencoder, train_loader, num_epochs=300, lr=1e-4,
                      lambda_cls=0.1, lambda_center=0.05, lambda_vgg=0.4, lambda_gan=0.2,
                      kl_weight_start=0.001, kl_weight_end=0.05,
                      visualize_every=10, save_dir="./results"):
    print("Starting VAE-GAN training with perceptual loss enhancement...")
    os.makedirs(save_dir, exist_ok=True)
    device = next(autoencoder.parameters()).device
    vgg_loss = VGGPerceptualLoss(device)
    optimizer = optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
    discriminator = Discriminator64().to(device)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    gan_criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=num_epochs * len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )
    loss_history = {'total': [], 'recon': [], 'kl': [], 'class': [], 'center': [], 'perceptual': [], 'gan': []}
    best_loss = float('inf')
    lambda_recon = 1.0
    for epoch in range(num_epochs):
        autoencoder.train()
        discriminator.train()
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        epoch_class_loss = 0
        epoch_center_loss = 0
        epoch_perceptual_loss = 0
        epoch_gan_loss = 0
        epoch_total_loss = 0
        kl_weight = min(kl_weight_end, kl_weight_start + (epoch / (num_epochs * 0.6)) * (kl_weight_end - kl_weight_start))
        autoencoder.kl_weight = kl_weight
        print(f"Epoch {epoch + 1}/{num_epochs} - KL Weight: {kl_weight:.6f}")
        # Now each batch returns (data, flower_labels, color_labels) – use flower_labels for classifier loss.
        for batch_idx, (data, flower_labels, color_labels) in enumerate(tqdm(train_loader, desc=f"Training")):
            data = data.to(device)
            flower_labels = flower_labels.to(device)
            valid = torch.ones(data.size(0), device=device)
            fake = torch.zeros(data.size(0), device=device)
            optimizer.zero_grad()
            recon_x, mu, logvar, z = autoencoder(data)
            if epoch < 40:
                kl_factor = 0.0
                cls_factor = 0.0
                center_factor = 0.0
            elif epoch < 80:
                kl_factor = min(1.0, (epoch - 20) / 20)
                cls_factor = 0.0
                center_factor = 0.0
            elif epoch < 160:
                kl_factor = 1.0
                cls_factor = min(0.2, (epoch - 40) / 20)
                center_factor = 0.0
            else:
                kl_factor = 1.0
                cls_factor = 1.0
                center_factor = min(1.0, (epoch - 60) / 20)
            recon_loss = euclidean_distance_loss(recon_x, data)
            perceptual_loss = vgg_loss(recon_x, data)
            kl_loss = autoencoder.kl_divergence(mu, logvar) if kl_factor > 0 else torch.tensor(0.0, device=device)
            class_loss = F.cross_entropy(autoencoder.classify(z), flower_labels) if cls_factor > 0 else torch.tensor(0.0, device=device)
            center_loss = autoencoder.compute_center_loss(z, flower_labels) if center_factor > 0 else torch.tensor(0.0, device=device)
            d_real_loss = gan_criterion(discriminator(data), valid)
            d_fake_loss = gan_criterion(discriminator(recon_x.detach()), fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            adv_loss = gan_criterion(discriminator(recon_x), valid)

            recon_scale = 1.0
            if recon_loss.item() > 1e-8:
                perceptual_scale = min(1.0, recon_loss.item() / (perceptual_loss.item() + 1e-8))
                kl_scale = min(1.0, recon_loss.item() / (kl_loss.item() + 1e-8)) if kl_loss.item() > 0 else 1.0
                gan_scale = min(1.0, recon_loss.item() / (adv_loss.item() + 1e-8))
            else:
                perceptual_scale = 1.0
                kl_scale = 1.0
                gan_scale = 1.0
            total_loss = (
                lambda_recon * recon_scale * recon_loss +
                lambda_vgg * perceptual_scale * perceptual_loss +
                kl_weight * kl_scale * kl_factor * kl_loss +
                lambda_cls * cls_factor * class_loss +
                lambda_center * center_factor * center_loss +
                lambda_gan * gan_scale * adv_loss
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                if epoch >= 60 and center_factor > 0:
                    autoencoder.update_centers(z.detach(), flower_labels, momentum=0.9)
            epoch_recon_loss += recon_loss.item()
            epoch_perceptual_loss += perceptual_loss.item()
            epoch_kl_loss += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else 0
            epoch_class_loss += class_loss.item() if isinstance(class_loss, torch.Tensor) else 0
            epoch_center_loss += center_loss.item() if isinstance(center_loss, torch.Tensor) else 0
            epoch_total_loss += total_loss.item()
            epoch_gan_loss += adv_loss.item()
        num_batches = len(train_loader)
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_perceptual_loss = epoch_perceptual_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        avg_class_loss = epoch_class_loss / num_batches
        avg_center_loss = epoch_center_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches
        avg_gan_loss = epoch_gan_loss / num_batches
        loss_history['recon'].append(avg_recon_loss)
        loss_history['perceptual'].append(avg_perceptual_loss)
        loss_history['kl'].append(avg_kl_loss)
        loss_history['class'].append(avg_class_loss)
        loss_history['center'].append(avg_center_loss)
        loss_history['total'].append(avg_total_loss)
        loss_history['gan'].append(avg_gan_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Recon Lambda * Scale: {lambda_recon * recon_scale:.6f}, "
              f"Perceptual Lambda * Scale: {lambda_vgg * perceptual_scale:.6f}, "
              f"KL Lambda * Scale: {kl_factor * kl_weight:.6f}, GAN Lambda * Scale: {gan_scale * lambda_gan:.6f}")
        print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {avg_total_loss:.6f}, Recon Loss: {avg_recon_loss:.6f}, "
              f"Perceptual Loss: {avg_perceptual_loss:.6f}, KL Loss: {avg_kl_loss:.6f}, GAN Loss: {avg_gan_loss:.6f}, "
              f"Class Loss: {avg_class_loss:.6f}, Center Loss: {avg_center_loss:.6f}")
        if loss_history['total'][-1] < best_loss:
            best_loss = loss_history['total'][-1]
            torch.save({
                'autoencoder': autoencoder.state_dict(),
                'discriminator': discriminator.state_dict(),
            }, f"{save_dir}/vae_gan_best.pt")
        if (epoch + 1) % visualize_every == 0 or epoch == num_epochs - 1:
            visualize_reconstructions(autoencoder, epoch + 1, save_dir)
            visualize_latent_space(autoencoder, epoch + 1, save_dir)
    torch.save({
        'autoencoder': autoencoder.state_dict(),
        'discriminator': discriminator.state_dict(),
    }, f"{save_dir}/vae_gan_final.pt")
    print("Training complete.")
    return autoencoder, discriminator, loss_history

def check_and_normalize_latent(autoencoder, data):
    mu, logvar = autoencoder.encode_with_params(data)
    z = autoencoder.reparameterize(mu, logvar)
    mean = z.mean(dim=0, keepdim=True)
    std = z.std(dim=0, keepdim=True)
    z_normalized = (z - mean) / (std + 1e-8)
    return z_normalized, mean, std

def visualize_latent_comparison(autoencoder, diffusion, data_loader, save_path):
    device = next(autoencoder.parameters()).device
    autoencoder.eval()
    diffusion.eps_model.eval()
    images, labels = next(iter(data_loader))
    images = images.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        mu, logvar = autoencoder.encode_with_params(images)
        z = autoencoder.reparameterize(mu, logvar)
        recon = autoencoder.decode(z)
        latent_shape = (images.size(0), autoencoder.latent_dim)
        # For diffusion sampling in latent space, use default color 0.
        z_denoised = diffusion.sample(latent_shape, device, flower_label=labels, color_label=torch.zeros_like(labels))
        gen = autoencoder.decode(z_denoised)
    n = images.size(0)
    fig, axes = plt.subplots(3, n, figsize=(n * 2, 6))
    for i in range(n):
        axes[0, i].imshow(recon[i].cpu().permute(1, 2, 0).numpy())
        axes[0, i].axis("off")
        axes[1, i].imshow(gen[i].cpu().permute(1, 2, 0).numpy())
        axes[1, i].axis("off")
        axes[2, i].imshow(images[i].cpu().permute(1, 2, 0).numpy())
        axes[2, i].axis("off")
    axes[0, 0].set_title("VAE reconstruction（actual latent）", fontsize=10)
    axes[1, 0].set_title("Diffusion（denoised latent）", fontsize=10)
    axes[2, 0].set_title("Original", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    autoencoder.train()
    diffusion.eps_model.train()

def train_conditional_diffusion(autoencoder, unet, train_loader, num_epochs=100, lr=1e-3, visualize_every=10,
                                save_dir="./results", device=None, start_epoch=0):
    print("Starting Class-Conditional Diffusion Model training with improved strategies...")
    os.makedirs(save_dir, exist_ok=True)
    autoencoder.eval()
    diffusion = ConditionalDenoiseDiffusion(unet, n_steps=1000, device=device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    loss_history = []
    visualization_loader = train_loader
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_loss = 0
        for batch_idx, (data, flower_labels, color_labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}")):
            data = data.to(device)
            flower_labels = flower_labels.to(device)
            # color_labels is now already a tensor of integers (or can be converted easily)
            color_labels = torch.tensor(color_labels, dtype=torch.long).to(device)
            with torch.no_grad():
                mu, logvar = autoencoder.encode_with_params(data)
                z = autoencoder.reparameterize(mu, logvar)
            loss = diffusion.loss(z, flower_labels, color_labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{start_epoch + num_epochs}, Average Loss: {avg_loss:.6f}")
        scheduler.step()
        if (epoch + 1) % visualize_every == 0 or epoch == start_epoch + num_epochs - 1:
            for class_idx in [4, 52]:
                create_diffusion_animation(autoencoder, diffusion, class_idx=class_idx, num_frames=50,
                                           save_path=f"{save_dir}/diffusion_animation_class_{class_names[class_idx]}_epoch_{epoch + 1}.gif")
                sp = f"{save_dir}/sample_class_{class_names[class_idx]}_epoch_{epoch + 1}.png"
                generate_class_samples(autoencoder, diffusion, target_class=class_idx, num_samples=5, save_path=sp)
                sp = f"{save_dir}/sample_class_color_{class_names[class_idx]}_pink_epoch_{epoch + 1}.png"
                generate_class_color_samples(autoencoder, diffusion, target_class=class_idx, target_color="pink", num_samples=5, save_path=sp)
                sp = f"{save_dir}/sample_class_color_{class_names[class_idx]}_purple_epoch_{epoch + 1}.png"
                generate_class_color_samples(autoencoder, diffusion, target_class=class_idx, target_color="purple",
                                             num_samples=5, save_path=sp)
                sp2 = f"{save_dir}/denoising_path_{class_names[class_idx]}_epoch_{epoch + 1}.png"
                visualize_denoising_steps(autoencoder, diffusion, class_idx=class_idx, save_path=sp2)
            torch.save(unet.state_dict(), f"{save_dir}/conditional_diffusion_epoch_{epoch + 1}.pt")
    torch.save(unet.state_dict(), f"{save_dir}/conditional_diffusion_final.pt")
    print(f"Saved final diffusion model after {start_epoch + num_epochs} epochs")
    return unet, diffusion, loss_history

# =============================================================================
# MAIN FUNCTION (ADAPTED FOR FLOWERS102 WITH AUTOMATED COLOR EXTRACTION)
# =============================================================================

def main(checkpoint_path=None, total_epochs=2000):
    print("Starting class-conditional diffusion model for Oxford 102 Flowers with improved architecture")
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    results_dir = "./oxford_flowers_conditional_improved_v3"
    os.makedirs(results_dir, exist_ok=True)
    print("Loading Oxford 102 Flowers dataset with automated color extraction...")
    # Use the custom dataset that returns (image, flower_label, color_label)
    flowers_train = Flowers102WithColor(root='./data', split='train', transform=transform_train, precompute_color=True)

    color_visualization_path = f"{results_dir}/color_visualization.png"
    if flowers_train is not None:
        create_flower_color_visualization(flowers_train, 100, color_visualization_path)

    global class_names
    class_names = flowers_train.flowers.classes if hasattr(flowers_train.flowers, 'classes') else [str(i) for i in range(102)]
    train_loader = DataLoader(flowers_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    autoencoder_path = f"{results_dir}/vae_gan_final.pt"
    diffusion_path = f"{results_dir}/conditional_diffusion_final.pt"
    autoencoder = SimpleAutoencoder(in_channels=3, latent_dim=256, num_classes=102).to(device)
    if os.path.exists(autoencoder_path):
        print(f"Loading existing autoencoder from {autoencoder_path}")
        checkpoint = torch.load(autoencoder_path, map_location=device)
        autoencoder.load_state_dict(checkpoint['autoencoder'], strict=False)
        autoencoder.eval()
    else:
        print("No existing autoencoder found. Training a new one with improved architecture...")
        autoencoder, discriminator, ae_losses = train_autoencoder(
            autoencoder,
            train_loader,
            num_epochs=1200,
            lr=1e-4,
            lambda_cls=0.3,
            lambda_center=0.1,
            lambda_vgg=0.4,
            visualize_every=50,
            save_dir=results_dir
        )
        torch.save(autoencoder.state_dict(), autoencoder_path)
        plt.figure(figsize=(10, 6))
        plt.plot(ae_losses['total'], label='Total Loss')
        plt.plot(ae_losses['recon'], label='Reconstruction Loss')
        plt.plot(ae_losses['kl'], label='KL Loss')
        plt.plot(ae_losses['class'], label='Classification Loss')
        plt.plot(ae_losses['center'], label='Center Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Autoencoder Training Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{results_dir}/autoencoder_losses.png")
        plt.close()
    # Build the conditional UNet with multi-condition (flower type and color)
    conditional_unet = ConditionalUNet(
        latent_dim=256,
        hidden_dims=[256, 512, 1024, 512, 256],
        time_emb_dim=256,
        num_classes=102,
        num_colors=10,
        dropout_rate=0.3
    ).to(device)
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            filename = os.path.basename(checkpoint_path)
            epoch_str = filename.split("epoch_")[1].split(".pt")[0]
            start_epoch = int(epoch_str)
            print(f"Continuing training from epoch {start_epoch}")
            conditional_unet.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
            diffusion = ConditionalDenoiseDiffusion(conditional_unet, n_steps=1000, device=device)
        except (IndexError, ValueError) as e:
            print(f"Could not extract epoch number from checkpoint filename: {e}")
            print("Starting from epoch 0")
            start_epoch = 0
    elif os.path.exists(diffusion_path):
        print(f"Loading existing diffusion model from {diffusion_path}")
        conditional_unet.load_state_dict(torch.load(diffusion_path, map_location=device))
        diffusion = ConditionalDenoiseDiffusion(conditional_unet, n_steps=1000, device=device)
    else:
        print("No existing diffusion model found. Training a new one with improved architecture...")
        conditional_unet.apply(init_weights)
    remaining_epochs = total_epochs - start_epoch
    if 'diffusion' not in globals():
        conditional_unet, diffusion, diff_losses = train_conditional_diffusion(
            autoencoder, conditional_unet, train_loader, num_epochs=remaining_epochs, lr=1e-3,
            visualize_every=50,
            save_dir=results_dir,
            device=device,
            start_epoch=start_epoch
        )
        torch.save(conditional_unet.state_dict(), diffusion_path)
        plt.figure(figsize=(8, 5))
        plt.plot(diff_losses)
        plt.title('Diffusion Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{results_dir}/diffusion_loss.png")
        plt.close()
    elif start_epoch > 0:
        conditional_unet, diffusion, diff_losses = train_conditional_diffusion(
            autoencoder, conditional_unet, train_loader, num_epochs=remaining_epochs, lr=1e-3,
            visualize_every=50,
            save_dir=results_dir,
            device=device,
            start_epoch=start_epoch
        )
        torch.save(conditional_unet.state_dict(), diffusion_path)
        plt.figure(figsize=(8, 5))
        plt.plot(range(start_epoch + 1, start_epoch + len(diff_losses) + 1), diff_losses)
        plt.title('Diffusion Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{results_dir}/diffusion_loss_continued.png")
        plt.close()
    print("Generating sample grid for a subset of classes...")
    grid_path = generate_samples_grid(autoencoder, diffusion, n_per_class=5, save_dir=results_dir)
    print(f"Sample grid saved to: {grid_path}")
    print("Generating denoising visualizations for a subset of classes...")
    denoising_paths = []
    for class_idx in range(min(len(class_names), 10)):
        sp = f"{results_dir}/denoising_path_{class_names[class_idx]}_final.png"
        path = visualize_denoising_steps(autoencoder, diffusion, class_idx, save_path=sp)
        denoising_paths.append(path)
        print(f"Generated visualization for {class_names[class_idx]}")
    print("Creating animations for a subset of classes...")
    for class_idx in range(min(len(class_names), 10)):
        animation_path = create_diffusion_animation(
            autoencoder, diffusion, class_idx=class_idx,
            num_frames=50, fps=15,
            save_path=f"{results_dir}/diffusion_animation_{class_names[class_idx]}_final.gif"
        )
        print(f"Created animation for {class_names[class_idx]}: {animation_path}")
    print("\nAll visualizations and models complete!")
    print(f"Results directory: {results_dir}")
    print(f"Sample grid: {grid_path}")
    print("Denoising visualizations:")
    for i, path in enumerate(denoising_paths):
        print(f"  - {class_names[i]}: {path}")

if __name__ == "__main__":
    main(total_epochs=10000)
