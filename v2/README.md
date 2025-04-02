# Class-Conditional Diffusion with GAN VAE for Oxford Flowers 102

This repository implements a **class-conditional diffusion model** in the latent space of a **Variational Autoencoder (VAE)** for the **Oxford Flowers 102** dataset. It integrates a VAE-GAN architecture enhanced with perceptual and center losses, and a custom diffusion model trained to denoise latent vectors conditioned on class labels.

## Features

- **Custom VAE architecture** with:
  - Residual blocks
  - Channel and spatial attention
  - LayerNorm for stability
- **Center loss** and **KL regularization** to improve latent space structure
- **VAE-GAN** training using:
  - Euclidean loss
  - Perceptual loss (VGG16)
  - GAN adversarial loss
- **Class-conditional diffusion model** trained in latent space
- Multiple **visualization utilities**:
  - Latent space exploration with t-SNE/PCA
  - Denoising path tracking
  - Sample grids and GIF animations

## Dataset

This project uses the [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) dataset, automatically downloaded via `torchvision.datasets.Flowers102`.

## Architecture Overview

- **Autoencoder**:
  - Encoder downsamples 64×64 images to a latent vector (dim = 256)
  - Decoder reconstructs images from latent vectors
  - Integrated classifier to predict flower class from latent vectors

- **Diffusion Model**:
  - Conditional UNet-like architecture adapted for 1D latent vectors
  - Time and class embeddings injected at each layer
  - Trained using standard denoising score matching

## Training

### 1. Train the VAE Autoencoder

```bash
python main.py
```

If no existing checkpoints are found, this will train the VAE from scratch and save results to:

```
./oxford_flowers_conditional_improved/
```

### 2. Train the Conditional Diffusion Model

After training the VAE, the diffusion model will be trained on latent vectors. Progress is visualized via sample images and latent path diagrams.

## Checkpoints

- `flowers_autoencoder.pt`: Trained VAE model
- `conditional_diffusion_final.pt`: Final trained diffusion model
- Intermediate checkpoints are saved during training

## Visualizations

Visual assets are saved in the results directory:

- **`vae_samples_grid_subset.png`**: Generated images per class
- **`vae_reconstruction_epoch_*.png`**: Input vs reconstruction
- **`vae_latent_space_epoch_*.png`**: t-SNE of latent embeddings
- **`diffusion_animation_*.gif`**: Animated denoising paths
- **`denoising_path_*.png`**: PCA of denoising path in latent space

 Model Component | Visualization | Description |
|-----------------|---------------|-------------|
| Autoencoder | ![Reconstructions](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v1/output/reconstruction/vae_reconstruction_epoch_1400.png) | Original images (top) and their reconstructions (bottom) |
| Latent Space | ![Latent Space](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v1/output/latent_space/vae_latent_space_epoch_1400.png) | t-SNE visualization of 1 to 10 classes latent representations |
| Class Samples | ![Class Samples](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v2/output/diffusion_sample_result/sample_class_4_epoch_1000.png)![Class Samples](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v2/output/diffusion_sample_result/sample_class_6_epoch_1000.png) | Generated samples for 4, 6 classes |
| Denoising Process | ![Denoising Class 0](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v2/output/diffusion_path/denoising_path_4_epoch_1000.png)![Denoising Class 1](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v2/output/diffusion_path/denoising_path_6_epoch_1000.png) | Visualization of 4, 6 classes generation process and latent path |
| Animation | ![Class 0 Animation](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v2/output/diffusion_animation_class_4_epoch_1000.gif)![Class 1 Animation](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v2/output/diffusion_animation_class_6_epoch_1000.gif) | Animation of the denoising process for 4, 6 classes generation |

## Dependencies

Ensure the following libraries are installed:

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm imageio
```

Optional for faster t-SNE:

```bash
pip install openTSNE
```

## File Structure

- `main.py`: Entry point for training
- `train_autoencoder(...)`: Trains VAE-GAN with classification and center loss
- `train_conditional_diffusion(...)`: Trains the conditional diffusion model
- `visualize_*.py`: Functions to visualize reconstructions, latent spaces, denoising, etc.

## Notes

- Trains on 64×64 resized images for speed and memory efficiency
- You can resume training from a specific epoch by providing a checkpoint
- Designed to run on GPU (`cuda` or `mps`); falls back to CPU if unavailable

