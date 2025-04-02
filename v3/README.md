# Multi-Conditional Flower Generation with VAE-Diffusion

This repository implements a novel multi-conditional generative model for high-quality flower image synthesis using a combination of Variational Autoencoder (VAE) and Diffusion Models. The system can generate realistic flower images conditioned on both flower type and color.

## Overview

This project builds a generative model for the Oxford 102 Flowers dataset with advanced conditioning capabilities. Unlike traditional conditional generative models that only condition on class labels, our approach adds automatic color extraction to enable multi-conditional generation - allowing users to specify both the flower type and desired color for generation.

## Features

- **Multi-conditional generation**: Control both flower type and color during generation
- **Automated color extraction**: K-means clustering to automatically extract dominant colors from flowers
- **Enhanced architecture**: Attention mechanisms, layer normalization, and swish activations
- **Comprehensive visualizations**: Sample grids, latent space projections, and denoising animations
- **Perceptual loss**: VGG-based perceptual loss for improved visual quality

## Architecture

The system consists of three main components:

1. **Variational Autoencoder (VAE)**:
   - Encoder with residual blocks and attention mechanisms
   - Decoder with skip connections for improved reconstruction
   - Classification head for flower type recognition

2. **Multi-Condition Embedding**:
   - Joint embedding of flower type and color information
   - Mapping categorical information to continuous representations

3. **Conditional Diffusion Model**:
   - UNet-based noise prediction network
   - Time embedding to capture denoising dynamics
   - Attention-enhanced architecture

## Sample Results

The model can generate realistic flower images based on:
- Flower type only (e.g., "rose", "daisy")
- Flower type and color (e.g., "red rose", "yellow daisy")

## Training Pipeline

### Step 1: VAE Training
```python
autoencoder, discriminator, loss_history = train_autoencoder(
    autoencoder, 
    train_loader,
    num_epochs=100,
    lr=1e-4,
    lambda_cls=0.3,
    lambda_center=0.1,
    lambda_vgg=0.4,
    visualize_every=20,
    save_dir=results_dir
)
```

### Step 2: Diffusion Model Training
```python
conditional_unet, diffusion, diff_losses = train_conditional_diffusion(
    autoencoder, 
    conditional_unet, 
    train_loader, 
    num_epochs=remaining_epochs, 
    lr=1e-3,
    visualize_every=50,
    save_dir=results_dir,
    device=device,
    start_epoch=start_epoch
)
```

## Key Innovations

- **Dual-conditioning mechanism**: Enables fine-grained control over generated images
- **Automated color extraction**: Enriches the dataset with color information without manual labeling
- **Enhanced attention**: Spatial and channel attention to capture complex flower patterns
- **Progressive training strategy**: Gradually introducing different loss components

## Requirements

- PyTorch
- torchvision
- tqdm
- scikit-learn
- matplotlib
- numpy
- imageio

## Usage

### Training the model
```python
python main.py --total_epochs 10000
```

### Generating images with specific conditions
```python
# Generate images of a specific flower type
samples = generate_class_samples(autoencoder, diffusion, target_class="rose", num_samples=5)

# Generate images with specific flower type and color
samples = generate_class_color_samples(autoencoder, diffusion, target_class="rose", target_color="red", num_samples=5)
```

### Creating visualizations
```python
# Generate a grid of samples for different classes
grid_path = generate_samples_grid(autoencoder, diffusion, n_per_class=5)

# Visualize the denoising process
path = visualize_denoising_steps(autoencoder, diffusion, class_idx=4)

# Create an animation of the denoising process
animation_path = create_diffusion_animation(autoencoder, diffusion, class_idx=5)
```

## Visualization Types

1. **Sample Grids**: Compare generated images across different flower classes
2. **Denoising Visualizations**: Show the progressive transformation from noise to image
3. **Latent Space Projections**: Visualize the structure of the learned latent space
4. **Diffusion Animations**: Animated visualization of the denoising process

 Model Component | Visualization | Description |
|-----------------|---------------|-------------|
| Autoencoder | ![Reconstructions](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v3/output/reconstruction/vae_reconstruction_epoch_2000.png) | Original images (top) and their reconstructions (bottom) |
| Latent Space | ![Latent Space](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v3/output/latent_space/vae_latent_space_epoch_2000.png) | t-SNE visualization of 4, 53, 68 latent representations |
| Class Samples | ![Class 4 Samples](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v3/output/diffusion_sample_result/sample_class_4_epoch_2750.png)![Class 53 Samples](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v3/output/diffusion_sample_result/sample_class_53_epoch_2750.png)![Class 68 Samples](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v3/output/diffusion_sample_result/sample_class_68_epoch_2750.png) | Generated samples for 4, 53, 68 classes |
| Class Color Samples | ![Class 4 Samples](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v3/output/diffusion_sample_result/sample_class_4_epoch_2750.png)![Class 53 Samples](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v3/output/diffusion_sample_result/sample_class_53_epoch_2750.png)![Class 68 Samples](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v3/output/diffusion_sample_result/sample_class_68_epoch_2750.png) | Generated samples for 4, 53, 68 classes and purple, yellow color |
| Denoising Process | ![Denoising Class 4](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v3/output/diffusion_path/denoising_path_4_epoch_2750.png)![Denoising Class 53](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v3/output/diffusion_path/denoising_path_53_epoch_2750.png)![Denoising Class 68](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v3/output/diffusion_path/denoising_path_68_epoch_2750.png) | Visualization of cat generation process and latent path |
| Animation | ![Class 4 Animation](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v3/output/diffusion_animation_class_4_epoch_2750.gif)![Class 53 Animation](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v3/output/diffusion_animation_class_53_epoch_2750.gif)![Class 68 Animation](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v3/output/diffusion_animation_class_68_epoch_2750.gif) | Animation of the denoising process for 4, 53, 68 classes generation |

## License

This project is licensed under the MIT License - see the LICENSE file for details.
