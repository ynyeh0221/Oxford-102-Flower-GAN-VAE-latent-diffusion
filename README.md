# Flower-GAN-VAE-Diffusion

A class-conditional generative modeling framework combining Variational Autoencoders (VAE) and Denoising Diffusion Probabilistic Models (DDPM), developed for the Oxford 102 Flowers dataset. This project integrates advanced components such as attention mechanisms, center loss regularization, VGG perceptual loss, and adversarial training to enable high-fidelity image generation and structured latent representation.

## Features

### VAE with Enhanced Architecture
- Residual blocks with Swish activation
- Channel and spatial attention modules
- LayerNorm for 2D convolutions
- Center loss with repulsion for flat latent distributions
- Class-conditional classifier head

### VAE-GAN Training
- Pixel-wise reconstruction loss
- Perceptual loss using pretrained VGG16 features
- Adversarial loss with PatchGAN-style discriminator

### Conditional Diffusion Model
- Diffusion operates in VAE latent space
- Temporal and class conditioning via embeddings
- Multi-head self-attention over latent representations
- Class-conditional sample generation

### Visualization Tools
- Reconstructions and denoising step plots
- Latent space visualizations (t-SNE, PCA)
- Animated GIFs of the denoising process
- Class-wise sample grids for visual inspection

## Dataset

- **Oxford 102 Flowers** dataset: 102 flower categories with segmented images.
- Automatically downloaded via `torchvision.datasets.Flowers102`.

## Training

### Autoencoder Training

```bash
python main.py --total_epochs 1200
```

Includes reconstruction, perceptual, classification, KL divergence, center loss, and adversarial losses. Visualizations are saved every `visualize_every` epochs.

### Conditional Diffusion Training

```bash
python main.py --checkpoint_path ./results/conditional_diffusion_epoch_600.pt --total_epochs 2000
```

Trains a transformer-inspired UNet operating in latent space to denoise class-conditioned samples.

## Output

- `vae_gan_best.pt`, `conditional_diffusion_final.pt`: Model checkpoints
- `autoencoder_losses.png`, `diffusion_loss.png`: Training loss plots
- `vae_samples_grid_subset.png`: Grid of generated samples per class
- `diffusion_animation_<class>.gif`: Animated denoising per class
- `denoising_path_<class>.png`: Latent space path during generation

Below example results were generated from [V2 model of this repo](https://github.com/ynyeh0221/Oxford-102-Flower-GAN-VAE-latent-diffusion/tree/main/v2).

 Model Component | Visualization | Description |
|-----------------|---------------|-------------|
| Autoencoder | ![Reconstructions](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v1/output/reconstruction/vae_reconstruction_epoch_1400.png) | Original images (top) and their reconstructions (bottom) |
| Latent Space | ![Latent Space](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v1/output/latent_space/vae_latent_space_epoch_1400.png) | t-SNE visualization of 1 to 10 classes latent representations |
| Class Samples | ![Class Samples](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v2/output/diffusion_sample_result/sample_class_4_epoch_1000.png)![Class Samples](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v2/output/diffusion_sample_result/sample_class_6_epoch_1000.png) | Generated samples for 4, 6 classes |
| Denoising Process | ![Denoising Class 0](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v2/output/diffusion_path/denoising_path_4_epoch_1000.png)![Denoising Class 1](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v2/output/diffusion_path/denoising_path_6_epoch_1000.png) | Visualization of 4, 6 classes generation process and latent path |
| Animation | ![Class 0 Animation](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v2/output/diffusion_animation_class_4_epoch_1000.gif)![Class 1 Animation](https://github.com/ynyeh0221/Oxford-120-Flower-GAN-VAE-latent-diffusion/blob/main/v2/output/diffusion_animation_class_6_epoch_1000.gif) | Animation of the denoising process for 4, 6 classes generation |

## Getting Started

1. Clone this repo and install dependencies (PyTorch, torchvision, scikit-learn, imageio, etc.).
2. Run `main.py` to train or resume training.
3. Visualizations and outputs are saved under `./results`.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- scikit-learn
- matplotlib
- tqdm
- imageio
