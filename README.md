# Flower-GAN-VAE-Diffusion

A class-conditional generative modeling framework combining Variational Autoencoders (VAE) and Denoising Diffusion Probabilistic Models (DDPM), developed for the Oxford 102 Flowers dataset. This project integrates advanced components such as attention mechanisms, center loss regularization, VGG perceptual loss, and adversarial training to enable high-fidelity image generation and structured latent representation.

[DEEPWIKI](https://deepwiki.com/ynyeh0221/Oxford-102-Flower-GAN-VAE-latent-diffusion)[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ynyeh0221/Oxford-102-Flower-GAN-VAE-latent-diffusion)

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
- Images are cached to the default `~/.cache/torch` directory on first run.

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

1. Clone this repository.
2. Set up the Python environment and install dependencies (see [Installation](#installation)).
3. Run `main.py` to train or resume training.
4. Visualizations and model checkpoints are written to the `./results` directory.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- scikit-learn
- matplotlib
- tqdm
- imageio

## Installation

1. Create a Python 3.8+ environment with your favorite tool (conda, venv, etc.).
2. Install the packages listed below, e.g.:

```bash
pip install torch torchvision scikit-learn matplotlib tqdm imageio
```

## Repository Structure

- `v1` – initial VAE-diffusion implementation
- `v2` – improved training utilities and logging
- `v3` – multi-conditional generation with color conditioning
- `v4` – pixel-space diffusion baseline
- `v5` – latest experimental code

Each folder contains a standalone `model_train_test.py` and example outputs.
