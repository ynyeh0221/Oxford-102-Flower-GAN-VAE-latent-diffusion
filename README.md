# Flower-VAE-Diffusion

A class-conditional generative modeling framework combining Variational Autoencoders (VAE) and Denoising Diffusion Probabilistic Models (DDPM), developed for the Oxford 102 Flowers dataset. This project integrates advanced components such as attention mechanisms, center loss regularization, VGG perceptual loss, and adversarial training to enable high-fidelity image generation and structured latent representation.

---

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

---

## Dataset

- **Oxford 102 Flowers** dataset: 102 flower categories with segmented images.
- Automatically downloaded via `torchvision.datasets.Flowers102`.

---

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

---

## Output

- `vae_gan_best.pt`, `conditional_diffusion_final.pt`: Model checkpoints
- `autoencoder_losses.png`, `diffusion_loss.png`: Training loss plots
- `vae_samples_grid_subset.png`: Grid of generated samples per class
- `diffusion_animation_<class>.gif`: Animated denoising per class
- `denoising_path_<class>.png`: Latent space path during generation

---

## Getting Started

1. Clone this repo and install dependencies (PyTorch, torchvision, scikit-learn, imageio, etc.).
2. Run `main.py` to train or resume training.
3. Visualizations and outputs are saved under `./results`.

---

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- scikit-learn
- matplotlib
- tqdm
- imageio

---
