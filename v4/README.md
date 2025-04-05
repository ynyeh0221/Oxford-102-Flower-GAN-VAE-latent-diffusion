# Pixel Space Diffusion Model

This repository contains an implementation of a diffusion model that operates in pixel space to generate flower images. The model is trained on the Flowers102 dataset and can generate realistic flower images through a denoising diffusion process.

## Overview

Diffusion models are a class of generative models that work by gradually adding noise to data and then learning to reverse this process. This implementation uses a UNet architecture with time conditioning to model the reverse diffusion process.

## Features

- Implementation of a diffusion model in PyTorch
- Custom UNet architecture with time conditioning
- Training on the Flowers102 dataset
- Image generation from random noise
- Visualization tools for generated samples
- Animation creation to visualize the denoising process

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- imageio

## Installation

```bash
pip install torch torchvision numpy matplotlib imageio
```

## Usage

### Training the model

```bash
python main.py
```

This will:
1. Download the Flowers102 dataset
2. Initialize a UNet model
3. Train the diffusion model for 300 epochs
4. Save the trained model weights to `diffusion_unet_pixels.pth`

### Model Architecture

The implementation uses a simplified UNet architecture with:
- Time embedding for conditioning the model on diffusion timesteps
- Skip connections between encoder and decoder blocks
- A configurable number of channels and embedding dimensions

### Diffusion Process

The diffusion process is implemented with:
- Forward process (q_sample): Gradually adds noise to images according to a fixed schedule
- Reverse process (p_sample): Removes noise step by step using the UNet model
- Loss calculation based on predicting the added noise

## Outputs

The script generates several outputs:
- `diffusion_unet_pixels.pth`: Trained model weights
- `samples_grid.png`: Grid of generated sample images
- `diffusion_animation.gif`: Animation showing the denoising process
- `generated_pixel_diffusion.png`: A single generated image

| | |
|-----------------|---------------|
| Epoch 30 | ![]() |
| Epoch 60 | ![]() |
| Epoch 90 | ![]() |
| Epoch 120 | ![]() |
| Epoch 150 | ![]() |
| Epoch 180 | ![]() |
| Epoch 210 | ![]() |
| Epoch 240 | ![]() |
| Epoch 270 | ![]() |

## Customization

You can customize the model by modifying these parameters:
- `img_size`: Image resolution (default: 64x64)
- `batch_size`: Batch size for training (default: 64)
- `n_steps`: Number of diffusion steps (default: 1000)
- `beta_start` and `beta_end`: Noise schedule parameters
- `base_channels`: Base number of channels in the UNet (default: 64)
- `time_emb_dim`: Dimension of time embeddings (default: 128)
- `num_epochs`: Number of training epochs (default: 300)

## Hardware Acceleration

The code automatically selects the available hardware acceleration:
- CUDA for NVIDIA GPUs
- MPS for Apple Silicon
- Falls back to CPU if neither is available

## Acknowledgments

This implementation is inspired by research on diffusion models:
- [Denoising Diffusion Probabilistic Models (Ho et al.)](https://arxiv.org/abs/2006.11239)
- [Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal)](https://arxiv.org/abs/2102.09672)
