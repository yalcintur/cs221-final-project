# AESAC: Autoencoder-Soft Actor-Critic for V2G Profit Maximization

## Overview

AESAC combines **Autoencoders** with the **Soft Actor-Critic (SAC)** algorithm to optimize Vehicle-to-Grid (V2G) profit maximization in electric vehicle (EV) charging station management. By leveraging dimensionality reduction through autoencoders and the stability of SAC in continuous action spaces, AESAC efficiently manages high-dimensional state spaces and dynamic environments.

## Features

- **Autoencoder for Dimensionality Reduction:** Compresses high-dimensional state vectors into a compact latent space.
- **Soft Actor-Critic (SAC) Integration:** Optimizes policies in continuous action spaces with enhanced stability and efficiency.
- **EV2Gym Simulation Environment:** Realistic modeling of EV charging scenarios for robust training and evaluation.
- **AFAP Heuristic:** Generates extensive datasets to train the autoencoder effectively.


## Usage

### Train the Autoencoder
To train the autoencoder using the AFAP heuristic-generated dataset:
```bash
bash scripts/start_training_ae.sh

To train the SAC agent within the EV2Gym environment:

```bash
bash scripts/start_training_rl.sh

## Configuration

input_dim: 112
latent_dim: 32
hidden_dims: [128, 64]
dropout_rate: 0.2
activation: ReLU
learning_rate: 0.002
epochs: 50
batch_size: 256
