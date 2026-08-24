# ATML Assignment 0

Code for **Assignment 0 — Advanced Topics in Machine Learning (EE-5102/CS-6304)**.

The assignment explores visual and multimodal representation learning using **ResNet-152, Vision Transformers (ViT), CLIP, and Variational Autoencoders (VAE)**.

## Repository Structure

```text
task1-resnet/
├── baseline.ipynb
├── skip_connections.ipynb
├── feature_analysis.ipynb
└── transfer_learning.ipynb

task2-vit/
├── task2_vit.ipynb
└── images/

task3-clip/
└── task3_clip.ipynb

task4-vae/
└── task4_vae.ipynb

Written_Notes.pdf
```

## Tasks

**Task 1 — ResNet-152**  
Frozen-backbone transfer learning, residual-connection ablation, feature visualization using t-SNE, and comparison of fine-tuning strategies on CIFAR-10.

**Task 2 — Vision Transformer**  
Image classification, patch-attention visualization, attention-head analysis, patch masking, and CLS versus mean-pooled representations.

**Task 3 — CLIP**  
Zero-shot classification on STL-10, prompt comparison, image-text modality-gap analysis, and Orthogonal Procrustes alignment.

**Task 4 — Variational Autoencoder**  
VAE implementation on MNIST, latent-space visualization, image reconstruction, and generation.

## Environment

The experiments were implemented in **Python/PyTorch**. Computationally intensive experiments were run using **Google Colab**.

Datasets and pretrained model weights are downloaded through the corresponding libraries when the notebooks are executed.

## Running

Open the required notebook locally or in Google Colab and run its cells sequentially. The notebooks have been saved with their experimental outputs for direct inspection.