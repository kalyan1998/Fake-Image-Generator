# Comparative Analysis of GAN Models: DCGAN, WGAN, and ACGAN

![Project Banner](https://via.placeholder.com/800x200.png?text=GAN+Comparative+Analysis)

## Project Description

This project examines three significant Generative Adversarial Network (GAN) models: Deep Convolutional GAN (DCGAN), Wasserstein GAN (WGAN), and Auxiliary Classifier GAN (ACGAN). The analysis focuses on understanding their architectural nuances, performance metrics, and the quality of generated images using the CIFAR-10 dataset.

## Workflow Overview

1. **Data Collection and Preparation**
   - **Dataset**: CIFAR-10 dataset containing 60,000 32x32 colored images across 10 categories.
   - **Preprocessing Steps**: Data is split into 50,000 training and 10,000 testing images.

2. **Model Architectures**
   - **DCGAN (Deep Convolutional GAN)**
     - **Generator**: Processes text input through linear layers and merges with noise vectors, feeding into transpose convolutional layers with batch normalization and ReLU activation, generating RGB images with Tanh normalization.
     - **Discriminator**: Processes images and text through convolutional layers with LeakyReLU activation and batch normalization, combining features through linear layers to produce authenticity scores.
     - **Hyperparameters**:
       - Optimizer: Adam, learning rate = 0.0002
       - Batch Size: 128
       - Noise Dimension: 100
       - Text Embedding Dimension: 256

   - **WGAN (Wasserstein GAN)**
     - **Generator**: Similar to DCGAN but optimized for Wasserstein distance.
     - **Critic (Discriminator)**: Outputs scores representing "realness" rather than probabilities, using gradient penalty for stability.
     - **Hyperparameters**:
       - Learning Rate: 0.0002
       - Critic Iterations: 5 per generator iteration
       - Gradient Penalty Weight: 10
       - Noise Dimension: 100
       - Epochs: 40

   - **ACGAN (Auxiliary Classifier GAN)**
     - **Generator**: Combines noise vectors with class labels, generating images through transpose convolutional layers with batch normalization and ReLU activation.
     - **Discriminator**: Outputs both validity scores and class predictions through convolutional layers, with heads for validity (sigmoid activation) and class predictions (log-softmax activation).
     - **Hyperparameters**:
       - Number of Classes: 10
       - Noise Dimension: 100
       - Batch Size: 128
       - Learning Rate: 0.0002
       - Beta1: 0.5
       - Epochs: 40

3. **Training and Evaluation**
   - **Training**: Models trained on CIFAR-10 dataset with visualizations of loss and FID (Fréchet Inception Distance) scores over epochs.
   - **Evaluation**: FID scores calculated using InceptionV3 model to compare statistical similarity between real and generated images.

## File Descriptions

### `dcgan.py`
- Implements DCGAN model with training and evaluation functions.

### `wgan.py`
- Implements WGAN model with training and evaluation functions.

### `acgan.py`
- Implements ACGAN model with training and evaluation functions.

### `fid_score.py`
- Calculates FID scores using InceptionV3 model.

## Instructions to Execute Code

### 1) Training the Models

To train the models, run the respective scripts with appropriate parameters. For example:
```bash
python dcgan.py --train --data_dir /path/to/cifar10
python wgan.py --train --data_dir /path/to/cifar10
python acgan.py --train --data_dir /path/to/cifar10
```

### 2) Calculating FID Scores
To calculate FID scores, use the fid_score.py script:
```bash
python fid_score.py --real /path/to/real_images --fake /path/to/generated_images
```

### 3)  Downloading Necessary Files
Ensure you have all necessary files, including pretrained models and datasets. Pretrained models can be downloaded from https://drive.google.com/drive/u/1/folders/1eP2yFAuo-JMGP-redinPNHFU47IiKqQb
