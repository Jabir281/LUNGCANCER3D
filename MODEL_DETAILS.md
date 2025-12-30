# LungCancer3DCNN Model Documentation

## Overview
The **LungCancer3DCNN** is a specialized deep learning model designed for the binary classification of 3D lung CT scan patches. It determines whether a given 3D volume of lung tissue contains a cancerous nodule (Positive) or is normal (Negative).

## Architecture Design
The model is a **Hybrid 3D Convolutional Neural Network (CNN)** that integrates **Attention Mechanisms** to improve performance. It is designed to process volumetric data (3D images) rather than standard 2D images.

### Key Components

1.  **Input Layer**
    *   **Input Shape:** `(Batch_Size, 1, Depth, Height, Width)`
    *   Takes a single-channel 3D volume (grayscale CT scan).

2.  **Convolutional Blocks (Feature Extraction)**
    The model consists of four main convolutional stages. Each stage increases the depth of the feature maps while reducing the spatial dimensions.
    *   **Block 1:** 1 -> 32 channels.
    *   **Block 2:** 32 -> 64 channels.
    *   **Block 3:** 64 -> 128 channels.
    *   **Block 4:** 128 -> 256 channels.
    *   *Each block typically contains:* `Conv3d` -> `BatchNorm3d` -> `ReLU` activation.

3.  **Attention Mechanisms (The "Hybrid" Part)**
    To make the model smarter, two types of attention modules are inserted between the convolutional blocks. These help the model focus on the most important parts of the scan.
    
    *   **Channel Attention 3D (SE-Block adapted):**
        *   *Function:* Focuses on **"What"** is important.
        *   *Mechanism:* It looks at the relationships between different feature channels and boosts the ones that contain relevant features (like edges or textures specific to nodules) while suppressing irrelevant ones.
    
    *   **Spatial Attention 3D (CBAM inspired):**
        *   *Function:* Focuses on **"Where"** the important features are.
        *   *Mechanism:* It creates a spatial map of importance, helping the model focus on the specific location of the nodule within the 3D volume and ignore the background lung tissue.

4.  **Global Pooling & Classification Head**
    *   **Global Average Pooling:** Converts the complex 3D feature maps into a single vector of numbers (feature vector).
    *   **Fully Connected Layers (MLP):**
        *   Linear Layer (256 -> 128) + ReLU + Dropout
        *   Linear Layer (128 -> 64) + ReLU + Dropout
        *   **Output Layer:** A single neuron that outputs a raw "logit" score.

## Technical Specifications

*   **Framework:** PyTorch
*   **Input Data:** `.npy` files (NumPy arrays) representing 3D volumes.
*   **Loss Function:** `BCEWithLogitsLoss` (Binary Cross Entropy).
*   **Optimizer:** Adam (Adaptive Moment Estimation).
*   **Mixed Precision:** Supports `torch.amp.autocast` for faster training on modern GPUs (like A100).

## How it Works
1.  **Input:** A 3D chunk of a CT scan is fed into the model.
2.  **Processing:** The 3D CNN layers extract shapes and textures. The Attention layers highlight the suspicious regions.
3.  **Output:** The model outputs a single number.
    *   We apply a **Sigmoid** function to this number to get a probability between 0 and 1.
    *   **> 0.5:** Classified as Cancer.
    *   **< 0.5:** Classified as Normal.
