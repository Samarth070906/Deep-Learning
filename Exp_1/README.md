# TensorFlow and Keras: MNIST Digit Classifier

A walkthrough of the core deep learning workflow using TensorFlow and Keras: loading a dataset, splitting into train/test sets, normalizing pixel values, building and training a neural network, and evaluating its performance.

## Overview

This notebook demonstrates the standard pipeline for an image classification task using the MNIST handwritten digits dataset:

1. **Load Data** - Import the MNIST dataset directly from `tf.keras.datasets`
2. **Preprocess** - Normalize pixel values from the 0-255 range to 0-1
3. **Visualize** - Inspect dataset shape and preview sample images
4. **Build Model** - A feedforward neural network (Flatten, Dense with 128 units and ReLU activation, Dense with 10 units and softmax activation)
5. **Train** - Fit the model over 3 epochs using the Adam optimizer
6. **Evaluate** - Measure accuracy on unseen test data
7. **Save** - Export the trained model as `my_mnist_model.keras`

## Tech Stack

- Python
- TensorFlow / Keras
- Matplotlib

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook "Tensorflow and Keras.ipynb"
```

## Model Architecture

| Layer | Type | Output Shape | Notes |
|---|---|---|---|
| 1 | Flatten | (784,) | Converts 28x28 images to 1D |
| 2 | Dense | (128,) | ReLU activation |
| 3 | Dense | (10,) | Softmax activation, 10 digit classes |

Optimizer: Adam. Loss: Sparse Categorical Crossentropy. Metric: Accuracy.

## Results

After 3 epochs, the model achieves solid accuracy on the MNIST test set. Exact figures will vary slightly between runs; check the notebook output for the `Test accuracy` value from your run.

## Repository Structure

```
├── Tensorflow and Keras.ipynb   # Main notebook
├── my_mnist_model.keras         # Saved trained model (generated after running)
└── README.md
```

