# Multilayer Perceptron (MLP) for Iris Dataset Classification

## Overview

This project demonstrates the implementation of a **Multilayer Perceptron (MLP)** using **TensorFlow/Keras** to classify the **Iris dataset**. The dataset is preprocessed, trained using a neural network, and evaluated using **accuracy** and a **confusion matrix**.

---

## Objective

- Implement a Multilayer Perceptron (MLP) for multi-class classification.
- Train the model using TensorFlow/Keras.
- Evaluate the model using accuracy and a confusion matrix.
- Visualize the training performance.

---

## Dataset

**Dataset:** Iris Dataset

The Iris dataset contains **150 flower samples** belonging to three species:

- Setosa
- Versicolor
- Virginica

Each sample contains four input features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Target classes:

- 0 → Setosa
- 1 → Versicolor
- 2 → Virginica

---

## Technologies Used

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Libraries Required

```python
numpy
matplotlib
tensorflow
scikit-learn
```

Install the required libraries using:

```bash
pip install numpy matplotlib tensorflow scikit-learn
```

---

## Project Workflow

### 1. Import Libraries

Import the required libraries for data preprocessing, visualization, model building, and evaluation.

---

### 2. Load the Dataset

- Load the Iris dataset using Scikit-learn.
- Display dataset shape.
- Display sample records.

---

### 3. Visualize the Dataset

Create a scatter plot to visualize the distribution of different Iris flower species.

---

### 4. Data Preprocessing

- Convert class labels into one-hot encoded vectors.
- Split the dataset into:
  - Training Set (80%)
  - Testing Set (20%)
- Standardize the features using StandardScaler.

---

### 5. Build the MLP Model

The neural network architecture is:

| Layer | Neurons | Activation |
|--------|---------|------------|
| Input | 4 | - |
| Hidden Layer 1 | 16 | ReLU |
| Hidden Layer 2 | 8 | ReLU |
| Output Layer | 3 | Softmax |

---

### 6. Compile the Model

Optimizer:
- Adam

Loss Function:
- Categorical Crossentropy

Metric:
- Accuracy

---

### 7. Train the Model

Training parameters:

- Epochs = 100
- Batch Size = 8
- Validation Split = 20%

---

### 8. Visualize Training

Plot:

- Training Accuracy
- Validation Accuracy

This helps observe the learning performance over epochs.

---

### 9. Model Evaluation

Evaluate the trained model using:

- Test Accuracy
- Confusion Matrix

---

## Output

The notebook produces:

- Dataset Information
- Scatter Plot of Iris Dataset
- One-Hot Encoded Labels
- Model Summary
- Training Accuracy Graph
- Test Accuracy
- Confusion Matrix

---

## Expected Results

- High classification accuracy (typically above 95%)
- Accurate prediction of Iris flower species
- Visualization of model learning through accuracy curves
- Confusion matrix showing correct and incorrect predictions

---

