# Forward Propagation and Backpropagation using TensorFlow/Keras

## Overview

This project demonstrates the implementation of a Multi-Layer Perceptron (MLP) using TensorFlow/Keras to understand the concepts of **Forward Propagation** and **Backpropagation**. The model is trained on the Iris dataset and its performance is analyzed by varying the **learning rate** and the **number of epochs**.

Although TensorFlow/Keras performs forward and backpropagation automatically, this project illustrates where these processes occur during model training.

---

## Objectives

- Implement a neural network using TensorFlow/Keras.
- Understand Forward Propagation.
- Understand Backpropagation.
- Train the model using different learning rates.
- Train the model using different epochs.
- Compare model performance.
- Visualize accuracy, loss, and confusion matrix.

---

## Dataset

**Dataset Used:** Iris Dataset

- Total Samples: 150
- Features: 4
- Classes: 3
  - Setosa
  - Versicolor
  - Virginica

The dataset is loaded directly from Scikit-Learn.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

---

## Project Structure

```
Forward_Backpropagation/
│
├── Forward and Backpropagation_tensorflow_keras.ipynb
├── README.md
```

---

## Neural Network Architecture

```
Input Layer
      │
      ▼
Dense Layer (16 neurons, ReLU)
      │
      ▼
Dense Layer (8 neurons, ReLU)
      │
      ▼
Output Layer (3 neurons, Softmax)
```

---

## Forward Propagation

During Forward Propagation, the input data passes through every layer of the neural network.

For every neuron:

```
Z = W × X + b
```

Activation:

```
A = Activation(Z)
```

Flow:

```
Input
   │
   ▼
Hidden Layer 1 (ReLU)
   │
   ▼
Hidden Layer 2 (ReLU)
   │
   ▼
Output Layer (Softmax)
   │
   ▼
Prediction
```

Forward propagation is automatically performed during:

```python
model.fit(...)
```

and

```python
model.predict(...)
```

---

## Backpropagation

After making predictions, the model computes the error using the loss function.

The optimizer then updates the weights using gradient descent.

Weight update equation:

```
New Weight = Old Weight − Learning Rate × Gradient
```

TensorFlow automatically performs Backpropagation during:

```python
model.fit(...)
```

---

## Learning Rates Used

- 0.1
- 0.01
- 0.001

### Observation

| Learning Rate | Observation |
|--------------|-------------|
| 0.1 | Fast learning but may overshoot the optimum solution. |
| 0.01 | Good balance between speed and stability. |
| 0.001 | Stable learning with better convergence. |

---

## Epochs Used

- 20
- 50
- 100

### Observation

| Epochs | Observation |
|---------|-------------|
| 20 | Lower accuracy due to underfitting. |
| 50 | Good training performance. |
| 100 | Highest accuracy with sufficient learning. |

---

## Model Compilation

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Model Training

```python
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2
)
```

During every epoch, TensorFlow internally performs:

1. Forward Propagation
2. Loss Calculation
3. Backpropagation
4. Weight Update

---

## Evaluation Metrics

- Accuracy
- Loss
- Confusion Matrix
- Training Accuracy Curve
- Validation Accuracy Curve
- Training Loss Curve
- Validation Loss Curve

---

## Expected Results

- Test Accuracy: **95%–100%**
- Decreasing loss with epochs
- Increasing accuracy with epochs
- Correct classification shown in the confusion matrix

---

## Applications

- Image Classification
- Medical Diagnosis
- Sentiment Analysis
- Handwritten Digit Recognition
- Fraud Detection
- Recommendation Systems

---

## Conclusion

This project successfully demonstrates the implementation of a neural network using TensorFlow/Keras. The framework automatically performs **Forward Propagation** to generate predictions and **Backpropagation** to update the model weights during training. Experimental analysis shows that the learning rate and number of epochs significantly affect convergence speed and model accuracy. A learning rate of **0.001** with **100 epochs** provides stable training and excellent performance on the Iris dataset.

---
