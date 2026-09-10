# Fire and Smoke Image Classification Using CNN

## 1. Project Overview

This project implements a Convolutional Neural Network (CNN) for classifying forest fire-related images using PyTorch.

The model classifies images into four categories:

- **Fire**
- **No Fire**
- **Smoke**
- **Smoke + Fire**

The project covers the complete deep learning workflow, including dataset preparation, image preprocessing, data augmentation, CNN model development, training, validation, testing, performance evaluation, and prediction on unseen images.

---

## 2. Objective

The main objective of this project is to design and implement a CNN-based image classification model that can identify different forest fire conditions from images.

The model learns visual features such as fire, smoke, textures, brightness, and environmental patterns to classify an input image into one of four categories.

---

## 3. Problem Statement

Forest fires can cause significant environmental and economic damage. Early detection of fire and smoke can help reduce response time and minimize damage.

This project aims to develop an automated image classification system that classifies forest images into:

1. Fire
2. No Fire
3. Smoke
4. Smoke + Fire

---

## 4. Dataset

The project uses the **Forest Fire C4 Image Classification Dataset**.

### Classes

| Class | Description |
|---|---|
| Fire | Images containing visible fire |
| No Fire | Images without fire |
| Smoke | Images containing smoke |
| Smoke + Fire | Images containing both smoke and fire |

### Dataset Split

| Dataset | Number of Images |
|---|---:|
| Training | 3,200 |
| Validation | 800 |
| Testing | 800 |
| **Total** | **4,800** |

An additional set of **23 unlabeled tester images** was used to demonstrate predictions on unseen images.


---

## 5. Technologies Used

### Software

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow
- Jupyter Notebook
- VS Code
- WSL2
- CUDA

### Hardware

- **GPU:** NVIDIA GeForce RTX 2050
- **VRAM:** 4 GB

---

## 6. Data Preprocessing

All input images were resized to:

**224 × 224 pixels**

### Training Augmentation

The training images were augmented using:

- Random Horizontal Flip
- Random Rotation
- Color Jitter

These techniques help the model generalize better to variations in image orientation, brightness, and appearance.

### Normalization

Images were converted to tensors and normalized using ImageNet mean and standard deviation values:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

Validation and test images were resized and normalized without random augmentation.

---

## 7. CNN Architecture

A custom Convolutional Neural Network was implemented from scratch using PyTorch.

### Architecture

```text
Input Image (224 × 224 × 3)
          |
          v
Conv2D (3 → 32)
          |
Batch Normalization
          |
ReLU
          |
Max Pooling
          |
          v
Conv2D (32 → 64)
          |
Batch Normalization
          |
ReLU
          |
Max Pooling
          |
          v
Conv2D (64 → 128)
          |
Batch Normalization
          |
ReLU
          |
Max Pooling
          |
          v
Conv2D (128 → 256)
          |
Batch Normalization
          |
ReLU
          |
Max Pooling
          |
          v
Adaptive Average Pooling
          |
          v
Dropout (0.5)
          |
          v
Fully Connected Layer
          |
          v
4 Output Classes
```

### Model Parameters

The model contains approximately **390,404 trainable parameters**.

---

## 8. Training and Model Configuration

### Loss Function

Cross Entropy Loss was used for multi-class classification.

```python
criterion = nn.CrossEntropyLoss()
```

### Optimizer

The Adam optimizer was used:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)
```

### Training Configuration

- **Epochs:** 20
- **Batch Size:** 32
- **Learning Rate:** 0.001
- **Optimizer:** Adam
- **Loss Function:** Cross Entropy Loss
- **Dropout:** 0.5
- **Image Size:** 224 × 224

The model with the highest validation accuracy was saved as:

```text
model/best_fire_smoke_cnn.pth
```

---

## 9. Results and Evaluation

### Overall Results

| Metric | Result |
|---|---:|
| Training Images | 3,200 |
| Validation Images | 800 |
| Test Images | 800 |
| Best Validation Accuracy | **89.62%** |
| Test Loss | **0.8212** |
| Test Accuracy | **73.38%** |

The best validation accuracy of **89.62%** was achieved at **Epoch 19**.

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Fire | 94.70% | 62.50% | 75.30% |
| No Fire | 96.35% | 92.50% | 94.39% |
| Smoke | 60.61% | 100.00% | 75.47% |
| Smoke + Fire | 52.74% | 38.50% | 44.51% |
| **Overall Accuracy** | | | **73.38%** |

### Result Analysis

The model performed best on the **No Fire** class, with an F1-score of **94.39%**.

The **Smoke** class achieved **100% recall**, meaning all actual smoke images in the test set were successfully detected.

The main challenge was the **Smoke + Fire** class, which achieved a recall of **38.50%** and an F1-score of **44.51%**. This indicates that the model sometimes confused images containing both smoke and fire with the individual Fire or Smoke classes.

A confusion matrix was also generated to visualize these classification errors.

---

## 10. Unseen Image Prediction

The trained model was also tested on **23 additional unlabeled images** that were not part of the training, validation, or test datasets.

### Prediction Summary

| Predicted Class | Number of Images |
|---|---:|
| Fire | 2 |
| No Fire | 6 |
| Smoke | 8 |
| Smoke + Fire | 7 |
| **Total** | **23** |

### Example Predictions

| Image | Predicted Class | Confidence |
|---|---|---:|
| 1.jpg | Smoke + Fire | 65.05% |
| 3.jpg | Smoke + Fire | 98.43% |
| 7.jpg | No Fire | 99.78% |
| 10.jpg | Smoke | 88.30% |
| 15.jpg | Smoke + Fire | 95.11% |
| 17.jpg | Smoke | 97.43% |
| abc192.jpg | Fire | 60.87% |
| abc223.jpg | Fire | 76.83% |

Since these 23 tester images do not have ground-truth labels, they were not used for calculating accuracy.

---

## 11. Evaluation Metrics

The following techniques were used to evaluate the CNN:

- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Training Loss
- Validation Loss
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Unseen Image Prediction

Training and validation accuracy/loss graphs were also generated to observe the model's learning behavior over the 20 epochs.

---

## 12. Project Structure

```text
Exp_6/
|
├── notebook/
│   └── fire_smoke_cnn.ipynb
|
├── model/
│   └── best_fire_smoke_cnn.pth
|
├── README.md
|
├── .gitignore
|
├── dataset/
│   └── Forest Fire/
│       ├── Forest_Fire_Dataset/
│       └── Forest_Fire_Tester/
|
└── fire_smoke.env/
```

---

## 13. How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/Samarth070906/Deep-Learning.git
```

### Step 2: Navigate to the Project

```bash
cd Deep-Learning/Exp_6
```

### Step 3: Create a Virtual Environment

```bash
python -m venv fire_smoke.env
```

### Step 4: Activate the Environment

For Linux/WSL:

```bash
source fire_smoke.env/bin/activate
```

### Step 5: Install Required Libraries

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow jupyter
```

### Step 6: Add the Dataset

Download the Forest Fire C4 dataset and place it inside:

```text
Exp_6/dataset/Forest Fire/
```

The dataset folder should contain:

```text
Forest_Fire_Dataset/
├── train/
├── val/
└── test/

Forest_Fire_Tester/
```

### Step 7: Open the Notebook

Open:

```text
notebook/fire_smoke_cnn.ipynb
```

Select the appropriate Python environment/kernel and run the notebook cells in order.

---

## 14. Future Improvements

The current model achieved a test accuracy of **73.38%**. The following improvements could potentially increase the accuracy and generalization of the model:

- Transfer learning using ResNet18 or another pretrained CNN
- Stronger image augmentation
- Learning-rate scheduling
- Hyperparameter tuning
- Increasing dataset size and diversity
- Class-specific augmentation
- Fine-tuning pretrained CNN models
- Better feature extraction
- Cross-validation

The primary improvement area is the **Smoke + Fire** class, which currently has the lowest recall and F1-score.

---

## 15. Conclusion

A CNN-based image classification system was successfully implemented using PyTorch to classify forest fire-related images into four categories:

**Fire, No Fire, Smoke, and Smoke + Fire.**

The project successfully demonstrates the complete deep learning workflow, from data preprocessing and augmentation to CNN model development, training, validation, testing, evaluation, and prediction on unseen images.

The model achieved:

- **89.62% Best Validation Accuracy**
- **73.38% Test Accuracy**

The model performed particularly well for the **No Fire** class, while **Smoke + Fire** was the most challenging class.

The project demonstrates that CNNs can effectively learn visual patterns from forest fire images and can be further improved using transfer learning, stronger augmentation, learning-rate scheduling, and hyperparameter optimization.

