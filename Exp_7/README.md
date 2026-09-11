# Transfer Learning Using Pre-trained CNN Models for Food Image Classification

## 1. Project Title

**Implementation and Comparison of Transfer Learning Using Pre-trained AlexNet, VGG16, ResNet50, and EfficientNet-B0 Models for Image Classification**

---

## 2. Project Overview

This project demonstrates the use of **Transfer Learning** for image classification using four pre-trained Convolutional Neural Network (CNN) architectures:

- AlexNet
- VGG16
- ResNet50
- EfficientNet-B0

The models are trained and evaluated on a selected subset of the **Food-101 dataset**.

The main purpose of this project is to understand how pre-trained deep learning models can be adapted to a new image classification problem and to compare their performance using different evaluation metrics.

---

## 3. Objective

The objective of this project is to:

1. Implement transfer learning using pre-trained CNN models.
2. Use AlexNet, VGG16, ResNet50, and EfficientNet-B0 for food image classification.
3. Use a subset of the Food-101 dataset.
4. Replace the original classification layer of each pre-trained model with a new classifier for 10 food classes.
5. Freeze the pre-trained layers and train only the newly added classification layer.
6. Use GPU acceleration for model training.
7. Compare the models based on:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Training Time
8. Save trained models and results to avoid unnecessary retraining.
9. Identify the best-performing model.

---

## 4. Dataset

The project uses the **Food-101 dataset** available through Hugging Face.

Dataset:

**Food-101 (`ethz/food101`)**

The complete Food-101 dataset contains 101 different food categories. However, due to hardware and time limitations, this project uses a selected subset of 10 classes.

The dataset is accessed using **Hugging Face streaming**, which avoids downloading the complete dataset to the local system before starting the experiment.

---

## 5. Selected Food Classes

The following 10 classes were selected:

1. Apple Pie
2. Cheesecake
3. Chicken Curry
4. French Fries
5. Fried Rice
6. Hamburger
7. Ice Cream
8. Pizza
9. Sushi
10. Tacos

Dataset labels:

```text
apple_pie
cheesecake
chicken_curry
french_fries
fried_rice
hamburger
ice_cream
pizza
sushi
tacos
```

---

## 6. Dataset Split

For each selected class:

- 100 images are used for training.
- 25 images are used for validation.

| Dataset | Images |
|---|---:|
| Training | 1000 |
| Validation | 250 |
| Total | 1250 |

There are 10 classes in total.

---

## 7. Why a Subset of Food-101 Was Used

The complete Food-101 dataset is large and requires significant storage and processing resources.

Since the experiment was performed on a laptop with limited storage and a GPU with 4 GB VRAM, a smaller subset was selected.

Using 10 classes with a limited number of images per class makes it possible to:

- Reduce storage requirements.
- Reduce training time.
- Run the experiment on a laptop GPU.
- Compare four different CNN architectures efficiently.

---

## 8. Hardware Used

The experiment was performed using an NVIDIA GPU.

### GPU

```text
NVIDIA GeForce RTX 2050
```

### GPU Memory

```text
4 GB VRAM
```

GPU acceleration was enabled using CUDA through PyTorch.

The final training run successfully detected:

```text
Device: cuda
GPU: NVIDIA GeForce RTX 2050
```

---

## 9. Software and Technologies

| Technology | Purpose |
|---|---|
| Python 3.10 | Programming language |
| PyTorch | Deep learning framework |
| Torchvision | Pre-trained CNN models |
| Hugging Face Datasets | Dataset loading and streaming |
| NumPy | Numerical operations |
| Pandas | Results processing |
| Matplotlib | Graph generation |
| Scikit-learn | Evaluation metrics |
| PIL | Image processing |
| CUDA | GPU acceleration |
| Git | Version control |
| GitHub | Code repository |

---

# 10. Transfer Learning

## What is Transfer Learning?

Transfer learning is a deep learning technique where a model that has already been trained on a large dataset is reused for a new task.

Instead of training a CNN completely from the beginning, a pre-trained model is used as a starting point.

In this project, the models are pre-trained on **ImageNet**.

The original classification layer is replaced with a new classification layer containing 10 output classes.

The pre-trained layers are frozen, meaning their weights are not updated during training.

Only the new final classification layer is trained.

---

## 11. Transfer Learning Architecture

The general process used in this project is:

```text
Food-101 Images
       |
       v
Image Preprocessing
       |
       v
Pre-trained CNN Model
       |
       v
Frozen Feature Extraction Layers
       |
       v
New 10-Class Classification Layer
       |
       v
Predicted Food Class
```

The same transfer learning approach is applied to:

```text
AlexNet
VGG16
ResNet50
EfficientNet-B0
```

---

# 12. Models Used

## 12.1 AlexNet

AlexNet is a convolutional neural network architecture that became well known for its performance in image classification.

In this project, the pre-trained AlexNet model is loaded and its final classifier layer is replaced with a new layer containing 10 output classes.

The pre-trained layers are frozen and only the final classification layer is trained.

---

## 12.2 VGG16

VGG16 is a deep convolutional neural network architecture containing multiple convolutional layers followed by fully connected layers.

The pre-trained VGG16 model is used as a feature extractor.

Its final classifier is replaced with a new 10-class classifier.

Only the newly replaced classification layer is trained.

---

## 12.3 ResNet50

ResNet50 is a deep CNN architecture that uses **residual connections**.

Residual connections help in training deeper neural networks by allowing information to pass through the network more effectively.

In this project, the original final fully connected layer is replaced with a 10-class classifier.

The pre-trained layers remain frozen.

---

## 12.4 EfficientNet-B0

EfficientNet-B0 is an efficient CNN architecture designed to achieve a good balance between model size and classification performance.

In this project, the original classifier is replaced with a new classifier containing 10 output classes.

The pre-trained layers are frozen.

---

# 13. Image Preprocessing

The input images are preprocessed before being provided to the models.

The preprocessing pipeline is:

```text
Original Image
      |
      v
Resize to 256 × 256
      |
      v
Center Crop to 224 × 224
      |
      v
Convert to Tensor
      |
      v
ImageNet Normalization
      |
      v
Model Input
```

The final input size is:

```text
224 × 224 × 3
```

---

## 14. ImageNet Normalization

The images are normalized using the mean and standard deviation used for ImageNet-trained models.

### Mean

```text
[0.485, 0.456, 0.406]
```

### Standard Deviation

```text
[0.229, 0.224, 0.225]
```

This preprocessing is appropriate because the four models were originally trained using ImageNet.

---

# 15. Project Structure

```text
7.Ttansfer_learning/
│
├── data_pipeline.py
├── models.py
├── train.py
├── show_results.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── saved_models/
│   ├── alexnet.pth
│   ├── vgg16.pth
│   ├── resnet50.pth
│   └── efficientnet_b0.pth
│
└── results/
    ├── transfer_learning_results.csv
    ├── training_histories.json
    ├── accuracy_comparison.png
    ├── metrics_comparison.png
    ├── training_time_comparison.png
    ├── alexnet_training.png
    ├── vgg16_training.png
    ├── resnet50_training.png
    └── efficientnet_b0_training.png
```

---

# 16. Description of Project Files

## `data_pipeline.py`

This file is responsible for the dataset pipeline.

It performs the following tasks:

- Loads Food-101 from Hugging Face.
- Uses streaming to reduce local storage requirements.
- Selects the required 10 food classes.
- Collects training and validation images.
- Converts images into RGB format.
- Applies image preprocessing.
- Creates a custom PyTorch Dataset.
- Creates PyTorch DataLoaders.
- Checks CUDA/GPU availability.
- Tests the image tensor shape.

---

## `models.py`

This file is responsible for loading and configuring the four pre-trained models.

It:

- Loads pre-trained AlexNet.
- Loads pre-trained VGG16.
- Loads pre-trained ResNet50.
- Loads pre-trained EfficientNet-B0.
- Freezes the pre-trained layers.
- Replaces the original classification layers.
- Configures each model for 10 classes.
- Moves the models to the available GPU.

---

## `train.py`

This is the main training program.

It performs the following tasks:

1. Creates directories for saved models and results.
2. Checks whether models have already been trained.
3. Loads the dataset.
4. Loads the four models.
5. Trains each model.
6. Evaluates each model.
7. Calculates accuracy.
8. Calculates precision.
9. Calculates recall.
10. Calculates F1-score.
11. Measures training time.
12. Saves model checkpoints.
13. Saves training history.
14. Saves results in CSV format.
15. Generates comparison graphs.
16. Identifies the best-performing model.

The program also checks for previously saved checkpoints.

If a model is already saved, it is not trained again.

---

## `show_results.py`

This script is used to view previously saved results.

It does not retrain the models.

It:

- Reads the saved CSV file.
- Displays model comparison results.
- Identifies the best model.
- Displays saved graphs.
- Shows the previously generated metrics.

---

# 17. Training Configuration

The following configuration was used:

| Parameter | Value |
|---|---|
| Number of classes | 10 |
| Training images | 1000 |
| Validation images | 250 |
| Image size | 224 × 224 |
| Batch size | 8 |
| Epochs | 3 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Device | CUDA / NVIDIA GPU |

---

# 18. Loss Function

The project uses:

```text
CrossEntropyLoss
```

Cross-entropy loss is commonly used for multi-class classification problems.

Since this project contains 10 different food classes, CrossEntropyLoss is suitable for measuring the difference between the predicted class probabilities and the actual class labels.

---

# 19. Optimizer

The project uses the:

```text
Adam Optimizer
```

with a learning rate of:

```text
0.001
```

Adam is used to update the trainable classification layer during training.

---

# 20. Frozen Layers

The pre-trained layers are frozen during training.

This means that their learned ImageNet weights are not changed.

Only the newly added final classification layer is trained.

This provides several benefits:

- Faster training.
- Fewer trainable parameters.
- Lower GPU memory requirements.
- Ability to use transfer learning with a smaller dataset.

---

# 21. Trainable Parameters

The approximate number of trainable parameters for each model is:

| Model | Trainable Parameters |
|---|---:|
| AlexNet | 40,970 |
| VGG16 | 40,970 |
| ResNet50 | 20,490 |
| EfficientNet-B0 | 12,810 |

EfficientNet-B0 has the lowest number of trainable parameters among the four models in this experiment.

---

# 22. Evaluation Metrics

The models are evaluated using the following metrics:

## Accuracy

Accuracy represents the percentage of correctly classified images.

```text
Accuracy =
Correct Predictions / Total Predictions
```

## Precision

Precision measures how many of the images predicted as a particular class actually belong to that class.

## Recall

Recall measures how many of the actual images belonging to a class were correctly identified.

## F1-Score

F1-score is the harmonic mean of precision and recall.

It provides a balanced measure when both precision and recall are important.

---

# 23. Experimental Results

The following are the results from the **latest successful training run**.

## Final Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|---|---:|---:|---:|---:|---:|
| **VGG16** | **80.00%** | **80.75%** | **80.00%** | **79.47%** | 0.82 min |
| **ResNet50** | **77.20%** | **77.92%** | **77.20%** | **77.19%** | 0.47 min |
| EfficientNet-B0 | 70.40% | 72.85% | 70.40% | 68.86% | 0.28 min |
| AlexNet | 68.40% | 71.30% | 68.40% | 68.04% | 0.26 min |

---

# 24. Training Results by Epoch

## AlexNet

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|---|---:|---:|---:|---:|
| 1 | 1.4432 | 54.50% | 1.0146 | 69.60% |
| 2 | 0.5813 | 80.20% | 0.9794 | 67.60% |
| 3 | 0.3662 | 87.70% | 1.0405 | 68.40% |

Best validation accuracy: **69.60%** at Epoch 1.

Final evaluation accuracy: **68.40%**.

---

## VGG16

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|---|---:|---:|---:|---:|
| 1 | 1.1378 | 61.00% | 0.6507 | 77.20% |
| 2 | 0.6220 | 77.00% | 0.6275 | 78.80% |
| 3 | 0.4805 | 84.10% | 0.6034 | 80.00% |

Best validation accuracy: **80.00%** at Epoch 3.

Final evaluation accuracy: **80.00%**.

---

## ResNet50

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|---|---:|---:|---:|---:|
| 1 | 1.9046 | 42.70% | 1.3959 | 70.40% |
| 2 | 1.2560 | 71.10% | 1.0689 | 75.20% |
| 3 | 1.0326 | 75.30% | 0.8664 | 77.20% |

Best validation accuracy: **77.20%** at Epoch 3.

Final evaluation accuracy: **77.20%**.

---

## EfficientNet-B0

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|---|---:|---:|---:|---:|
| 1 | 1.9282 | 41.00% | 1.4046 | 63.60% |
| 2 | 1.3566 | 63.80% | 1.0933 | 69.60% |
| 3 | 1.1464 | 68.30% | 0.9805 | 70.40% |

Best validation accuracy: **70.40%** at Epoch 3.

Final evaluation accuracy: **70.40%**.

---

# 25. Model Ranking

Based on the final evaluation accuracy:

```text
1. VGG16             80.00%
2. ResNet50          77.20%
3. EfficientNet-B0   70.40%
4. AlexNet           68.40%
```

Therefore, **VGG16 is the best-performing model** in this experiment.

---

# 26. Best Performing Model

The best-performing model is:

## VGG16

Final metrics:

```text
Accuracy : 80.00%
Precision: 80.75%
Recall   : 80.00%
F1-Score : 79.47%
Training Time: 0.82 minutes
```

VGG16 achieved the highest accuracy, precision, recall, and F1-score among the four models.

---

# 27. Model Observations

## AlexNet

AlexNet achieved a final evaluation accuracy of 68.40%.

Its training accuracy increased from 54.50% to 87.70%, while validation accuracy remained around 68–70%.

The increase in training accuracy combined with lower validation performance indicates some overfitting.

---

## VGG16

VGG16 achieved the highest final evaluation accuracy of 80.00%.

Its validation accuracy improved throughout the three epochs:

```text
77.20% → 78.80% → 80.00%
```

The validation loss also decreased:

```text
0.6507 → 0.6275 → 0.6034
```

Therefore, VGG16 provided the best overall performance in this experiment.

---

## ResNet50

ResNet50 achieved 77.20% final evaluation accuracy.

Its validation accuracy improved consistently:

```text
70.40% → 75.20% → 77.20%
```

The validation loss also decreased during training.

This indicates that the model was still improving at the end of the three epochs.

---

## EfficientNet-B0

EfficientNet-B0 achieved 70.40% final evaluation accuracy.

Its validation accuracy improved from:

```text
63.60% → 69.60% → 70.40%
```

It also had the lowest number of trainable parameters among the four models.

---

# 28. Training Time Comparison

The training times from the latest run were:

| Model | Training Time |
|---|---:|
| AlexNet | 0.26 minutes |
| VGG16 | 0.82 minutes |
| ResNet50 | 0.47 minutes |
| EfficientNet-B0 | 0.28 minutes |

AlexNet had the shortest training time, while VGG16 required the longest training time.

---

# 29. Generated Results

The project generates the following files:

### `transfer_learning_results.csv`

Contains the final model performance metrics.

### `training_histories.json`

Contains the training and validation history.

### `accuracy_comparison.png`

Compares the accuracy of all four models.

### `metrics_comparison.png`

Compares accuracy, precision, recall, and F1-score.

### `training_time_comparison.png`

Compares the training time of the models.

### Model training graphs

```text
alexnet_training.png
vgg16_training.png
resnet50_training.png
efficientnet_b0_training.png
```

These graphs show the training and validation performance across epochs.

---

# 30. Model Checkpoint Saving

The trained models are saved as checkpoint files:

```text
saved_models/
├── alexnet.pth
├── vgg16.pth
├── resnet50.pth
└── efficientnet_b0.pth
```

The checkpoint stores information such as:

- Model state dictionary
- Model name
- Training history
- Training time
- Evaluation metrics
- Selected classes
- Number of epochs
- Learning rate

---

# 31. Avoiding Retraining

The project includes checkpoint detection.

If a model has already been trained and its checkpoint exists, `train.py` skips training that model.

If all four checkpoints exist, the training script does not retrain the models.

Previously saved results can be viewed using:

```powershell
python show_results.py
```

This saves time and GPU resources.

---

# 32. Installation

## Step 1: Install Python

Python 3.10 was used for this project.

Check the installed Python version:

```powershell
python --version
```

Expected:

```text
Python 3.10.x
```

## Step 2: Create Virtual Environment

```powershell
python -m venv .venv
```

This creates an isolated Python environment for the project.

## Step 3: Activate Virtual Environment

On Windows:

```powershell
.venv\Scripts\activate
```

The terminal should show:

```text
(.venv)
```

## Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

---

# 33. CUDA / GPU Setup

PyTorch must have CUDA support to train using the NVIDIA GPU.

Verify CUDA using Python:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected output:

```text
True
NVIDIA GeForce RTX 2050
```

If `True` is displayed, CUDA is available.

---

# 34. Running the Project

## Step 1: Run the Training Program

```powershell
python train.py
```

This starts the transfer learning experiment.

The program trains models that do not already have saved checkpoints.

## Step 2: View Results

After training finishes:

```powershell
python show_results.py
```

This displays the saved results and comparison graphs without retraining the models.

---

# 35. Dataset Streaming

The project uses Hugging Face dataset streaming.

The main advantage is that the complete dataset does not have to be downloaded and stored locally before the program starts.

The program retrieves the required images while constructing the dataset subset.

This is useful for systems with limited storage.

---

# 36. Advantages of the Approach

The project provides several advantages:

### 1. Reduced Training Time

Pre-trained models already contain useful visual features.

### 2. Fewer Trainable Parameters

Most of the network is frozen.

### 3. Lower Computational Requirement

Only the new classification layer needs to be trained.

### 4. Suitable for Smaller Datasets

Transfer learning can work effectively when the available training data is limited.

### 5. Easy Model Comparison

Four different CNN architectures are evaluated using the same dataset and classification task.

### 6. Checkpoint Support

Trained models are saved so that completed models do not need to be trained again.

---

# 37. Limitations

The experiment has the following limitations:

1. Only 10 of the 101 Food-101 classes were used.
2. Only 100 training images were used per class.
3. Only 25 validation images were used per class.
4. Training was limited to 3 epochs.
5. The pre-trained layers were frozen.
6. The experiment was performed on a laptop GPU with 4 GB VRAM.
7. Results may vary depending on the selected images and execution environment.
8. The results obtained from this subset should not be considered representative of performance on the complete Food-101 dataset.
9. The Hugging Face dataset was accessed using unauthenticated streaming, which may result in lower download limits or slower data access.

---

# 38. Possible Future Improvements

The project can be improved by:

- Using all 101 Food-101 classes.
- Increasing the number of training images.
- Increasing the number of epochs.
- Fine-tuning selected layers of the pre-trained networks.
- Applying data augmentation.
- Using learning-rate scheduling.
- Performing hyperparameter tuning.
- Generating confusion matrices.
- Comparing per-class performance.
- Testing the models on a larger validation dataset.

---

# 39. Conclusion

This project successfully demonstrates transfer learning for food image classification using four pre-trained CNN architectures:

- AlexNet
- VGG16
- ResNet50
- EfficientNet-B0

The models were adapted for a 10-class Food-101 classification problem by replacing their original classification layers and training the new classifier.

The latest experimental results were:

```text
VGG16             80.00%
ResNet50          77.20%
EfficientNet-B0   70.40%
AlexNet           68.40%
```

Among the four models, **VGG16 achieved the highest validation/evaluation accuracy of 80.00%**.

The results show that VGG16 provided the best classification performance for the selected Food-101 subset in this experiment.

ResNet50 was the second-best model with 77.20% accuracy, while EfficientNet-B0 and AlexNet achieved 70.40% and 68.40%, respectively.

Overall, the project demonstrates that transfer learning can provide good image classification performance while reducing the amount of training required compared with training a CNN completely from scratch.

---

### Topic

**Transfer Learning using Pre-trained CNN Models**

### Models

- AlexNet
- VGG16
- ResNet50
- EfficientNet-B0

### Dataset

**Food-101**

### Number of Classes

**10**

### Training Images

**1000**

### Validation Images

**250**

### GPU

**NVIDIA GeForce RTX 2050 (4 GB VRAM)**

### Best Model

**VGG16**

### Best Accuracy

**80.00%**
