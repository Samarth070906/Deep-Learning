# Exp 9 — Image Classification Using Vision Transformer (ViT) vs CNN (ResNet-18) on CIFAR-10

## 1. Project Title

**Performance Comparison of Pre-trained Vision Transformer (ViT-B/16) and Convolutional Neural Network (ResNet-18) on CIFAR-10 Image Classification**

---

## 2. Project Overview

This experiment evaluates and compares two foundational paradigms in computer vision:
1. **Convolutional Neural Networks (CNNs)** represented by **ResNet-18**
2. **Vision Transformers (ViTs)** represented by **ViT-B/16**

Both models are initialized with pre-trained ImageNet weights and fine-tuned for multi-class image classification on the benchmark **CIFAR-10** dataset. The study examines how differences in architectural inductive biases affect learning dynamics, computational efficiency, convergence, and classification accuracy.

---

## 3. Objective

1. Implement image classification on CIFAR-10 using:
   - A pre-trained CNN (**ResNet-18**)
   - A pre-trained Vision Transformer (**ViT-B/16**)
2. Adapt pre-trained architectures for 10-class classification by replacing their classification heads.
3. Apply standard data augmentation, resizing (to 224×224), and normalization suited for ImageNet pre-trained backbones.
4. Train both models using GPU acceleration with PyTorch.
5. Evaluate and compare both architectures across key metrics:
   - Test Loss & Test Accuracy
   - Precision, Recall, and Macro F1-Score
   - Per-class Classification Report & Confusion Matrix
   - Training Time, Epoch Loss, and Epoch Accuracy Curves
6. Analyze the theoretical and empirical differences between CNN inductive bias (local receptive fields, translation equivariance) and Vision Transformer global self-attention.

---

## 4. Dataset: CIFAR-10

The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset is a standard computer vision benchmark consisting of 60,000 32×32 color images across 10 balanced classes:

| Class ID | Class Name | Description |
|:--------:|:----------:|:------------|
| 0 | `airplane` | Airplanes, jets, gliders |
| 1 | `automobile` | Cars, sedans, SUVs |
| 2 | `bird` | Birds of various species |
| 3 | `cat` | Feline domestic pets |
| 4 | `deer` | Wild deer and stags |
| 5 | `dog` | Domestic canines |
| 6 | `frog` | Amphibians |
| 7 | `horse` | Horses |
| 8 | `ship` | Boats, cargo ships |
| 9 | `truck` | Commercial transport vehicles |

- **Training Set:** 50,000 images (5,000 per class)
- **Test Set:** 10,000 images (1,000 per class)
- **Channels:** 3 (RGB)

---

## 5. Architectural Comparison

| Feature | CNN (ResNet-18) | Vision Transformer (ViT-B/16) |
|---|---|---|
| **Core Mechanism** | Convolutional layers with residual skip connections | Multi-Head Self-Attention (MHSA) + MLP blocks |
| **Input Representation** | Continuous 2D pixel grid with hierarchical feature maps | Flattened non-overlapping 16×16 patches + linear projection |
| **Inductive Bias** | Strong (spatial locality, translation equivariance) | Weak (must learn spatial relationships from positional embeddings) |
| **Receptive Field** | Grows progressively layer by layer | Global across all patches from the very first layer |
| **Input Resolution** | Flexible, native 32×32 or resized to 224×224 | Fixed patch grid (typically 224×224 for 14×14 = 196 patches) |
| **Pre-trained Weights** | `ResNet18_Weights.DEFAULT` (ImageNet-1K) | `ViT_B_16_Weights.DEFAULT` (ImageNet-1K) |
| **Classification Head** | `nn.Linear(512, 10)` | `nn.Linear(768, 10)` |

---

## 6. Data Preprocessing & Augmentation

Both models require an input size of 224×224 to match their ImageNet pre-training specifications:

- **Training Transformations:**
  - `transforms.Resize((224, 224))`
  - `transforms.RandomHorizontalFlip()`
  - `transforms.RandomCrop(224, padding=4)`
  - `transforms.ToTensor()`
  - `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
- **Testing Transformations:**
  - `transforms.Resize((224, 224))`
  - `transforms.ToTensor()`
  - `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

---

## 7. Hyperparameters & Training Setup

| Parameter | Value |
|---|---|
| **Batch Size** | 64 |
| **Optimizer** | Adam (`lr=1e-4`) |
| **Loss Function** | Cross-Entropy Loss (`nn.CrossEntropyLoss`) |
| **Epochs** | 5 epochs per model |
| **Dataloader Workers** | 2 with pinned memory |
| **Hardware** | NVIDIA GPU (CUDA enabled) |

---

## 8. Experiment Workflow

```
       [CIFAR-10 Dataset]
               │
     ┌─────────┴─────────┐
     ▼                   ▼
[Train Loader]      [Test Loader]
(224×224, Augment)  (224×224, Standard)
     │                   │
     ├───────────────────┼───────────────────┐
     ▼                                       ▼
[Part A: ResNet-18 CNN]             [Part B: ViT-B/16 Transformer]
 - Replace fc layer (512 -> 10)      - Replace head layer (768 -> 10)
 - Adam optimizer (lr=1e-4)          - Adam optimizer (lr=1e-4)
 - Train 5 epochs                    - Train 5 epochs
 - Compute Loss & Acc History        - Compute Loss & Acc History
     │                                       │
     └───────────────────┬───────────────────┘
                         ▼
             [Part C: Comparative Analysis]
              - Accuracy, Precision, Recall, F1
              - Confusion Matrix Heatmaps
              - Training Time & Resource Usage
              - Loss & Accuracy Convergence Curves
```

---

## 9. Technologies Used

- **Language:** Python
- **Deep Learning Framework:** PyTorch (`torch`, `torchvision`)
- **Data Handling:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Metrics & Reporting:** Scikit-Learn (`classification_report`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`)
- **Environment:** Jupyter Notebook, CUDA / cuDNN

---

## 10. Key Insights

1. **Local vs Global Features:** ResNet-18 uses convolutional kernels that naturally prioritize nearby pixels, making it highly data-efficient and fast to train. ViT-B/16 processes pairwise patch relationships globally via self-attention, capturing long-range contextual relationships across the entire image.
2. **Computational Footprint:** ViT-B/16 has ~86M parameters compared to ResNet-18's ~11M parameters. Consequently, ViT requires significantly higher memory and compute per training step.
3. **Transfer Learning Effectiveness:** Pre-trained representations allow both architectures to achieve high accuracy on CIFAR-10, demonstrating the versatility of transfer learning from ImageNet to domain-specific classification tasks.

---

## 11. How to Run

1. Open the notebook in your Jupyter environment:
   ```bash
   jupyter notebook CNN_vs_ViT_CIFAR10.ipynb
   ```
2. Run cells sequentially to:
   - Download the CIFAR-10 dataset
   - Fine-tune ResNet-18
   - Fine-tune ViT-B/16
   - Generate comparative tables, loss/accuracy curves, and confusion matrices
