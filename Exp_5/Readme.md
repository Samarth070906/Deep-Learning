# RNN vs LSTM vs GRU — Sequence Classification on Retail Sales Data

## Overview

This project implements and compares three recurrent neural network architectures — **Simple RNN**, **LSTM (Long Short-Term Memory)**, and **GRU (Gated Recurrent Unit)** — for **binary sequence classification** on retail sales data. The task is to predict whether the next day's sales will **increase** or **decrease** based on the previous 30 days of sales history.

---

## Objectives

- Load and preprocess a large-scale retail sales dataset.
- Aggregate transaction-level data into daily total sales.
- Create a binary classification target (increase vs decrease).
- Build and train Simple RNN, LSTM, and GRU models under identical conditions.
- Compare all three models using classification metrics.
- Visualize training curves, confusion matrices, and metric comparisons.

---

## Dataset

**Dataset Used:** Retail Sales Dataset (`sales.csv`)

- **Records:** ~3.8 million transactions
- **Key Columns:**
  - `date` — Transaction date
  - `sum_total` — Total sale amount
- **Preprocessing:** Aggregated to daily total sales
- **Target Variable:** Binary — `1` (sales increased next day), `0` (sales decreased or same)

> **Note:** The dataset is not included in this repository due to its large size (~379 MB).

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

---

## Project Structure

```
Exp_5/
│
├── RNN_LSTM_GRU_Retail_Sales_Classification.ipynb
├── Dataset/
│   ├── sales.csv
│   ├── catalog.csv
│   ├── discounts_history.csv
│   ├── online.csv
│   ├── price_history.csv
│   ├── stores.csv
│   ├── markdowns.csv
│   └── actual_matrix.csv
└── Readme.md
```

---

## Data Preprocessing Steps

1. **Load** the sales dataset from CSV.
2. **Strip** whitespace from column names.
3. **Convert** the `date` column to datetime format.
4. **Sort** data chronologically.
5. **Handle** missing values — drop rows with missing `sum_total`.
6. **Aggregate** daily sales — group by date and sum `sum_total`.
7. **Create binary target** — using `shift(-1)` to compare each day's sales with the next day.
8. **Visualize** daily sales trends and class distribution.

---

## Binary Classification Target

```python
daily_sales["target"] = (daily_sales["sum_total"].shift(-1) > daily_sales["sum_total"]).astype(int)
```

| Target | Meaning |
|--------|---------|
| **0** | Sales decreased or stayed the same the next day |
| **1** | Sales increased the next day |

---

## Train/Test Split

- **Split Ratio:** 80% Training, 20% Testing
- **Method:** Chronological split (no shuffling) to preserve temporal order.

---

## Feature Scaling

- **Scaler Used:** `MinMaxScaler` (range 0 to 1)
- **Important:** Scaler is fit **only on training data** to prevent data leakage.

---

## Sequence Creation

- **Window Size:** 30 days
- Each input sample contains the previous 30 days of scaled sales values.
- The target is the binary classification label for the last day in the window.

```python
WINDOW_SIZE = 30

def create_sequences(scaled_values, targets, window_size):
    X, y = [], []
    for i in range(window_size, len(scaled_values)):
        X.append(scaled_values[i - window_size:i])
        y.append(targets[i])
    return np.array(X), np.array(y)
```

- **Input Shape:** `(samples, 30, 1)`
- **Output Shape:** `(samples,)` — binary labels

---

## Model Architectures

All three models share the **same architecture depth** for a fair comparison:

### Simple RNN

```
Input (30, 1)
      │
      ▼
SimpleRNN (64 units)
      │
      ▼
Dropout (0.2)
      │
      ▼
Dense (32 neurons, ReLU)
      │
      ▼
Dense (1 neuron, Sigmoid)
```

### LSTM

```
Input (30, 1)
      │
      ▼
LSTM (64 units)
      │
      ▼
Dropout (0.2)
      │
      ▼
Dense (32 neurons, ReLU)
      │
      ▼
Dense (1 neuron, Sigmoid)
```

### GRU

```
Input (30, 1)
      │
      ▼
GRU (64 units)
      │
      ▼
Dropout (0.2)
      │
      ▼
Dense (32 neurons, ReLU)
      │
      ▼
Dense (1 neuron, Sigmoid)
```

---

## Model Compilation

All models use the same compilation settings:

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy

---

## Model Training

```python
EPOCHS = 30
BATCH_SIZE = 32

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

- **Epochs:** 30
- **Batch Size:** 32
- **Early Stopping:** Patience of 5 epochs on validation loss
- **Training time** is recorded for each model for comparison.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | How many predicted positives are actually positive |
| **Recall** | How many actual positives are correctly predicted |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **Training Time** | Time taken to train each model (in seconds) |

```python
def evaluate_model(y_true, y_pred, model_name, train_time):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {'Model': model_name, 'Accuracy': acc, ...}
```

---

## Visualizations

- **Daily Sales Plot** — Complete time series of aggregated daily sales.
- **Class Distribution** — Bar chart showing balance of increase vs decrease classes.
- **Training Curves** — Training vs validation accuracy and loss for each model.
- **Validation Accuracy Comparison** — All three models on the same plot.
- **Validation Loss Comparison** — All three models on the same plot.
- **Confusion Matrices** — Side-by-side heatmaps for RNN, LSTM, and GRU.
- **Metric Comparison Bar Charts** — Accuracy, Precision, Recall, F1 Score.
- **Training Time Comparison** — Bar chart of training durations.

---

## Key Concepts

### Simple RNN
Processes sequential data by maintaining a hidden state across time steps. Prone to the **vanishing gradient problem**, making it less effective for long sequences.

### LSTM (Long Short-Term Memory)
Uses three gates (**forget**, **input**, **output**) and a cell state to selectively remember or forget information over long sequences. Solves the vanishing gradient problem.

### GRU (Gated Recurrent Unit)
A simplified variant of LSTM with two gates (**reset** and **update**). Fewer parameters than LSTM, often trains faster while achieving comparable performance.

---

## Algorithm

1. **Load** the `sales.csv` dataset.
2. **Convert** the `date` column to datetime and sort chronologically.
3. **Handle** missing values by dropping rows with null `sum_total`.
4. **Aggregate** daily sales by summing `sum_total` per date.
5. **Create binary target** — `1` if next day's sales increase, `0` otherwise.
6. **Split** data chronologically into 80% train / 20% test.
7. **Scale** features using MinMaxScaler (fit on training data only).
8. **Create sequences** — sliding window of 30 days as input.
9. **Build** three models (RNN, LSTM, GRU) with identical architecture.
10. **Train** each model with early stopping on validation loss.
11. **Predict** on the test set and convert probabilities to binary classes (threshold = 0.5).
12. **Evaluate** using Accuracy, Precision, Recall, and F1 Score.
13. **Visualize** training curves, confusion matrices, and comparison charts.
14. **Compare** all models in a final summary table.

---

## Applications

- Retail Sales Trend Prediction
- Stock Market Direction Forecasting
- Customer Demand Classification
- Inventory Replenishment Decisions
- Financial Risk Assessment
- Anomaly Detection in Time Series

---

## Conclusion

This project successfully implements and compares three recurrent neural network architectures — Simple RNN, LSTM, and GRU — for binary sequence classification on retail sales data. All models were trained under identical conditions to ensure a fair comparison. The experiment demonstrates how different recurrent architectures handle temporal dependencies in sales data, with evaluation across multiple classification metrics. The comparison of training times, accuracy, and F1 scores provides practical insights into the trade-offs between model complexity and performance for sequence classification tasks.

---
