# LSTM-Based Time Series Forecasting on Sales Data

## Overview

This project demonstrates the implementation of an **LSTM (Long Short-Term Memory)** neural network for **time-series forecasting** using a real-world sales dataset. The model learns temporal patterns from historical daily sales data and predicts future sales values.

---

## Objectives

- Load and preprocess a large-scale retail sales dataset.
- Aggregate transactional data into a daily time series.
- Normalize the data using Min-Max Scaling.
- Create sliding window sequences for LSTM input.
- Build and train a multi-layer LSTM model.
- Evaluate model performance using RMSE, MAE, and MAPE.
- Visualize actual vs predicted sales and training loss curves.

---

## Dataset

**Dataset Used:** Retail Sales Dataset (`sales.csv`)

- **Records:** ~3.8 million transactions
- **Key Columns:**
  - `date` — Transaction date
  - `sum_total` — Total sale amount
- **Date Range:** Multiple years of daily sales data
- **Preprocessing:** Aggregated to daily total sales

> **Note:** The dataset is not included in this repository due to its large size (~379 MB).

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

---

## Project Structure

```
Exp_4/
│
├── LSTM_Time_Series_Foreecasting_.ipynb
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
2. **Remove** unnecessary index columns (`Unnamed: 0`).
3. **Convert** the `date` column to datetime format.
4. **Aggregate** sales by date (`sum_total` grouped by `date`).
5. **Sort** data chronologically.
6. **Check** for missing values and missing dates.
7. **Compute** basic statistics (mean, min, max, std).

---

## Train/Test Split

- **Split Ratio:** 80% Training, 20% Testing
- **Method:** Chronological split (no shuffling) to preserve temporal order.

---

## Data Normalization

- **Scaler Used:** `MinMaxScaler` (range 0 to 1)
- **Important:** Scaler is fit only on the training data to prevent data leakage.

---

## LSTM Sequence Creation

- **Time Steps (Window Size):** 30 days
- Each input sample consists of the previous 30 days of sales values.
- The target is the sales value on day 31.

```python
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
```

---

## Neural Network Architecture

```
Input (30 time steps, 1 feature)
        │
        ▼
LSTM Layer (64 units, return_sequences=True)
        │
        ▼
LSTM Layer (32 units, return_sequences=False)
        │
        ▼
Dense Layer (16 neurons, ReLU)
        │
        ▼
Output Layer (1 neuron, Linear)
```

---

## Model Compilation

```python
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)
```

- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)

---

## Model Training

```python
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)
```

- **Epochs:** 50
- **Batch Size:** 32
- **Validation Split:** 10% of training data

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error — penalizes large errors |
| **MAE** | Mean Absolute Error — average absolute deviation |
| **MAPE** | Mean Absolute Percentage Error — percentage-based error |

```python
rmse = np.sqrt(mean_squared_error(actual, predictions))
mae = mean_absolute_error(actual, predictions)
mape = np.mean(np.abs((actual - predictions) / actual)) * 100
```

---

## Visualizations

- **Daily Sales Plot** — Complete time series of aggregated daily sales.
- **Training vs Validation Loss** — Loss curve over 50 epochs.
- **Actual vs Predicted Sales** — Comparison of real and forecasted values on the test set.
- **Results Table** — DataFrame showing actual, predicted, and absolute error for each test sample.

---

## Key Concepts

### Why LSTM for Time Series?

LSTMs are a type of Recurrent Neural Network (RNN) designed to learn long-term dependencies in sequential data. Unlike standard RNNs, LSTMs use **gates** (forget, input, output) to control information flow, making them effective for time-series forecasting.

### Sliding Window Approach

The model uses a sliding window of 30 days to predict the next day's sales. This approach converts a time-series problem into a supervised learning problem.

---

## Applications

- Sales Forecasting
- Stock Price Prediction
- Weather Forecasting
- Energy Demand Prediction
- Inventory Management
- Financial Planning

---

## Conclusion

This project successfully demonstrates the use of LSTM networks for time-series forecasting on retail sales data. The model captures temporal patterns in daily sales and generates predictions with measurable accuracy. The evaluation metrics (RMSE, MAE, MAPE) confirm the model's ability to learn and generalize from historical sales trends. The sliding window approach with a 30-day lookback period proves effective for capturing short-to-medium term sales patterns.

---
