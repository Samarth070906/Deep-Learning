# Exp 8 — BERT Sentiment Analysis on Amazon Reviews

## Objective

Implement a pre-trained **BERT** model for **binary sentiment classification** on the **Amazon Polarity** dataset. Fine-tune `bert-base-uncased` to classify Amazon product reviews as **Positive** or **Negative**.

---

## Dataset

**Dataset Used:** [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) (via Hugging Face `datasets`)

- **Full Dataset:** 3,600,000 training + 400,000 test reviews
- **Subset Used:**
  - **Training:** 8,000 reviews
  - **Validation:** 2,000 reviews
  - **Test:** 2,000 reviews
- **Classes:**
  - `0` → Negative
  - `1` → Positive
- **Input:** Review title + review content combined into a single text field

> **Note:** The full dataset is downloaded at runtime via `load_dataset("fancyzhx/amazon_polarity")`. A stratified subset is used for practical fine-tuning.

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

---

## Project Structure

```
Exp_8/
├── BERT_Amazon_Reviews.ipynb
├── bert_amazon_sentiment_model/   (saved model — not in repo)
├── bert_amazon_results/           (training outputs — not in repo)
├── bert_venv/                     (virtual environment — not in repo)
└── README.md
```

---

## Data Preprocessing Steps

1. **Load** the Amazon Polarity dataset from Hugging Face.
2. **Convert** to Pandas DataFrames for exploration.
3. **Combine** review `title` and `content` into a single `text` column.
4. **Visualize** class distribution to confirm balance.
5. **Sample** a stratified subset (10,000 train + 2,000 test).
6. **Split** training subset into 80% train / 20% validation.
7. **Convert** DataFrames back to Hugging Face `Dataset` objects.

---

## Tokenization

- **Tokenizer:** `bert-base-uncased` (AutoTokenizer)
- **Max Sequence Length:** 256 tokens
- **Truncation:** Enabled for reviews exceeding 256 tokens
- **Padding:** Dynamic padding via `DataCollatorWithPadding` (pads each batch to the longest sequence in that batch)

---

## Model Architecture

**Pre-trained Model:** `bert-base-uncased` with a sequence classification head

```
BERT (bert-base-uncased)
    │
    ▼
12 Transformer Encoder Layers
    │
    ▼
[CLS] Token Representation (768-dim)
    │
    ▼
Linear Classification Head (768 → 2)
    │
    ▼
Output: [Negative, Positive]
```

| Component | Details |
|-----------|---------|
| **Base Model** | `bert-base-uncased` (110M parameters) |
| **Hidden Size** | 768 |
| **Attention Heads** | 12 |
| **Transformer Layers** | 12 |
| **Output Classes** | 2 (Negative, Positive) |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW (default in Hugging Face Trainer) |
| **Learning Rate** | 2e-5 |
| **Epochs** | 3 |
| **Batch Size (Train)** | 16 |
| **Batch Size (Eval)** | 64 |
| **Weight Decay** | 0.01 |
| **Evaluation Strategy** | Per epoch |
| **Seed** | 42 |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | How many predicted positives are actually positive |
| **Recall** | How many actual positives are correctly predicted |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | Breakdown of TP, TN, FP, FN |

---

## Visualizations

- **Class Distribution** — Bar chart showing balance of Negative vs Positive reviews.
- **Confusion Matrix** — Heatmap of true vs predicted sentiment labels.
- **Training & Validation Loss** — Line plot showing loss over training progress.
- **Evaluation Metrics Bar Chart** — Accuracy, Precision, Recall, and F1-Score comparison.

---

## Key Concepts

### BERT (Bidirectional Encoder Representations from Transformers)
A pre-trained Transformer model that reads text **bidirectionally** (both left-to-right and right-to-left simultaneously). BERT learns deep contextualized word representations and can be **fine-tuned** on downstream tasks like sentiment classification with minimal architecture changes.

### Fine-Tuning
The process of taking a model pre-trained on a large general corpus and further training it on a smaller task-specific dataset. Only the final classification head (and optionally the full model weights) are updated during fine-tuning.

### Tokenization
BERT uses **WordPiece tokenization**, which splits words into subword units. This allows it to handle out-of-vocabulary words by breaking them into known subword tokens (e.g., "unhappiness" → "un", "##happiness").

### [CLS] Token
A special token prepended to every input. Its final hidden state is used as the aggregate sequence representation for classification tasks.

---

## Algorithm

1. **Load** the Amazon Polarity dataset from Hugging Face.
2. **Combine** review title and content into a single text field.
3. **Sample** a stratified subset (10,000 training + 2,000 test reviews).
4. **Split** training data into 80% train / 20% validation.
5. **Tokenize** all text using the BERT tokenizer (max length 256).
6. **Load** `bert-base-uncased` with a 2-class classification head.
7. **Fine-tune** BERT using Hugging Face `Trainer` for 3 epochs.
8. **Evaluate** on the validation set after each epoch.
9. **Predict** on the held-out test set.
10. **Calculate** Accuracy, Precision, Recall, and F1-Score.
11. **Visualize** confusion matrix, loss curves, and metric bar charts.
12. **Test** on custom user-written reviews for live sentiment prediction.
13. **Save** the fine-tuned model and tokenizer for future use.

---

## Custom Review Prediction

The notebook includes a function to classify any custom review:

```python
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).item()
    label = "Positive" if prediction == 1 else "Negative"
    confidence = probabilities[0][prediction].item()
    return label, confidence
```

**Example Predictions:**

| Review | Prediction |
|--------|------------|
| "This product is excellent. I am very happy with the quality." | Positive ✅ |
| "The product stopped working quickly and I regret buying it." | Negative ✅ |
| "Amazing quality and fast delivery. I would definitely buy it again." | Positive ✅ |
| "Very disappointing experience. The product was poorly made." | Negative ✅ |

---

## Applications

- E-commerce Review Sentiment Analysis
- Customer Feedback Classification
- Brand Reputation Monitoring
- Social Media Sentiment Tracking
- Product Quality Assessment
- Automated Review Moderation

---

## Conclusion

This experiment demonstrates how a pre-trained **BERT (`bert-base-uncased`)** model can be fine-tuned for binary sentiment classification on Amazon product reviews. The reviews were tokenized using the BERT WordPiece tokenizer, and the model was adapted to distinguish between **Negative** and **Positive** sentiments. Evaluation was performed using accuracy, precision, recall, F1-score, and a confusion matrix. Custom review examples were also tested to demonstrate real-world sentiment prediction capability.

### Key Learning Outcomes
- Understanding pre-trained Transformer models (BERT)
- Using a BERT tokenizer for text preprocessing
- Fine-tuning BERT for text classification using Hugging Face Trainer
- Leveraging GPU acceleration for deep learning
- Evaluating NLP classification models with multiple metrics
- Performing sentiment prediction on new, unseen text

---
