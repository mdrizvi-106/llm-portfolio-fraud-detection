# LLM Financial Anomaly Detection Pipeline

A hybrid ML + LLM pipeline that detects fraudulent credit card transactions and generates plain-English, audit-ready explanations for each flagged case — mirroring real-world compliance workflows.

---

## Overview

Most fraud detection systems produce a score or a binary flag — but regulators and compliance teams need *reasons*. This project solves that by combining:

- A **Random Forest classifier** to detect fraud with high precision
- A **Large Language Model (Llama 3.1 via Groq)** to explain *why* each flagged transaction looks suspicious

The result is a pipeline that is both accurate and interpretable.

---

## Dataset

**[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** — ULB Machine Learning Group via Kaggle

| Stat | Value |
|------|-------|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (0.17%) |
| Features | V1–V28 (PCA-anonymised), Amount, Time |
| Target column | `Class` (0 = Legitimate, 1 = Fraud) |

> The dataset is highly imbalanced — only 0.17% of transactions are fraudulent.

---

## Pipeline Steps

### 1. Data Loading
Downloaded via `kagglehub`. The dataset contains anonymised PCA features (V1–V28), transaction `Amount`, `Time`, and the ground truth `Class` label.

### 2. Feature Engineering
All columns except `Class` (target) and `Time` (irrelevant for pattern detection) are used as input features — resulting in 29 features: V1–V28 and Amount.

### 3. Model Training
A `RandomForestClassifier` is trained with:
- `class_weight='balanced'` to handle class imbalance
- 80/20 stratified train/test split
- `n_jobs=-1` for parallel training

**Test set performance:**

| Metric | Legitimate | Fraud |
|--------|-----------|-------|
| Precision | 1.00 | 0.96 |
| Recall | 1.00 | 0.76 |
| F1-Score | 1.00 | 0.85 |

### 4. Flagging Suspicious Transactions
The trained model flags transactions predicted as fraud from the test set. To control API costs, a random sample of **10 flagged transactions** is selected for LLM analysis.

Of the 77 flagged transactions:
- **74** were true fraud (True Positives)
- **3** were legitimate (False Positives)

### 5. LLM Explanation (via Groq)
Each sampled transaction is sent to **Llama 3.1 8B Instant** with a structured prompt containing:
- Top fraud-driving feature values: `V14`, `V10`, `V4`, `V11`, `V12`, `Amount`
- Context about average transaction amount (£88.40)
- Instructions to explain what looks suspicious and what a compliance analyst should investigate

The LLM returns a 3–4 sentence plain-English explanation for each flagged transaction.

### 6. Pipeline Evaluation
Of the 10 transactions sent to the LLM:
- **9** were actual fraud ✅
- **1** was a false positive ❌
- **Precision on LLM input: 90%**

### 7. Audit Report Export
All results are saved to `audit_report.csv` containing:

| Column | Description |
|--------|-------------|
| `Transaction_ID` | Original dataframe index |
| `Amount` | Transaction amount in GBP |
| `True_Label` | Ground truth (FRAUD / Legitimate) |
| `ML_Flag` | Always "Flagged" (only flagged transactions are exported) |
| `LLM_Explanation` | Plain-English explanation from Llama 3.1 |

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3 |
| ML Model | `scikit-learn` RandomForestClassifier |
| LLM | Llama 3.1 8B Instant via [Groq](https://groq.com/) |
| Data | `kagglehub`, `pandas` |
| Environment | Google Colab |

---

## Setup & Usage

### Prerequisites
```bash
pip install groq kagglehub scikit-learn pandas numpy
```

### API Key
This project uses the Groq API. Store your key as a Colab secret named `GROQ_API_KEY`:
```python
from google.colab import userdata
client = Groq(api_key=userdata.get('GROQ_API_KEY'))
```
Never hardcode API keys in notebooks.

### Run
Open `llm_portfolio_v2-2.ipynb` in Google Colab and run all cells in order.

---

## Key Design Decisions

- **Why Random Forest?** Strong baseline for tabular fraud data; handles class imbalance well with `class_weight='balanced'`; fast to train.
- **Why sample only 10 for LLM?** LLM API calls have cost and latency — in production this would be batched or throttled.
- **Why exclude `Time`?** Transaction time is not a meaningful fraud signal in this dataset and adds noise.
- **Why Groq / Llama 3.1?** Fast inference, free tier available, sufficient for structured analytical prompts.

---

## Output

`audit_report.csv` — a compliance-ready report linking each ML flag to a human-readable justification, suitable for audit trails or analyst review queues.

---

## Limitations

- The LLM explanations are generated from anonymised PCA features — real-world compliance analysts would have access to merchant names, locations, and raw amounts.
- The 10-transaction sample is for demonstration; production pipelines would process all flagged cases.
- LLM explanations are not verified for factual accuracy — they should be treated as analyst prompts, not conclusions.

---

## Author

Built as a portfolio project demonstrating the integration of classical ML with LLM explainability for financial compliance use cases.
