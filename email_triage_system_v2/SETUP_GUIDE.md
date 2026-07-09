# 📧 Email Classification and Prioritization System - Setup Guide

## 🎯 Overview

This guide explains how to set up, train, and run the **Email Classification and Prioritization System** locally on your computer.

The system uses:
1. **BERT (Deep Learning)** to classify emails/messages into 8 distinct departments.
2. **VADER (Rule-based NLP)** to evaluate sentiment, emotional tone, and urgency.
3. **SQLite** to log all analysis metrics locally.

---

## 🚀 Step-by-Step Installation

### 1. Install System Dependencies

Make sure you have **Python 3.8+** installed. Then, open your terminal and install the required libraries:

```bash
pip install -r requirements.txt
```

### 2. Verify Your CUDA Setup (Optional)
If you have an NVIDIA GPU and want to run BERT classification faster:
- Ensure PyTorch is installed with CUDA support.
- If no GPU is available, the system will automatically fall back to CPU with no code changes needed.

---

## 🧠 Model Training & Customization

The system comes with a pre-trained BERT model checkpoint. If you want to train it from scratch or retrain it on new custom training data, follow these steps:

### 1. Prepare Training Data
The training data is stored in `data/email_training_data.csv`. The file requires two columns:
- `message`: The content of the text/email.
- `category`: One of the 8 business categories (*sales, support, billing, hr, it, legal, complaint, general*).

### 2. Run the Training Script

To start fine-tuning the BERT model:

```bash
python src/train_bert.py
```

This script will:
- Load the base model weights from `bert-base-uncased`.
- Fine-tune it on the dataset for 4 epochs.
- Output metrics (Accuracy, Precision, Recall, F1).
- Save the trained checkpoint and label encoder inside `models/bert/`.

### 3. Run Benchmarking (Optional)

To evaluate classification performance:

```bash
python src/evaluate_models.py
```

---

## 💻 Running the Application

### Option A: Command Line Demonstration

To run a quick command-line demo that processes 5 sample texts and displays their predicted categories, priority levels, sentiment scores, and urgency keywords:

```bash
python demo.py
```

---

### Option B: Interactive Web Dashboard

To run the interactive Streamlit dashboard:

```bash
streamlit run src/app.py
```

Access the user interface in your web browser at: **`http://localhost:8501`**

#### Web Dashboard Features:
- **Manual Classification Tab**: Type any subject and body, click "Classify Email", and inspect category probabilities, priority levels, and sentiment metrics immediately.
- **Bulk Processing Tab**: Upload a CSV file (e.g. `emails_dataset_unique.csv`) containing `subject` and `body` columns to process hundreds of texts at once, view summary charts, and download a finalized Excel/CSV report.

---

## 📁 File Structure Reference

```
email_triage_system_v2/
├── src/
│   ├── app.py                   # Streamlit dashboard interface
│   ├── bert_classifier.py       # BERT classification inference
│   ├── sentiment_analyzer.py    # VADER sentiment and priority heuristics
│   ├── database.py              # SQLite local log manager
│   ├── train_bert.py            # BERT model training script
│   └── evaluate_models.py       # Performance evaluation script
├── data/
│   ├── email_training_data.csv  # Training dataset
│   └── email_triage.db          # SQLite audit log file
├── models/
│   └── bert/                    # Saved BERT model weights & tokenizers
├── demo.py                      # CLI test script
└── requirements.txt             # Project requirements
```

---

## 🚨 Heuristics Logic

- **Category**: Derived from the BERT Softmax probability output.
- **Sentiment**: Calculated by VADER. Positive (compound $\ge$ 0.05), Neutral (between -0.05 and 0.05), Negative (compound $\le$ -0.05).
- **Urgency Level**: Derived from keyword matching (e.g. *critical, urgent, ASAP, immediately*).
- **Priority**:
  - **High**: Negative sentiment + urgent keywords.
  - **Medium**: Negative sentiment OR medium urgency.
  - **Low**: Positive/neutral sentiment + no urgency.
