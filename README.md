📧 Intelligent Email Classification and Prioritization System
An AI-powered classification and prioritization engine for business communication.

This system leverages Bidirectional Encoder Representations from Transformers (BERT) and VADER Sentiment Analysis to automatically categorize incoming text/emails and determine their priority level in real-time. It runs entirely locally, requiring no external APIs.

✨ Key Features
🤖 BERT-based Categorization - Deep learning classifier fine-tuned on custom business communication for high accuracy.
📊 Sentiment Analysis - Determines emotional tone (Positive, Neutral, Negative) using VADER.
🚨 Priority & Urgency Detection - Mixed heuristics model mapping category, sentiment, and urgency keywords to Low, Medium, and High priority levels.
📂 Bulk CSV Processing - Upload text datasets (with subject/body columns), classify/prioritize them in batch, and download results instantly.
📝 Local Data Logging - Stores audit trails in a local SQLite database with multi-threaded connection pooling.
💻 Web Interface - Sleek Streamlit dashboard for testing individual texts manually or bulk processing files.
🚀 Quick Start
1. Install Dependencies
Ensure Python 3.8+ is installed, then run:

pip install -r requirements.txt
2. Launch the Web Dashboard
Start the local Streamlit server:

streamlit run src/app.py
Open your browser and navigate to: http://localhost:8501

📊 Classification Categories
The model is trained on 8 distinct business departments/topics:

Sales - Product inquiries, pricing, quotes, and demo requests.
Support - Technical issues, troubleshooting, and bug reports.
Billing - Payment failures, invoices, and refund requests.
HR - Employee benefits, job inquiries, and internal policies.
IT - User credentials, software installation, and hardware access.
Legal - Contracts, agreements, and compliance inquiries.
Complaint - Critical customer escalations and service delivery issues.
General - General feedback, spam/unrelated items, or misc inquiries.
🧠 System Architecture
                      ┌──────────────────────┐
                      │    Text Input        │
                      │ (Subject + Body/Msg) │
                      └──────────┬───────────┘
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │   Processing Engine    │
                     └──────────┬─────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌──────────────┐                                ┌──────────────┐
│  BERT Model  │                                │ VADER Engine │
│ (Category)   │                                │ (Sentiment)  │
└───────┬──────┘                                └──────┬───────┘
        │                                              │
        └───────────────────────┬──────────────────────┘
                                │
                                ▼
                     ┌────────────────────────┐
                     │   Priority Heuristics  │
                     │  (Category + Sentiment │
                     │  + Urgency Keywords)   │
                     └──────────┬─────────────┘
                                │
                                ▼
                     ┌────────────────────────┐
                     │    Final Decision      │
                     │ (Category + Priority)  │
                     └──────────┬─────────────┘
                                │
                                ▼
                     ┌────────────────────────┐
                     │ SQLite Log Database   │
                     └────────────────────────┘
🔧 Technical details & Tech Stack
Component	Technology	Description
Deep Learning Engine	BERT (bert-base-uncased)	Fine-tuned on a custom dataset for multi-class classification.
Sentiment Analysis	VADER	Computes compound sentiment score to detect emotional urgency.
Frontend UI	Streamlit	Multi-threaded local web app for interactive validation.
Database	SQLite	Local audit logger with thread-safe configuration (check_same_thread=False).
Visualization	Plotly	Renders dynamic metrics and category distributions.
📁 Project Structure
email_triage_system_v2/
├── src/                      # Source code
│   ├── app.py                # Streamlit web interface
│   ├── bert_classifier.py    # BERT inference module
│   ├── sentiment_analyzer.py # VADER sentiment & urgency detection
│   ├── database.py           # SQLite audit logging database
│   ├── train_bert.py         # BERT model fine-tuning script
│   └── evaluate_models.py    # Performance benchmarking script
├── data/                     # Training datasets & local SQLite database
├── models/                   # Saved fine-tuned model checkpoints
├── demo.py                   # Command-line classification demonstration
├── requirements.txt          # Project dependencies
└── USER_GUIDE.md             # Guide on using the Web Dashboard
🔒 Security & Privacy
100% Local Execution: All computations (BERT inference, VADER analysis) run on your local CPU/GPU. No external server API calls are made, ensuring complete data privacy.
SQLite Storage: Data logs are kept in a local database file (data/email_triage.db).
