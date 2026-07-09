# 🚀 Email Classification & Prioritization System - User Guide

This user guide describes how to operate the Web Dashboard to classify and prioritize emails or text messages.

---

## 🌐 Launching the Dashboard

1. Open your terminal in the project directory.
2. Run the command:
   ```bash
   streamlit run src/app.py
   ```
3. Open your browser and navigate to: **`http://localhost:8501`**

---

## 📊 Dashboard Features

### 1. Manual Classification Tab
This tab lets you analyze individual text/email messages in real-time.

**How to use:**
1. Enter the **Email Subject** (optional).
2. Enter the **Email Body** (required).
3. Click **🔍 Classify Email**.

**Results Displayed:**
- **Category**: The department category predicted by the BERT model (e.g. Sales, Support, Billing).
- **Confidence**: How confident the BERT model is in its category prediction.
- **Priority**: System-calculated priority (High, Medium, Low) based on urgency and emotional sentiment.
- **Sentiment**: Tone analysis (Positive, Neutral, Negative) and compound score.
- **Urgency Analysis**: Highlighted keywords indicating time-sensitivity (e.g. "ASAP").
- **Detailed Probability Chart**: A horizontal bar chart showing probability scores for all 8 categories.

**Test Example to Try:**
- **Subject**: `URGENT: Charged twice for my order`
- **Body**: `I need a refund immediately. Please fix this ASAP.`
- **Expected Result**: Category: *Billing*, Priority: *High*, Sentiment: *Negative*.

---

### 2. Bulk Processing Tab
This tab is designed to process multiple records simultaneously using a CSV file.

**How to use:**
1. Upload a CSV file containing `subject` and `body` columns.
   - *A sample file `sample_messages.csv` is provided in the project root folder for demonstration.*
2. Verify the preview of the loaded dataset.
3. Click **🚀 Classify All Emails**.
4. The system will iterate through all rows, updating the progress bar.
5. Once complete, a results table will appear showing classification, confidence, priority, and sentiment columns added to your original file.
6. Click **📥 Download Results** to export the updated dataset as a CSV.
7. Review the **Category Summary** and **Priority Summary** charts to view distribution statistics.

---

## 🎯 Classification & Priority Logic

### Email Categories
- **Sales**: Quotes, Pricing, Demo Requests, Product Interest.
- **Support**: Bug reports, how-to questions, technical troubleshooting.
- **Billing**: Refund requests, billing discrepancies, invoices.
- **HR**: Recruitment, employee benefits, internal policies.
- **IT**: Active directory logins, password resets, hardware setup.
- **Legal**: Non-Disclosure Agreements, terms of service, compliance.
- **Complaint**: Service delivery issues, customer escalations.
- **General**: Miscellaneous inquiries.

### Priority Detection Matrix
The system determines priority levels using a combination of **sentiment tone** (computed by VADER) and **urgency keywords** (e.g., *urgent, critical, immediately, ASAP, emergency*):

- 🔴 **High Priority**: Message has **negative sentiment** AND contains **high-urgency keywords**.
- 🟡 **Medium Priority**: Message has **negative sentiment** OR contains **medium-urgency keywords**.
- 🟢 **Low Priority**: Message has **positive/neutral sentiment** AND contains **no urgency keywords**.

---

## 📝 Local Logging Database

All classification runs are logged in the background into the SQLite database file:
`data/email_triage.db`

The database logs:
- **Timestamp** of the classification request.
- **Sender/Subject** details.
- **Predicted Category** and model confidence score.
- **Sentiment label** and score.
- **Urgency level** and calculated Priority.
- **Execution processing time** (in milliseconds).

*Note: Since the system runs entirely offline, all logged database entries remain secure on your local workstation.*
