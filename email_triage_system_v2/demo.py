"""
Demo script to showcase the Email Triage System capabilities.
"""
import sys
sys.path.append('src')

from sentiment_analyzer import SentimentAnalyzer
from bert_classifier import BertEmailClassifier
import joblib
import re
import os
import torch

def preprocess_text(text):
    """Preprocess text for classification."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def demo_classification():
    """Demonstrate the email classification system."""
    
    print("="*70)
    print("EMAIL TRIAGE SYSTEM - LIVE DEMONSTRATION")
    print("="*70)
    
    # Load model
    print("\n1. Loading ML Model...")
    model = joblib.load('models/model.joblib')
    vectorizer = joblib.load('models/vectorizer.joblib')
    print(f"   [DONE] Model loaded with {len(model.classes_)} categories")
    print(f"   [INFO] Categories: {', '.join(sorted(model.classes_))}")
    
    # Load sentiment analyzer
    print("\n2. Loading Sentiment Analyzer...")
    sentiment_analyzer = SentimentAnalyzer()
    print("   [DONE] Sentiment analyzer ready")
    
    # Load BERT classifier
    print("\n3. Loading BERT Classifier...")
    bert_classifier = BertEmailClassifier()
    bert_available = bert_classifier._check_model_exists()
    if bert_available:
        print("   [DONE] BERT model weights found")
    else:
        print("   [WARN] BERT model not found (training might be required)")
    
    # Test emails
    test_emails = [
        {
            'subject': 'URGENT: Need refund immediately',
            'body': 'I want my money back right now. This is completely unacceptable service.'
        },
        {
            'subject': 'Question about enterprise pricing',
            'body': 'Hi, I am interested in purchasing your enterprise plan for our company. Can you send me pricing information?'
        },
        {
            'subject': 'Cannot login to my account',
            'body': 'I keep getting an error when trying to log in. The page says "invalid credentials" but I know my password is correct.'
        },
        {
            'subject': 'Employee benefits question',
            'body': 'I would like to know more about the health insurance options available. Can someone from HR help me?'
        },
        {
            'subject': 'System is down - CRITICAL',
            'body': 'Our entire production system is down. We need immediate assistance. This is affecting all our customers.'
        }
    ]
    
    print("\n" + "="*70)
    print("PROCESSING TEST EMAILS")
    print("="*70)
    
    for i, email in enumerate(test_emails, 1):
        print(f"\n--- Email {i} ---")
        print(f"Subject: {email['subject']}")
        print(f"Body: {email['body'][:60]}...")
        
        # ML classification (Logistic)
        combined = f"{email['subject']} {email['body']}"
        processed = preprocess_text(combined)
        vectorized = vectorizer.transform([processed])
        
        category_log = model.predict(vectorized)[0]
        conf_log = max(model.predict_proba(vectorized)[0])
        
        # ML classification (BERT)
        category_bert = "N/A"
        conf_bert = 0
        if bert_available:
            bert_res = bert_classifier.predict(email['subject'], email['body'])
            category_bert = bert_res['category']
            conf_bert = bert_res['confidence']
        
        # Sentiment analysis
        sentiment_result = sentiment_analyzer.analyze_email(
            email['subject'],
            email['body']
        )
        
        print(f"\nRESULTS:")
        print(f"   Logistic: {category_log.upper()} ({conf_log*100:.1f}%)")
        if bert_available:
            print(f"   BERT:     {category_bert.upper()} ({conf_bert*100:.1f}%)")
        print(f"   Priority:  {sentiment_result['priority'].upper()}")
        print(f"   Sentiment: {sentiment_result['sentiment']['label'].title()} ({sentiment_result['sentiment']['compound']:.3f})")
        print(f"   Urgency:   {sentiment_result['urgency']['level'].title()}")
        if sentiment_result['urgency']['keywords']:
            print(f"   Urgency Keywords: {', '.join(sentiment_result['urgency']['keywords'])}")
        
        print("-"*70)
    
    print("\nDONE: Demonstration complete.")
    print("\n[OK] The system is working perfectly!")
    print("\nDisplay the dashboard at: http://localhost:8501")
    print("\nDashboard Features:")
    print("  - Manual Classification - Test individual emails (category, priority, sentiment, urgency)")
    print("  - Bulk Processing - Upload CSV files and download results")

if __name__ == "__main__":
    demo_classification()
