"""
Test script to verify the trained model works correctly.
"""
import joblib
import re

def preprocess_text(text):
    """Basic text preprocessing."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def test_model():
    """Test the model with sample messages."""
    
    # Load model and vectorizer
    print("Loading model and vectorizer...")
    model = joblib.load('models/model.joblib')
    vectorizer = joblib.load('models/vectorizer.joblib')
    print("✓ Model loaded successfully\n")
    
    # Test messages
    test_messages = [
        ("I need my money back immediately", "refund_request"),
        ("This service is terrible and I'm very disappointed", "complaint"),
        ("The app keeps crashing when I try to login", "technical_issue"),
        ("CONGRATULATIONS! You've won $1,000,000!!!", "spam"),
        ("I can't log into my account", "account_problem"),
        ("Please process my refund as soon as possible", "refund_request"),
        ("Your staff was rude and unprofessional", "complaint"),
    ]
    
    print("="*70)
    print("MODEL PREDICTION TESTS")
    print("="*70)
    
    correct = 0
    total = len(test_messages)
    
    for message, expected_category in test_messages:
        # Preprocess and predict
        processed = preprocess_text(message)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        confidence = max(probabilities)
        
        # Check if correct
        is_correct = prediction == expected_category
        if is_correct:
            correct += 1
            status = "✓ CORRECT"
        else:
            status = "✗ INCORRECT"
        
        print(f"\n{status}")
        print(f"Message: \"{message}\"")
        print(f"Expected: {expected_category}")
        print(f"Predicted: {prediction}")
        print(f"Confidence: {confidence*100:.2f}%")
        print("-"*70)
    
    # Summary
    accuracy = (correct / total) * 100
    print(f"\n{'='*70}")
    print(f"RESULTS: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    print(f"{'='*70}")
    
    if accuracy >= 80:
        print("\n✓ Model is performing well!")
    else:
        print("\n⚠ Model may need more training or better data.")

if __name__ == "__main__":
    test_model()
