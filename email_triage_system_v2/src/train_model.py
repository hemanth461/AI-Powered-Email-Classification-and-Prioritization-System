"""
Enhanced training script for email triage system.
Trains model on business email categories with improved features.
"""
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import re

def preprocess_text(text):
    """Basic text preprocessing."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_prepare_data(data_path='data/email_training_data.csv'):
    """Load and preprocess the dataset."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Preprocess messages
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    print(f"✓ Loaded {len(df)} messages")
    print(f"✓ Categories: {sorted(df['category'].unique().tolist())}")
    
    return df

def train_model(df, test_size=0.2, random_state=42):
    """Train the classification model."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_message'], 
        df['category'], 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['category']
    )
    
    print(f"\n✓ Training set: {len(X_train)} messages")
    print(f"✓ Test set: {len(X_test)} messages")
    
    # Create TF-IDF vectorizer
    print("\nVectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Use unigrams and bigrams
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"✓ Feature matrix shape: {X_train_tfidf.shape}")
    
    # Train Logistic Regression model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        multi_class='multinomial',
        solver='lbfgs',
        C=1.0
    )
    
    model.fit(X_train_tfidf, y_train)
    print("✓ Model trained successfully")
    
    # Evaluate model
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    categories = sorted(df['category'].unique())
    
    # Print header
    print(f"{'':15}", end='')
    for cat in categories:
        print(f"{cat[:10]:12}", end='')
    print()
    
    # Print matrix
    for i, cat in enumerate(categories):
        print(f"{cat[:15]:15}", end='')
        for j in range(len(categories)):
            print(f"{cm[i][j]:12}", end='')
        print()
    
    return model, vectorizer

def save_model(model, vectorizer, model_dir='models'):
    """Save the trained model and vectorizer."""
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'model.joblib')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"\n✓ Model saved to: {model_path}")
    print(f"✓ Vectorizer saved to: {vectorizer_path}")

def main():
    """Main training pipeline."""
    print("="*70)
    print("EMAIL TRIAGE MODEL TRAINING")
    print("="*70)
    
    # Load data
    df = load_and_prepare_data()
    
    # Train model
    model, vectorizer = train_model(df)
    
    # Save artifacts
    save_model(model, vectorizer)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nThe model is now ready for email triage.")
    print("Categories:", sorted(model.classes_.tolist()))

if __name__ == "__main__":
    main()
