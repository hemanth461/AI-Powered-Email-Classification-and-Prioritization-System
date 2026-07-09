"""
Generate synthetic dataset for message categorization training.
Creates a CSV file with labeled messages across different categories.
"""
import pandas as pd
import random
import os

# Define categories and example messages
CATEGORIES = {
    'complaint': [
        "This service is terrible and I'm very disappointed",
        "I've been waiting for hours with no response",
        "Your product quality has declined significantly",
        "I'm extremely unhappy with the customer service",
        "This is the worst experience I've ever had",
        "Your staff was rude and unprofessional",
        "I demand to speak to a manager immediately",
        "The quality does not match what was advertised",
        "I'm filing a formal complaint about this",
        "This is completely unacceptable service"
    ],
    'refund_request': [
        "I would like to request a full refund",
        "Please process my refund as soon as possible",
        "I need my money back for this purchase",
        "Can I get a refund for order #12345",
        "I want to return this and get my money back",
        "Please refund my payment immediately",
        "How do I get a refund for this transaction",
        "I'm requesting a complete refund",
        "I need to cancel and get my money returned",
        "Please issue a refund to my account"
    ],
    'technical_issue': [
        "The app keeps crashing when I try to login",
        "I'm getting an error message on the checkout page",
        "The website won't load properly on my browser",
        "I can't access my account dashboard",
        "The payment system is not working",
        "I'm experiencing technical difficulties",
        "The page keeps showing a 404 error",
        "My password reset link isn't working",
        "The app freezes whenever I click submit",
        "I'm having trouble uploading files"
    ],
    'spam': [
        "CONGRATULATIONS! You've won $1,000,000!!!",
        "Click here for amazing weight loss pills",
        "Make money fast working from home",
        "You have been selected for a special offer",
        "Buy cheap medications online now",
        "Increase your income overnight",
        "Free gift cards click here now",
        "You won't believe this one weird trick",
        "Limited time offer act now or miss out",
        "Get rich quick with this simple method"
    ],
    'account_problem': [
        "I can't log into my account",
        "My account has been locked",
        "I forgot my username and password",
        "Someone else is using my account",
        "I need to update my account information",
        "My account was suspended without notice",
        "I can't change my email address",
        "There's unauthorized activity on my account",
        "I need to recover my account access",
        "My account settings won't save"
    ]
}

def generate_variations(base_messages, num_variations=20):
    """Generate variations of base messages to create more training data."""
    variations = []
    prefixes = ["", "Hi, ", "Hello, ", "Urgent: ", "Help: ", "Question: "]
    suffixes = ["", " Thanks.", " Please help.", " Urgent!", " ASAP", " Thank you."]
    
    for _ in range(num_variations):
        msg = random.choice(base_messages)
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        variations.append(f"{prefix}{msg}{suffix}")
    
    return variations

def generate_dataset(output_path='data/labeled_messages.csv', samples_per_category=100):
    """Generate a synthetic dataset with labeled messages."""
    
    data = []
    
    for category, base_messages in CATEGORIES.items():
        # Generate variations for each category
        messages = generate_variations(base_messages, samples_per_category)
        
        for message in messages:
            data.append({
                'message': message,
                'category': category
            })
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated {len(df)} labeled messages")
    print(f"✓ Saved to: {output_path}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())
    
    return df

if __name__ == "__main__":
    generate_dataset()
