"""
Enhanced Streamlit application for Email Triage System.
Includes real-time monitoring, analytics dashboard, and manual classification.
"""
import streamlit as st
import pandas as pd
import joblib
import os
import re
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Import custom modules
import sys
sys.path.append('src')
from sentiment_analyzer import SentimentAnalyzer
from database import TriageDatabase
from bert_classifier import BertEmailClassifier

# Page configuration
st.set_page_config(
    page_title="Email Triage System",
    page_icon="📧",
    layout="wide"
)

# BERT model is loaded by default


@st.cache_resource
def load_bert_model():
    """Load the BERT classifier."""
    return BertEmailClassifier()

@st.cache_resource
def load_sentiment_analyzer():
    """Load sentiment analyzer."""
    return SentimentAnalyzer()

@st.cache_resource
def load_database():
    """Load database connection."""
    return TriageDatabase()

def preprocess_text(text):
    """Basic text preprocessing (must match training preprocessing)."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def classify_email(subject, body, model_type, models, sentiment_analyzer):
    """Classify email with sentiment analysis."""
    # Combine subject and body
    combined = f"{subject} {body}"
    
    if model_type == "Deep (BERT)":
        bert_model = models['bert']
        if not bert_model.model:
            return None
        ml_result = bert_model.predict(subject, body)
        category = ml_result['category']
        confidence = ml_result['confidence']
        category_probs = ml_result['category_probs']
    else:
        # Fallback to BERt if Deep (BERT) is not selected (should not happen in current version)
        bert_model = models['bert']
        if not bert_model.model:
            return None
        ml_result = bert_model.predict(subject, body)
        category = ml_result['category']
        confidence = ml_result['confidence']
        category_probs = ml_result['category_probs']
    
    # Sentiment analysis
    sentiment_result = sentiment_analyzer.analyze_email(subject, body)
    
    return {
        'category': category,
        'confidence': confidence,
        'category_probs': category_probs,
        'sentiment': sentiment_result['sentiment'],
        'urgency': sentiment_result['urgency'],
        'priority': sentiment_result['priority']
    }

def main():
    """Main application."""
    
    # Header
    st.title("📧 Automated Email Triage System")
    st.markdown("---")
    
    # Load resources
    with st.spinner("Loading BERT model..."):
        bert_model = load_bert_model()
        sentiment_analyzer = load_sentiment_analyzer()
        db = load_database()
        
        models = {
            'bert': bert_model
        }
        model_selection = "Deep (BERT)"
    
    # Sidebar
    with st.sidebar:
        st.header("🧠 NLP Engine")
        
        if not bert_model.model:
            st.warning("⚠️ BERT model not found. Training might be in progress.")
        else:
            st.success("✓ BERT model loaded")

        st.markdown("---")
        st.header("ℹ️ System Info")
        st.info(f"""
        **Current Model**: {model_selection}  
        **Features**: Transformers + VADER  
        **Categories**: 8 business types
        """)
        
        st.header("📊 Categories")
        # List categories manually since log_model is gone
        categories = ["Sales", "Support", "Billing", "HR", "IT", "Legal", "Complaint", "General"]
        for cat in sorted(categories):
            st.write(f"• {cat}")
    
    # Main tabs
    tab1, tab4 = st.tabs([
        "📝 Manual Classification",
        "📂 Bulk Processing"
    ])
    
    # Tab 1: Manual Classification
    with tab1:
        st.header("Classify Email Manually")
        
        subject_input = st.text_input(
            "Email Subject:",
            placeholder="Enter email subject..."
        )
        
        body_input = st.text_area(
            "Email Body:",
            placeholder="Enter email body text...",
            height=200
        )
        
        if st.button("🔍 Classify Email", type="primary"):
            if subject_input or body_input:
                status_placeholder = st.empty()
                status_placeholder.info(f"Using {model_selection} model... This may take a moment.")
                
                result = classify_email(
                    subject_input,
                    body_input,
                    model_selection,
                    models,
                    sentiment_analyzer
                )
                
                status_placeholder.empty()
                
                if result is None:
                    st.error("❌ Classification failed. Please ensure the selected model is trained and loaded.")
                    return
                
                st.success("✅ Classification Complete!")
                
                # Display results in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Category",
                        result['category'].replace('_', ' ').title()
                    )
                
                with col2:
                    st.metric(
                        "Confidence",
                        f"{result['confidence']*100:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Priority",
                        result['priority'].upper()
                    )
                
                with col4:
                    st.metric(
                        "Sentiment",
                        result['sentiment']['label'].title()
                    )
                
                # Detailed analysis
                st.subheader("Detailed Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Category Probabilities**")
                    prob_df = pd.DataFrame([
                        {
                            'Category': cat.replace('_', ' ').title(),
                            'Probability': f"{prob*100:.1f}%",
                            'Score': prob
                        }
                        for cat, prob in sorted(
                            result['category_probs'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                    ])
                    
                    # Bar chart
                    fig = px.bar(
                        prob_df,
                        x='Score',
                        y='Category',
                        orientation='h',
                        text='Probability'
                    )
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**Sentiment Details**")
                    st.write(f"• **Label**: {result['sentiment']['label'].title()}")
                    st.write(f"• **Compound Score**: {result['sentiment']['compound']:.3f}")
                    st.write(f"• **Positive**: {result['sentiment']['positive']:.3f}")
                    st.write(f"• **Negative**: {result['sentiment']['negative']:.3f}")
                    st.write(f"• **Neutral**: {result['sentiment']['neutral']:.3f}")
                    
                    st.write("**Urgency Analysis**")
                    st.write(f"• **Level**: {result['urgency']['level'].title()}")
                    st.write(f"• **Score**: {result['urgency']['score']}")
                    if result['urgency']['keywords']:
                        st.write(f"• **Keywords**: {', '.join(result['urgency']['keywords'])}")
            else:
                st.warning("⚠️ Please enter at least a subject or body.")
    
    # Removed Analytics and Recent Activity tabs

    
    # Tab 4: Bulk Processing
    with tab4:
        st.header("Bulk Email Classification")
        st.markdown("""
        Upload a CSV file with columns `subject` and `body` to classify multiple emails at once.
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'subject' not in df.columns and 'body' not in df.columns:
                    st.error("⚠️ CSV must contain 'subject' and/or 'body' columns.")
                else:
                    # Fill missing columns
                    if 'subject' not in df.columns:
                        df['subject'] = ''
                    if 'body' not in df.columns:
                        df['body'] = ''
                    
                    st.success(f"✅ Loaded {len(df)} emails")
                    
                    with st.expander("📋 Preview Data"):
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    if st.button("🚀 Classify All Emails", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        results = []
                        
                        for idx, row in df.iterrows():
                            status_text.text(f"Processing {idx+1}/{len(df)}...")
                            
                            result = classify_email(
                                str(row.get('subject', '')),
                                str(row.get('body', '')),
                                model_selection,
                                models,
                                sentiment_analyzer
                            )
                            
                            if result is None:
                                results.append({
                                    'category': 'error',
                                    'confidence': '0%',
                                    'priority': 'unknown',
                                    'sentiment': 'unknown'
                                })
                            else:
                                results.append({
                                    'category': result['category'],
                                    'confidence': f"{result['confidence']*100:.1f}%",
                                    'priority': result['priority'],
                                    'sentiment': result['sentiment']['label']
                                })
                            
                            progress_bar.progress((idx + 1) / len(df))
                        
                        status_text.text("✅ Complete!")
                        
                        # Combine results - Drop existing result columns if they exist to avoid duplicates
                        cols_to_drop = ['category', 'confidence', 'priority', 'sentiment']
                        df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
                        results_df = pd.concat([df_clean, pd.DataFrame(results)], axis=1)
                        
                        st.subheader("Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download button
                        csv_output = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv_output,
                            file_name="classified_emails.csv",
                            mime="text/csv",
                            type="primary"
                        )
                        
                        # Summary
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Category Summary")
                            category_counts = results_df['category'].value_counts()
                            st.bar_chart(category_counts)
                        
                        with col2:
                            st.subheader("Priority Summary")
                            priority_counts = results_df['priority'].value_counts()
                            
                            # Ensure all priorities are shown even if count is zero
                            all_priorities = ['high', 'medium', 'low']
                            counts_dict = priority_counts.to_dict()
                            final_counts = {p: counts_dict.get(p, 0) for p in all_priorities}
                            
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=list(final_counts.keys()),
                                    y=list(final_counts.values()),
                                    marker_color=['#ff4444', '#ffaa00', '#44ff44'],
                                    text=list(final_counts.values()),
                                    textposition='auto',
                                )
                            ])
                            fig.update_layout(xaxis_title="Priority", yaxis_title="Count")
                            st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>Automated Email Triage System • Built with Streamlit & Scikit-learn</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
