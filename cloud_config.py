# Cloud Configuration for AfzaAssistant
import os
import streamlit as st

def get_ai_provider():
    """Determine which AI provider to use based on environment"""
    
    # Check for API keys in Streamlit secrets
    if hasattr(st, 'secrets'):
        if 'OPENAI_API_KEY' in st.secrets:
            return 'openai'
        elif 'GOOGLE_API_KEY' in st.secrets:
            return 'google'
        elif 'ANTHROPIC_API_KEY' in st.secrets:
            return 'anthropic'
    
    # Check environment variables
    if os.getenv('OPENAI_API_KEY'):
        return 'openai'
    elif os.getenv('GOOGLE_API_KEY'):
        return 'google'
    elif os.getenv('ANTHROPIC_API_KEY'):
        return 'anthropic'
    
    # Default to Hugging Face transformers (free)
    return 'transformers'

def chat_with_openai(message):
    """Chat using OpenAI API"""
    try:
        import openai
        openai.api_key = st.secrets.get('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": message}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API error: {str(e)}"

def chat_with_google(message):
    """Chat using Google Gemini API"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=st.secrets.get('GOOGLE_API_KEY') or os.getenv('GOOGLE_API_KEY'))
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(message)
        return response.text
    except Exception as e:
        return f"Google API error: {str(e)}"

def chat_with_transformers(message):
    """Chat using Hugging Face transformers (free, local)"""
    try:
        from transformers import pipeline
        
        # Use a lightweight model for cloud deployment
        generator = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-small",
            device=-1  # CPU only
        )
        
        response = generator(
            message,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=50256
        )
        
        return response[0]['generated_text']
    except Exception as e:
        return f"Transformers error: {str(e)}"

# Model recommendations for different use cases
CLOUD_MODELS = {
    'free': {
        'name': 'Hugging Face Transformers',
        'models': ['DialoGPT-small', 'GPT2', 'DistilBERT'],
        'setup': 'Add transformers>=4.30.0 to requirements.txt'
    },
    'paid_best': {
        'name': 'OpenAI GPT-4',
        'models': ['gpt-4', 'gpt-3.5-turbo'],
        'setup': 'Add OPENAI_API_KEY to Streamlit secrets'
    },
    'paid_free_tier': {
        'name': 'Google Gemini',
        'models': ['gemini-pro', 'gemini-pro-vision'],
        'setup': 'Add GOOGLE_API_KEY to Streamlit secrets'
    }
}