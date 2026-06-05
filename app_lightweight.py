import streamlit as st
import os
import json
import time
import datetime
from io import BytesIO

# Minimal environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AfzaAssistant",
    page_icon="🤖",
    layout="wide"
)

# Lightweight module manager
def safe_import(module_name):
    try:
        return __import__(module_name)
    except ImportError:
        return None

# Check available modules
transformers = safe_import('transformers')
torch = safe_import('torch')
whisper = safe_import('whisper')
gtts = safe_import('gtts')

# Lightweight AI Assistant
class LightweightAI:
    def __init__(self):
        self.chat_model = None
        self.speech_model = None
    
    def load_chat_model(self):
        if self.chat_model or not transformers:
            return self.chat_model is not None
        
        try:
            from transformers import pipeline
            self.chat_model = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",  # Smaller model
                device=-1
            )
            return True
        except Exception:
            return False
    
    def chat(self, message):
        if not self.load_chat_model():
            return "AI chat not available. Install transformers."
        
        try:
            response = self.chat_model(message, max_length=150)
            return response[0]['generated_text']
        except Exception as e:
            return f"Error: {str(e)}"
    
    def load_speech_model(self):
        if self.speech_model or not whisper:
            return self.speech_model is not None
        
        try:
            self.speech_model = whisper.load_model("tiny")  # Smallest model
            return True
        except Exception:
            return False
    
    def speech_to_text(self, audio_file):
        if not self.load_speech_model():
            return "Speech recognition not available"
        
        try:
            result = self.speech_model.transcribe(audio_file)
            return result["text"]
        except Exception as e:
            return f"Error: {str(e)}"
    
    def text_to_speech(self, text):
        if not gtts:
            return None
        
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang='en')
            buffer = BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)
            return buffer
        except Exception:
            return None

# Initialize AI
@st.cache_resource
def get_ai():
    return LightweightAI()

ai = get_ai()

# Session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "Chat"

# Header
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <h1 style="color: #2563eb;">🤖 AfzaAssistant</h1>
    <p style="color: #64748b;">Lightweight AI Assistant</p>
</div>
""", unsafe_allow_html=True)

# Status
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write(f"{'✅' if transformers else '❌'} Chat")
with col2:
    st.write(f"{'✅' if whisper else '❌'} Speech")
with col3:
    st.write(f"{'✅' if gtts else '❌'} TTS")
with col4:
    st.write(f"{'✅' if torch else '❌'} PyTorch")

# Mode selection
mode = st.selectbox("Mode:", ["Chat", "Voice", "About"])

if mode == "Chat":
    st.subheader("💬 Chat")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        if isinstance(msg, dict):
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**AI:** {msg['content']}")
    
    # Chat input
    with st.form("chat"):
        user_input = st.text_input("Message:")
        submitted = st.form_submit_button("Send")
    
    if submitted and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("Thinking..."):
            response = ai.chat(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

elif mode == "Voice":
    st.subheader("🎤 Voice")
    
    uploaded_file = st.file_uploader("Upload audio:", type=['wav', 'mp3', 'm4a'])
    
    if uploaded_file:
        st.audio(uploaded_file)
        
        if st.button("Process"):
            temp_file = f"temp_{int(time.time())}.wav"
            
            try:
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                text = ai.speech_to_text(temp_file)
                
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                if text:
                    st.success(f"Text: {text}")
                    
                    response = ai.chat(text)
                    st.write(f"AI: {response}")
                    
                    # TTS
                    audio = ai.text_to_speech(response)
                    if audio:
                        st.audio(audio, format='audio/mp3')
                    
                    # Save to history
                    st.session_state.chat_history.extend([
                        {"role": "user", "content": f"[Voice] {text}"},
                        {"role": "assistant", "content": response}
                    ])
                else:
                    st.error("No speech detected")
            except Exception as e:
                st.error(f"Error: {str(e)}")

elif mode == "About":
    st.subheader("ℹ️ About")
    
    st.markdown("""
    **AfzaAssistant Lightweight**
    
    Features:
    - 💬 AI Chat (Flan-T5-Base)
    - 🎤 Speech-to-Text (Whisper-Tiny)
    - 🔊 Text-to-Speech (gTTS)
    
    Optimized for Streamlit Cloud's 2.7GB limit.
    
    **Dependencies:**
    - transformers (Chat AI)
    - whisper (Speech recognition)
    - gtts (Text-to-speech)
    - torch (ML backend)
    
    **Memory Usage:**
    - Flan-T5-Base: ~250MB
    - Whisper-Tiny: ~39MB
    - Total: <500MB
    """)
    
    # Clear data
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Cleared!")
        st.rerun()

# Footer
st.markdown("---")
st.markdown(f"**Stats:** {len(st.session_state.chat_history)} messages")