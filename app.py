import streamlit as st
import os
import json
import time
import datetime
import hashlib
import pickle
from io import BytesIO

# Environment variables for CPU-only devices
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Cached lazy import function
_import_cache = {}

def lazy_import(module_name, package=None):
    cache_key = f"{module_name}.{package}" if package else module_name
    
    if cache_key in _import_cache:
        return _import_cache[cache_key]
    
    try:
        if package:
            module = __import__(module_name, fromlist=[package])
            result = getattr(module, package, None)
        else:
            result = __import__(module_name)
        
        _import_cache[cache_key] = result
        return result
    except (ImportError, AttributeError):
        _import_cache[cache_key] = None
        return None

# Page configuration
st.set_page_config(
    page_title="AfzaAssistant - Local AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data
def get_default_prompts():
    return {
        "chat": [
            "Explain quantum computing in simple terms",
            "Write a professional email template",
            "Create a study plan for learning Python",
            "Summarize renewable energy benefits",
            "How to start a small business?",
            "Best practices for time management"
        ],
        "image": [
            "Futuristic city at sunset with flying cars",
            "Serene mountain landscape with crystal lake",
            "Modern minimalist office with plants",
            "Abstract geometric art with vibrant colors",
            "Cyberpunk street scene with neon lights",
            "Beautiful garden with colorful flowers"
        ],
        "code": [
            "Create a Python web scraper with BeautifulSoup",
            "Build a FastAPI REST API with authentication",
            "Write a React component with TypeScript",
            "Design a database schema for e-commerce",
            "Create a machine learning model for prediction",
            "Build a responsive CSS grid layout"
        ]
    }

# Persistent storage functions
def load_gallery_data():
    """Load saved gallery data from outputs folder"""
    try:
        # Load images
        images = []
        if os.path.exists("outputs/images"):
            for filename in os.listdir("outputs/images"):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    filepath = os.path.join("outputs/images", filename)
                    metadata_file = filepath.replace('.png', '_metadata.json')
                    try:
                        from PIL import Image
                        image = Image.open(filepath)
                        
                        # Load metadata if exists
                        prompt = "Generated Image"
                        if os.path.exists(metadata_file):
                            try:
                                with open(metadata_file, 'r', encoding='utf-8') as f:
                                    metadata = json.load(f)
                                    prompt = metadata.get('prompt', prompt)
                            except Exception:
                                pass
                        
                        images.append({
                            "image": image,
                            "prompt": prompt,
                            "filepath": filepath,
                            "timestamp": datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
                        })
                    except Exception:
                        continue
        
        # Load code
        code_snippets = []
        if os.path.exists("outputs/code"):
            for filename in os.listdir("outputs/code"):
                if filename.endswith(('.py', '.js', '.java', '.cpp', '.go', '.rs', '.html')):
                    filepath = os.path.join("outputs/code", filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            code = f.read()
                        language = filename.split('.')[-1]
                        code_snippets.append({
                            "code": code,
                            "prompt": f"Code from {filename}",
                            "language": language,
                            "timestamp": datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
                        })
                    except Exception:
                        continue
        
        # Load chat history
        chat_history = []
        if os.path.exists("outputs/chat"):
            chat_files = [f for f in os.listdir("outputs/chat") if f.endswith('.json')]
            if chat_files:
                # Load most recent chat file
                latest_file = max(chat_files, key=lambda x: os.path.getmtime(os.path.join("outputs/chat", x)))
                try:
                    with open(os.path.join("outputs/chat", latest_file), 'r', encoding='utf-8') as f:
                        chat_history = json.load(f)
                except Exception:
                    pass
        
        return images, code_snippets, chat_history
    except Exception:
        return [], [], []

def save_gallery_data():
    """Save current gallery data to outputs folder"""
    try:
        # Save chat history
        if st.session_state.chat_history:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            chat_file = os.path.join("outputs/chat", f"chat_history_{timestamp}.json")
            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.chat_history, f, indent=2, ensure_ascii=False)
        
        # Save code snippets
        for i, code_data in enumerate(st.session_state.generated_code):
            if 'saved' not in code_data:
                timestamp = code_data['timestamp'].strftime("%Y%m%d_%H%M%S")
                ext = {'Python': 'py', 'JavaScript': 'js', 'Java': 'java', 'C++': 'cpp', 'Go': 'go', 'Rust': 'rs', 'HTML/CSS': 'html'}.get(code_data['language'], 'txt')
                filename = f"code_{timestamp}_{i}.{ext}"
                filepath = os.path.join("outputs/code", filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(code_data['code'])
                code_data['saved'] = True
    except Exception:
        pass

# Initialize session state
def init_session_state():
    defaults = {
        'chat_history': [],
        'current_mode': "Chat",
        'generated_images': [],
        'generated_code': [],
        'listening': False,
        'selected_model': "phi3",
        'selected_prompt': None,
        'saved_prompts': get_default_prompts()
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Load saved gallery data on first run
    if 'gallery_loaded' not in st.session_state:
        images, code_snippets, chat_history = load_gallery_data()
        if images:
            st.session_state.generated_images = images
        if code_snippets:
            st.session_state.generated_code = code_snippets
        if chat_history:
            st.session_state.chat_history = chat_history
        st.session_state.gallery_loaded = True

init_session_state()

@st.cache_data
def get_css_styles():
    return """
    <style>
    .stDeployButton {display: none !important;}
    header[data-testid="stHeader"] {display: none !important;}
    .main .block-container {
        padding: 1rem;
        max-width: 100%;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    :root {
        --primary-color: #2563eb;
        --secondary-color: #64748b;
        --success-color: #059669;
        --warning-color: #d97706;
        --error-color: #dc2626;
        --background-light: #f8fafc;
        --background-white: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .panel {
        background: var(--background-white);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-md);
        height: fit-content;
    }
    .panel-title {
        color: var(--text-primary);
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
        border-bottom: 2px solid var(--border-color);
        padding-bottom: 0.75rem;
    }
    .section {
        margin-bottom: 1.5rem;
    }
    .section-title {
        color: var(--text-primary);
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stButton > button {
        background: var(--background-white) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
        box-shadow: var(--shadow-sm) !important;
    }
    .stButton > button:hover {
        background: var(--primary-color) !important;
        color: white !important;
        border-color: var(--primary-color) !important;
        box-shadow: var(--shadow-md) !important;
        transform: translateY(-1px) !important;
    }
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--background-white) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        box-shadow: var(--shadow-sm) !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgb(37 99 235 / 0.1) !important;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.75rem 0;
        border-radius: 12px;
        max-width: 80%;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-color);
    }
    .user-message {
        background: #eff6ff;
        border-color: #bfdbfe;
        margin-left: auto;
        text-align: right;
    }
    .ai-message {
        background: var(--background-light);
        border-color: var(--border-color);
        margin-right: auto;
    }
    .stSuccess {
        background: #f0fdf4 !important;
        border: 1px solid #bbf7d0 !important;
        color: var(--success-color) !important;
        border-radius: 8px !important;
    }
    .stError {
        background: #fef2f2 !important;
        border: 1px solid #fecaca !important;
        color: var(--error-color) !important;
        border-radius: 8px !important;
    }
    .stInfo {
        background: #eff6ff !important;
        border: 1px solid #bfdbfe !important;
        color: var(--primary-color) !important;
        border-radius: 8px !important;
    }
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.5rem;
        }
        .panel {
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
        .chat-message {
            max-width: 95%;
        }
        .element-container {
            width: 100% !important;
        }
    }
    @media (max-width: 480px) {
        .panel-title {
            font-size: 1rem;
        }
        .section-title {
            font-size: 0.75rem;
        }
    }
    </style>
    """

# Load CSS once with persistent caching
st.markdown(get_css_styles(), unsafe_allow_html=True)

# AI Assistant Class with Persistent Caching
class LocalAIAssistant:
    def __init__(self):
        self.setup_directories()
        # Models cached persistently
        self.sd_pipeline = None
        self.whisper_model = None
        
    def setup_directories(self):
        directories = ["outputs", "outputs/images", "outputs/code", "outputs/audio", "outputs/chat"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def check_ollama_connection(self):
        return check_ollama_connection_cached()
    
    def load_image_model(self):
        if self.sd_pipeline is not None:
            return True
        
        self.sd_pipeline = load_stable_diffusion_model()
        return self.sd_pipeline is not None
    
    def load_whisper_model(self):
        if self.whisper_model is not None:
            return True
        
        self.whisper_model = load_whisper_model_cached()
        return self.whisper_model is not None
    
    def chat(self, message, model=None):
        # Try cloud-friendly models first
        if self.use_cloud_model():
            return self.chat_with_transformers(message)
        
        # Fallback to Ollama for local deployment
        ollama = lazy_import('ollama')
        if not ollama:
            return self.chat_with_transformers(message)
        
        if model is None:
            model = st.session_state.selected_model
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': 'You are an AI assistant. Provide clear, detailed, and helpful responses.'},
                    {'role': 'user', 'content': message}
                ]
            )
            return response['message']['content']
        except Exception as e:
            return self.chat_with_transformers(message)
    
    def generate_image(self, prompt, width=512, height=512):
        # Check dependencies first
        if not lazy_import('diffusers') or not lazy_import('torch'):
            return None, "Image generation requires diffusers and torch libraries"
        
        pipeline = load_stable_diffusion_model()
        if not pipeline:
            return None, "Failed to load image generation model. Check if dependencies are installed."
        
        try:
            with st.spinner("Generating image (this may take 1-2 minutes)..."):
                enhanced_prompt = f"{prompt}, high quality, detailed"
                negative_prompt = "blurry, low quality, distorted, ugly"
                
                # Use smaller settings for better performance on limited resources
                image = pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    width=min(width, 512),
                    height=min(height, 512),
                    num_inference_steps=15,  # Slightly more steps for better quality
                    guidance_scale=7.5,
                    generator=None  # For reproducible results, could add seed here
                ).images[0]
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_image_{timestamp}.png"
                filepath = os.path.join("outputs/images", filename)
                image.save(filepath)
                
                return image, filepath
        except Exception as e:
            return None, f"Image generation failed: {str(e)}. Try a simpler prompt or restart the app."
    
    def speech_to_text(self, audio_file):
        model = load_whisper_model_cached()
        if not model:
            return "Speech recognition not available"
        
        try:
            result = model.transcribe(audio_file)
            return result["text"]
        except Exception as e:
            return f"Speech recognition failed: {str(e)}"
    
    def text_to_speech(self, text):
        try:
            gTTS = lazy_import('gtts', 'gTTS')
            if not gTTS:
                return None
            
            tts = gTTS(text=text, lang='en', slow=False)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer
        except Exception as e:
            st.error(f"Text-to-speech failed: {str(e)}")
            return None
    
    def save_chat_history(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.json"
        filepath = os.path.join("outputs/chat", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.chat_history, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def use_cloud_model(self):
        """Check if running on cloud platform"""
        return (
            os.getenv('STREAMLIT_SHARING_MODE') or 
            'streamlit.io' in os.getenv('HOSTNAME', '') or
            'streamlit' in os.getenv('SERVER_NAME', '') or
            os.path.exists('/.streamlit') or
            not os.path.exists('/usr/local/bin/ollama')
        )
    
    def chat_with_transformers(self, message):
        """Chat using Hugging Face transformers"""
        try:
            pipeline = lazy_import('transformers', 'pipeline')
            if not pipeline:
                return "AI chat requires transformers library. Add 'transformers' to requirements.txt"
            
            # Use cached model
            if not hasattr(self, 'chat_model'):
                with st.spinner("Loading AI model (first time only)..."):
                    self.chat_model = pipeline(
                        "text-generation",
                        model="gpt2",  # Smaller, faster model
                        device=-1,  # CPU only
                        max_length=150
                    )
            
            # Generate response
            prompt = f"Human: {message}\nAI:"
            response = self.chat_model(
                prompt, 
                max_length=len(prompt.split()) + 50,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=50256,
                do_sample=True
            )
            
            # Extract only the AI response part
            full_response = response[0]['generated_text']
            ai_response = full_response.split("AI:")[-1].strip()
            return ai_response if ai_response else "I'm here to help! Please ask me anything."
            
        except Exception as e:
            return f"AI temporarily unavailable: {str(e)}. Please try again."
    
    def auto_save_all(self):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            session_data = {
                "chat_history": st.session_state.chat_history,
                "generated_code": [{
                    "code": item["code"],
                    "prompt": item["prompt"],
                    "language": item["language"],
                    "timestamp": item["timestamp"].isoformat()
                } for item in st.session_state.generated_code],
                "saved_prompts": st.session_state.saved_prompts,
                "timestamp": timestamp
            }
            
            autosave_path = os.path.join("outputs", "autosave_session.json")
            with open(autosave_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception:
            return False

@st.cache_data(ttl=300)
def check_ollama_connection_cached():
    # Force cloud mode detection
    is_cloud = (
        os.getenv('STREAMLIT_SHARING_MODE') or 
        'streamlit.io' in os.getenv('HOSTNAME', '') or
        'streamlit' in os.getenv('SERVER_NAME', '') or
        os.path.exists('/.streamlit') or
        not os.path.exists('/usr/local/bin/ollama')
    )
    
    if is_cloud:
        transformers = lazy_import('transformers')
        if transformers:
            return True, "🌐 Cloud AI Model (Hugging Face)", ["DialoGPT-small"]
        else:
            return False, "❌ Add 'transformers' to requirements.txt", []
    
    # Local Ollama check (only for local deployment)
    ollama = lazy_import('ollama')
    if not ollama:
        transformers = lazy_import('transformers')
        if transformers:
            return True, "🔄 Fallback AI Model (Hugging Face)", ["DialoGPT-small"]
        return False, "❌ Install Ollama or add 'transformers' to requirements.txt", []
    
    try:
        models = ollama.list()
        if not models or not models.get('models'):
            return False, "❌ No models installed. Run: ollama pull phi3", []
        model_names = [model['name'] for model in models['models']]
        return True, f"✅ Ollama - {len(model_names)} models available", model_names
    except Exception as e:
        transformers = lazy_import('transformers')
        if transformers:
            return True, "🔄 Fallback AI Model (Hugging Face)", ["DialoGPT-small"]
        return False, f"❌ Ollama failed. Install transformers as fallback: {str(e)}", []

@st.cache_resource
def load_stable_diffusion_model():
    try:
        torch = lazy_import('torch')
        StableDiffusionPipeline = lazy_import('diffusers', 'StableDiffusionPipeline')
        
        if not torch or not StableDiffusionPipeline:
            return None
        
        # For Streamlit Cloud, use a smaller model or handle memory constraints
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        pipeline = pipeline.to("cpu")
        pipeline.enable_attention_slicing()
        pipeline.enable_sequential_cpu_offload()
        return pipeline
    except Exception as e:
        # Don't show error in UI here, let the calling function handle it
        return None

@st.cache_resource
def load_whisper_model_cached():
    try:
        whisper = lazy_import('whisper')
        if not whisper:
            return None
        return whisper.load_model("base")
    except Exception as e:
        st.error(f"Failed to load Whisper model: {str(e)}")
        return None

@st.cache_resource
def get_ai_assistant():
    return LocalAIAssistant()

ai = get_ai_assistant()

# Main App
def main():
    # Responsive layout
    if st.session_state.get('mobile_view', False) or st.sidebar.button("📱 Toggle Mobile View"):
        st.session_state.mobile_view = not st.session_state.get('mobile_view', False)
    
    # Layout based on screen size
    if st.session_state.get('mobile_view', False):
        render_mobile_layout()
    else:
        render_desktop_layout()

def render_desktop_layout():
    # Three-column layout for desktop
    left_col, main_col, right_col = st.columns([1.2, 2.5, 1.3])
    
    with left_col:
        render_control_panel()
    
    with main_col:
        render_main_content()
    
    with right_col:
        render_prompts_panel()

def render_mobile_layout():
    # Single column layout for mobile
    render_control_panel()
    render_main_content()
    render_prompts_panel()

def render_control_panel():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🔧 Controls</div>', unsafe_allow_html=True)
    
    # Model Status and Selection - Cached
    st.markdown('<div class="section"><div class="section-title">🔌 Model Selection</div>', unsafe_allow_html=True)
    
    # Only check connection once per session or on refresh
    if 'connection_checked' not in st.session_state:
        st.session_state.connection_result = check_ollama_connection_cached()
        st.session_state.connection_checked = True
    
    connection_result = st.session_state.connection_result
    if len(connection_result) == 3:
        ollama_status, ollama_msg, available_models = connection_result
    else:
        ollama_status, ollama_msg = connection_result
        available_models = []
    
    if ollama_status and available_models:
        st.success(f"✅ {ollama_msg}")
        if st.session_state.selected_model not in available_models:
            st.session_state.selected_model = available_models[0]
        
        st.session_state.selected_model = st.selectbox(
            "Choose AI Model:", 
            available_models, 
            index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
        )
        st.info(f"🤖 Using: {st.session_state.selected_model}")
    else:
        st.error(f"❌ {ollama_msg}")
        st.info("💡 Start Ollama: `ollama serve` then `ollama pull phi3`")
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = "phi3"
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", key="refresh_conn"):
            if 'connection_checked' in st.session_state:
                del st.session_state.connection_checked
            st.cache_data.clear()
            st.success("Connection refreshed!")
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Cache", key="clear_cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            _import_cache.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("All cache cleared! Refresh page.")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Mode Selection
    st.markdown('<div class="section"><div class="section-title">🎯 Select Mode</div>', unsafe_allow_html=True)
    modes = ["Chat", "Image Generation", "Code Helper", "Voice Chat", "Gallery"]
    st.session_state.current_mode = st.selectbox("Choose mode:", modes, index=0)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown('<div class="section"><div class="section-title">⚡ Quick Actions</div>', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat History", width='stretch'):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
    
    if st.button("💾 Save Chat History", width='stretch'):
        if st.session_state.chat_history:
            filepath = ai.save_chat_history()
            st.success(f"Saved to: {filepath}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistics
    st.markdown('<div class="section"><div class="section-title">📊 Statistics</div>', unsafe_allow_html=True)
    st.metric("Chat Messages", len(st.session_state.chat_history))
    st.metric("Generated Images", len(st.session_state.generated_images))
    st.metric("Code Snippets", len(st.session_state.generated_code))
    st.metric("Saved Prompts", sum(len(prompts) for prompts in st.session_state.saved_prompts.values()))
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_prompts_panel():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">💡 Quick Prompts</div>', unsafe_allow_html=True)
    
    current_prompts_key = "chat" if "Chat" in st.session_state.current_mode else "image" if "Image" in st.session_state.current_mode else "code"
    
    st.markdown(f'<div class="section-title">{current_prompts_key.title()} Prompts</div>', unsafe_allow_html=True)
    
    # Display saved prompts
    for i, prompt in enumerate(st.session_state.saved_prompts[current_prompts_key]):
        button_text = prompt if len(prompt) <= 50 else prompt[:47] + "..."
        if st.button(button_text, key=f"prompt_{current_prompts_key}_{i}", width='stretch', help=prompt):
            st.session_state.selected_prompt = prompt
    
    # Add new prompt
    with st.expander("➕ Add New Prompt"):
        new_prompt = st.text_area("Enter new prompt:", height=60)
        if st.button("Save Prompt", width='stretch'):
            if new_prompt.strip():
                st.session_state.saved_prompts[current_prompts_key].append(new_prompt.strip())
                st.success("Prompt saved!")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_main_content():
    # Header only in main column
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="font-size: 2.5rem; color: #2563eb; margin-bottom: 0.5rem;">
            🤖 AfzaAssistant
        </h1>
        <p style="font-size: 1rem; color: #64748b; margin-bottom: 1rem;">
            Your Local AI Assistant - Chat, Generate Images, Code & More
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Route to appropriate mode
    if st.session_state.current_mode == "Chat":
        render_chat_mode()
    elif st.session_state.current_mode == "Image Generation":
        render_image_mode()
    elif st.session_state.current_mode == "Code Helper":
        render_code_mode()
    elif st.session_state.current_mode == "Voice Chat":
        render_voice_mode()
    elif st.session_state.current_mode == "Gallery":
        render_gallery_mode()

def render_chat_mode():
    st.subheader("💬 AfzaAssistant Chat")
    
    # Ensure chat_history is always a list
    if not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = []
    
    # Handle selected prompt
    if st.session_state.get('selected_prompt'):
        prompt = st.session_state.selected_prompt
        st.session_state.selected_prompt = None
        
        # Clear other mode prompts
        if 'image_prompt_value' in st.session_state:
            del st.session_state.image_prompt_value
        if 'code_prompt_value' in st.session_state:
            del st.session_state.code_prompt_value
        
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.spinner("AI is thinking..."):
            response = ai.chat(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            # Handle both dict and string formats for backward compatibility
            if isinstance(message, dict):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message ai-message">
                        <strong>AI:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)

    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input("Type your message:", placeholder="Ask me anything...")
        with col2:
            submitted = st.form_submit_button("Send", width='stretch')
    
    if submitted and user_input:
        if len(user_input.strip()) < 2:
            st.error("⚠️ Please enter a valid message (at least 2 characters)")
            st.info("💡 Tip: Try asking 'What is AI?' or 'Help me with Python'")
        else:
            try:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                with st.spinner("AI is thinking..."):
                    response = ai.chat(user_input)
                    
                    if response and not response.startswith("Error:"):
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        st.rerun()
                    else:
                        st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, I encountered an issue. Please try again."})
                        st.error(f"❌ {response}")
                        st.info("💡 Tip: Make sure Ollama is running and try a simpler question")
                        st.rerun()
            except Exception as e:
                st.error(f"❌ Chat error: {str(e)}")
                if "connection" in str(e).lower():
                    st.info("💡 Tip: Start Ollama with: `ollama serve`")
                else:
                    st.info("💡 Tip: Try restarting the application")
    elif submitted:
        st.warning("⚠️ Please enter a message to chat")
        st.info("💡 Tip: Click on a saved prompt from the right panel or type your question")

def render_image_mode():
    st.subheader("🎨 AI Image Generation")
    
    # Check for required libraries
    diffusers_available = lazy_import('diffusers')
    torch_available = lazy_import('torch')
    
    if not diffusers_available or not torch_available:
        st.error("🚫 Image generation requires additional libraries")
        st.markdown("""
        **Missing Dependencies:**
        ```bash
        pip install diffusers torch transformers accelerate
        ```
        
        **For Streamlit Cloud deployment, add to requirements.txt:**
        ```
        diffusers>=0.21.0
        torch>=2.0.0
        transformers>=4.30.0
        accelerate>=0.20.0
        ```
        """)
        
        st.info("💡 After installing dependencies, restart the application to enable image generation.")
        return
    
    # Handle selected prompt
    if st.session_state.get('selected_prompt'):
        st.session_state.image_prompt_value = st.session_state.selected_prompt
        st.session_state.selected_prompt = None
        # Clear other mode prompts
        if 'code_prompt_value' in st.session_state:
            del st.session_state.code_prompt_value
    
    # Image generation form
    with st.form("image_form"):
        prompt = st.text_area(
            "Describe the image you want to generate:",
            value=st.session_state.get('image_prompt_value', ''),
            placeholder="A beautiful sunset over mountains, digital art style",
            height=100
        )
        
        # Update the stored value when form is submitted
        if prompt != st.session_state.get('image_prompt_value', ''):
            st.session_state.image_prompt_value = prompt
        
        col1, col2, col3 = st.columns(3)
        with col1:
            width = st.selectbox("Width", [256, 512], index=1)  # Removed large sizes for speed
        with col2:
            height = st.selectbox("Height", [256, 512], index=1)  # Removed large sizes for speed
        with col3:
            submitted = st.form_submit_button("🎨 Generate Image", width='stretch')
        
        st.info("💡 Tip: 256x256 generates in ~30 seconds, 512x512 takes ~60 seconds on CPU")
    
    if submitted and prompt:
        if len(prompt.strip()) < 3:
            st.error("⚠️ Please enter a more detailed prompt (at least 3 characters)")
            st.info("💡 Tip: Try 'A beautiful sunset over mountains' or 'Modern city skyline at night'")
        else:
            try:
                image, result = ai.generate_image(prompt, width, height)
                
                if image:
                    st.success("✅ Image generated successfully!")
                    st.image(image, caption=prompt, width='stretch')
                    
                    st.session_state.generated_images.append({
                        "image": image,
                        "prompt": prompt,
                        "filepath": result,
                        "timestamp": datetime.datetime.now()
                    })
                    
                    # Save metadata
                    metadata_file = result.replace('.png', '_metadata.json')
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump({"prompt": prompt, "timestamp": datetime.datetime.now().isoformat()}, f)
                    
                    # Download button
                    img_buffer = BytesIO()
                    image.save(img_buffer, format="PNG")
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download Image",
                        data=img_buffer,
                        file_name=f"generated_image_{int(time.time())}.png",
                        mime="image/png",
                        width='stretch'
                    )
                else:
                    st.error(f"❌ Failed to generate image: {result}")
                    if "not available" in str(result).lower():
                        st.info("💡 Tip: Add 'diffusers>=0.21.0' and 'torch>=2.0.0' to requirements.txt for Streamlit Cloud")
                    elif "memory" in str(result).lower():
                        st.info("💡 Tip: Try smaller image size (256x256) or restart the app")
                    else:
                        st.info("💡 Tip: Check if you have enough disk space and try a simpler prompt")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.info("💡 Tip: Try restarting the application or check your system resources")
    elif submitted:
        st.warning("⚠️ Please enter a prompt to generate an image")
        st.info("💡 Tip: Click on a saved prompt from the right panel or type your own description")
    
    # Recent images preview
    if st.session_state.generated_images:
        st.subheader("🖼️ Recent Images")
        cols = st.columns(3)
        for i, img_data in enumerate(st.session_state.generated_images[-3:]):
            with cols[i % 3]:
                st.image(img_data["image"], caption=img_data["prompt"][:50] + "...", use_container_width=True)

def render_code_mode():
    st.subheader("💻 AI Code Helper")
    
    # Handle selected prompt
    if st.session_state.get('selected_prompt'):
        st.session_state.code_prompt_value = st.session_state.selected_prompt
        st.session_state.selected_prompt = None
        # Clear other mode prompts
        if 'image_prompt_value' in st.session_state:
            del st.session_state.image_prompt_value
    
    # Code generation form
    with st.form("code_form"):
        prompt = st.text_area(
            "Describe the code you need:",
            value=st.session_state.get('code_prompt_value', ''),
            placeholder="Create a Python function that sorts a list of dictionaries by a specific key",
            height=100
        )
        
        # Update the stored value when form is submitted
        if prompt != st.session_state.get('code_prompt_value', ''):
            st.session_state.code_prompt_value = prompt
        
        col1, col2 = st.columns(2)
        with col1:
            language = st.selectbox("Programming Language", 
                                  ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "HTML/CSS"])
        with col2:
            style = st.selectbox("Code Style", 
                               ["Clean & Simple", "With Comments", "Production Ready", "With Tests"])
        
        submitted = st.form_submit_button("💻 Generate Code", width='stretch')
    
    if submitted and prompt:
        if len(prompt.strip()) < 5:
            st.error("⚠️ Please provide a more detailed code description (at least 5 characters)")
            st.info("💡 Tip: Try 'Create a function to sort data' or 'Build a REST API endpoint'")
        else:
            try:
                enhanced_prompt = f"Write {style.lower()} {language} code for: {prompt}. Include proper error handling and follow best practices."
                
                with st.spinner("Generating code..."):
                    code = ai.chat(enhanced_prompt)
                    
                    if code and len(code.strip()) > 10:
                        st.success("✅ Code generated successfully!")
                        st.code(code, language=language.lower())
                        
                        st.session_state.generated_code.append({
                            "code": code,
                            "prompt": prompt,
                            "language": language,
                            "timestamp": datetime.datetime.now()
                        })
                        
                        # Auto-save code
                        save_gallery_data()
                        

                        
                        # Download button
                        st.download_button(
                            label="📥 Download Code",
                            data=code,
                            file_name=f"generated_code_{int(time.time())}.{language.lower()}",
                            mime="text/plain",
                            width='stretch'
                        )
                    else:
                        st.error("❌ Failed to generate valid code")
                        st.info("💡 Tip: Try a more specific prompt or check if Ollama is running")
            except Exception as e:
                st.error(f"❌ Code generation failed: {str(e)}")
                if "connection" in str(e).lower():
                    st.info("💡 Tip: Make sure Ollama is running with: `ollama serve`")
                else:
                    st.info("💡 Tip: Try a simpler request or restart the application")
    elif submitted:
        st.warning("⚠️ Please describe the code you need")
        st.info("💡 Tip: Click on a code template below or use a saved prompt from the right panel")
    
    # Code templates
    st.subheader("📋 Code Templates")
    templates = {
        "Python Web Scraper": "Create a Python web scraper using BeautifulSoup that extracts data from a website",
        "REST API": "Build a FastAPI REST API with CRUD operations and database integration",
        "React Component": "Create a React functional component with hooks and state management",
        "Database Schema": "Design a database schema for a social media application"
    }
    
    cols = st.columns(2)
    for i, (title, template_prompt) in enumerate(templates.items()):
        with cols[i % 2]:
            if st.button(title, key=f"template_{i}", width='stretch'):
                enhanced_prompt = f"Write clean Python code for: {template_prompt}. Include proper error handling and follow best practices."
                with st.spinner("Generating code..."):
                    code = ai.chat(enhanced_prompt)
                    st.code(code, language="python")

def render_voice_mode():
    st.subheader("🎙️ Voice Chat")
    
    # Check if running on cloud
    is_cloud = os.getenv('STREAMLIT_SHARING_MODE') or 'streamlit.io' in os.getenv('HOSTNAME', '')
    
    if not lazy_import('whisper'):
        st.error("Voice chat requires whisper. Add 'openai-whisper>=20230314' to requirements.txt")
        return
    
    if is_cloud:
        st.info("🌐 Running on Streamlit Cloud - Live recording disabled. Upload audio files instead.")
    
    # Audio file upload
    st.subheader("📁 Upload Audio File")
    uploaded_audio = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'm4a', 'ogg'])
    
    if uploaded_audio:
        st.audio(uploaded_audio, format='audio/wav')
        
        if st.button("🎯 Process Audio", width='stretch'):
            try:
                with st.spinner("Converting speech to text..."):
                    temp_path = f"temp_audio_{int(time.time())}.wav"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_audio.getbuffer())
                    
                    text = ai.speech_to_text(temp_path)
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    if text and len(text.strip()) > 1:
                        st.success("✅ Speech converted successfully!")
                        st.write(f"**Transcribed Text:** {text}")
                        
                        with st.spinner("AI is responding..."):
                            response = ai.chat(text)
                            st.write(f"**AI Response:** {response}")
                            
                            if lazy_import('gtts'):
                                audio_buffer = ai.text_to_speech(response)
                                if audio_buffer:
                                    st.audio(audio_buffer, format='audio/mp3')
                                else:
                                    st.info("💡 Tip: Text-to-speech failed, but you can read the response above")
                        
                        st.session_state.chat_history.extend([
                            {"role": "user", "content": f"[Voice] {text}"},
                            {"role": "assistant", "content": response}
                        ])
                        
                        # Auto-save after voice chat
                        save_gallery_data()
    
                    else:
                        st.error("❌ No speech detected or transcription failed")
                        st.info("💡 Tip: Make sure the audio is clear and contains speech. Try a different file format.")
            except Exception as e:
                st.error(f"❌ Audio processing failed: {str(e)}")
                st.info("💡 Tip: Check if the audio file is valid and not corrupted. Supported formats: WAV, MP3, M4A, OGG")
    
    # Live recording (only for local deployment)
    if not is_cloud:
        st.subheader("🎤 Live Recording")
        
        if not lazy_import('sounddevice') or not lazy_import('soundfile'):
            st.warning("Live recording requires: pip install sounddevice soundfile")
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎤 Start Recording", use_container_width=True, disabled=st.session_state.listening):
                    st.session_state.listening = True
            
            with col2:
                if st.button("⏹️ Stop Recording", use_container_width=True, disabled=not st.session_state.listening):
                    st.session_state.listening = False
            
            if st.session_state.listening:
                st.warning("🔴 Recording... Speak now!")
                
                duration = 5
                sample_rate = 16000
                
                try:
                    sd = lazy_import('sounddevice')
                    sf = lazy_import('soundfile')
                    
                    with st.spinner(f"Recording for {duration} seconds..."):
                        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
                        sd.wait()
                        
                        temp_path = f"temp_recording_{int(time.time())}.wav"
                        sf.write(temp_path, audio_data, sample_rate)
                        
                        st.session_state.listening = False
                        
                        with st.spinner("Processing speech..."):
                            text = ai.speech_to_text(temp_path)
                            
                            if text and text.strip():
                                st.success("Speech processed successfully!")
                                st.write(f"**You said:** {text}")
                                
                                with st.spinner("AI is responding..."):
                                    response = ai.chat(text)
                                    st.write(f"**AI Response:** {response}")
                                    
                                    if lazy_import('gtts'):
                                        audio_buffer = ai.text_to_speech(response)
                                        if audio_buffer:
                                            st.audio(audio_buffer, format='audio/mp3')
                                
                                st.session_state.chat_history.extend([
                                    {"role": "user", "content": f"[Live Recording] {text}"},
                                    {"role": "assistant", "content": response}
                                ])
                            else:
                                st.warning("No speech detected. Please try again.")
                        
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                except Exception as e:
                    st.error(f"Recording failed: {str(e)}")
                    st.session_state.listening = False
    else:
        st.subheader("🎤 Live Recording")
        st.info("🌐 Live recording is not available on Streamlit Cloud. Use file upload instead.")
        st.markdown("""
        **For live recording, run locally:**
        ```bash
        pip install sounddevice soundfile
        streamlit run app.py
        ```
        """)

def render_gallery_mode():
    st.subheader("🖼️ Content Gallery")
    
    tab1, tab2, tab3 = st.tabs(["🎨 Images", "💻 Code", "💬 Chat History"])
    
    with tab1:
        if st.session_state.generated_images:
            st.write(f"**Total Images:** {len(st.session_state.generated_images)}")
            
            cols = st.columns(3)
            for i, img_data in enumerate(st.session_state.generated_images):
                with cols[i % 3]:
                    st.image(img_data["image"], caption=img_data["prompt"][:50] + "...", use_container_width=True)
                    st.caption(f"Generated: {img_data['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        else:
            st.info("No images generated yet. Go to Image Generation mode to create some!")
    
    with tab2:
        if st.session_state.generated_code:
            st.write(f"**Total Code Snippets:** {len(st.session_state.generated_code)}")
            
            for i, code_data in enumerate(st.session_state.generated_code):
                with st.expander(f"{code_data['language']} - {code_data['prompt'][:50]}..."):
                    st.code(code_data["code"], language=code_data["language"].lower())
                    st.caption(f"Generated: {code_data['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        else:
            st.info("No code generated yet. Go to Code Helper mode to create some!")
    
    with tab3:
        if st.session_state.chat_history:
            st.write(f"**Total Messages:** {len(st.session_state.chat_history)}")
            
            chat_history = st.session_state.get('chat_history', [])
            if isinstance(chat_history, list) and chat_history:
                for message in chat_history[-10:]:
                    if isinstance(message, dict):
                        if message["role"] == "user":
                            st.markdown(f"**You:** {message['content']}")
                        else:
                            st.markdown(f"**AI:** {message['content']}")

                    st.markdown("---")
            else:
                st.info("No chat history yet. Start a conversation in Chat mode!")

if __name__ == "__main__":
    main()