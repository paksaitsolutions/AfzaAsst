import streamlit as st
import os
import json
import time
import datetime
from io import BytesIO

# Environment setup
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="AfzaAssistant - AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Module management system
class ModuleManager:
    def __init__(self):
        self._cache = {}
        self.available_modules = self._check_modules()
    
    def _check_modules(self):
        modules = {
            'transformers': self._try_import('transformers'),
            'torch': self._try_import('torch'),
            'diffusers': self._try_import('diffusers'),
            'whisper': self._try_import('whisper'),
            'gtts': self._try_import('gtts'),
            'PIL': self._try_import('PIL'),
            'requests': self._try_import('requests')
        }
        return modules
    
    def _try_import(self, module_name):
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def get_module(self, module_name, package=None):
        cache_key = f"{module_name}.{package}" if package else module_name
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            if package:
                module = __import__(module_name, fromlist=[package])
                result = getattr(module, package, None)
            else:
                result = __import__(module_name)
            
            self._cache[cache_key] = result
            return result
        except (ImportError, AttributeError):
            self._cache[cache_key] = None
            return None
    
    def is_available(self, module_name):
        return self.available_modules.get(module_name, False)

# Initialize module manager
modules = ModuleManager()

# Cloud detection
def is_cloud():
    return (
        os.getenv('STREAMLIT_SHARING_MODE') or 
        'streamlit.io' in os.getenv('HOSTNAME', '') or
        'streamlit' in os.getenv('SERVER_NAME', '') or
        os.path.exists('/.streamlit')
    )

# AI Assistant with proper module management
class AfzaAssistant:
    def __init__(self):
        self.is_cloud = is_cloud()
        self.models = {}
        self.model_configs = {
            'chat': {
                'model': 'google/flan-t5-large',
                'task': 'text2text-generation',
                'size': '780MB'
            },
            'image': {
                'model': 'runwayml/stable-diffusion-v1-5',
                'size': '1.2GB'
            },
            'speech': {
                'model': 'base',
                'size': '290MB'
            }
        }
    
    def load_chat_model(self):
        if 'chat' in self.models:
            return True
        
        if not modules.is_available('transformers'):
            return False
        
        try:
            pipeline = modules.get_module('transformers', 'pipeline')
            with st.spinner("🤖 Loading AI chat model..."):
                self.models['chat'] = pipeline(
                    self.model_configs['chat']['task'],
                    model=self.model_configs['chat']['model'],
                    device=-1,
                    max_length=512
                )
            return True
        except Exception as e:
            st.error(f"Failed to load chat model: {str(e)}")
            return False
    
    def chat(self, message):
        if not self.load_chat_model():
            return "❌ Chat model not available. Check dependencies."
        
        try:
            response = self.models['chat'](
                message,
                max_length=200,
                temperature=0.7,
                do_sample=True
            )
            return response[0]['generated_text'].strip()
        except Exception as e:
            return f"❌ Chat error: {str(e)}"
    
    def load_image_model(self):
        if 'image' in self.models:
            return True
        
        if not modules.is_available('diffusers') or not modules.is_available('torch'):
            return False
        
        try:
            torch = modules.get_module('torch')
            StableDiffusionPipeline = modules.get_module('diffusers', 'StableDiffusionPipeline')
            
            with st.spinner("🎨 Loading image generation model..."):
                self.models['image'] = StableDiffusionPipeline.from_pretrained(
                    self.model_configs['image']['model'],
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    low_cpu_mem_usage=True
                )
                
                if torch.cuda.is_available():
                    self.models['image'] = self.models['image'].to("cuda")
                else:
                    self.models['image'] = self.models['image'].to("cpu")
                    self.models['image'].enable_attention_slicing()
                    self.models['image'].enable_model_cpu_offload()
            return True
        except Exception as e:
            st.error(f"Failed to load image model: {str(e)}")
            return False
    
    def generate_image(self, prompt, width=512, height=512):
        if not self.load_image_model():
            return None, "❌ Image model not available"
        
        try:
            with st.spinner("🎨 Generating image..."):
                image = self.models['image'](
                    prompt=f"{prompt}, high quality",
                    negative_prompt="blurry, low quality",
                    width=min(width, 512),
                    height=min(height, 512),
                    num_inference_steps=20,
                    guidance_scale=7.5
                ).images[0]
                return image, "✅ Success"
        except Exception as e:
            return None, f"❌ Generation failed: {str(e)}"
    
    def load_speech_model(self):
        if 'speech' in self.models:
            return True
        
        if not modules.is_available('whisper'):
            return False
        
        try:
            whisper = modules.get_module('whisper')
            with st.spinner("🎤 Loading speech model..."):
                self.models['speech'] = whisper.load_model(self.model_configs['speech']['model'])
            return True
        except Exception as e:
            st.error(f"Failed to load speech model: {str(e)}")
            return False
    
    def speech_to_text(self, audio_file):
        if not self.load_speech_model():
            return "❌ Speech model not available"
        
        try:
            result = self.models['speech'].transcribe(audio_file)
            return result["text"]
        except Exception as e:
            return f"❌ Speech recognition failed: {str(e)}"
    
    def text_to_speech(self, text):
        if not modules.is_available('gtts'):
            return None
        
        try:
            gTTS = modules.get_module('gtts', 'gTTS')
            tts = gTTS(text=text, lang='en', slow=False)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer
        except Exception:
            return None

# Initialize AI assistant
@st.cache_resource
def get_assistant():
    return AfzaAssistant()

ai = get_assistant()

# Session state management
def init_session_state():
    defaults = {
        'chat_history': [],
        'current_mode': "Chat",
        'generated_images': [],
        'generated_code': [],
        'saved_prompts': {
            "chat": [
                "Explain quantum computing in simple terms",
                "Write a professional email template",
                "Create a study plan for learning Python",
                "How to start a small business?",
                "Best practices for time management"
            ],
            "image": [
                "Futuristic city at sunset with flying cars",
                "Serene mountain landscape with crystal lake",
                "Modern minimalist office with plants",
                "Abstract geometric art with vibrant colors",
                "Cyberpunk street scene with neon lights"
            ],
            "code": [
                "Create a Python web scraper with BeautifulSoup",
                "Build a FastAPI REST API with authentication",
                "Write a React component with TypeScript",
                "Design a database schema for e-commerce",
                "Create a machine learning model for prediction"
            ]
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# UI Components
def render_header():
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="font-size: 2.5rem; color: #2563eb; margin-bottom: 0.5rem;">
            🤖 AfzaAssistant
        </h1>
        <p style="font-size: 1rem; color: #64748b; margin-bottom: 1rem;">
            Your Complete AI Assistant - Chat, Images, Code & Voice
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_system_status():
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        # Module status
        status_items = [
            ("🤖 AI Chat", modules.is_available('transformers')),
            ("🎨 Image Gen", modules.is_available('diffusers') and modules.is_available('torch')),
            ("🎤 Speech", modules.is_available('whisper')),
            ("🔊 TTS", modules.is_available('gtts')),
            ("🖼️ Images", modules.is_available('PIL'))
        ]
        
        for name, status in status_items:
            st.write(f"{'✅' if status else '❌'} {name}")
        
        # Model info
        st.markdown("### 📊 Model Info")
        for model_type, config in ai.model_configs.items():
            loaded = model_type in ai.models
            st.write(f"{'🟢' if loaded else '⚪'} {model_type.title()}: {config['size']}")
        
        # Statistics
        st.markdown("### 📈 Statistics")
        st.metric("Chat Messages", len(st.session_state.chat_history))
        st.metric("Generated Images", len(st.session_state.generated_images))
        st.metric("Code Snippets", len(st.session_state.generated_code))
        
        # Environment info
        st.markdown("### 🌐 Environment")
        st.write(f"{'☁️ Cloud' if ai.is_cloud else '💻 Local'}")

def render_chat_mode():
    st.subheader("💬 AI Chat")
    
    # Handle selected prompt
    if st.session_state.get('selected_prompt'):
        prompt = st.session_state.selected_prompt
        st.session_state.selected_prompt = None
        
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.spinner("AI is thinking..."):
            response = ai.chat(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    # Chat display
    for message in st.session_state.chat_history:
        if isinstance(message, dict):
            if message["role"] == "user":
                st.markdown(f"""
                <div style="background: #eff6ff; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; margin-left: 20%; text-align: right;">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #f8fafc; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; margin-right: 20%;">
                    <strong>AI:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input("Type your message:", placeholder="Ask me anything...")
        with col2:
            submitted = st.form_submit_button("Send", use_container_width=True)
    
    if submitted and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("AI is thinking..."):
            response = ai.chat(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

def render_image_mode():
    st.subheader("🎨 AI Image Generation")
    
    if not modules.is_available('diffusers') or not modules.is_available('torch'):
        st.error("🚫 Image generation requires diffusers and torch")
        st.code("pip install diffusers torch accelerate")
        return
    
    # Handle selected prompt
    if st.session_state.get('selected_prompt'):
        st.session_state.image_prompt = st.session_state.selected_prompt
        st.session_state.selected_prompt = None
    
    with st.form("image_form"):
        prompt = st.text_area(
            "Describe your image:",
            value=st.session_state.get('image_prompt', ''),
            placeholder="A beautiful sunset over mountains, digital art style",
            height=100
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            width = st.selectbox("Width", [256, 512], index=1)
        with col2:
            height = st.selectbox("Height", [256, 512], index=1)
        with col3:
            submitted = st.form_submit_button("🎨 Generate", use_container_width=True)
    
    if submitted and prompt:
        image, result = ai.generate_image(prompt, width, height)
        
        if image:
            st.success("✅ Image generated successfully!")
            st.image(image, caption=prompt, use_container_width=True)
            
            st.session_state.generated_images.append({
                "image": image,
                "prompt": prompt,
                "timestamp": datetime.datetime.now()
            })
            
            # Download button
            img_buffer = BytesIO()
            image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            
            st.download_button(
                "📥 Download Image",
                data=img_buffer,
                file_name=f"generated_image_{int(time.time())}.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.error(result)

def render_voice_mode():
    st.subheader("🎙️ Voice Chat")
    
    if not modules.is_available('whisper'):
        st.error("🚫 Voice features require whisper")
        st.code("pip install openai-whisper")
        return
    
    if ai.is_cloud:
        st.info("🌐 Cloud mode: Upload audio files only")
    
    uploaded_audio = st.file_uploader("Choose audio file", type=['wav', 'mp3', 'm4a', 'ogg'])
    
    if uploaded_audio:
        st.audio(uploaded_audio)
        
        if st.button("🎯 Process Audio", use_container_width=True):
            temp_path = f"temp_audio_{int(time.time())}.wav"
            
            try:
                with open(temp_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                
                text = ai.speech_to_text(temp_path)
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                if text and len(text.strip()) > 1:
                    st.success("✅ Speech converted!")
                    st.write(f"**Text:** {text}")
                    
                    response = ai.chat(text)
                    st.write(f"**AI:** {response}")
                    
                    # Text-to-speech
                    audio_buffer = ai.text_to_speech(response)
                    if audio_buffer:
                        st.audio(audio_buffer, format='audio/mp3')
                    
                    st.session_state.chat_history.extend([
                        {"role": "user", "content": f"[Voice] {text}"},
                        {"role": "assistant", "content": response}
                    ])
                else:
                    st.error("❌ No speech detected")
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")

def render_code_mode():
    st.subheader("💻 AI Code Helper")
    
    # Handle selected prompt
    if st.session_state.get('selected_prompt'):
        st.session_state.code_prompt = st.session_state.selected_prompt
        st.session_state.selected_prompt = None
    
    with st.form("code_form"):
        prompt = st.text_area(
            "Describe the code you need:",
            value=st.session_state.get('code_prompt', ''),
            placeholder="Create a Python function that sorts a list of dictionaries",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            language = st.selectbox("Language", 
                                  ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "HTML/CSS"])
        with col2:
            style = st.selectbox("Style", 
                               ["Clean & Simple", "With Comments", "Production Ready"])
        
        submitted = st.form_submit_button("💻 Generate Code", use_container_width=True)
    
    if submitted and prompt:
        enhanced_prompt = f"Write {style.lower()} {language} code for: {prompt}. Include proper error handling."
        
        with st.spinner("Generating code..."):
            code = ai.chat(enhanced_prompt)
            
            if code and len(code.strip()) > 10:
                st.success("✅ Code generated!")
                st.code(code, language=language.lower())
                
                st.session_state.generated_code.append({
                    "code": code,
                    "prompt": prompt,
                    "language": language,
                    "timestamp": datetime.datetime.now()
                })
                
                st.download_button(
                    "📥 Download Code",
                    data=code,
                    file_name=f"generated_code_{int(time.time())}.{language.lower()}",
                    mime="text/plain",
                    use_container_width=True
                )

def render_gallery_mode():
    st.subheader("🖼️ Content Gallery")
    
    tab1, tab2, tab3 = st.tabs(["🎨 Images", "💻 Code", "💬 Chat"])
    
    with tab1:
        if st.session_state.generated_images:
            cols = st.columns(3)
            for i, img_data in enumerate(st.session_state.generated_images):
                with cols[i % 3]:
                    st.image(img_data["image"], caption=img_data["prompt"][:50] + "...", use_container_width=True)
        else:
            st.info("No images generated yet")
    
    with tab2:
        if st.session_state.generated_code:
            for code_data in st.session_state.generated_code:
                with st.expander(f"{code_data['language']} - {code_data['prompt'][:50]}..."):
                    st.code(code_data["code"], language=code_data["language"].lower())
        else:
            st.info("No code generated yet")
    
    with tab3:
        if st.session_state.chat_history:
            for message in st.session_state.chat_history[-10:]:
                if isinstance(message, dict):
                    role = "You" if message["role"] == "user" else "AI"
                    st.markdown(f"**{role}:** {message['content']}")
                    st.markdown("---")
        else:
            st.info("No chat history yet")

def render_prompts_panel():
    st.markdown("### 💡 Quick Prompts")
    
    current_mode = st.session_state.current_mode.lower()
    prompt_key = "chat" if "chat" in current_mode else "image" if "image" in current_mode else "code"
    
    for i, prompt in enumerate(st.session_state.saved_prompts[prompt_key]):
        button_text = prompt if len(prompt) <= 50 else prompt[:47] + "..."
        if st.button(button_text, key=f"prompt_{i}", use_container_width=True):
            st.session_state.selected_prompt = prompt
            st.rerun()

# Main application
def main():
    # CSS
    st.markdown("""
    <style>
    .stDeployButton {display: none !important;}
    header[data-testid="stHeader"] {display: none !important;}
    .main .block-container {
        padding: 1rem;
        max-width: 100%;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)
    
    render_header()
    render_system_status()
    
    # Main layout
    left_col, main_col, right_col = st.columns([1, 3, 1])
    
    with left_col:
        st.markdown("### 🎯 Mode")
        modes = ["Chat", "Image Generation", "Code Helper", "Voice Chat", "Gallery"]
        st.session_state.current_mode = st.selectbox("Choose:", modes, index=0)
        
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.generated_images = []
            st.session_state.generated_code = []
            st.success("Cleared!")
            st.rerun()
    
    with main_col:
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
    
    with right_col:
        render_prompts_panel()

if __name__ == "__main__":
    main()