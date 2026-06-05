import streamlit as st
import os
import json
import time
import datetime
from io import BytesIO
import base64

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

# Lazy import cache
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

# Cloud detection
def is_cloud_environment():
    return (
        os.getenv('STREAMLIT_SHARING_MODE') or 
        'streamlit.io' in os.getenv('HOSTNAME', '') or
        'streamlit' in os.getenv('SERVER_NAME', '') or
        os.path.exists('/.streamlit')
    )

# Initialize session state
def init_session_state():
    defaults = {
        'chat_history': [],
        'current_mode': "Chat",
        'generated_images': [],
        'generated_code': [],
        'listening': False,
        'ai_model_loaded': False,
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

# CSS Styles
st.markdown("""
<style>
.stDeployButton {display: none !important;}
header[data-testid="stHeader"] {display: none !important;}
.main .block-container {
    padding: 1rem;
    max-width: 100%;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
}
.chat-message {
    padding: 1rem;
    margin: 0.75rem 0;
    border-radius: 12px;
    max-width: 80%;
    box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    border: 1px solid #e2e8f0;
}
.user-message {
    background: #eff6ff;
    border-color: #bfdbfe;
    margin-left: auto;
    text-align: right;
}
.ai-message {
    background: #f8fafc;
    border-color: #e2e8f0;
    margin-right: auto;
}
.stButton > button {
    background: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #2563eb !important;
    color: white !important;
    border-color: #2563eb !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

# AI Assistant Class
class CloudAIAssistant:
    def __init__(self):
        self.is_cloud = is_cloud_environment()
        self.chat_model = None
        self.whisper_model = None
        self.image_pipeline = None
        
    def load_chat_model(self):
        if self.chat_model is not None:
            return True
            
        try:
            pipeline = lazy_import('transformers', 'pipeline')
            if not pipeline:
                return False
                
            with st.spinner("🤖 Loading AI model (first time only)..."):
                self.chat_model = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-large",  # 780MB - best quality for space
                    device=-1,
                    max_length=512
                )
            st.session_state.ai_model_loaded = True
            return True
        except Exception as e:
            st.error(f"Failed to load AI model: {str(e)}")
            return False
    
    def chat(self, message):
        if not self.load_chat_model():
            return "AI model not available. Please check dependencies."
            
        try:
            response = self.chat_model(
                message,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            return response[0]['generated_text'].strip()
                
        except Exception as e:
            return f"AI temporarily unavailable: {str(e)}"
    
    def load_image_model(self):
        if self.image_pipeline is not None:
            return True
            
        try:
            torch = lazy_import('torch')
            StableDiffusionPipeline = lazy_import('diffusers', 'StableDiffusionPipeline')
            
            if not torch or not StableDiffusionPipeline:
                return False
                
            with st.spinner("🎨 Loading image generation model..."):
                self.image_pipeline = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    low_cpu_mem_usage=True
                )
                
                if torch.cuda.is_available():
                    self.image_pipeline = self.image_pipeline.to("cuda")
                else:
                    self.image_pipeline = self.image_pipeline.to("cpu")
                    self.image_pipeline.enable_attention_slicing()
                    self.image_pipeline.enable_model_cpu_offload()
            return True
        except Exception as e:
            st.error(f"Failed to load image model: {str(e)}")
            return False
    
    def generate_image(self, prompt, width=512, height=512):
        if not self.load_image_model():
            return None, "Image generation model not available"
            
        try:
            with st.spinner("🎨 Generating image (this may take 1-2 minutes)..."):
                enhanced_prompt = f"{prompt}, high quality, detailed"
                negative_prompt = "blurry, low quality, distorted, ugly"
                
                image = self.image_pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    width=min(width, 512),
                    height=min(height, 512),
                    num_inference_steps=20,
                    guidance_scale=7.5
                ).images[0]
                
                return image, "success"
        except Exception as e:
            return None, f"Image generation failed: {str(e)}"
    
    def load_whisper_model(self):
        if self.whisper_model is not None:
            return True
            
        try:
            whisper = lazy_import('whisper')
            if not whisper:
                return False
                
            with st.spinner("🎤 Loading speech recognition model..."):
                self.whisper_model = whisper.load_model("base")
            return True
        except Exception as e:
            st.error(f"Failed to load Whisper model: {str(e)}")
            return False
    
    def speech_to_text(self, audio_file):
        if not self.load_whisper_model():
            return "Speech recognition not available"
            
        try:
            result = self.whisper_model.transcribe(audio_file)
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
            return None

# Initialize AI Assistant
@st.cache_resource
def get_ai_assistant():
    return CloudAIAssistant()

ai = get_ai_assistant()

# Microphone permission component
def render_microphone_permission():
    st.markdown("""
    <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
        <h4 style="color: #92400e; margin: 0 0 0.5rem 0;">🎤 Microphone Access</h4>
        <p style="color: #92400e; margin: 0;">
            For voice features to work, please allow microphone access when prompted by your browser.
            This app runs locally and doesn't send audio to external servers.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main layout functions
def render_header():
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="font-size: 2.5rem; color: #2563eb; margin-bottom: 0.5rem;">
            🤖 AfzaAssistant
        </h1>
        <p style="font-size: 1rem; color: #64748b; margin-bottom: 1rem;">
            Your AI Assistant - Chat, Generate Images, Code & Voice
        </p>
        <div style="background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 8px; padding: 0.5rem; margin: 1rem auto; max-width: 600px;">
            <p style="color: #0369a1; margin: 0; font-size: 0.9rem;">
                🌐 <strong>Cloud Mode:</strong> Running on Streamlit Cloud with optimized performance
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_status_panel():
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        # Check dependencies
        transformers_ok = lazy_import('transformers') is not None
        torch_ok = lazy_import('torch') is not None
        diffusers_ok = lazy_import('diffusers') is not None
        whisper_ok = lazy_import('whisper') is not None
        gtts_ok = lazy_import('gtts') is not None
        
        st.write("**Dependencies:**")
        st.write(f"{'✅' if transformers_ok else '❌'} Transformers (AI Chat)")
        st.write(f"{'✅' if torch_ok else '❌'} PyTorch")
        st.write(f"{'✅' if diffusers_ok else '❌'} Diffusers (Image Gen)")
        st.write(f"{'✅' if whisper_ok else '❌'} Whisper (Speech)")
        st.write(f"{'✅' if gtts_ok else '❌'} gTTS (Text-to-Speech)")
        
        st.write("**Statistics:**")
        st.metric("Chat Messages", len(st.session_state.chat_history))
        st.metric("Generated Images", len(st.session_state.generated_images))
        st.metric("Code Snippets", len(st.session_state.generated_code))
        
        if st.button("🗑️ Clear All Data"):
            st.session_state.chat_history = []
            st.session_state.generated_images = []
            st.session_state.generated_code = []
            st.success("All data cleared!")
            st.rerun()

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
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
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
            submitted = st.form_submit_button("Send", use_container_width=True)
    
    if submitted and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("AI is thinking..."):
            response = ai.chat(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

def render_image_mode():
    st.subheader("🎨 AI Image Generation")
    
    # Check dependencies
    if not lazy_import('diffusers') or not lazy_import('torch'):
        st.error("🚫 Image generation requires additional libraries")
        st.info("💡 Dependencies are being installed. Please wait and refresh the page.")
        st.code("pip install diffusers torch accelerate")
        return
    
    # Handle selected prompt
    if st.session_state.get('selected_prompt'):
        st.session_state.image_prompt_value = st.session_state.selected_prompt
        st.session_state.selected_prompt = None
    
    # Image generation form
    with st.form("image_form"):
        prompt = st.text_area(
            "Describe the image you want to generate:",
            value=st.session_state.get('image_prompt_value', ''),
            placeholder="A beautiful sunset over mountains, digital art style",
            height=100
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            width = st.selectbox("Width", [256, 512], index=1)
        with col2:
            height = st.selectbox("Height", [256, 512], index=1)
        with col3:
            submitted = st.form_submit_button("🎨 Generate Image", use_container_width=True)
        
        st.info("💡 Tip: Generation takes 1-2 minutes on cloud servers")
    
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
                label="📥 Download Image",
                data=img_buffer,
                file_name=f"generated_image_{int(time.time())}.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.error(f"❌ Failed to generate image: {result}")

def render_voice_mode():
    st.subheader("🎙️ Voice Chat")
    
    if not lazy_import('whisper'):
        st.error("Voice chat requires whisper. Dependencies are being installed.")
        return
    
    if is_cloud_environment():
        render_microphone_permission()
        st.info("🌐 Running on Streamlit Cloud - Live recording disabled. Upload audio files instead.")
    
    # Audio file upload
    st.subheader("📁 Upload Audio File")
    uploaded_audio = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'm4a', 'ogg'])
    
    if uploaded_audio:
        st.audio(uploaded_audio, format='audio/wav')
        
        if st.button("🎯 Process Audio", use_container_width=True):
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
                            
                            # Text-to-speech
                            audio_buffer = ai.text_to_speech(response)
                            if audio_buffer:
                                st.audio(audio_buffer, format='audio/mp3')
                        
                        st.session_state.chat_history.extend([
                            {"role": "user", "content": f"[Voice] {text}"},
                            {"role": "assistant", "content": response}
                        ])
                    else:
                        st.error("❌ No speech detected or transcription failed")
            except Exception as e:
                st.error(f"❌ Audio processing failed: {str(e)}")

def render_code_mode():
    st.subheader("💻 AI Code Helper")
    
    # Handle selected prompt
    if st.session_state.get('selected_prompt'):
        st.session_state.code_prompt_value = st.session_state.selected_prompt
        st.session_state.selected_prompt = None
    
    # Code generation form
    with st.form("code_form"):
        prompt = st.text_area(
            "Describe the code you need:",
            value=st.session_state.get('code_prompt_value', ''),
            placeholder="Create a Python function that sorts a list of dictionaries by a specific key",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            language = st.selectbox("Programming Language", 
                                  ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "HTML/CSS"])
        with col2:
            style = st.selectbox("Code Style", 
                               ["Clean & Simple", "With Comments", "Production Ready", "With Tests"])
        
        submitted = st.form_submit_button("💻 Generate Code", use_container_width=True)
    
    if submitted and prompt:
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
                
                # Download button
                st.download_button(
                    label="📥 Download Code",
                    data=code,
                    file_name=f"generated_code_{int(time.time())}.{language.lower()}",
                    mime="text/plain",
                    use_container_width=True
                )

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
            
            for message in st.session_state.chat_history[-10:]:
                if isinstance(message, dict):
                    if message["role"] == "user":
                        st.markdown(f"**You:** {message['content']}")
                    else:
                        st.markdown(f"**AI:** {message['content']}")
                    st.markdown("---")
        else:
            st.info("No chat history yet. Start a conversation in Chat mode!")

def render_prompts_panel():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 💡 Quick Prompts")
    
    current_mode = st.session_state.current_mode.lower()
    prompt_key = "chat" if "chat" in current_mode else "image" if "image" in current_mode else "code"
    
    st.write(f"**{prompt_key.title()} Prompts:**")
    
    for i, prompt in enumerate(st.session_state.saved_prompts[prompt_key]):
        button_text = prompt if len(prompt) <= 50 else prompt[:47] + "..."
        if st.button(button_text, key=f"prompt_{prompt_key}_{i}", use_container_width=True):
            st.session_state.selected_prompt = prompt
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main application
def main():
    render_header()
    render_status_panel()
    
    # Main layout
    left_col, main_col, right_col = st.columns([1, 3, 1])
    
    with left_col:
        st.markdown("### 🎯 Mode Selection")
        modes = ["Chat", "Image Generation", "Code Helper", "Voice Chat", "Gallery"]
        st.session_state.current_mode = st.selectbox("Choose mode:", modes, index=0)
    
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