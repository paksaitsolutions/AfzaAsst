import streamlit as st
import os
import json
import time
import datetime
from io import BytesIO
import gc

# Optimized environment setup
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AfzaAssistant - Complete AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Smart Module Manager with Memory Optimization
class SmartModuleManager:
    def __init__(self):
        self._cache = {}
        self.available = self._check_all_modules()
        self.memory_usage = {}
    
    def _check_all_modules(self):
        modules = {}
        test_modules = [
            'transformers', 'torch', 'diffusers', 'whisper', 
            'gtts', 'PIL', 'requests', 'accelerate'
        ]
        
        for module in test_modules:
            try:
                __import__(module)
                modules[module] = True
            except ImportError:
                modules[module] = False
        
        return modules
    
    def get(self, module_name, package=None):
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
        return self.available.get(module_name, False)
    
    def clear_cache(self):
        self._cache.clear()
        gc.collect()

modules = SmartModuleManager()

# Cloud detection
def is_cloud():
    return (
        os.getenv('STREAMLIT_SHARING_MODE') or 
        'streamlit.io' in os.getenv('HOSTNAME', '') or
        'streamlit' in os.getenv('SERVER_NAME', '') or
        os.path.exists('/.streamlit')
    )

# Perfect AI Assistant with Smart Memory Management
class PerfectAIAssistant:
    def __init__(self):
        self.is_cloud = is_cloud()
        self.models = {}
        self.model_status = {}
        
        # Optimized model configurations for 2.7GB limit
        self.configs = {
            'chat': {
                'model': 'google/flan-t5-base',  # 250MB - perfect balance
                'task': 'text2text-generation',
                'memory': '250MB'
            },
            'image': {
                'model': 'runwayml/stable-diffusion-v1-5',
                'memory': '1.2GB',
                'optimizations': ['attention_slicing', 'cpu_offload', 'low_mem']
            },
            'speech': {
                'model': 'base',  # 290MB - good quality
                'memory': '290MB'
            }
        }
    
    def get_memory_info(self):
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0
    
    def load_chat_model(self):
        if 'chat' in self.models:
            return True
        
        if not modules.is_available('transformers'):
            self.model_status['chat'] = "❌ transformers not available"
            return False
        
        try:
            pipeline = modules.get('transformers', 'pipeline')
            
            with st.spinner("🤖 Loading chat model (Flan-T5-Base)..."):
                self.models['chat'] = pipeline(
                    self.configs['chat']['task'],
                    model=self.configs['chat']['model'],
                    device=-1,
                    model_kwargs={"low_cpu_mem_usage": True}
                )
            
            self.model_status['chat'] = "✅ Loaded"
            return True
            
        except Exception as e:
            self.model_status['chat'] = f"❌ Error: {str(e)[:50]}"
            return False
    
    def chat(self, message):
        if not self.load_chat_model():
            return "❌ Chat model not available. Check transformers installation."
        
        try:
            response = self.models['chat'](
                message,
                max_length=200,
                temperature=0.7,
                do_sample=True,
                early_stopping=True
            )
            return response[0]['generated_text'].strip()
        except Exception as e:
            return f"❌ Chat error: {str(e)}"
    
    def load_image_model(self):
        if 'image' in self.models:
            return True
        
        if not modules.is_available('diffusers') or not modules.is_available('torch'):
            self.model_status['image'] = "❌ diffusers/torch not available"
            return False
        
        # Check available memory before loading
        current_memory = self.get_memory_info()
        if current_memory > 1500:  # If already using >1.5GB, skip image model
            self.model_status['image'] = "❌ Insufficient memory"
            return False
        
        try:
            torch = modules.get('torch')
            StableDiffusionPipeline = modules.get('diffusers', 'StableDiffusionPipeline')
            
            with st.spinner("🎨 Loading image model (this may take a moment)..."):
                # Use optimized settings for cloud
                self.models['image'] = StableDiffusionPipeline.from_pretrained(
                    self.configs['image']['model'],
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                )
                
                # Apply memory optimizations
                if torch.cuda.is_available():
                    self.models['image'] = self.models['image'].to("cuda")
                else:
                    self.models['image'] = self.models['image'].to("cpu")
                    self.models['image'].enable_attention_slicing()
                    self.models['image'].enable_model_cpu_offload()
                    
                    # Additional memory optimizations for cloud
                    if hasattr(self.models['image'], 'enable_sequential_cpu_offload'):
                        self.models['image'].enable_sequential_cpu_offload()
            
            self.model_status['image'] = "✅ Loaded with optimizations"
            return True
            
        except Exception as e:
            self.model_status['image'] = f"❌ Error: {str(e)[:50]}"
            return False
    
    def generate_image(self, prompt, width=512, height=512):
        if not self.load_image_model():
            return None, "❌ Image model not available"
        
        try:
            with st.spinner("🎨 Generating image (1-2 minutes on cloud)..."):
                # Optimized generation settings
                image = self.models['image'](
                    prompt=f"{prompt}, high quality, detailed",
                    negative_prompt="blurry, low quality, distorted, ugly, bad anatomy",
                    width=min(width, 512),
                    height=min(height, 512),
                    num_inference_steps=20,  # Good quality/speed balance
                    guidance_scale=7.5,
                    generator=None
                ).images[0]
                
                return image, "✅ Generated successfully"
                
        except Exception as e:
            # Clear memory and retry once
            gc.collect()
            try:
                image = self.models['image'](
                    prompt=prompt,
                    width=256, height=256,  # Smaller fallback
                    num_inference_steps=15,
                    guidance_scale=7.0
                ).images[0]
                return image, "✅ Generated (reduced size due to memory)"
            except:
                return None, f"❌ Generation failed: {str(e)}"
    
    def load_speech_model(self):
        if 'speech' in self.models:
            return True
        
        if not modules.is_available('whisper'):
            self.model_status['speech'] = "❌ whisper not available"
            return False
        
        try:
            whisper = modules.get('whisper')
            
            with st.spinner("🎤 Loading speech model (Whisper-Base)..."):
                self.models['speech'] = whisper.load_model(
                    self.configs['speech']['model'],
                    download_root=None
                )
            
            self.model_status['speech'] = "✅ Loaded"
            return True
            
        except Exception as e:
            self.model_status['speech'] = f"❌ Error: {str(e)[:50]}"
            return False
    
    def speech_to_text(self, audio_file):
        if not self.load_speech_model():
            return "❌ Speech model not available"
        
        try:
            result = self.models['speech'].transcribe(
                audio_file,
                fp16=False,  # Better compatibility
                language='en'  # Faster processing
            )
            return result["text"].strip()
        except Exception as e:
            return f"❌ Speech recognition failed: {str(e)}"
    
    def text_to_speech(self, text):
        if not modules.is_available('gtts'):
            return None
        
        try:
            gTTS = modules.get('gtts', 'gTTS')
            tts = gTTS(text=text[:500], lang='en', slow=False)  # Limit text length
            buffer = BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)
            return buffer
        except Exception:
            return None
    
    def unload_model(self, model_name):
        """Unload model to free memory"""
        if model_name in self.models:
            del self.models[model_name]
            self.model_status[model_name] = "⚪ Unloaded"
            gc.collect()
    
    def get_status(self):
        """Get system status"""
        return {
            'memory_mb': self.get_memory_info(),
            'models_loaded': len(self.models),
            'model_status': self.model_status,
            'is_cloud': self.is_cloud
        }

# Initialize AI Assistant
@st.cache_resource
def get_ai_assistant():
    return PerfectAIAssistant()

ai = get_ai_assistant()

# Enhanced Session State Management
def init_session_state():
    defaults = {
        'chat_history': [],
        'current_mode': "Chat",
        'generated_images': [],
        'generated_code': [],
        'selected_prompt': None,
        'saved_prompts': {
            "chat": [
                "Explain quantum computing in simple terms",
                "Write a professional email template",
                "Create a study plan for learning Python",
                "How to start a small business?",
                "Best practices for time management",
                "Explain machine learning concepts"
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
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Enhanced UI Components
def render_header():
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="font-size: 2.5rem; color: #2563eb; margin-bottom: 0.5rem;">
            🤖 AfzaAssistant Perfect
        </h1>
        <p style="font-size: 1rem; color: #64748b; margin-bottom: 1rem;">
            Complete AI Assistant - Optimized for Cloud Performance
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_system_dashboard():
    with st.sidebar:
        st.markdown("### 🔧 System Dashboard")
        
        # Get system status
        status = ai.get_status()
        
        # Memory usage
        memory_mb = status['memory_mb']
        memory_color = "🟢" if memory_mb < 1500 else "🟡" if memory_mb < 2000 else "🔴"
        st.metric("Memory Usage", f"{memory_mb:.0f}MB", delta=f"{memory_color}")
        
        # Module availability
        st.markdown("**📦 Modules:**")
        module_status = [
            ("🤖 Chat AI", modules.is_available('transformers')),
            ("🎨 Image Gen", modules.is_available('diffusers') and modules.is_available('torch')),
            ("🎤 Speech", modules.is_available('whisper')),
            ("🔊 TTS", modules.is_available('gtts')),
            ("🖼️ Images", modules.is_available('PIL'))
        ]
        
        for name, available in module_status:
            st.write(f"{'✅' if available else '❌'} {name}")
        
        # Model status
        st.markdown("**🧠 Models:**")
        for model_name, config in ai.configs.items():
            status_text = ai.model_status.get(model_name, "⚪ Not loaded")
            st.write(f"{status_text} {model_name.title()}")
        
        # Statistics
        st.markdown("### 📊 Statistics")
        st.metric("Chat Messages", len(st.session_state.chat_history))
        st.metric("Generated Images", len(st.session_state.generated_images))
        st.metric("Code Snippets", len(st.session_state.generated_code))
        
        # Memory management
        st.markdown("### 🧹 Memory Management")
        if st.button("🗑️ Clear Cache", use_container_width=True):
            modules.clear_cache()
            st.cache_resource.clear()
            st.success("Cache cleared!")
        
        if st.button("🔄 Unload Models", use_container_width=True):
            for model_name in list(ai.models.keys()):
                ai.unload_model(model_name)
            st.success("Models unloaded!")
        
        # Environment info
        st.markdown("### 🌐 Environment")
        env_icon = "☁️" if ai.is_cloud else "💻"
        st.write(f"{env_icon} {'Cloud' if ai.is_cloud else 'Local'}")

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
    
    # Chat display with better styling
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if isinstance(message, dict):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                color: white; padding: 1rem; border-radius: 15px; 
                                margin: 0.5rem 0; margin-left: 20%; text-align: right;">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                color: white; padding: 1rem; border-radius: 15px; 
                                margin: 0.5rem 0; margin-right: 20%;">
                        <strong>AI:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Enhanced chat input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input(
                "Type your message:", 
                placeholder="Ask me anything... I'm powered by Flan-T5!",
                help="Try asking about science, coding, business, or creative writing!"
            )
        with col2:
            submitted = st.form_submit_button("Send 🚀", use_container_width=True)
    
    if submitted and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("AI is thinking..."):
            response = ai.chat(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

def render_image_mode():
    st.subheader("🎨 AI Image Generation")
    
    # Check dependencies with helpful messages
    if not modules.is_available('diffusers') or not modules.is_available('torch'):
        st.error("🚫 Image generation requires additional libraries")
        st.markdown("""
        **Missing Dependencies:**
        ```bash
        pip install diffusers torch accelerate
        ```
        
        **For Streamlit Cloud:** Add to requirements.txt:
        ```
        diffusers>=0.21.0
        torch>=2.0.0
        accelerate>=0.20.0
        ```
        """)
        return
    
    # Memory warning
    memory_mb = ai.get_memory_info()
    if memory_mb > 1800:
        st.warning("⚠️ High memory usage detected. Image generation may fail. Try clearing cache first.")
    
    # Handle selected prompt
    if st.session_state.get('selected_prompt'):
        st.session_state.image_prompt = st.session_state.selected_prompt
        st.session_state.selected_prompt = None
    
    with st.form("image_form"):
        prompt = st.text_area(
            "Describe your image:",
            value=st.session_state.get('image_prompt', ''),
            placeholder="A majestic dragon flying over a medieval castle at sunset, fantasy art style",
            height=100,
            help="Be descriptive! Include style, mood, colors, and details."
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            width = st.selectbox("Width", [256, 512], index=1)
        with col2:
            height = st.selectbox("Height", [256, 512], index=1)
        with col3:
            submitted = st.form_submit_button("🎨 Generate Magic", use_container_width=True)
        
        st.info("💡 Tip: 512x512 takes 1-2 minutes on cloud. 256x256 is faster!")
    
    if submitted and prompt:
        if len(prompt.strip()) < 5:
            st.error("⚠️ Please provide a more detailed description (at least 5 characters)")
            return
        
        image, result = ai.generate_image(prompt, width, height)
        
        if image:
            st.success("✅ Image generated successfully!")
            st.image(image, caption=prompt, use_container_width=True)
            
            # Save to gallery
            st.session_state.generated_images.append({
                "image": image,
                "prompt": prompt,
                "timestamp": datetime.datetime.now(),
                "size": f"{width}x{height}"
            })
            
            # Download options
            col1, col2 = st.columns(2)
            
            with col1:
                img_buffer = BytesIO()
                image.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                
                st.download_button(
                    "📥 Download PNG",
                    data=img_buffer,
                    file_name=f"afza_generated_{int(time.time())}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col2:
                img_buffer_jpg = BytesIO()
                image.convert('RGB').save(img_buffer_jpg, format="JPEG", quality=95)
                img_buffer_jpg.seek(0)
                
                st.download_button(
                    "📥 Download JPG",
                    data=img_buffer_jpg,
                    file_name=f"afza_generated_{int(time.time())}.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
        else:
            st.error(result)
            if "memory" in result.lower():
                st.info("💡 Try: 1) Clear cache 2) Use 256x256 size 3) Restart app")

def render_voice_mode():
    st.subheader("🎙️ Voice Chat")
    
    if not modules.is_available('whisper'):
        st.error("🚫 Voice features require Whisper")
        st.code("pip install openai-whisper")
        return
    
    # Cloud-specific info
    if ai.is_cloud:
        st.info("🌐 Cloud Mode: Upload audio files for processing")
        st.markdown("""
        **Supported formats:** WAV, MP3, M4A, OGG  
        **Max file size:** 200MB  
        **Best quality:** Clear speech, minimal background noise
        """)
    
    # Audio upload
    uploaded_audio = st.file_uploader(
        "Choose an audio file", 
        type=['wav', 'mp3', 'm4a', 'ogg'],
        help="Upload clear audio with speech for best results"
    )
    
    if uploaded_audio:
        # Show audio player
        st.audio(uploaded_audio, format='audio/wav')
        
        # File info
        file_size = len(uploaded_audio.getbuffer()) / 1024 / 1024
        st.caption(f"File: {uploaded_audio.name} ({file_size:.1f}MB)")
        
        if st.button("🎯 Process Audio", use_container_width=True):
            temp_path = f"temp_audio_{int(time.time())}.wav"
            
            try:
                # Save uploaded file
                with open(temp_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                
                # Process speech
                with st.spinner("🎤 Converting speech to text..."):
                    text = ai.speech_to_text(temp_path)
                
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                if text and len(text.strip()) > 1:
                    st.success("✅ Speech converted successfully!")
                    
                    # Display transcription
                    st.markdown("**🎤 Transcribed Text:**")
                    st.markdown(f"*\"{text}\"*")
                    
                    # Get AI response
                    with st.spinner("🤖 AI is responding..."):
                        response = ai.chat(text)
                    
                    st.markdown("**🤖 AI Response:**")
                    st.markdown(response)
                    
                    # Text-to-speech
                    audio_buffer = ai.text_to_speech(response)
                    if audio_buffer:
                        st.markdown("**🔊 Listen to AI Response:**")
                        st.audio(audio_buffer, format='audio/mp3')
                    
                    # Save to chat history
                    st.session_state.chat_history.extend([
                        {"role": "user", "content": f"[Voice] {text}"},
                        {"role": "assistant", "content": response}
                    ])
                    
                    st.success("💾 Conversation saved to chat history!")
                    
                else:
                    st.error("❌ No speech detected or transcription failed")
                    st.info("💡 Tips: Ensure clear audio, minimal background noise, and supported format")
                    
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)

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
            placeholder="Create a Python function that validates email addresses using regex",
            height=100,
            help="Be specific about functionality, input/output, and any special requirements"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            language = st.selectbox(
                "Programming Language", 
                ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "HTML/CSS", "SQL"],
                help="Choose your preferred programming language"
            )
        with col2:
            style = st.selectbox(
                "Code Style", 
                ["Clean & Simple", "With Comments", "Production Ready", "With Tests"],
                help="Select the level of detail and documentation"
            )
        
        submitted = st.form_submit_button("💻 Generate Code", use_container_width=True)
    
    if submitted and prompt:
        if len(prompt.strip()) < 5:
            st.error("⚠️ Please provide a more detailed description (at least 5 characters)")
            return
        
        # Enhanced prompt engineering
        enhanced_prompt = f"""
        Write {style.lower()} {language} code for: {prompt}
        
        Requirements:
        - Include proper error handling
        - Follow best practices for {language}
        - Add appropriate comments if requested
        - Make code readable and maintainable
        """
        
        with st.spinner("🔧 Generating code..."):
            code = ai.chat(enhanced_prompt)
        
        if code and len(code.strip()) > 10:
            st.success("✅ Code generated successfully!")
            
            # Display code with syntax highlighting
            st.code(code, language=language.lower())
            
            # Save to gallery
            st.session_state.generated_code.append({
                "code": code,
                "prompt": prompt,
                "language": language,
                "style": style,
                "timestamp": datetime.datetime.now()
            })
            
            # Download options
            col1, col2 = st.columns(2)
            
            with col1:
                file_ext = {
                    'Python': 'py', 'JavaScript': 'js', 'Java': 'java', 
                    'C++': 'cpp', 'Go': 'go', 'Rust': 'rs', 
                    'HTML/CSS': 'html', 'SQL': 'sql'
                }.get(language, 'txt')
                
                st.download_button(
                    f"📥 Download .{file_ext}",
                    data=code,
                    file_name=f"afza_code_{int(time.time())}.{file_ext}",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                # Create a formatted version with metadata
                formatted_code = f"""/*
Generated by AfzaAssistant
Prompt: {prompt}
Language: {language}
Style: {style}
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
*/

{code}"""
                
                st.download_button(
                    "📥 Download with Info",
                    data=formatted_code,
                    file_name=f"afza_code_info_{int(time.time())}.{file_ext}",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.error("❌ Failed to generate valid code")
            st.info("💡 Try: 1) More specific prompt 2) Different language 3) Simpler request")

def render_gallery_mode():
    st.subheader("🖼️ Content Gallery")
    
    tab1, tab2, tab3 = st.tabs(["🎨 Images", "💻 Code", "💬 Chat History"])
    
    with tab1:
        if st.session_state.generated_images:
            st.write(f"**Total Images:** {len(st.session_state.generated_images)}")
            
            # Image grid
            cols = st.columns(3)
            for i, img_data in enumerate(st.session_state.generated_images):
                with cols[i % 3]:
                    st.image(
                        img_data["image"], 
                        caption=f"{img_data['prompt'][:40]}...", 
                        use_container_width=True
                    )
                    st.caption(f"🕒 {img_data['timestamp'].strftime('%m/%d %H:%M')}")
                    st.caption(f"📐 {img_data.get('size', 'Unknown')}")
        else:
            st.info("🎨 No images generated yet. Go to Image Generation to create some!")
    
    with tab2:
        if st.session_state.generated_code:
            st.write(f"**Total Code Snippets:** {len(st.session_state.generated_code)}")
            
            for i, code_data in enumerate(st.session_state.generated_code):
                with st.expander(f"🔧 {code_data['language']} - {code_data['prompt'][:50]}..."):
                    st.caption(f"Style: {code_data.get('style', 'Unknown')} | Generated: {code_data['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                    st.code(code_data["code"], language=code_data["language"].lower())
        else:
            st.info("💻 No code generated yet. Go to Code Helper to create some!")
    
    with tab3:
        if st.session_state.chat_history:
            st.write(f"**Total Messages:** {len(st.session_state.chat_history)}")
            
            # Show recent messages
            for message in st.session_state.chat_history[-15:]:
                if isinstance(message, dict):
                    role = "You" if message["role"] == "user" else "AI"
                    icon = "👤" if message["role"] == "user" else "🤖"
                    st.markdown(f"**{icon} {role}:** {message['content']}")
                    st.markdown("---")
        else:
            st.info("💬 No chat history yet. Start a conversation in Chat mode!")

def render_prompts_panel():
    st.markdown("### 💡 Quick Prompts")
    
    current_mode = st.session_state.current_mode.lower()
    prompt_key = "chat" if "chat" in current_mode else "image" if "image" in current_mode else "code"
    
    st.caption(f"**{prompt_key.title()} Prompts:**")
    
    for i, prompt in enumerate(st.session_state.saved_prompts[prompt_key]):
        button_text = prompt if len(prompt) <= 45 else prompt[:42] + "..."
        if st.button(
            button_text, 
            key=f"prompt_{prompt_key}_{i}", 
            use_container_width=True,
            help=prompt
        ):
            st.session_state.selected_prompt = prompt
            st.rerun()
    
    # Add custom prompt
    with st.expander("➕ Add Custom Prompt"):
        new_prompt = st.text_area("Enter new prompt:", height=60)
        if st.button("💾 Save Prompt", use_container_width=True):
            if new_prompt.strip():
                st.session_state.saved_prompts[prompt_key].append(new_prompt.strip())
                st.success("Prompt saved!")
                st.rerun()

# Main Application
def main():
    # Enhanced CSS
    st.markdown("""
    <style>
    .stDeployButton {display: none !important;}
    header[data-testid="stHeader"] {display: none !important;}
    .main .block-container {
        padding: 1rem;
        max-width: 100%;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .stTextArea > div > div > textarea {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    render_header()
    render_system_dashboard()
    
    # Main layout
    left_col, main_col, right_col = st.columns([1, 3, 1])
    
    with left_col:
        st.markdown("### 🎯 Mode Selection")
        modes = ["Chat", "Image Generation", "Code Helper", "Voice Chat", "Gallery"]
        st.session_state.current_mode = st.selectbox(
            "Choose mode:", 
            modes, 
            index=0,
            help="Select the AI feature you want to use"
        )
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.generated_images = []
            st.session_state.generated_code = []
            st.success("All data cleared!")
            st.rerun()
        
        if st.button("💾 Export Data", use_container_width=True):
            export_data = {
                "chat_history": st.session_state.chat_history,
                "generated_images_count": len(st.session_state.generated_images),
                "generated_code_count": len(st.session_state.generated_code),
                "export_time": datetime.datetime.now().isoformat()
            }
            
            st.download_button(
                "📥 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"afza_export_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )
    
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