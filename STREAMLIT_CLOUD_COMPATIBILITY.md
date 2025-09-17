# Streamlit Cloud Compatibility Matrix

## ✅ WORKS ON STREAMLIT CLOUD

### 1. **AI Chat** 
- ✅ **Hugging Face Transformers** (Free)
- ✅ **OpenAI API** (Paid - with API key)
- ✅ **Google Gemini API** (Free tier available)
- ❌ **Ollama** (Local only)

### 2. **Image Generation**
- ⚠️ **Stable Diffusion** (Limited - may timeout/crash due to memory)
- ✅ **OpenAI DALL-E API** (Paid)
- ✅ **Stability AI API** (Paid)

### 3. **Speech-to-Text**
- ✅ **File Upload + Whisper** (Works with uploaded audio files)
- ❌ **Live Recording** (No microphone access on cloud)
- ✅ **OpenAI Whisper API** (Paid)

### 4. **Text-to-Speech**
- ✅ **gTTS (Google Text-to-Speech)** (Free)
- ✅ **OpenAI TTS API** (Paid)

### 5. **Code Generation**
- ✅ **All AI models work** (Same as chat)

### 6. **File Operations**
- ✅ **Upload/Download** (Works perfectly)
- ✅ **Gallery/History** (Works perfectly)

## ❌ DOESN'T WORK ON STREAMLIT CLOUD

1. **Ollama** - Requires local installation
2. **Live Audio Recording** - No microphone access
3. **Large Image Models** - Memory limitations
4. **Local File System** - Limited persistent storage

## 🔧 CLOUD-OPTIMIZED SETUP

### Minimal Working Setup:
```txt
streamlit>=1.28.0
transformers>=4.30.0
torch>=2.0.0
Pillow>=10.0.0
openai-whisper>=20230314
gtts>=2.3.0
requests>=2.31.0
```

### Full Featured Setup (with APIs):
```txt
streamlit>=1.28.0
transformers>=4.30.0
torch>=2.0.0
diffusers>=0.21.0
Pillow>=10.0.0
openai-whisper>=20230314
gtts>=2.3.0
openai>=1.0.0
google-generativeai>=0.3.0
requests>=2.31.0
```