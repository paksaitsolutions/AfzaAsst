# Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud

1. **Fork/Clone this repository** to your GitHub account

2. **Create requirements.txt** with these dependencies:
```txt
streamlit>=1.28.0
requests>=2.31.0
Pillow>=10.0.0
diffusers>=0.21.0
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
openai-whisper>=20230314
gtts>=2.3.0
sounddevice>=0.4.6
soundfile>=0.12.1
ollama>=0.1.0
```

3. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

## Important Notes for Cloud Deployment

### ⚠️ Limitations on Streamlit Cloud:
- **No Ollama Support**: Local AI models (Ollama) won't work on Streamlit Cloud
- **Limited Resources**: Image generation may be slow or fail due to memory constraints
- **No Audio Recording**: Live audio recording requires local microphone access

### ✅ What Works on Streamlit Cloud:
- Basic UI and interface
- File uploads for audio processing
- Image generation (if enough memory)
- Code templates and prompts

### 🔧 Cloud-Optimized Version

For a cloud-friendly version, consider:

1. **Replace Ollama with API-based models**:
```python
# Instead of Ollama, use OpenAI API or Hugging Face API
import openai
# or
from transformers import pipeline
```

2. **Use lighter image models**:
```python
# Use smaller Stable Diffusion models
pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",  # Smaller model
    torch_dtype=torch.float16,        # Half precision
    low_cpu_mem_usage=True
)
```

3. **Add environment variables** for API keys:
```python
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

## Local Development vs Cloud

| Feature | Local | Streamlit Cloud |
|---------|-------|-----------------|
| Ollama AI Chat | ✅ | ❌ |
| Image Generation | ✅ | ⚠️ (Limited) |
| Voice Processing | ✅ | ⚠️ (Upload only) |
| File Operations | ✅ | ✅ |
| UI/UX | ✅ | ✅ |

## Recommended Cloud Setup

For production cloud deployment, use:
- **Heroku** or **Railway** for full control
- **Docker containers** for consistent environment
- **External AI APIs** instead of local models
- **Cloud storage** for generated content

## Troubleshooting Cloud Deployment

### Memory Errors:
```bash
# Add to requirements.txt
--find-links https://download.pytorch.org/whl/torch_stable.html
torch==2.0.0+cpu
```

### Import Errors:
- Ensure all dependencies are in requirements.txt
- Use specific version numbers
- Check Streamlit Cloud logs for detailed errors

### Performance Issues:
- Reduce image generation steps
- Use smaller models
- Implement caching with `@st.cache_resource`