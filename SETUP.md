# AfzaAssistant Setup Guide

## Quick Start (Windows)

### 1. Prerequisites
- **Python 3.8+** - Download from [python.org](https://python.org)
- **Ollama** - Download from [ollama.ai](https://ollama.ai)

### 2. Install Ollama and Models
```bash
# Install Ollama (Windows)
winget install Ollama.Ollama

# Or download installer from https://ollama.ai

# Start Ollama service
ollama serve

# Install AI model (in new terminal)
ollama pull phi3
```

### 3. Run AfzaAssistant
```bash
# Easy way - just double-click
start.bat

# Or manually
pip install -r requirements.txt
streamlit run app.py
```

### 4. Open Browser
Navigate to: `http://localhost:8501`

## Features

### ✅ Working Features
- **💬 AI Chat** - Powered by Ollama (phi3, llama3, etc.)
- **🎨 Image Generation** - Stable Diffusion (CPU optimized)
- **💻 Code Generation** - Programming assistance
- **🎙️ Voice Chat** - Speech-to-text with Whisper
- **🖼️ Gallery** - View all generated content
- **💾 Save/Export** - Download images, code, chat history

### 🔧 CPU Optimizations
- Attention slicing for memory efficiency
- Reduced inference steps for faster generation
- Automatic model caching
- Progressive loading

## Troubleshooting

### Common Issues

**1. "Ollama not running"**
```bash
# Start Ollama service
ollama serve

# Check if models are installed
ollama list

# Install a model if none exist
ollama pull phi3
```

**2. "Image generation failed"**
- First run takes 5-10 minutes to download models
- Requires 4-8GB RAM for image generation
- Use smaller image sizes (512x512) for faster generation

**3. "Voice chat not working"**
```bash
# Install audio dependencies
pip install sounddevice soundfile
```

**4. "Port already in use"**
```bash
# Run on different port
streamlit run app.py --server.port 8502
```

### Performance Tips

**For Low-End Systems:**
- Use smaller Ollama models: `ollama pull gemma2:2b`
- Generate smaller images (256x256 or 512x512)
- Close other applications while generating

**For Better Performance:**
- Use faster models: `ollama pull phi3` (recommended)
- Increase system RAM
- Use SSD storage for model caching

## Model Recommendations

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| phi3 | 2.3GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Recommended** |
| gemma2:2b | 1.6GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Fast & Light |
| llama3 | 4.7GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High Quality |
| qwen2:1.5b | 934MB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Ultra Fast |

## File Structure
```
AfzaAssistant/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── start.bat          # Windows launcher
├── SETUP.md           # This file
├── README.md          # Project documentation
└── outputs/           # Generated content
    ├── images/        # Generated images
    ├── code/          # Generated code
    ├── audio/         # Audio files
    └── chat/          # Chat history
```

## Support

If you encounter issues:
1. Check this setup guide
2. Ensure Ollama is running: `ollama serve`
3. Verify models are installed: `ollama list`
4. Check Python version: `python --version`
5. Try restarting the application

## Updates

To update AfzaAssistant:
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

To update Ollama models:
```bash
ollama pull phi3
```