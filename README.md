# AfzaAssistant - Advanced Local AI Assistant

A powerful local AI assistant with chat, voice, and image generation capabilities using free, open-source models.

![AfzaAssistant](https://img.shields.io/badge/AI-Assistant-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)

## 🚀 Features

- **💬 AI Chat** - Powered by Microsoft Phi-3 (3-5x faster than Llama3)
- **🎤 Voice Chat** - Speech-to-text using OpenAI Whisper
- **🎨 Image Generation** - Create images with Stable Diffusion
- **💻 Code Helper** - Programming assistance and code generation
- **📝 Quick Prompts** - Pre-built prompts for common tasks
- **💾 Chat History** - Persistent conversation memory
- **🔒 100% Local** - No data sent to external servers

## 📋 Requirements

### System Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **GPU**: Optional (NVIDIA GPU recommended for faster image generation)

### Software Requirements
- **Python**: 3.8 or higher
- **Ollama**: For running local AI models

## ⚡ Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/AfzaAssistant.git
cd AfzaAssistant
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama
**Windows:**
```bash
# Download and install from https://ollama.ai
# Or use PowerShell:
winget install Ollama.Ollama
```

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 4. Install AI Model
```bash
# Start Ollama service
ollama serve

# Install fast Phi-3 model (recommended)
ollama pull phi3
```

### 5. Run Application
```bash
# Easy start (Windows)
start.bat

# Or manually
streamlit run app.py
```

### 6. Open Browser
Navigate to: `http://localhost:8501`

## 🎯 Usage Guide

### Chat Interface
1. Type your question in the main search box
2. Click "AI Chat" for extended conversations
3. Use quick prompts on the right panel for common tasks

### Image Generation
1. Click "Image Generation"
2. Describe your desired image
3. Click "Generate" and wait for creation
4. Images appear instantly in the interface

### Voice Chat
1. Click "Voice Chat"
2. Upload audio file (WAV, MP3, M4A)
3. Click "Process" to transcribe and get AI response

### Code Helper
1. Click "Code Helper"
2. Describe your coding task
3. Get clean, commented code with examples

## 🔧 Configuration

### Switch AI Models
Edit `app.py` line 95 to change models:
```python
response = ollama.chat(model="phi3", messages=[...])
```

### Available Models (Speed Comparison)
```bash
# Recommended (Best balance)
ollama pull phi3              # 3-5x faster than Llama3

# Ultra Fast Options
ollama pull gemma2:2b         # 5-8x faster, 1.6GB
ollama pull qwen2:1.5b        # 8-10x faster, 934MB
ollama pull tinyllama         # 10x+ faster, 637MB

# High Quality Options
ollama pull llama3            # Baseline, 4.7GB
ollama pull mistral           # Good quality, 4.1GB
```

### GPU Acceleration
For faster image generation, ensure CUDA is installed:
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

## 📁 Project Structure
```
AfzaAssistant/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── start.bat          # Windows launcher
├── test_app.py        # Testing script
└── README.md          # This file
```

## 🛠️ Troubleshooting

### Common Issues

**1. Ollama Connection Error**
```bash
# Start Ollama service
ollama serve

# Check if running
ollama list
```

**2. Model Not Found**
```bash
# Install the model
ollama pull phi3

# Verify installation
ollama list
```

**3. Port Already in Use**
```bash
# Run on different port
streamlit run app.py --server.port 8502
```

**4. Memory Issues**
- Use smaller models: `gemma2:2b` or `tinyllama`
- Close other applications
- Reduce image generation steps

**5. GPU Not Detected**
- Install CUDA toolkit
- Update GPU drivers
- Restart application

### Performance Optimization

**For Low-End Systems:**
```bash
# Use ultra-fast model
ollama pull gemma2:2b
```

**For High-End Systems:**
```bash
# Use high-quality model
ollama pull llama3
```

## 🔄 Updates

### Update Models
```bash
# Update to latest version
ollama pull phi3

# List all models
ollama list

# Remove old models
ollama rm old_model_name
```

### Update Application
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## 📊 Performance Benchmarks

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| Phi-3 | 2.3GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Recommended |
| Gemma2:2b | 1.6GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Ultra Fast |
| Qwen2:1.5b | 934MB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Lightning |
| TinyLlama | 637MB | ⭐⭐⭐⭐⭐ | ⭐⭐ | Testing |
| Llama3 | 4.7GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High Quality |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft** - Phi-3 model
- **OpenAI** - Whisper speech recognition
- **Stability AI** - Stable Diffusion image generation
- **Ollama** - Local model serving
- **Streamlit** - Web interface framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/AfzaAssistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/AfzaAssistant/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/AfzaAssistant/wiki)

## 🔮 Roadmap

- [ ] Multi-language support
- [ ] Plugin system
- [ ] Mobile app
- [ ] Docker deployment
- [ ] API endpoints
- [ ] Custom model training

---

**Made with ❤️ for the AI community**

*Privacy-first • Open-source • Lightning-fast*