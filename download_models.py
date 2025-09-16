#!/usr/bin/env python3
"""
Pre-download all models for AfzaAssistant
Run this once to cache models locally
"""

print("Downloading TinyLlama model...")
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print("✓ TinyLlama downloaded")

print("Downloading Whisper model...")
import whisper
whisper_model = whisper.load_model("base")
print("✓ Whisper downloaded")

print("Downloading Stable Diffusion model...")
from diffusers import StableDiffusionXLPipeline
sd_model = StableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-turbo")
print("✓ Stable Diffusion downloaded")

print("All models downloaded successfully!")