# app.py
# Storytelling App: Image to Children's Story with Audio

import streamlit as st
from transformers import pipeline
import torch
import tempfile
import os
import numpy as np
import scipy.io.wavfile as wav
from transformers import AutoModelForCausalLM, AutoTokenizer

# Cache models
@st.cache_resource
def load_img2text_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_model():
    # Using TinyStories-33M (smaller, more stable on Streamlit Cloud)
    model_name = "roneneldan/TinyStories-33M"
    
    # Add timeout and retry handling
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=None
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    return {"tokenizer": tokenizer, "model": model}

@st.cache_resource
def load_tts_model():
    from transformers import VitsModel, AutoTokenizer
    model_name = "facebook/mms-tts-eng"
    model = VitsModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return {"model": model, "tokenizer": tokenizer}

# Image to text
def img2text(model, image_path):
    result = model(image_path)
    return result[0]["generated_text"]

# Generate story (50-100 words)
def generate_story(story_model, image_desc):
    prompt = f"Write a short children's story for ages 3-8 about {image_desc}. Use simple words, happy ending, 50-100 words: Once upon a time,"
    
    tokenizer = story_model["tokenizer"]
    model = story_model["model"]
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=150)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=130,
            min_new_tokens=45,
            temperature=0.85,
            top_p=0.92,
            repetition_penalty=1.18,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    story = story[len(prompt):].strip()
    
    if not story.startswith("Once"):
        story = "Once upon a time, " + story
    
    # Ensure story has proper ending
    if not story.endswith(('.', '!', '?')):
        story += " The end!"
    
    return story

# Text to audio
def text_to_audio(tts_model, text):
    model = tts_model["model"]
    tokenizer = tts_model["tokenizer"]
    
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        output = model(**inputs).waveform
    
    audio_array = output.squeeze().cpu().numpy()
    sampling_rate = model.config.sampling_rate
    
    # Normalize
    max_val = np.max(np.abs(audio_array))
    if max_val > 0:
        audio_array = audio_array / max_val
    
    return audio_array, sampling_rate

# Main app
st.set_page_config(page_title="StoryTeller", page_icon="📖")
st.title("📖 Turn Your Image into an Audio Story 🎧")
st.markdown("🖼️ Upload an image → 📝 Get a 50-100 word children's story → 🔊 Listen & download")

uploaded_file = st.file_uploader("📸 Select an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    st.image(uploaded_file, caption="🖼️ Your Uploaded Image", use_container_width=True)
    
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Stage 1: Image to Text
    with st.spinner("🔍 Analyzing image..."):
        img_model = load_img2text_model()
        image_desc = img2text(img_model, tmp_path)
    st.success("✅ Image analyzed!")
    st.info(f"📷 **What I see:** {image_desc}")
    
    # Stage 2: Text to Story
    with st.spinner("✍️ Writing a magical story..."):
        story_model = load_story_model()
        story = generate_story(story_model, image_desc)
    
    st.success("✨ Story generated!")
    st.write(f"📖 **Your Story** ({len(story.split())} words):")
    st.info(story)
    
    # Stage 3: Text to Audio
    with st.spinner("🔊 Generating audio narration..."):
        tts_model = load_tts_model()
        audio_array, sample_rate = text_to_audio(tts_model, story)
    
    # Play audio
    st.success("🎵 Audio ready!")
    st.audio(audio_array, sample_rate=sample_rate)
    
    # Download button
    audio_int16 = (audio_array * 32767).astype(np.int16)
    wav.write("story.wav", sample_rate, audio_int16)
    with open("story.wav", "rb") as f:
        st.download_button("💾 Download Audio (WAV)", f, file_name="story.wav")
    
    # Cleanup
    os.unlink(tmp_path)
    os.unlink("story.wav")
    
    # Footer
    st.markdown("---")
    st.caption("⭐ Powered by TinyStories-33M & Hugging Face | 🎯 50-100 word stories for ages 3-8")
