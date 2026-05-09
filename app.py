# app.py
import streamlit as st
from transformers import pipeline
import torch
import tempfile
import os
import numpy as np  # ✅ 添加 numpy 导入
import io
import scipy.io.wavfile as wav
import time

# 缓存模型
@st.cache_resource
def load_img2text_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_model():
    return pipeline(
        "text-generation", 
        model="distilgpt2",  # 更小更快，适合儿童故事
        max_new_tokens=120,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )

@st.cache_resource
def load_tts_model():
    """
    使用轻量级 TTS 模型
    选项 1: VITS (推荐 - 快速且稳定)
    选项 2: MMS-TTS (更小)
    """
    try:
        # 方案1: 使用 VITS 模型（效果最好）
        from transformers import VitsModel, AutoTokenizer
        
        model_name = "facebook/mms-tts-eng"  # 更小更快
        model = VitsModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        return {
            "type": "vits",
            "model": model,
            "tokenizer": tokenizer
        }
    except Exception as e:
        st.warning(f"Falling back to alternative TTS: {e}")
        # 方案2: 使用简单的 TTS pipeline
        return {
            "type": "pipeline",
            "model": pipeline("text-to-speech", model="facebook/mms-tts-eng")
        }

def img2text(model, image_path):
    """图片转文字"""
    result = model(image_path)
    return result[0]["generated_text"]

def generate_story(model, prompt, target_words=75):
    """生成指定长度的故事"""
    story_prompt = f"Write a short children's story about {prompt}. The story is for young kids, simple and happy end:"
    
    result = model(
        story_prompt,
        max_new_tokens=120,
        truncation=True,
        pad_token_id=50256  # GPT-2 的 pad token
    )
    
    full_text = result[0]['generated_text']
    story = full_text[len(story_prompt):].strip()
    
    # 确保故事完整
    if len(story) > 0:
        # 在最后一个句号处截断
        last_period = story.rfind('.')
        if last_period > 50:
            story = story[:last_period + 1]
    
    return story if story else f"Once upon a time, there was {prompt}."

def text_to_audio_fast(tts_model, text):
    """快速生成音频（优化版本）"""
    
    if tts_model["type"] == "vits":
        # 使用 VITS 模型
        model = tts_model["model"]
        tokenizer = tts_model["tokenizer"]
        
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            output = model(**inputs).waveform
        
        # 转换为 numpy 数组
        audio_array = output.squeeze().cpu().numpy()
        sampling_rate = model.config.sampling_rate
        
        # 确保音频格式正确
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        
        # 归一化到 [-1, 1] 范围
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val
        
    else:
        # 使用 pipeline
        result = tts_model["model"](text)
        audio_array = result["audio"][0] if isinstance(result["audio"], list) else result["audio"]
        sampling_rate = result["sampling_rate"]
        
        # 确保是 1D 数组
        if len(audio_array.shape) > 1:
            audio_array = audio_array.squeeze()
    
    return audio_array, sampling_rate

# 主应用
st.set_page_config(page_title="StoryTeller: Image to Audio Story", page_icon="📖")
st.title("📖 StoryTeller App")
st.markdown("Upload an image for a **50-100 word children's story** with audio!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示图片
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption="Your Image", use_container_width=True)
    
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # 进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Stage 1: Image to Text
    status_text.text("📷 Analyzing image...")
    progress_bar.progress(20)
    
    with st.spinner("Extracting image description..."):
        img2text_model = load_img2text_model()
        image_description = img2text(img2text_model, tmp_path)
    
    st.info(f"📷 **Image description:** {image_description}")
    
    # Stage 2: Text to Story
    status_text.text("📝 Writing story...")
    progress_bar.progress(40)
    
    with st.spinner("Creating a magical story..."):
        story_model = load_story_model()
        story = generate_story(story_model, image_description)
    
    st.success("📝 **Your Story**")
    st.write(f"> {story}")
    
    word_count = len(story.split())
    if 40 <= word_count <= 110:
        st.caption(f"✅ Word count: {word_count} words (within target)")
    else:
        st.caption(f"⚠️ Word count: {word_count} words")
    
    # Stage 3: Story to Audio (优化加载时间)
    status_text.text("🔊 Loading TTS model (first time only)...")
    progress_bar.progress(60)
    
    # 添加一个"生成音频"按钮，让用户控制何时生成（更友好）
    if st.button("🎵 Generate Audio", type="primary"):
        with st.spinner("Generating audio narration (may take 10-20 seconds)..."):
            progress_bar.progress(75)
            
            # 加载 TTS 模型（首次加载后缓存）
            tts_model = load_tts_model()
            
            status_text.text("🔊 Creating voice narration...")
            progress_bar.progress(80)
            
            start_time = time.time()
            audio_array, sample_rate = text_to_audio_fast(tts_model, story)
            gen_time = time.time() - start_time
            
            progress_bar.progress(95)
            status_text.text("✅ Audio ready!")
            
            st.success(f"🎧 Audio generated in {gen_time:.1f} seconds")
            
            # 播放音频
            st.audio(audio_array, sample_rate=sample_rate)
            
            # 下载按钮（修复 np 错误）
            try:
                # 转换为 int16 格式用于保存
                audio_int16 = (audio_array * 32767).astype(np.int16)
                
                wav_buffer = io.BytesIO()
                wav.write(wav_buffer, sample_rate, audio_int16)
                
                st.download_button(
                    label="💾 Download Audio (WAV)",
                    data=wav_buffer.getvalue(),
                    file_name="story_audio.wav",
                    mime="audio/wav"
                )
            except Exception as e:
                st.error(f"Audio download failed: {e}")
            
            progress_bar.progress(100)
            status_text.text("✨ Complete!")
    
    # 清理
    os.unlink(tmp_path)
    progress_bar.empty()
    status_text.empty()

# 页脚
st.markdown("---")
st.caption("Powered by Hugging Face Transformers | ⚡ Optimized for Streamlit Cloud")
