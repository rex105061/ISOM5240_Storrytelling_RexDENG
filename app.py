# app.py
import streamlit as st
from transformers import pipeline
import torch
import tempfile
import os

# 缓存模型，避免重复加载（关键优化！）
@st.cache_resource
def load_img2text_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_model():
    return pipeline(
        "text-generation", 
        model="gpt2",  # 改用 GPT-2，故事质量更好
        max_new_tokens=120,  # 约70-100词
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )

@st.cache_resource
def load_tts_model():
    return pipeline(
        "text-to-speech",
        model="suno/bark-small",  # 或 "facebook/fastspeech2-en-ljspeech"
        device="cpu"
    )

def img2text(model, image_path):
    result = model(image_path)
    return result[0]["generated_text"]

def generate_story(model, prompt, target_word_count=75):
    """生成指定长度的故事"""
    # 添加故事引导词
    enhanced_prompt = f"Write a short children's story about {prompt}. The story should be simple and fun:"
    
    result = model(enhanced_prompt, max_new_tokens=120, truncation=True)
    full_text = result[0]['generated_text']
    
    # 提取新生成的部分（去掉prompt）
    story = full_text[len(enhanced_prompt):].strip()
    
    # 确保故事以句号或换行结束
    if story and story[-1] not in '.!?':
        story += '.'
    
    # 统计词数并调整（可选）
    word_count = len(story.split())
    if word_count < 40:
        story += " The end. What a wonderful day it was!"
    
    return story

def text_to_audio(model, text):
    """生成音频并返回可播放的格式"""
    result = model(text)
    
    # 根据不同模型格式处理
    if "audio" in result:
        audio_array = result["audio"]
        sampling_rate = result["sampling_rate"]
    elif "array" in result:
        audio_array = result["array"]
        sampling_rate = result["sampling_rate"]
    else:
        # 备用方案：使用更简单的 TTS
        from transformers import VitsModel, AutoTokenizer
        import scipy.io.wavfile as wav
        
        model_vits = VitsModel.from_pretrained("facebook/mms-tts-eng")
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
        
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model_vits(**inputs).waveform
        
        audio_array = output.squeeze().cpu().numpy()
        sampling_rate = model_vits.config.sampling_rate
    
    return audio_array, sampling_rate

# 主应用
st.set_page_config(page_title="StoryTeller: Image to Audio Story", page_icon="📖")
st.title("📖 StoryTeller App")
st.markdown("Upload an image, and I'll create a **50-100 word children's story** with audio!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示上传的图片
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption="Your Image", use_container_width=True)
    
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # 进度提示
    progress_bar = st.progress(0, text="Starting...")
    
    # Stage 1: Image to Text
    progress_bar.progress(20, text="📷 Analyzing image...")
    with st.spinner("Extracting details from your image..."):
        img2text_model = load_img2text_model()
        image_description = img2text(img2text_model, tmp_path)
    st.success(f"📷 Image describes: _{image_description}_")
    
    # Stage 2: Text to Story
    progress_bar.progress(50, text="📝 Writing story...")
    with st.spinner("Crafting a magical story..."):
        story_model = load_story_model()
        story = generate_story(story_model, image_description)
    
    st.success("📝 Story generated!")
    st.write("### 📖 Your Story")
    st.write(f"> {story}")
    
    # 显示词数统计
    word_count = len(story.split())
    st.caption(f"Word count: {word_count} words {'✅' if 40 <= word_count <= 110 else '⚠️'}")
    
    # Stage 3: Story to Audio
    progress_bar.progress(75, text="🔊 Generating audio...")
    with st.spinner("Creating voice narration..."):
        tts_model = load_tts_model()
        audio_array, sample_rate = text_to_audio(tts_model, story)
    
    progress_bar.progress(100, text="✅ Ready!")
    progress_bar.empty()
    
    # 音频播放器
    st.audio(audio_array, sample_rate=sample_rate, format="audio/wav")
    
    # 可选：下载按钮
    import io
    import scipy.io.wavfile as wav
    wav_buffer = io.BytesIO()
    wav.write(wav_buffer, sample_rate, (audio_array * 32767).astype(np.int16) if audio_array.dtype == np.float32 else audio_array)
    st.download_button(
        label="⬇️ Download Story Audio",
        data=wav_buffer.getvalue(),
        file_name="story_audio.wav",
        mime="audio/wav"
    )
    
    # 清理临时文件
    os.unlink(tmp_path)

# 页脚
st.markdown("---")
st.caption("Powered by Hugging Face Transformers | Story length: ~50-100 words")
