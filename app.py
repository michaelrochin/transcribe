import streamlit as st
import whisper
import tempfile
import os
import time
import ffmpeg
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Audio Transcription",
    page_icon="🎙️",
    layout="centered"
)

# Title
st.title("🎙️ Audio Transcription")
st.write("Upload your audio file and get the transcription instantly!")

# Initialize whisper model
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

def transcribe_audio(audio_file):
    # Create a unique temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / f"audio{os.path.splitext(audio_file.name)[1]}"
    
    try:
        # Save the uploaded file
        with open(temp_path, 'wb') as f:
            f.write(audio_file.getvalue())
        
        # Transcribe the audio
        result = model.transcribe(str(temp_path))
        return result["text"]
    
    finally:
        # Clean up with retry
        try:
            if temp_path.exists():
                temp_path.unlink()
            os.rmdir(temp_dir)
        except Exception as e:
            st.warning(f"Note: Temporary file cleanup will be handled by the system later.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=['mp3', 'wav', 'm4a', 'ogg', 'flac'],
    help="Upload your audio file here. Supported formats: MP3, WAV, M4A, OGG, FLAC"
)

if uploaded_file is not None:
    # Add a transcribe button
    if st.button("🎯 Transcribe"):
        with st.spinner('Transcribing your audio...'):
            try:
                # Get transcription
                transcription = transcribe_audio(uploaded_file)
                
                # Display results
                st.success("Transcription completed!")
                st.subheader("📝 Transcription:")
                st.write(transcription)
                
                # Download button
                st.download_button(
                    label="📥 Download Transcription",
                    data=transcription,
                    file_name=f"{uploaded_file.name}_transcription.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"An error occurred during transcription. Please try again.")
                st.error(f"Error details: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using OpenAI's Whisper")
