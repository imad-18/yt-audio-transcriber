import os
import streamlit as st
import whisper
import yt_dlp
import static_ffmpeg

# Add FFmpeg binaries to system PATH automatically
static_ffmpeg.add_paths()

# Set page config
st.set_page_config(page_title="YouTube Audio Transcriber", page_icon="🎙️")


@st.cache_resource
def load_whisper_model():
    """Cache model loading so it stays in memory across reruns."""
    return whisper.load_model("base")


def download_youtube_audio(url: str, output_path: str = "temp_audio") -> str:
    """Extract audio stream from YouTube URL and save as WAV file."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return f"{output_path}.wav"


def cleanup_file(file_path: str):
    """Remove temporary audio file after transcription."""
    if os.path.exists(file_path):
        os.remove(file_path)


# UI Layout
st.title("🎙️ YouTube Audio Transcriber")
st.write("Convert any YouTube video audio into text instantly.")

url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Generate Transcript", type="primary"):
    if not url:
        st.warning("Please enter a valid YouTube URL.")
    else:
        audio_file = None
        try:
            model = load_whisper_model()

            with st.spinner("Downloading audio stream..."):
                audio_file = download_youtube_audio(url)

            with st.spinner("Transcribing audio (using OpenAI Whisper)..."):
                result = model.transcribe(audio_file)

            st.success("Transcription complete!")
            st.subheader("Transcript")
            st.write(result["text"])

            # Download transcript option
            st.download_button(
                label="Download Transcript (.txt)",
                data=result["text"],
                file_name="transcript.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")

        finally:
            if audio_file:
                cleanup_file(audio_file)