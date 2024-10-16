import os
import json
import asyncio
import requests
import soundfile as sf
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, SpeakOptions
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
from pydub import AudioSegment
import streamlit as st

load_dotenv()

st.title("AI-Powered Audio Replacement")
st.write("Upload a video file with audio to start the processing.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

# Deepgram API key
DEEPGRAM_API_KEY = os.getenv("DG_API_KEY")

def correct_transcription(transcript):
    """Correct grammatical mistakes in the provided transcript using GPT-4o."""
    url = "https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": os.getenv("OPENAI_API_KEY"),
    }

    payload = {
        "messages": [
            {"role": "user", "content": f"Please correct any grammatical mistakes in the following text and don't write anything else except the corrected text: {transcript}"}
        ],
        "temperature": 0.5,
        "max_tokens": 1000,
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        response_json = response.json()
        corrected_text = response_json['choices'][0]['message']['content']
        return corrected_text
    else:
        st.error("Error: Unable to get a response from OpenAI API.")
        return None

async def transcribe_audio(audio_file_path):
    """Transcribe audio using Deepgram's Speech-to-Text API."""
    with open(audio_file_path, "rb") as audio_file:
        buffer_data = audio_file.read()

    payload = {"buffer": buffer_data}
    options = PrerecordedOptions(model="nova-2", smart_format=True)

    try:
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        return response.to_json()
    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")
        return None

def process_video_audio_sync(video_clip, audio_path, segments):
    """Process the video to sync it with the cleaned audio based on transcription segments."""
    # Load audio for processing
    audio = AudioSegment.from_file(audio_path)

    # Filler words to mute
    filler_words = ["um", "uh", "hmm", "umm"]
    cleaned_audio_segments = []
    cleaned_timestamps = []

    # Process each segment from transcription
    for segment in segments:
        word = segment['word']
        start = segment['start'] * 1000
        end = segment['end'] * 1000

        chunk = audio[start:end]

        # Check if the word is a filler word
        if word in filler_words:
            chunk = AudioSegment.silent(duration=len(chunk))  # Mute this chunk if it's a filler word
        
        # Append cleaned chunk and record the timestamps
        cleaned_audio_segments.append(chunk)
        cleaned_timestamps.append((segment['start'], segment['end']))

    # Concatenate cleaned audio segments
    cleaned_audio = sum(cleaned_audio_segments)
    cleaned_audio_path = "cleaned_audio.mp3"
    cleaned_audio.export(cleaned_audio_path, format="mp3")

    # Create new video clips based on cleaned timestamps
    new_video_clips = []
    for start, end in cleaned_timestamps:
        new_video_clips.append(video_clip.subclip(start, end))

    # Concatenate video clips and set the cleaned audio
    final_video = concatenate_videoclips(new_video_clips)
    final_audio = AudioFileClip(cleaned_audio_path)
    final_video = final_video.set_audio(final_audio)

    return final_video, cleaned_audio_path

def convert_text_to_speech(text):
    """Convert the provided text to speech using Deepgram's TTS API."""
    try:
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        
        SPEAK_OPTIONS = {"text": text}
        filename = "output_audio.wav"

        # Configure the options
        options = SpeakOptions(
            model="aura-helios-en",
            encoding="linear16",
            container="wav"
        )

        # Call the save method on the speak property
        response = deepgram.speak.v("1").save(filename, SPEAK_OPTIONS, options)
        return filename 
    
    except Exception as e:
        st.error(f"Error converting text to speech: {e}")
        return None

def adjust_audio_speed_pydub(audio_path, target_duration):
    """Adjust the speed of the audio to match the target duration using pydub."""
    # Load the audio file
    audio = AudioSegment.from_file(audio_path)

    # Calculate the current duration in milliseconds
    current_duration = len(audio)

    # Calculate the speed adjustment factor
    speed_factor = current_duration / (target_duration * 1000)

    # Speed up or slow down the audio
    adjusted_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed_factor)
    }).set_frame_rate(audio.frame_rate)

    # Save the adjusted audio
    adjusted_audio_path = "adjusted_ai_audio_pydub.wav"
    adjusted_audio.export(adjusted_audio_path, format="wav")

    return adjusted_audio_path

if uploaded_file is not None:
    st.write("Video uploaded successfully. Processing will start soon.")

    temp_video_path = "temp_uploaded_video.mp4"
    temp_audio_path = "extracted_audio.mp3"
    
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    video_clip = VideoFileClip(temp_video_path)
    video_clip.audio.write_audiofile(temp_audio_path, codec='libmp3lame')

    st.write("Audio extracted successfully.")
    st.write("Transcribing the audio with Deepgram...")

    transcription = asyncio.run(transcribe_audio(temp_audio_path))

    if transcription:
        try:
            if isinstance(transcription, str):
                transcription = json.loads(transcription)

            # Extract transcript and timestamps
            if transcription and "results" in transcription and "channels" in transcription["results"]:
                segments = transcription["results"]["channels"][0]["alternatives"][0]["words"]
                st.write("Transcription completed:")
                transcript = " ".join([segment['word'] for segment in segments])
                st.text_area("Transcribed Text", transcript, height=200)

                # Correct the transcription
                corrected_transcript = correct_transcription(transcript)

                if corrected_transcript:
                    st.write("Corrected Transcription:")
                    st.text_area("Corrected Text", corrected_transcript, height=200)

                    # Process video and sync cleaned audio
                    synced_video, cleaned_audio_path = process_video_audio_sync(video_clip, temp_audio_path, segments)
                    
                    st.write("Cleaned audio synced with video. Generating AI audio...")

                    # Generate AI audio from corrected transcript
                    ai_audio_path = convert_text_to_speech(corrected_transcript)

                    if ai_audio_path:
                        # Get synced video duration
                        synced_video_duration = synced_video.duration

                        # Adjust the AI audio to match the video duration using pydub
                        adjusted_ai_audio_path = adjust_audio_speed_pydub(ai_audio_path, synced_video_duration)

                        # Replace the audio in the synced video with the adjusted AI audio
                        final_video = synced_video.set_audio(AudioFileClip(adjusted_ai_audio_path))

                        # Save the final video
                        final_video_path = "final_output_video.mp4"
                        final_video.write_videofile(final_video_path, codec='libx264', audio_codec='aac')

                        st.write("Final video generated with AI audio.")
                        st.video(final_video_path)

        except Exception as e:
            st.error(f"Error while processing transcription: {e}")

    video_clip.close()
    try:
        os.remove(temp_audio_path)
        os.remove(temp_video_path)
    except Exception as e:
        st.error(f"Error during cleanup: {e}")

else:
    st.write("Please upload a video file to begin.")
