import os
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get API key
api_key = os.getenv("ELEVENLABS_API_KEY")
if not api_key:
    raise ValueError("ELEVENLABS_API_KEY not found in environment variables")

# Initialize ElevenLabs client with API key
client = ElevenLabs(api_key=api_key)

def text_to_speech(text):
    """
    Convert input text to speech using ElevenLabs streaming API and return audio bytes.
    
    Args:
        text (str): The text to convert to speech.
        
    Returns:
        bytes: Audio data in MP3 format, or None if an error occurs.
    """
    try:
        # Generate audio stream with ElevenLabs
        audio_stream = client.text_to_speech.convert_as_stream(
            text=text,
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel (seductive female voice)
            model_id="eleven_multilingual_v2",  # Multilingual model
            output_format="mp3_44100_128",  # Standard MP3 format
            optimize_streaming_latency=1  # Normal latency optimization
        )

        # Collect audio stream into bytes
        audio_bytes = b""
        for chunk in audio_stream:
            if isinstance(chunk, bytes):
                audio_bytes += chunk

        return audio_bytes

    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        return None