# speech_utils.py

import os
import io
from pathlib import Path
from typing import Union, BinaryIO
from functools import lru_cache

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs


# Resolve configuration relative to this file, independent of Streamlit's cwd.
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env.local")
load_dotenv(PROJECT_DIR / ".env")

@lru_cache(maxsize=1)
def get_client():
    """Create the client on demand so text chat works without speech credentials."""
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set ELEVENLABS_API_KEY in VivaSensei/.env to use speech.")
    return ElevenLabs(api_key=api_key)


# ---------------------------------------------------------
# TEXT -> SPEECH
# ---------------------------------------------------------

def text_to_speech(
    text: str,
    voice_id: str = "kLhAstPcnnPxqzk6gS5i",
    model_id: str = "eleven_v4",
) -> bytes:
    """
    Convert text into speech.

    Returns:
        bytes: Raw audio bytes.

    These bytes can be:
        - saved to a file
        - returned from FastAPI
        - played directly in Streamlit
    """

    if not text.strip():
        raise ValueError("Text cannot be empty.")

    response = get_client().text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=model_id,
        output_format="mp3_44100_128",
    )

    audio_bytes = b"".join(response)
    if not audio_bytes:
        raise RuntimeError("ElevenLabs returned empty audio.")

    return audio_bytes


# ---------------------------------------------------------
# SPEECH -> TEXT
# ---------------------------------------------------------

def speech_to_text(
    audio: Union[str, Path, bytes, BinaryIO],
    model_id: str = "scribe_v2",
) -> str:
    """
    Convert speech/audio into text.

    audio can be:
        - path to a file
        - bytes
        - file-like object
          (FastAPI UploadFile.file / Streamlit UploadedFile)

    Returns:
        str: Transcribed text.
    """

    client = get_client()
    if isinstance(audio, (str, Path)):
        with open(audio, "rb") as f:
            response = client.speech_to_text.convert(
                file=f,
                model_id=model_id,
            )

    elif isinstance(audio, bytes):
        if not audio:
            raise ValueError("Audio cannot be empty.")
        audio_file = io.BytesIO(audio)

        # Some APIs expect a filename on file-like objects
        audio_file.name = "audio.wav"

        response = client.speech_to_text.convert(
            file=audio_file,
            model_id=model_id,
        )

    else:
        # Already a file-like object
        position = audio.tell() if audio.seekable() else None
        try:
            if position is not None:
                audio.seek(0)
            response = client.speech_to_text.convert(file=audio, model_id=model_id)
        finally:
            if position is not None:
                audio.seek(position)

    text = response.text
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("No speech was recognized. Try recording your answer again.")
    return text.strip()
