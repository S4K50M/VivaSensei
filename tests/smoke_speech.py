"""Use one real TTS and one STT request to check the configured ElevenLabs key."""

from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from speech_utils import speech_to_text, text_to_speech


def main():
    output = PROJECT_DIR / ".cache" / "speech" / "smoke.mp3"
    audio = text_to_speech("What is a current mirror?")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(audio)
    print(f"Speech synthesis passed: {len(audio)} bytes", flush=True)
    transcript = speech_to_text(output)
    print(f"Transcription passed: {transcript}", flush=True)
    print(f"Audio saved to {output}")


if __name__ == "__main__":
    main()
