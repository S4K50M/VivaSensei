"""Validate speech configuration and SDK inputs without using API credits."""

import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import speech_utils


class SpeechUtilityTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.speech_to_text.convert.return_value.text = " Recognized answer "
        loader = patch("speech_utils.get_client", return_value=self.client)
        loader.start()
        self.addCleanup(loader.stop)

    def test_synthesis_returns_mp3_and_rejects_empty_output(self):
        self.client.text_to_speech.convert.return_value = iter([b"one", b"two"])
        self.assertEqual(speech_utils.text_to_speech("A question?"), b"onetwo")
        self.assertEqual(self.client.text_to_speech.convert.call_args.kwargs["output_format"], "mp3_44100_128")
        self.client.text_to_speech.convert.return_value = iter([])
        with self.assertRaisesRegex(RuntimeError, "empty audio"):
            speech_utils.text_to_speech("A question?")
        with self.assertRaisesRegex(ValueError, "Text cannot be empty"):
            speech_utils.text_to_speech("  ")

    def test_transcription_accepts_bytes_paths_and_reused_streams(self):
        self.assertEqual(speech_utils.speech_to_text(b"wave"), "Recognized answer")
        uploaded = self.client.speech_to_text.convert.call_args.kwargs["file"]
        self.assertEqual(uploaded.getvalue(), b"wave")
        self.assertEqual(uploaded.name, "audio.wav")
        stream = io.BytesIO(b"wave")
        stream.seek(4)
        self.client.speech_to_text.convert.side_effect = lambda **kwargs: (
            self.assertEqual(kwargs["file"].read(), b"wave") or Mock(text="Answer")
        )
        self.assertEqual(speech_utils.speech_to_text(stream), "Answer")
        self.assertEqual(stream.tell(), 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            path.write_bytes(b"wave")
            self.assertEqual(speech_utils.speech_to_text(path), "Answer")
            self.assertTrue(self.client.speech_to_text.convert.call_args.kwargs["file"].closed)

    def test_empty_transcription_is_an_error(self):
        self.client.speech_to_text.convert.return_value.text = " "
        with self.assertRaisesRegex(RuntimeError, "No speech was recognized"):
            speech_utils.speech_to_text(b"wave")
        with self.assertRaisesRegex(ValueError, "Audio cannot be empty"):
            speech_utils.speech_to_text(b"")


class SpeechConfigurationTests(unittest.TestCase):
    def tearDown(self):
        speech_utils.get_client.cache_clear()

    def test_missing_key_fails_only_when_speech_is_requested(self):
        speech_utils.get_client.cache_clear()
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": " "}), patch("speech_utils.ElevenLabs") as constructor:
            with self.assertRaisesRegex(ValueError, "VivaSensei/.env"):
                speech_utils.get_client()
            constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
