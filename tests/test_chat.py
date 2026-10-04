"""Exercise session lifecycle through Streamlit's UI, without loading the GPU."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


class ChatMemoryTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.received = []
        self.examiner = Mock()

        def respond(messages):
            self.received.append(deepcopy(messages))
            return "What is channel length modulation?" if len(messages) == 2 else "That needs correction. What happens to output resistance?"

        self.examiner.generate.side_effect = respond
        self.loader = patch("inference.QwenExaminer", return_value=self.examiner)
        self.constructor = self.loader.start()
        self.addCleanup(self.loader.stop)
        self.addCleanup(st.cache_resource.clear)
        self.app = AppTest.from_file(str(PROJECT_DIR / "app.py")).run()
        self.assertEqual(len(self.app.exception), 0)

    def test_history_survives_reruns_and_settings_changes(self):
        self.app.button(key="begin_viva").click().run()
        self.app.run()
        answer = "The effective channel gets longer as VDS increases."
        self.app.chat_input[0].set_value(answer).run()
        self.assertEqual(len(self.app.exception), 0)
        supplied = self.received[-1]
        self.assertEqual(supplied[-2]["content"], "What is channel length modulation?")
        self.assertEqual(supplied[-1]["content"], answer)
        self.assertIn("Output resistance", supplied[0]["content"])
        self.assertEqual([m["role"] for m in self.app.session_state.messages], ["system", "user", "assistant", "user", "assistant"])
        self.assertEqual(self.constructor.call_count, 1)

        history = deepcopy(self.app.session_state.messages[1:])
        self.app.sidebar.text_input[1].set_value("Cascode Mirrors")
        self.app.sidebar.text_area[0].set_value("Voltage headroom")
        self.app.button(key="apply_settings").click().run()
        self.assertEqual(self.app.session_state.messages[1:], history)
        self.app.chat_input[0].set_value("As I said earlier, the channel gets longer.").run()
        self.assertIn("Cascode Mirrors", self.received[-1][0]["content"])
        self.assertIn("Voltage headroom", self.received[-1][0]["content"])
        self.assertEqual(self.received[-1][1:-1], history)

    def test_new_viva_and_other_sessions_do_not_share_history(self):
        self.app.button(key="begin_viva").click().run()
        other = AppTest.from_file(str(PROJECT_DIR / "app.py")).run()
        self.assertEqual(len(other.session_state.messages), 1)
        other.session_state.viva["student_weaknesses"].append("Another student's weakness")
        self.assertNotIn("Another student's weakness", self.app.session_state.viva["student_weaknesses"])
        self.app.button(key="new_viva").click().run()
        self.assertEqual(len(self.app.session_state.messages), 1)
        self.assertEqual(len(self.app.chat_message), 0)

    def test_failed_generation_can_retry_without_duplicate_user_turns(self):
        self.examiner.generate.side_effect = RuntimeError("Simulated GPU error")
        self.app.chat_input[0].set_value("Start my viva.").run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.session_state.messages), 2)
        self.assertTrue(self.app.chat_input[0].disabled)
        self.assertIn("Simulated GPU error", self.app.error[0].value)
        self.app.run()
        self.assertEqual(self.examiner.generate.call_count, 1)
        self.examiner.generate.side_effect = None
        self.examiner.generate.return_value = "What is output resistance?"
        self.app.button(key="retry_response").click().run()
        self.assertEqual([m["role"] for m in self.app.session_state.messages], ["system", "user", "assistant"])
        self.assertFalse(self.app.chat_input[0].disabled)

    def test_failed_answer_can_be_edited_without_duplicating_history(self):
        self.examiner.generate.side_effect = ValueError("Input too long")
        self.app.chat_input[0].set_value("A very long answer").run()
        self.app.text_area(key="pending_answer").set_value("Short answer")
        self.examiner.generate.side_effect = None
        self.examiner.generate.return_value = "Next question?"
        self.app.button(key="edit_response").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(self.app.session_state.messages[1]["content"], "Short answer")
        self.assertEqual(len(self.app.session_state.messages), 3)
        self.assertFalse(self.app.chat_input[0].disabled)
        self.examiner.generate.side_effect = RuntimeError("Another failure")
        self.app.chat_input[0].set_value("A different answer").run()
        self.assertEqual(self.app.text_area(key="pending_answer").value, "A different answer")

    def test_empty_response_retains_answer_for_retry(self):
        self.examiner.generate.side_effect = None
        self.examiner.generate.return_value = "  "
        self.app.button(key="begin_viva").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.session_state.messages), 2)
        self.assertIn("empty response", self.app.error[0].value)

    def test_invalid_settings_preserve_chat_and_current_viva(self):
        self.app.button(key="begin_viva").click().run()
        history = deepcopy(self.app.session_state.messages)
        viva = deepcopy(self.app.session_state.viva)
        self.app.sidebar.text_input[0].set_value(" ")
        self.app.button(key="new_viva").click().run()
        self.assertEqual(self.app.session_state.messages, history)
        self.assertEqual(self.app.session_state.viva, viva)
        self.assertIn("Enter a subject", self.app.error[0].value)

    def test_discard_failed_message_restores_chat_input(self):
        self.examiner.generate.side_effect = RuntimeError("Failure")
        self.app.chat_input[0].set_value("An answer").run()
        self.app.button(key="discard_pending").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.session_state.messages), 1)
        self.assertFalse(self.app.chat_input[0].disabled)
        self.assertEqual(len(self.app.error), 0)


if __name__ == "__main__":
    unittest.main()
