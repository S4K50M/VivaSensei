"""Streamlit viva with conversation memory scoped to the current session."""

from hashlib import sha256

import streamlit as st

from inference import QwenExaminer
from speech_utils import speech_to_text, text_to_speech
from viva import default_viva, system_message


@st.cache_resource
def get_examiner():
    # Cache only model resources. Never put a student's chat in this shared object.
    return QwenExaminer()


def clear_voice_answer():
    st.session_state.pop("voice_transcript", None)
    st.session_state.pop("voice_recording_hash", None)
    st.session_state.voice_recording_id = st.session_state.get("voice_recording_id", 0) + 1


def render_reply_audio(index, text):
    audio = st.session_state.setdefault("examiner_audio", {})
    automatic = st.session_state.get("speak_reply") == index
    requested = st.button("Listen" if index not in audio else "Regenerate audio", key=f"listen_{index}")
    if automatic or requested:
        st.session_state.pop("speak_reply", None)
        try:
            with st.spinner("Creating examiner audio…"):
                audio[index] = text_to_speech(text)
        except Exception as error:
            st.warning(str(error) if isinstance(error, ValueError) else
                       "Couldn't create speech. Check your ElevenLabs connection, key, voice access, and credits.")
    if index in audio:
        st.audio(audio[index], format="audio/mpeg", autoplay=automatic or requested)


def render_voice_answer(disabled):
    """Transcribe only on request and let the student review before sending."""
    with st.expander("Answer by voice"):
        st.caption("Record an answer, transcribe it with ElevenLabs, then review and send the text.")
        recording = st.audio_input(
            "Record your answer", sample_rate=16000, disabled=disabled,
            key=f"voice_recording_{st.session_state.get('voice_recording_id', 0)}",
        )
        if recording is not None:
            data = recording.getvalue()
            recording_hash = sha256(data).hexdigest()
            if recording_hash != st.session_state.get("voice_recording_hash"):
                st.session_state.voice_recording_hash = recording_hash
                st.session_state.pop("voice_transcript", None)
        if st.button("Transcribe recording", key="transcribe_recording", disabled=disabled or recording is None):
            try:
                with st.spinner("Transcribing your answer…"):
                    st.session_state.voice_transcript = speech_to_text(data)
            except Exception as error:
                st.error(str(error) if isinstance(error, (ValueError, RuntimeError)) else
                         "Couldn't transcribe this recording. Check your ElevenLabs connection, key, and credits, then retry.")
        if "voice_transcript" in st.session_state:
            with st.form("voice_answer"):
                transcript = st.text_area("Review your transcript", key="voice_transcript", disabled=disabled)
                send = st.form_submit_button("Send voice answer", key="send_voice_answer", disabled=disabled)
            if send:
                if transcript.strip():
                    return transcript.strip()
                st.error("Enter an answer before sending.")
    return None


def main():
    st.set_page_config(page_title="VivaSensei", page_icon="🎓")
    st.title("VivaSensei")
    st.caption("Practice your viva one question at a time. Your examiner remembers this conversation.")
    st.caption("AI feedback can be inaccurate. Check technical explanations against your course material.")

    if "viva" not in st.session_state:
        st.session_state.viva = default_viva()
    if "messages" not in st.session_state:
        st.session_state.messages = [system_message(st.session_state.viva)]
    if st.session_state.pop("clear_voice_on_rerun", False):
        clear_voice_answer()

    with st.sidebar:
        st.header("Current viva")
        with st.form("viva_settings"):
            current = st.session_state.viva
            subject = st.text_input("Subject", value=current["subject"])
            difficulties = ["Easy", "Medium", "Hard"]
            difficulty = st.selectbox("Difficulty", difficulties, index=difficulties.index(current["difficulty"]))
            topic = st.text_input("Current topic", value=current["current_topic"])
            weaknesses = st.text_area("Student weaknesses (one per line)", value="\n".join(current["student_weaknesses"]))
            apply = st.form_submit_button("Apply settings", key="apply_settings")
            reset = st.form_submit_button("Start new viva", key="new_viva")
        if apply or reset:
            if not subject.strip() or not topic.strip():
                st.error("Enter a subject and current topic.")
            else:
                st.session_state.viva = {
                    "subject": subject.strip(),
                    "difficulty": difficulty,
                    "current_topic": topic.strip(),
                    "student_weaknesses": [line.strip() for line in weaknesses.splitlines() if line.strip()],
                }
                if reset:
                    st.session_state.messages = [system_message(st.session_state.viva)]
                    st.session_state.pop("response_error", None)
                    st.session_state.pop("pending_answer", None)
                    st.session_state.pop("examiner_audio", None)
                    st.session_state.pop("speak_reply", None)
                    clear_voice_answer()
        st.caption("Settings apply to this chat. Starting a new viva clears its conversation.")
        st.toggle("Read new replies aloud", value=False, key="read_aloud")

    # Refresh just the system context when settings change; retain all chat turns.
    st.session_state.messages[0] = system_message(st.session_state.viva)
    viva = st.session_state.viva
    st.subheader(f"{viva['subject']} · {viva['current_topic']}")
    st.caption(f"Difficulty: {viva['difficulty']}")
    for index, message in enumerate(st.session_state.messages[1:], start=1):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_reply_audio(index, message["content"])

    prompt = None
    if len(st.session_state.messages) == 1 and st.button("Ask first question", key="begin_viva"):
        prompt = "Start the viva with one question about the current topic."

    pending = st.session_state.messages[-1]["role"] == "user"
    if not pending:
        st.session_state.pop("pending_answer", None)
    if pending and "response_error" in st.session_state:
        st.error(st.session_state.response_error)
    retry = pending and st.button("Retry response", key="retry_response")
    if pending:
        with st.form("edit_pending"):
            edited_answer = st.text_area(
                "Edit unanswered message", value=st.session_state.messages[-1]["content"],
                key="pending_answer",
            )
            resubmit = st.form_submit_button("Submit edited message", key="edit_response")
        if resubmit:
            if edited_answer.strip():
                st.session_state.messages[-1]["content"] = edited_answer.strip()
                st.session_state.pop("response_error", None)
                retry = True
            else:
                st.error("Enter a message before submitting.")
    if pending and st.button("Discard unanswered message", key="discard_pending"):
        st.session_state.messages.pop()
        st.session_state.pop("response_error", None)
        st.session_state.pop("pending_answer", None)
        st.rerun()
    answer = st.chat_input("Your answer or a question for the examiner", disabled=pending)
    if answer and answer.strip():
        prompt = answer.strip()
    voice_answer = render_voice_answer(disabled=pending or bool(prompt))
    if voice_answer:
        prompt = voice_answer
    if prompt:
        st.session_state.clear_voice_on_rerun = True
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    if prompt or retry:
        st.session_state.pop("response_error", None)
        try:
            with st.spinner("Your examiner is thinking…"):
                response = get_examiner().generate(st.session_state.messages)
                if not isinstance(response, str) or not response.strip():
                    raise RuntimeError("The model returned an empty response.")
        except Exception as error:
            st.session_state.response_error = f"Couldn't generate a response: {error}"
            # Retain the unanswered user turn and offer retry without duplicating it.
            st.rerun()
        else:
            st.session_state.messages.append({"role": "assistant", "content": response.strip()})
            if st.session_state.read_aloud:
                st.session_state.speak_reply = len(st.session_state.messages) - 1
            st.rerun()


if __name__ == "__main__":
    main()
