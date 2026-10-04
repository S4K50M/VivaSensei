"""Streamlit viva with conversation memory scoped to the current session."""

import streamlit as st

from inference import QwenExaminer
from viva import default_viva, system_message


@st.cache_resource
def get_examiner():
    # Cache only model resources. Never put a student's chat in this shared object.
    return QwenExaminer()


def main():
    st.set_page_config(page_title="VivaSensei", page_icon="🎓")
    st.title("VivaSensei")
    st.caption("Practice your viva one question at a time. Your examiner remembers this conversation.")
    st.caption("AI feedback can be inaccurate. Check technical explanations against your course material.")

    if "viva" not in st.session_state:
        st.session_state.viva = default_viva()
    if "messages" not in st.session_state:
        st.session_state.messages = [system_message(st.session_state.viva)]

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
        st.caption("Settings apply to this chat. Starting a new viva clears its conversation.")

    # Refresh just the system context when settings change; retain all chat turns.
    st.session_state.messages[0] = system_message(st.session_state.viva)
    viva = st.session_state.viva
    st.subheader(f"{viva['subject']} · {viva['current_topic']}")
    st.caption(f"Difficulty: {viva['difficulty']}")
    for message in st.session_state.messages[1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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
    if prompt:
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
            st.rerun()


if __name__ == "__main__":
    main()
