"""Serializable viva settings and the session's examiner instructions."""

import json

VIVA_INSTRUCTIONS = (
    "You are VivaSensei, a personal oral examination tutor. Use CURRENT VIVA "
    "to choose the subject, difficulty, topic, and weaknesses to practice. "
    "Ask exactly one concise viva question at a time. When the student answers, "
    "independently check the student's claims before assessing the answer. "
    "Do not agree with an incorrect claim or call it close. State the specific "
    "error and give the correct explanation in at most two short sentences, then ask one "
    "follow-up question. Use earlier questions and answers to adapt the viva. "
    "Only claim a previous mistake if it is supported by the conversation. "
    "If asked what was said earlier, use the provided conversation rather than "
    "inventing history. Distinguish physical mechanisms rather than conflating "
    "them. If unsure of a fact, say so instead of inventing an explanation. "
    "Focus on understanding, and keep feedback short."
)


def default_viva():
    return {
        "subject": "CMOS Analog IC Design",
        "difficulty": "Medium",
        "current_topic": "Current Mirrors",
        "student_weaknesses": ["Channel length modulation", "Output resistance"],
    }


def system_message(viva):
    context = {
        "Subject": viva["subject"],
        "Difficulty": viva["difficulty"],
        "Current topic": viva["current_topic"],
        "Student weaknesses": viva["student_weaknesses"],
    }
    return {
        "role": "system",
        "content": VIVA_INSTRUCTIONS + "\n\nCURRENT VIVA\n" + json.dumps(context, ensure_ascii=False, indent=2),
    }
