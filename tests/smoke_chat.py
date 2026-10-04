"""Check real GPU generation and history formatting, not factual accuracy."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import QwenExaminer, encode_chat
from viva import default_viva, system_message


def main():
    examiner = QwenExaminer()
    messages = [system_message(default_viva())]
    prompts = [
        "Ask me one viva question about channel length modulation.",
        "The channel becomes longer as drain-source voltage increases, so current decreases.",
        "What did I get wrong in my earlier explanation?",
    ]
    for prompt in prompts:
        messages.append({"role": "user", "content": prompt})
        # Check that the actual model input retains every earlier turn.
        inputs = encode_chat(examiner.tokenizer, messages)
        rendered = examiner.tokenizer.decode(inputs["input_ids"][0])
        assert all(message["content"] in rendered for message in messages)
        response = examiner.generate(messages)
        print(f"Student: {prompt}\nExaminer: {response}\n", flush=True)
        messages.append({"role": "assistant", "content": response})
    assert len(messages) == 7
    print("Generation and conversation formatting passed. Review factual accuracy separately.")


if __name__ == "__main__":
    main()
