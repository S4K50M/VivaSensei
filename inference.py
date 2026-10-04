"""Ask one viva question using Qwen2.5-7B-Instruct."""

import argparse
import os
from pathlib import Path
import sys
from threading import Lock

# Keep downloaded model assets inside the project, including when invoked elsewhere.
PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".cache" / "huggingface"))
# Avoid preloading full-precision tensors concurrently before 4-bit conversion.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_PROMPT = "Ask me one viva question about CMOS current mirrors."
SYSTEM_PROMPT = (
    "You are VivaSensei, an oral examination tutor. Follow the student's topic "
    "request and ask exactly one concise viva question. Output only the question, "
    "without a greeting, answer, explanation, or list."
)
MAX_INPUT_TOKENS = 2048


def select_runtime(device="auto", precision="auto", check_memory=True):
    """Choose the same runtime for the CLI and Streamlit app."""
    import psutil
    import torch

    if device not in ("auto", "cpu", "cuda"):
        raise ValueError(f"Unsupported device: {device}")
    if precision not in ("auto", "bf16", "4bit"):
        raise ValueError(f"Unsupported precision: {precision}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Check the NVIDIA driver and PyTorch installation.")
    if precision == "auto":
        precision = "4bit" if device == "cuda" else "bf16"
    if precision == "4bit" and device != "cuda":
        raise ValueError("This script's 4-bit mode requires CUDA.")
    if device == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bfloat16. Use --precision 4bit.")
    if check_memory:
        if device == "cpu" and psutil.virtual_memory().available < 16 * 1024**3:
            raise ValueError("Full CPU inference needs about 16 GiB of available RAM. Run with GPU access; --check validates setup without loading weights.")
        if device == "cuda" and precision == "bf16" and torch.cuda.mem_get_info()[0] < 16 * 1024**3:
            raise ValueError("Bfloat16 inference needs about 16 GiB of available VRAM. Use --precision 4bit on this GPU.")
    return device, precision


def encode_chat(tokenizer, messages):
    """Encode the complete conversation without dropping previous turns."""
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if inputs["input_ids"].shape[-1] > MAX_INPUT_TOKENS:
        raise ValueError(
            f"This viva exceeds the {MAX_INPUT_TOKENS}-token input limit for local inference. "
            "Start a new viva or shorten the current answer. Your history has been kept."
        )
    return inputs


class QwenExaminer:
    """Reusable local model; conversation state belongs to the caller."""

    def __init__(self, device="auto", precision="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        device, precision = select_runtime(device, precision)
        compute_dtype = (
            torch.float16
            if device == "cuda" and not torch.cuda.is_bf16_supported()
            else torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        quantization_config = None
        if precision == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        print(f"Loading {MODEL_ID} on {device} ({precision})...", file=sys.stderr)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=compute_dtype,
            device_map={"": device},
            quantization_config=quantization_config,
        ).eval()
        # Streamlit shares the cached model across sessions, so serialize GPU work.
        self._lock = Lock()

    def generate(self, messages, max_new_tokens=192):
        import torch

        with self._lock:
            inputs = encode_chat(self.tokenizer, messages).to(self.model.device)
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(
                output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            ).strip()
            if not response:
                raise RuntimeError("The model returned an empty response.")
            return response


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--check", action="store_true", help="Validate the tokenizer and chat template without loading model weights.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("auto", "bf16", "4bit"), default="auto", help="Auto uses 4-bit on CUDA and bfloat16 on CPU.")
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("prompt must not be empty")

    import torch
    from transformers import AutoTokenizer

    try:
        device, precision = select_runtime(args.device, args.precision, check_memory=not args.check)
    except ValueError as error:
        parser.error(str(error))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.prompt},
    ]
    if args.check:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        inputs = encode_chat(tokenizer, messages)
        print(f"Model: {MODEL_ID}")
        print(f"Device: {device}; precision: {precision}; CUDA available: {torch.cuda.is_available()}")
        print(f"Chat template validated: {inputs['input_ids'].shape[-1]} input tokens")
        print("Model weights were not loaded; generation has not been verified.")
        return 0

    print(QwenExaminer(device, precision).generate(messages, max_new_tokens=96))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
