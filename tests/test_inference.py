"""Check runtime selection and token limits without loading model weights."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import MAX_INPUT_TOKENS, QwenExaminer, encode_chat, select_runtime


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.torch = Mock()
        self.torch.cuda.is_available.return_value = True
        self.torch.cuda.is_bf16_supported.return_value = False
        self.torch.cuda.mem_get_info.return_value = (20 * 1024**3, 24 * 1024**3)
        self.psutil = Mock()
        self.psutil.virtual_memory.return_value.available = 20 * 1024**3
        self.modules = patch.dict(sys.modules, {"torch": self.torch, "psutil": self.psutil})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_auto_runtime_and_unsupported_bfloat16(self):
        self.assertEqual(select_runtime(), ("cuda", "4bit"))
        with self.assertRaisesRegex(ValueError, "does not support bfloat16"):
            select_runtime("cuda", "bf16")
        self.torch.cuda.is_available.return_value = False
        self.assertEqual(select_runtime(), ("cpu", "bf16"))
        with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
            select_runtime("cuda")
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            select_runtime("cpu", "4bit")

    def test_memory_checks_and_invalid_arguments(self):
        self.psutil.virtual_memory.return_value.available = 1024**3
        with self.assertRaisesRegex(ValueError, "available RAM"):
            select_runtime("cpu")
        self.assertEqual(select_runtime("cpu", check_memory=False), ("cpu", "bf16"))
        self.torch.cuda.is_bf16_supported.return_value = True
        self.torch.cuda.mem_get_info.return_value = (1024**3, 24 * 1024**3)
        with self.assertRaisesRegex(ValueError, "available VRAM"):
            select_runtime("cuda", "bf16")
        for args in [("bad-device", "auto"), ("cpu", "bad-precision")]:
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                select_runtime(*args)

    def test_four_bit_loading_falls_back_to_float16(self):
        transformers = Mock()
        with patch.dict(sys.modules, {"transformers": transformers}):
            QwenExaminer()
        options = transformers.BitsAndBytesConfig.call_args.kwargs
        self.assertEqual(options["bnb_4bit_compute_dtype"], self.torch.float16)
        self.assertEqual(transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["dtype"], self.torch.float16)

    def test_context_limit_preserves_messages(self):
        tokenizer = Mock()
        messages = [{"role": "user", "content": "An answer"}]
        inputs = {"input_ids": SimpleNamespace(shape=(1, MAX_INPUT_TOKENS))}
        tokenizer.apply_chat_template.return_value = inputs
        self.assertIs(encode_chat(tokenizer, messages), inputs)
        inputs["input_ids"].shape = (1, MAX_INPUT_TOKENS + 1)
        with self.assertRaisesRegex(ValueError, "history has been kept"):
            encode_chat(tokenizer, messages)
        self.assertEqual(messages, [{"role": "user", "content": "An answer"}])


if __name__ == "__main__":
    unittest.main()
