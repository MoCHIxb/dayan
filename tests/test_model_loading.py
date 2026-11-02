import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
    from personal_llm_data import model_loading as loading
except ImportError:
    loading = None


@unittest.skipIf(loading is None, "Training dependencies are optional")
class ModelLoadingTests(unittest.TestCase):
    def load_mocked(self, mode, **flags):
        model = SimpleNamespace(**flags)
        with patch.object(loading.AutoModelForCausalLM, "from_pretrained", return_value=model) as factory:
            with patch.object(loading.torch.cuda, "is_available", return_value=True):
                with patch.object(loading.torch.cuda, "current_device", return_value=0):
                    result = loading.load_causal_model("local-model", mode)
        self.assertIs(result, model)
        return factory.call_args.kwargs

    def test_bf16_preserves_original_loading_arguments(self):
        kwargs = self.load_mocked("none")
        self.assertEqual(kwargs, {"dtype": torch.bfloat16, "attn_implementation": "sdpa"})

    def test_int8_is_explicitly_single_gpu(self):
        kwargs = self.load_mocked("int8", is_loaded_in_8bit=True)
        self.assertEqual(kwargs["device_map"], {"": 0})
        self.assertTrue(kwargs["quantization_config"].load_in_8bit)
        self.assertFalse(kwargs["quantization_config"].load_in_4bit)

    def test_nf4_uses_bf16_compute_and_double_quantization(self):
        kwargs = self.load_mocked("nf4", is_loaded_in_4bit=True)
        config = kwargs["quantization_config"]
        self.assertTrue(config.load_in_4bit)
        self.assertEqual(config.bnb_4bit_quant_type, "nf4")
        self.assertTrue(config.bnb_4bit_use_double_quant)
        self.assertEqual(config.bnb_4bit_compute_dtype, torch.bfloat16)

    def test_quantization_cannot_silently_fall_back(self):
        with self.assertRaisesRegex(RuntimeError, "was not quantized"):
            self.load_mocked("int8", is_loaded_in_8bit=False)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported quantization"):
            loading.load_causal_model("local-model", "unknown")


if __name__ == "__main__":
    unittest.main()
