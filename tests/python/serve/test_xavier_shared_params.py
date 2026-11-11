"""CPU mocks verify short VM buckets alias only identical parameter schemas."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from bench_xavier_mtp import Program

class SharedParamsTest(unittest.TestCase):
    def build(self, metadata, shared=None):
        vm = {"vm_initialization": lambda *args: None, "_metadata": lambda: json.dumps(metadata)}
        with patch("bench_xavier_mtp.tvm_ffi.load_module", return_value={"vm_load_executable": lambda: vm}), patch("bench_xavier_mtp.tvm_ffi.get_global_func") as get:
            get.return_value.return_value = object()
            result = Program("unused", "mock", share_params_from=shared)
            return result, get

    def test_matching_buckets_share_without_loading_again(self):
        params = [{"name": "weight", "shape": [2, 3], "dtype": "float16"}]
        main, loaded = self.build({"context_window_size": 2048, "params": params})
        self.assertTrue(loaded.called)
        short, loaded = self.build({"context_window_size": 256, "params": params}, main)
        self.assertIs(short.params, main.params)
        self.assertEqual(short.context_length, 256)
        loaded.assert_not_called()

    def test_mismatched_parameters_rejected(self):
        main, _ = self.build({"context_window_size": 2048, "params": [{"name": "a", "shape": [2], "dtype": "float16"}]})
        for params in ([{"name": "b", "shape": [2], "dtype": "float16"}], [{"name": "a", "shape": [3], "dtype": "float16"}], [{"name": "a", "shape": [2], "dtype": "float32"}]):
            with self.subTest(params=params), self.assertRaisesRegex(ValueError, "metadata"):
                self.build({"context_window_size": 256, "params": params}, main)

if __name__ == "__main__":
    unittest.main()
