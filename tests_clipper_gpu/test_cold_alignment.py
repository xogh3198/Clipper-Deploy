"""No GPU required: verify timing boundaries and keep original routing untouched."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, Mock, patch

from clipper_gpu import deploy, doctor, download
from clipper_gpu.cache import MODEL_FILES
from clipper_gpu.config import compose_spec, load
from clipper_gpu.worker import GPUModel, Recorder


class ColdAlignmentTests(unittest.TestCase):
    def make_modules(self):
        tensor = MagicMock()
        tensor.__matmul__.return_value.__getitem__.return_value.item.return_value = 2
        torch = types.SimpleNamespace(
            __version__="test", version=types.SimpleNamespace(cuda="test"),
            cuda=types.SimpleNamespace(is_available=Mock(return_value=True),
                                       synchronize=Mock(), get_device_name=Mock(return_value="V100")),
            set_num_threads=Mock(), set_num_interop_threads=Mock(), ones=Mock(return_value=tensor),
            inference_mode=contextlib.nullcontext,
            backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=True, allow_tf32=True),
                cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True))))
        processor = Mock(return_value={"pixel_values": MagicMock()})
        model = MagicMock()
        model.float.return_value = model
        model.eval.return_value = model
        model.to.return_value = model
        model.return_value.logits.argmax.return_value.cpu.return_value.tolist.return_value = [0]
        auto_processor = Mock(from_pretrained=Mock(return_value=processor))
        auto_model = Mock(from_pretrained=Mock(return_value=model))
        pil = types.SimpleNamespace(Image=types.SimpleNamespace(new=Mock()))
        modules = {"torch": torch, "PIL": pil, "transformers": types.SimpleNamespace(
            AutoImageProcessor=auto_processor, AutoModelForImageClassification=auto_model)}
        return modules, torch, auto_processor, auto_model, model

    def test_runtime_initializes_before_readiness_without_loading_model(self):
        modules, torch, processor, auto_model, _ = self.make_modules()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules), \
             patch.dict(os.environ, {"MODEL_ID": "microsoft/resnet-50", "MODEL_REVISION": "a" * 40}), \
             contextlib.redirect_stdout(io.StringIO()):
            recorder = Recorder(directory, "run")
            model = GPUModel(recorder)
            snapshot = recorder.snapshot()
            self.assertTrue(snapshot["runtime_initialized"])
            self.assertFalse(snapshot["model_loaded"])
            self.assertIsNone(model.model)
            self.assertEqual(snapshot["warmup_queries"], 10)
            self.assertGreaterEqual(snapshot["startup_ms"], 0)
            processor.from_pretrained.assert_not_called()
            auto_model.from_pretrained.assert_not_called()
            torch.ones.assert_called_once_with((2, 2), device="cuda:0")
            self.assertFalse(torch.backends.cudnn.benchmark)
            event = json.loads(recorder.path.read_text().splitlines()[0])
            self.assertEqual(event["event"], "runtime_ready")
            self.assertEqual(snapshot["forward_calls"], 0)

    def test_first_load_measures_read_transfer_and_warmup_then_reuses_model(self):
        modules, torch, processor, auto_model, gpu_model = self.make_modules()
        env = {"MODEL_ID": "microsoft/resnet-50", "MODEL_REVISION": "a" * 40, "WARMUP_QUERIES": "10"}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules), \
             patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
            recorder = Recorder(directory, "run")
            runtime = GPUModel(recorder)
            elapsed = runtime.ensure_loaded()
            self.assertEqual(gpu_model.call_count, 10)
            self.assertEqual(set(runtime.load_stages), {"checkpoint_read_ms", "model_h2d_ms", "warmup_ms"})
            self.assertAlmostEqual(sum(runtime.load_stages.values()), elapsed)
            self.assertTrue(recorder.snapshot()["model_loaded"])
            self.assertEqual(recorder.snapshot()["forward_calls"], 0)
            self.assertEqual(recorder.snapshot()["warmup_forward_calls"], 10)
            self.assertEqual(runtime.ensure_loaded(), 0)
            auto_model.from_pretrained.assert_called_once_with(
                "microsoft/resnet-50", revision="a" * 40, local_files_only=True, use_safetensors=True)
            self.assertEqual(gpu_model.call_count, 10)

    def test_warmups_can_be_disabled_explicitly(self):
        modules, _, _, _, gpu_model = self.make_modules()
        env = {"MODEL_ID": "microsoft/resnet-50", "MODEL_REVISION": "a" * 40, "WARMUP_QUERIES": "0"}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules), \
             patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
            runtime = GPUModel(Recorder(directory, "run"))
            runtime.ensure_loaded()
            gpu_model.assert_not_called()

    def test_no_cpu_fallback_when_cuda_unavailable(self):
        modules, torch, _, auto_model, _ = self.make_modules()
        torch.cuda.is_available.return_value = False
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                GPUModel(Recorder(directory, "run"))
        auto_model.from_pretrained.assert_not_called()

    def test_warmup_setting_reaches_worker(self):
        with patch.dict(os.environ, {"WARMUP_QUERIES": "3"}):
            self.assertEqual(compose_spec(load())["services"]["worker"]["environment"]["WARMUP_QUERIES"], "3")
        with patch.dict(os.environ, {"WARMUP_QUERIES": "-1"}):
            with self.assertRaisesRegex(ValueError, "WARMUP_QUERIES"):
                load()


class OfflineCacheTests(unittest.TestCase):
    def test_download_and_doctor_use_identical_inference_file_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"fixture")
            # README/Flax/TensorFlow files deliberately do not exist.
            snapshots = Mock(return_value=directory)
            tensor = MagicMock()
            tensor.__mul__.return_value.item.return_value = 2
            torch = types.SimpleNamespace(set_num_threads=Mock(), ones=Mock(return_value=tensor),
                version=types.SimpleNamespace(cuda="test"), cuda=types.SimpleNamespace(
                    get_device_name=lambda _: "V100", get_arch_list=lambda: ["sm_70"]))
            modules = {"torch": torch, "huggingface_hub": types.SimpleNamespace(snapshot_download=snapshots)}
            with patch.dict(sys.modules, modules), patch.dict(os.environ, {
                "MODEL_ID": "microsoft/resnet-50", "MODEL_REVISION": "a" * 40}), \
                 patch("clipper_gpu.rpc_bridge.load_rpc", return_value=types.SimpleNamespace(RPC_VERSION=3)), \
                 patch.object(doctor.importlib.metadata, "version", return_value="test"), \
                 contextlib.redirect_stdout(io.StringIO()):
                download.main()
                doctor.main()
            self.assertEqual(snapshots.call_count, 2)
            online, offline = [call.kwargs for call in snapshots.call_args_list]
            self.assertEqual(online["allow_patterns"], MODEL_FILES)
            self.assertEqual(offline["allow_patterns"], MODEL_FILES)
            self.assertTrue(offline["local_files_only"])


class FirstColdRunTests(unittest.TestCase):
    def test_first_cold_test_does_not_require_old_compose_file(self):
        with tempfile.TemporaryDirectory() as directory:
            absent = Path(directory) / "compose.json"
            with patch.object(sys, "argv", ["clipper", "cold-test", "-n", "11"]), \
                 patch.object(deploy, "paths", return_value=(absent, None, None)), \
                 patch.object(deploy, "compose") as compose, patch.object(deploy, "up") as up, \
                 patch.object(deploy.client, "measure", return_value={"warm_requests": 10,
                     "warm_real_predictions": 10, "warm_load_ms": [0] * 10}) as measure:
                self.assertEqual(deploy.main(), 0)
            compose.assert_not_called()
            up.assert_called_once()
            self.assertEqual(measure.call_args.kwargs["cold"], True)

    def test_existing_cold_test_still_stops_only_its_project(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "compose.json"
            existing.write_text("{}")
            with patch.object(sys, "argv", ["clipper", "cold-test", "-n", "11"]), \
                 patch.object(deploy, "paths", return_value=(existing, None, None)), \
                 patch.object(deploy, "compose") as compose, patch.object(deploy, "up"), \
                 patch.object(deploy.client, "measure", return_value={"warm_requests": 10,
                     "warm_real_predictions": 10, "warm_load_ms": [0] * 10}):
                self.assertEqual(deploy.main(), 0)
            self.assertEqual(compose.call_args.args[1:], ("down",))


if __name__ == "__main__":
    unittest.main()
