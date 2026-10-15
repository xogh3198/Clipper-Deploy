"""Preinitialized runtime, model-cold FP32 inference through original Clipper RPC."""
import base64
from collections import deque
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import threading
import time

from .rpc_bridge import load_rpc


class Recorder:
    def __init__(self, result_dir, run_id):
        self.run_id = run_id
        self.lock = threading.Lock()
        self.state = {"run_id": run_id, "model_loaded": False, "rpc_started": False,
                      "batches": 0, "forward_calls": 0, "executed_inputs": 0, "errors": 0,
                      "max_observed_batch": 0, "warmup_forward_calls": 0}
        self.recent = deque(maxlen=5000)
        self.path = Path(result_dir) / ("worker-" + run_id + ".jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def append(self, event):
        event = dict(event, run_id=self.run_id, unix_time=time.time())
        with self.lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                stream.flush()
            if event["event"] == "batch_complete":
                self.state["batches"] += 1
                self.state["forward_calls"] += 1
                self.state["executed_inputs"] += event["batch_size"]
                self.state["max_observed_batch"] = max(self.state["max_observed_batch"], event["batch_size"])
                self.recent.append(event)
            elif event["event"] == "batch_error":
                self.state["errors"] += 1
        print(json.dumps(event, ensure_ascii=False), flush=True)

    def snapshot(self):
        with self.lock:
            return dict(self.state, recent_batches=list(self.recent))


class GPUModel:
    def __init__(self, recorder):
        startup = time.perf_counter()
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        imports_done = time.perf_counter()
        self.torch = torch
        self.Image = Image
        self.AutoProcessor = AutoImageProcessor
        self.AutoModel = AutoModelForImageClassification
        self.recorder = recorder
        self.model = None
        self.processor = None
        self.load_stages = {}
        self.warmups = int(os.environ.get("WARMUP_QUERIES", "10"))
        if self.warmups < 0:
            raise ValueError("WARMUP_QUERIES must be nonnegative")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU inference fallback is forbidden")
        torch.set_num_threads(int(os.environ.get("CPU_THREADS", "2")))
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Pay framework/CUDA initialization before advertising RPC readiness.
        # This does NOT load a model or run a model prediction.
        tensor = torch.ones((2, 2), device="cuda:0")
        if (tensor @ tensor)[0, 0].item() != 2:
            raise RuntimeError("CUDA math check failed")
        torch.cuda.synchronize()
        del tensor
        startup_ms = (time.perf_counter() - startup) * 1000
        startup_stages = {"framework_import_ms": (imports_done - startup) * 1000,
                          "cuda_setup_ms": startup_ms - (imports_done - startup) * 1000}
        recorder.update(precision="fp32", preprocessing="PIL + Hugging Face CPU processor",
                        model_id=os.environ["MODEL_ID"], model_revision=os.environ["MODEL_REVISION"],
                        torch=torch.__version__, cuda=torch.version.cuda,
                        runtime_initialized=True, startup_ms=startup_ms,
                        startup_stages=startup_stages, warmup_queries=self.warmups,
                        cold_definition="model non-resident; framework/CUDA initialized; caches retained")
        recorder.append({"event": "runtime_ready", "startup_ms": startup_ms,
                         "startup_stages": startup_stages, "model_loaded": False})

    def ensure_loaded(self):
        if self.model is not None:
            return 0.0
        start = time.perf_counter()
        args = {"revision": os.environ["MODEL_REVISION"], "local_files_only": True}
        processor = self.AutoProcessor.from_pretrained(os.environ["MODEL_ID"], use_fast=False, **args)
        model = self.AutoModel.from_pretrained(
            os.environ["MODEL_ID"], use_safetensors=True, **args).float().eval()
        read_end = time.perf_counter()
        model = model.to("cuda:0")
        self.torch.cuda.synchronize()
        transfer_end = time.perf_counter()
        # Same default warmup count as the Proteus GPU comparison runtime.
        # Keep it inside load_ms rather than hiding it before the first request.
        if self.warmups:
            probe = self.Image.new("RGB", (224, 224), (128, 128, 128))
            tensors = processor(images=[probe], return_tensors="pt")
            with self.torch.inference_mode():
                for _ in range(self.warmups):
                    gpu_inputs = {key: value.to("cuda:0") for key, value in tensors.items()}
                    model(**gpu_inputs).logits.argmax(dim=-1).cpu().tolist()
                    self.torch.cuda.synchronize()
        self.torch.cuda.synchronize()
        ready = time.perf_counter()
        self.processor, self.model = processor, model
        self.load_stages = {"checkpoint_read_ms": (read_end - start) * 1000,
                            "model_h2d_ms": (transfer_end - read_end) * 1000,
                            "warmup_ms": (ready - transfer_end) * 1000}
        elapsed = (ready - start) * 1000
        self.recorder.update(model_loaded=True, gpu=self.torch.cuda.get_device_name(0),
                             initial_load_ms=elapsed, load_stages=dict(self.load_stages),
                             warmup_forward_calls=self.warmups)
        return elapsed

    def predict_strings(self, inputs):
        start = time.perf_counter()
        from PIL import Image
        ids = []
        hashes = []
        try:
            load_ms = self.ensure_loaded()
            preprocess_start = time.perf_counter()
            images = []
            for item in inputs:
                payload = json.loads(item)
                ids.append(payload["request_id"])
                raw = base64.b64decode(payload["image_b64"], validate=True)
                hashes.append(hashlib.sha256(raw).hexdigest())
                with Image.open(io.BytesIO(raw)) as image:
                    images.append(image.convert("RGB"))
            tensors = self.processor(images=images, return_tensors="pt")
            preprocess_ms = (time.perf_counter() - preprocess_start) * 1000
            transfer_start = time.perf_counter()
            tensors = {key: value.to("cuda:0") for key, value in tensors.items()}
            self.torch.cuda.synchronize()
            h2d_ms = (time.perf_counter() - transfer_start) * 1000
            forward_start = time.perf_counter()
            with self.torch.inference_mode():
                logits = self.model(**tensors).logits
            self.torch.cuda.synchronize()
            forward_ms = (time.perf_counter() - forward_start) * 1000
            post_start = time.perf_counter()
            indices = logits.argmax(dim=-1).cpu().tolist()
            outputs = [{"request_id": request_id, "label": self.model.config.id2label[index],
                        "class_id": index, "model": os.environ["CLIPPER_MODEL_NAME"],
                        "precision": "fp32", "batch_size": len(inputs), "load_ms": load_ms,
                        "preprocess_ms": preprocess_ms, "h2d_ms": h2d_ms, "forward_ms": forward_ms}
                       for request_id, index in zip(ids, indices)]
            encoded = [json.dumps(output, ensure_ascii=False) for output in outputs]
            event = {"event": "batch_complete", "request_ids": ids, "image_sha256": hashes,
                     "batch_size": len(inputs), "load_ms": load_ms, "preprocess_ms": preprocess_ms,
                     "load_stages": dict(self.load_stages) if load_ms else {},
                     "h2d_ms": h2d_ms, "forward_ms": forward_ms,
                     "postprocess_ms": (time.perf_counter() - post_start) * 1000,
                     "worker_total_ms": (time.perf_counter() - start) * 1000}
            self.recorder.append(event)
            return encoded
        except Exception as exc:
            self.recorder.append({"event": "batch_error", "request_ids": ids,
                                  "batch_size": len(inputs), "error": repr(exc)})
            raise


def start_stats(recorder):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/health", "/state"):
                self.send_error(404)
                return
            body = json.dumps(recorder.snapshot()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("0.0.0.0", 9090), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    recorder = Recorder("/results", os.environ["RUN_ID"])
    rpc = load_rpc()
    model = GPUModel(recorder)
    start_stats(recorder)
    recorder.update(rpc_started=True)
    # Framework/CUDA are initialized, but model weights remain non-resident.
    # This is an added model-cold diagnostic, not the original container's eager load.
    rpc.RPCService(collect_metrics=False).start(model)


if __name__ == "__main__":
    main()
