"""Attach the common FP32 backend to Clipper's original RPC and C++ batching."""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time

from .preparation import Preparation


class GatewayProcessor:
    def __init__(self):
        self.preparation = Preparation(Path("/inputs/image.jpg").read_bytes(),
            os.environ["EXPERIMENT_PHASE"], int(os.environ["PREPROCESS_WORKERS"]))

    def prepare(self, payload, received, deadline):
        raw = base64.b64decode(payload["image_b64"], validate=True)
        if time.time_ns() // 1000 >= deadline:
            raise TimeoutError("SLO expired in CPU queue")
        started = time.perf_counter()
        item = self.preparation.prepare(raw)
        elapsed = (time.perf_counter()-started)*1000
        tensor = item["inputs"]["pixel_values"]
        return {"pipeline": "cpu-ready-v1", "contract": item["contract"],
                "request_id": payload["request_id"], "image_size": list(item["image_size"]),
                "shape": list(tensor.shape), "tensor_b64": base64.b64encode(tensor.astype("<f4", copy=False).tobytes()).decode(),
                "received_unix_us": received, "deadline_unix_us": deadline,
                "cpu_queue_ms": max(0, (time.time_ns()/1000-received)/1000-elapsed),
                "decode_ms": item["timings"]["decode"], "preprocess_ms": item["timings"]["preprocess"]}


def gateway():
    import torch
    torch.set_num_threads(int(os.environ["CPU_THREADS"]))
    torch.set_num_interop_threads(int(os.environ["CPU_INTEROP_THREADS"]))
    from clipper_gpu.gateway import create_server
    processor = GatewayProcessor()
    server = create_server(processor, "imagenet", os.environ["SLO_MS"],
        limit=0, workers=int(os.environ["PREPROCESS_WORKERS"]))
    handler = server.RequestHandlerClass
    original_get = handler.do_GET
    def get(request):
        if request.path == "/preparation":
            return request.send({"phase": processor.preparation.phase,
                "fixture_sha256": processor.preparation.fixture_sha256,
                "decode_calls_including_startup_fixture": processor.preparation.decode_calls})
        return original_get(request)
    handler.do_GET = get
    try:
        server.serve_forever()
    finally:
        server.server_close()


class Model:
    def __init__(self, runtime):
        self.runtime = runtime
        self.name = os.environ["MODEL_NAME"]
        self.batches = self.inputs = 0
        self.lock = threading.RLock()

    def predict_strings(self, inputs):
        with self.lock:
            entered = time.perf_counter()
            load = self.runtime.load(self.name)
            payloads = [json.loads(item) for item in inputs]
            prepared,ready = [],[]
            for p in payloads:
                # Application preprocessing stays in the model callback, after
                # native Clipper batching. No gateway/deadline rewrite is used.
                raw = base64.b64decode(p["image_b64"],validate=True)
                prepared.append(self.runtime.preparation._prepare(raw))
                ready.append(time.perf_counter())
            started = time.perf_counter()
            results = self.runtime.infer_prepared(prepared)
            finished = time.perf_counter()
            gpu_ms = (finished-started)*1000
            self.batches += 1
            self.inputs += len(inputs)
            return [json.dumps({"request_id": p["request_id"], "label": r["label"], "model": self.name,
                               "batch_size": len(inputs), "load_ms": load["load_ms"],
                               "server_e2e_ms": (finished-entered)*1000,
                               "post_preprocess_ms": (finished-prepared_at)*1000,
                               "timings": {"worker.cpu_service": r["timings"]["decode"]+r["timings"]["preprocess"],
                                   "measurement.post_preprocess": (finished-prepared_at)*1000,
                                   "worker.gpu_service": gpu_ms, "worker.load": load["load_ms"],
                                   "worker.batch_size": len(inputs)}})
                    for p, r, prepared_at in zip(payloads, results, ready)]


def stats(model):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in {"/state", "/health"}:
                return self.send_error(404)
            body = json.dumps({**model.runtime.describe(), "engine": "clipper",
                "request_preprocessing": True, "native_core_modified": False,
                "baseline_source_commit": "9f25e3fc7f8edc891615e81c5b80d3d8aed72608",
                "preprocessing_placement": "model callback after native batch selection",
                "rpc_started": True, "fixed_model": model.name,
                "batches": model.batches, "processed": model.inputs}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path not in {"/reset", "/warmup"}:
                return self.send_error(404)
            with model.lock:
                if self.path == "/reset":
                    model.runtime.reset()
                else:
                    model.runtime.load(model.name)
            body = json.dumps({"resident": self.path == "/warmup"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("0.0.0.0", 9090), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


def worker():
    from .runtime import Runtime
    from clipper_gpu.rpc_bridge import load_rpc
    model = Model(Runtime())
    stats(model)
    load_rpc().RPCService(collect_metrics=False).start(model)


if __name__ == "__main__":
    gateway()
