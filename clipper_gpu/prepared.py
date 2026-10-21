"""CPU image preparation and a bounded, validated float32 wire contract."""
import base64
import hashlib
import io
import json
import math
import os
import threading
import time


PIPELINE = "cpu-ready-v1"


def processor_contract(processor):
    config = {key: value for key, value in processor.to_dict().items()
              if key not in {"_commit_hash", "_name_or_path", "name_or_path"}}
    value = json.dumps({"class": type(processor).__name__, "config": config}, sort_keys=True)
    return hashlib.sha256(value.encode()).hexdigest()


class CpuProcessor:
    def __init__(self):
        import torch
        from transformers import AutoImageProcessor
        self.torch = torch
        torch.set_num_threads(int(os.environ.get("CPU_THREADS", "2")))
        interop = int(os.environ.get("CPU_INTEROP_THREADS", "1"))
        if torch.get_num_interop_threads() != interop:
            torch.set_num_interop_threads(interop)
        self.processor = AutoImageProcessor.from_pretrained(os.environ["MODEL_ID"],
            revision=os.environ["MODEL_REVISION"], local_files_only=True, use_fast=False)
        self.contract = processor_contract(self.processor)
        self.lock = threading.Lock()

    def prepare(self, payload, received, deadline):
        if time.time_ns() // 1000 >= deadline:
            raise TimeoutError("SLO expired in CPU queue")
        from PIL import Image
        started = time.perf_counter()
        raw = base64.b64decode(payload["image_b64"], validate=True)
        if not raw or len(raw) > 16 * 1024 * 1024:
            raise ValueError("Image must be in 1..16 MiB")
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGB")
        decoded = time.perf_counter()
        with self.lock:
            inputs = self.processor(images=[image], return_tensors="pt")
        tensor = inputs["pixel_values"].contiguous()
        if tensor.device.type != "cpu" or tensor.dtype != self.torch.float32:
            raise ValueError("Preparation requires CPU float32 tensors")
        processed = time.perf_counter()
        return {"pipeline": PIPELINE, "request_id": payload["request_id"],
            "image_sha256": hashlib.sha256(raw).hexdigest(), "image_size": list(image.size),
            "shape": list(tensor.shape), "contract": self.contract,
            "tensor_b64": base64.b64encode(tensor.numpy().astype("<f4", copy=False).tobytes()).decode("ascii"),
            "received_unix_us": received, "deadline_unix_us": deadline,
            "cpu_queue_ms": max(0, time.time_ns() / 1000 - received - (time.perf_counter() - started) * 1e6) / 1000,
            "decode_ms": (decoded - started) * 1000,
            "preprocess_ms": (processed - decoded) * 1000}


def decode_tensor(payload, contract, torch):
    import numpy as np
    if payload.get("pipeline") != PIPELINE or payload.get("contract") != contract:
        raise ValueError("Prepared input contract mismatch; JPEGs must go through the CPU gateway")
    shape = payload.get("shape")
    if (not isinstance(shape, list) or len(shape) != 4 or shape[:2] != [1, 3] or
            any(type(n) is not int or n <= 0 or n > 2048 for n in shape)):
        raise ValueError("Invalid prepared tensor shape")
    count = math.prod(shape)
    if count * 4 > 16 * 1024 * 1024:
        raise ValueError("Prepared tensor exceeds 16 MiB")
    encoded = payload["tensor_b64"]
    if not isinstance(encoded, str) or len(encoded) > (count * 4 + 2) // 3 * 4:
        raise ValueError("Prepared tensor size mismatch")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != count * 4:
        raise ValueError("Prepared tensor byte length mismatch")
    values = np.frombuffer(raw, dtype="<f4").copy()
    if not np.isfinite(values).all():
        raise ValueError("Prepared tensor contains non-finite values")
    return torch.from_numpy(values).reshape(shape)
