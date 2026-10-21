"""Literal env parsing: no shell evaluation and no system Docker changes."""
import os
import re
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_env(path):
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError("Invalid env line: " + line)
        values[key] = value.strip().strip("\"'")
    return values


def load(root=ROOT):
    root = Path(root).resolve()
    values = read_env(root / "deploy/docker/clipper-docker.env")
    local = read_env(root / "deploy/docker/clipper-docker.local.env")
    if set(local) - set(values):
        raise ValueError("Unknown local settings: " + ", ".join(sorted(set(local) - set(values))))
    values.update(local)
    for key in list(values):
        if key in os.environ:
            values[key] = os.environ[key]
    for key in ("PROJECT", "APPLICATION", "MODEL_NAME", "MODEL_VERSION"):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", values[key]):
            raise ValueError("Unsafe name: " + key)
    if not values["GPU"].isdigit():
        raise ValueError("GPU must be one numeric GPU index")
    for key in ("ALLOW_GPU_SHARING", "PULL_LIB_BASE", "BUILD_LIB_BASE"):
        if values[key] not in ("0", "1"):
            raise ValueError(key + " must be 0 or 1")
    for key in ("CPU_THREADS", "CPU_INTEROP_THREADS", "PREPROCESS_WORKERS",
                "QUERY_THREADS", "QUERY_PORT", "MANAGEMENT_PORT", "STATS_PORT", "BASE_BUILD_JOBS"):
        if int(values[key]) <= 0:
            raise ValueError(key + " must be positive")
    if int(values["PREPROCESS_WORKERS"]) > 32:
        raise ValueError("PREPROCESS_WORKERS must be in 1..32")
    if int(values["MAX_QUEUE_SIZE"]) < 0:
        raise ValueError("MAX_QUEUE_SIZE must be >= 0 (0 keeps the original unbounded queues)")
    if len({values[k] for k in ("QUERY_PORT", "MANAGEMENT_PORT", "STATS_PORT")}) != 3:
        raise ValueError("Host ports must be distinct")
    if any(int(values[k]) > 65535 for k in ("QUERY_PORT", "MANAGEMENT_PORT", "STATS_PORT")):
        raise ValueError("Ports must be <= 65535")
    if not math.isfinite(float(values["SLO_MS"])) or float(values["SLO_MS"]) <= 0 or \
            int(values["BATCH_SIZE"]) == 0 or int(values["BATCH_SIZE"]) < -1:
        raise ValueError("SLO must be positive; BATCH_SIZE must be -1 or positive")
    if not re.fullmatch(r"[0-9a-f]{40}", values["MODEL_REVISION"]):
        raise ValueError("Use a pinned model revision, not a mutable branch")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*/[a-zA-Z0-9][a-zA-Z0-9_.-]*", values["MODEL_ID"]):
        raise ValueError("MODEL_ID must be a Hugging Face owner/model ID")
    values["ROOT"] = str(root)
    for key in ("MODEL_DIR", "RESULT_DIR", "INPUT_IMAGE"):
        path = Path(values[key]).expanduser()
        values[key] = str((root / path).resolve() if not path.is_absolute() else path.resolve())
    return values


def compose_spec(c, worker_user="0:0", run_id="pending"):
    project = c["PROJECT"]
    base = {"restart": "no", "labels": {"org.clipper.gpu.project": project}}
    services = {
        "redis": dict(base, image=c["REDIS_IMAGE"],
                      command=["redis-server", "--save", "", "--appendonly", "no",
                               "--protected-mode", "no", "--notify-keyspace-events", "AKE"],
                      healthcheck={"test": ["CMD", "redis-cli", "ping"], "interval": "2s", "timeout": "2s", "retries": 30}),
        "management": dict(base, image=c["MANAGEMENT_IMAGE"],
                           command=["--redis_ip=redis", "--redis_port=6379"],
                           ports=["127.0.0.1:" + c["MANAGEMENT_PORT"] + ":1338"],
                           depends_on={"redis": {"condition": "service_healthy"}}),
        "query": dict(base, image=c["QUERY_IMAGE"],
                      environment={"CLIPPER_MAX_READY_QUEUE": c["MAX_QUEUE_SIZE"]},
                      command=["--redis_ip=redis", "--redis_port=6379", "--rpc_service_port=7000",
                               "--prediction_cache_size=0", "--thread_pool_size=" + c["QUERY_THREADS"]],
                      depends_on={"redis": {"condition": "service_healthy"}}),
    }
    services["worker"] = dict(base, image=c["WORKER_IMAGE"], user=worker_user,
        ports=["127.0.0.1:" + c["STATS_PORT"] + ":9090"], depends_on=["query"],
        volumes=[{"type": "bind", "source": c["MODEL_DIR"], "target": "/models", "read_only": True},
                 {"type": "bind", "source": c["RESULT_DIR"], "target": "/results"}],
        environment={"CLIPPER_MODEL_NAME": c["MODEL_NAME"], "CLIPPER_MODEL_VERSION": c["MODEL_VERSION"],
                     "CLIPPER_MODEL_PATH": "/models", "CLIPPER_INPUT_TYPE": "strings", "CLIPPER_IP": "query",
                     "CLIPPER_PORT": "7000", "MODEL_ID": c["MODEL_ID"], "MODEL_REVISION": c["MODEL_REVISION"],
                     "CPU_THREADS": c["CPU_THREADS"], "CPU_INTEROP_THREADS": c["CPU_INTEROP_THREADS"],
                     "RUN_ID": run_id,
                     "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        deploy={"resources": {"reservations": {"devices": [
            {"driver": "nvidia", "device_ids": [c["GPU"]], "capabilities": ["gpu"]}]}}})
    services["gateway"] = dict(base, image=c["WORKER_IMAGE"], user=worker_user,
        command=["python", "-m", "clipper_gpu.gateway"],
        ports=["127.0.0.1:" + c["QUERY_PORT"] + ":8080"], depends_on=["query"],
        volumes=[{"type": "bind", "source": c["MODEL_DIR"], "target": "/models", "read_only": True}],
        environment={"MODEL_ID": c["MODEL_ID"], "MODEL_REVISION": c["MODEL_REVISION"],
            "CPU_THREADS": c["CPU_THREADS"], "CPU_INTEROP_THREADS": c["CPU_INTEROP_THREADS"],
            "PREPROCESS_WORKERS": c["PREPROCESS_WORKERS"], "MAX_QUEUE_SIZE": c["MAX_QUEUE_SIZE"],
            "APPLICATION": c["APPLICATION"], "SLO_MS": c["SLO_MS"],
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    return {"services": services}
