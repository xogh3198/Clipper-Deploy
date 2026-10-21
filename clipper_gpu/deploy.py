"""Docker lifecycle using the selected daemon, never a global daemon edit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from . import BASE_COMMIT
from . import client
from .config import ROOT, compose_spec, load

CORE_PATHS = ["configure", "dockerfiles/QueryFrontendDockerfile", "dockerfiles/ManagementFrontendDockerfile",
              "CMakeLists.txt", "cmake/DownloadProject.cmake", "cmake/DownloadProject.CMakeLists.cmake.in",
              "containers/python/rpc.py", "src/libclipper/src/containers.cpp",
              "src/libclipper/src/task_executor.cpp", "src/libclipper/include/clipper/task_executor.hpp",
              "src/frontends/src/query_frontend_main.cpp"]
CPU_READY_ADAPTER_SOURCES = {
    "src/libclipper/include/clipper/datatypes.hpp": "c7fd8eeb57e3a458e88ac53166813d2d081fa2c71cef413ee9784fc2a7fb028f",
    "src/frontends/src/query_frontend.hpp": "44d742cc66ff12b8534760faf91d67a32822f0c5854e82b15d1f737995710aa0",
    "src/libclipper/src/query_processor.cpp": "8f25c943c2bc0cf3cd3286300f52787312fe0d149a922ac7b125509599ca5cd7",
    "src/libclipper/include/clipper/task_executor.hpp": "9ddc09c7577087cc3aab10f47cb2ca785fc603286356e2fc3652a60277531c0b",
    "src/frontends/src/query_frontend_tests.cpp": "a342cb7e789caa26b955b1cd3473eaa34662ac3531fd95eb4966add2bc9a64b1",
    "src/libclipper/test/task_executor_test.cpp": "7c79e66f691f1b5b9c49b3bf980996ae8f9b89845ee0d9f633fee9f61a7ff284",
}
CORE_PATHS += [p for p in CPU_READY_ADAPTER_SOURCES if p not in CORE_PATHS]


def run(args, capture=False, cwd=ROOT):
    result = subprocess.run([str(x) for x in args], cwd=str(cwd), check=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None, text=True)
    return result.stdout.strip() if capture else None


def docker_info():
    info = json.loads(run(["docker", "info", "--format", "{{json .}}"], capture=True))
    print("Docker data-root:", info["DockerRootDir"])
    print("Docker security:", info.get("SecurityOptions", []))
    return info


def docker_user(info):
    if any("rootless" in item for item in info.get("SecurityOptions", [])):
        return "0:0"
    return "%s:%s" % (os.getuid(), os.getgid())


def paths(c):
    result = Path(c["RESULT_DIR"])
    result.mkdir(parents=True, exist_ok=True)
    Path(c["MODEL_DIR"]).mkdir(parents=True, exist_ok=True)
    return result / "compose.json", result / "active.json", result / "build.json"


def compose(c, *args):
    path, _, _ = paths(c)
    if not path.exists():
        raise RuntimeError("No active compose file. Run up first.")
    run(["docker", "compose", "-p", c["PROJECT"], "-f", path] + list(args))


def check_original(c):
    root = Path(c["ROOT"])
    run(["git", "cat-file", "-e", BASE_COMMIT + "^{commit}"], cwd=root)
    # Validate all original core paths, not just a branch name or repository URL.
    changed = run(["git", "diff", "--name-only", BASE_COMMIT, "--", "src", "configure", "dockerfiles",
                   "CMakeLists.txt", "cmake",
                   "containers/python/rpc.py", "containers/query_frontend", "containers/mgmt_frontend"],
                  capture=True, cwd=root)
    # New Docker ignore files are not tracked until user commits, and do not change source.
    unexpected = [p for p in changed.splitlines()
                  if not p.endswith("Dockerfile.dockerignore") and p not in CPU_READY_ADAPTER_SOURCES]
    if unexpected:
        raise RuntimeError("Original source differs from required commit: " + ", ".join(unexpected))
    for path in CORE_PATHS:
        if not (root / path).is_file():
            raise RuntimeError("Original file missing: " + path)
    for path, expected in CPU_READY_ADAPTER_SOURCES.items():
        actual = hashlib.sha256((root / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise RuntimeError("CPU-ready deadline adapter differs from audited source: " + path)


def prepare_base(c):
    base = c["LIB_BASE_IMAGE"]
    registry, sep, version = base.rpartition("/lib_base:")
    if not sep or not registry or not version:
        raise ValueError("LIB_BASE_IMAGE must have the form registry/lib_base:tag")
    if c["BUILD_LIB_BASE"] == "1":
        print("Building dependency image locally (no pull of clipper/lib_base:develop).")
        run(["docker", "build", "--build-arg", "BUILD_JOBS=" + c["BASE_BUILD_JOBS"],
             "-t", base, "-f", "deploy/docker/lib-base.Dockerfile", "."], cwd=c["ROOT"])
    elif c["PULL_LIB_BASE"] == "1":
        run(["docker", "pull", base])
    else:
        run(["docker", "image", "inspect", base], capture=True)
    return registry, version


def build(c):
    info = docker_info()
    check_original(c)
    print("Building Clipper with the audited CPU-ingress deadline adapter; native batch selection is retained.")
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=c["ROOT"])
    status = run(["git", "submodule", "status", "--recursive"], capture=True, cwd=c["ROOT"])
    if any(line.startswith(("+", "-", "U")) for line in status.splitlines()):
        raise RuntimeError("Submodules are not pinned to the original checkout")
    for path in ("src/libs/spdlog", "src/libs/redox"):
        if run(["git", "status", "--porcelain"], capture=True, cwd=Path(c["ROOT"]) / path):
            raise RuntimeError("Modified dependency checkout: " + path)
    base = c["LIB_BASE_IMAGE"]
    registry, version = prepare_base(c)
    for dockerfile, image in (("QueryFrontendDockerfile", c["QUERY_IMAGE"]),
                              ("ManagementFrontendDockerfile", c["MANAGEMENT_IMAGE"])):
        run(["docker", "build", "--build-arg", "REGISTRY=" + registry,
             "--build-arg", "CODE_VERSION=" + version, "-t", image,
             "-f", "dockerfiles/" + dockerfile, "."], cwd=c["ROOT"])
    run(["docker", "build", "--build-arg", "PYTHON_IMAGE=" + c["PYTHON_IMAGE"],
         "--build-arg", "TORCH_VERSION=" + c["TORCH_VERSION"],
         "--build-arg", "TORCHVISION_VERSION=" + c["TORCHVISION_VERSION"],
         "-t", c["WORKER_IMAGE"], "-f", "deploy/docker/worker.Dockerfile", "."], cwd=c["ROOT"])
    run(["docker", "pull", c["REDIS_IMAGE"]])
    images = {}
    base_labels = {}
    for name in (base, c["QUERY_IMAGE"], c["MANAGEMENT_IMAGE"], c["WORKER_IMAGE"], c["REDIS_IMAGE"]):
        details = json.loads(run(["docker", "image", "inspect", name], capture=True))[0]
        images[name] = details["Id"]
        if name == base:
            base_labels = details.get("Config", {}).get("Labels") or {}
    _, _, manifest = paths(c)
    recipe_path = Path(c["ROOT"]) / "deploy/docker/lib-base.Dockerfile"
    manifest.write_text(json.dumps({"base_commit": BASE_COMMIT, "images": images,
                        "pipeline": "cpu-ready-v1", "deadline_adapter": CPU_READY_ADAPTER_SOURCES,
                        "dependency_base_locally_built": c["BUILD_LIB_BASE"] == "1",
                        "dependency_googletest_revision": base_labels.get("org.clipper.gpu.googletest"),
                        "dependency_event_header_compatibility": base_labels.get("org.clipper.gpu.event-header-compatibility"),
                        "dependency_recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
                        "docker_data_root": info["DockerRootDir"], "submodules": status,
                        "source_sha256": {p: hashlib.sha256((Path(c["ROOT"]) / p).read_bytes()).hexdigest()
                                          for p in CORE_PATHS}}, indent=2), encoding="utf-8")
    print("Build complete. Manifest:", manifest)


def verify_images(c):
    _, _, path = paths(c)
    if not path.exists():
        raise RuntimeError("No build manifest; run build")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("pipeline") != "cpu-ready-v1":
        raise RuntimeError("Old JPEG-after-batching images; rebuild for cpu-ready-v1 before running")
    for key in ("QUERY_IMAGE", "MANAGEMENT_IMAGE", "WORKER_IMAGE", "REDIS_IMAGE"):
        name = c[key]
        expected = manifest["images"].get(name)
        actual = json.loads(run(["docker", "image", "inspect", name], capture=True))[0]["Id"]
        if expected != actual:
            raise RuntimeError("Image changed after build: " + name)


def oneoff(c, module, gpu=False, offline=False):
    info = docker_info()
    args = ["docker", "run", "--rm", "--user", docker_user(info),
            "--mount", "type=bind,src=" + c["MODEL_DIR"] + ",dst=/models" + (",readonly" if offline else ""),
            "-e", "MODEL_ID=" + c["MODEL_ID"], "-e", "MODEL_REVISION=" + c["MODEL_REVISION"],
            "-e", "CPU_THREADS=" + c["CPU_THREADS"],
            "-e", "CPU_INTEROP_THREADS=" + c["CPU_INTEROP_THREADS"]]
    if gpu:
        args += ["--gpus", "device=" + c["GPU"]]
    if offline:
        args += ["-e", "HF_HUB_OFFLINE=1"]
    args += [c["WORKER_IMAGE"], "python", "-m", "clipper_gpu." + module]
    run(args)


def endpoint(c, name):
    return "http://127.0.0.1:" + c["MANAGEMENT_PORT"] + "/admin/" + name


def register(c):
    app = {"name": c["APPLICATION"], "input_type": "strings", "default_output": "CLIPPER_SLO_FALLBACK",
           "latency_slo_micros": int(float(c["SLO_MS"]) * 1000)}
    existing = client.http_json(endpoint(c, "get_all_applications"), {"verbose": True})
    match = next((a for a in existing if a["name"] == app["name"]), None)
    if match:
        if any(str(match.get(key)) != str(value) for key, value in app.items()):
            raise RuntimeError("Application settings differ; down/up this project before changing SLO")
    else:
        client.http_json(endpoint(c, "add_app"), app)
    model = {"model_name": c["MODEL_NAME"], "model_version": c["MODEL_VERSION"], "labels": [],
             "input_type": "strings", "container_name": c["WORKER_IMAGE"],
             "model_data_path": "DEPRECATED", "batch_size": int(c["BATCH_SIZE"])}
    existing = client.http_json(endpoint(c, "get_all_models"), {"verbose": True})
    match = next((m for m in existing if m["model_name"] == model["model_name"]
                  and str(m["model_version"]) == model["model_version"]), None)
    if match:
        # This commit does not expose batch_size in the management JSON response.
        _, active_path, _ = paths(c)
        active = json.loads(active_path.read_text(encoding="utf-8"))
        if active["config"]["BATCH_SIZE"] != c["BATCH_SIZE"]:
            raise RuntimeError("Batch setting changed; down/up this project")
    else:
        client.http_json(endpoint(c, "add_model"), model)
    client.http_json(endpoint(c, "set_model_version"),
                     {"model_name": c["MODEL_NAME"], "model_version": c["MODEL_VERSION"]})
    linked = next((a.get("linked_models", []) for a in
        client.http_json(endpoint(c, "get_all_applications"), {"verbose": True})
        if a["name"] == c["APPLICATION"]), [])
    if not linked:
        client.http_json(endpoint(c, "add_model_links"),
                         {"app_name": c["APPLICATION"], "model_names": [c["MODEL_NAME"]]})
    elif linked != [c["MODEL_NAME"]]:
        raise RuntimeError("Application linked to an unexpected model: " + repr(linked))
    print("Registered:", c["APPLICATION"], c["MODEL_NAME"], "SLO=" + c["SLO_MS"] + "ms batch=" + c["BATCH_SIZE"])


def wait_for(check, timeout=90):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            value = check()
            if value:
                return value
        except Exception as exc:
            last = exc
        time.sleep(0.5)
    raise TimeoutError("Service did not become ready: " + str(last))


def worker_registered(c):
    replicas = client.http_json(endpoint(c, "get_all_containers"), {"verbose": True})
    return any(r.get("model_name") == c["MODEL_NAME"] and str(r.get("model_version")) == c["MODEL_VERSION"]
               for r in replicas)


def gpu_idle(c):
    name = run(["nvidia-smi", "-i", c["GPU"], "--query-gpu=name", "--format=csv,noheader"], capture=True)
    if c["GPU_TYPE"] == "v100" and "V100" not in name:
        raise RuntimeError("Selected GPU is not a V100: " + name)
    processes = run(["nvidia-smi", "-i", c["GPU"], "--query-compute-apps=pid,process_name,used_memory",
                     "--format=csv,noheader"], capture=True)
    if processes and c["ALLOW_GPU_SHARING"] != "1":
        raise RuntimeError("GPU already has compute processes; no process was stopped:\n" + processes)


def up(c):
    verify_images(c)
    running = run(["docker", "ps", "-q", "--filter", "label=com.docker.compose.project=" + c["PROJECT"]],
                  capture=True)
    if running:
        raise RuntimeError("This project already has running services. Use state; use down/up to restart only this project.")
    gpu_idle(c)
    info = docker_info()
    compose_path, active_path, _ = paths(c)
    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    compose_path.write_text(json.dumps(compose_spec(c, docker_user(info), run_id), indent=2), encoding="utf-8")
    active_path.write_text(json.dumps({"run_id": run_id, "config": c}, indent=2), encoding="utf-8")
    compose(c, "up", "-d", "redis", "management", "query", "gateway")
    wait_for(lambda: isinstance(client.http_json(endpoint(c, "get_all_applications"), {"verbose": True}), list))
    # Health is a GET, not a prediction. This does not warm the model.
    wait_for(lambda: client.http_json("http://127.0.0.1:" + c["QUERY_PORT"] + "/metrics") is not None)
    register(c)
    compose(c, "up", "-d", "worker")
    wait_for(lambda: client.state(c)["rpc_started"] and worker_registered(c))
    print("READY: original frontend has registered the GPU RPC worker. Model resident:", client.state(c)["model_loaded"])


def state(c):
    compose(c, "ps")
    print("Frontend replicas:", json.dumps(client.http_json(endpoint(c, "get_all_containers"), {"verbose": True})))
    snapshot = client.state(c)
    snapshot.pop("recent_batches", None)
    print(json.dumps(snapshot, indent=2))


def sample(c, source=None):
    dest = Path(c["INPUT_IMAGE"])
    if source:
        source = Path(source).expanduser().resolve()
    elif dest.exists():
        source = dest
    else:
        candidates = [Path.home() / "taeho/infaas_deploy/inputs/test-dog.jpg",
                      Path.home() / "taeho/serving-repos/Proteus/inputs/test-dog.jpg"]
        source = next((p for p in candidates if p.is_file()), None)
    if source is None or not source.is_file():
        raise RuntimeError("No sample image. Run sample --from /absolute/path/test-dog.jpg")
    if source != dest:
        if dest.exists():
            raise RuntimeError("Refusing to overwrite input image: " + str(dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    print("Image:", dest, "sha256=" + hashlib.sha256(dest.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description="Original Clipper + independent GPU worker")
    parser.add_argument("command", choices=["config", "settings", "build", "download", "sample", "doctor", "up", "register",
                         "query", "test", "cold-test", "batching-test", "state", "logs", "down"])
    parser.add_argument("service", nargs="?", choices=["redis", "management", "query", "worker"])
    parser.add_argument("--models", help="Optional name; must match MODEL_NAME in settings")
    parser.add_argument("--from", dest="source")
    parser.add_argument("-n", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--watch", type=float, default=0)
    args = parser.parse_args()
    c = load()
    if args.models and args.models != c["MODEL_NAME"]:
        parser.error("--models must match MODEL_NAME; change the model in clipper-docker.local.env")
    if args.n is not None and args.n < 1 or args.concurrency < 1 or args.watch < 0:
        parser.error("Counts and concurrency must be positive; watch must be nonnegative")
    if args.command in ("config", "settings"):
        docker_info()
        print(json.dumps(c, indent=2))
    elif args.command == "build":
        build(c)
    elif args.command == "sample":
        sample(c, args.source)
    elif args.command in ("download", "doctor"):
        paths(c)
        verify_images(c)
        oneoff(c, args.command, gpu=args.command == "doctor", offline=args.command == "doctor")
    elif args.command == "up":
        up(c)
    elif args.command == "register":
        register(c)
    elif args.command == "state":
        while True:
            state(c)
            if not args.watch:
                break
            time.sleep(args.watch)
    elif args.command == "logs":
        compose(c, "logs", "--tail", "100", args.service or "worker")
    elif args.command == "down":
        compose(c, "down")
        print("Only this project's containers/network removed. Images, weights and results retained.")
    elif args.command == "query":
        raw = Path(c["INPUT_IMAGE"]).read_bytes()
        failed = False
        for _ in range(args.n or 1):
            result = client.query_one(c, raw)
            client.print_query(result)
            failed |= result["default"]
        return 2 if failed else 0
    else:
        if args.command == "cold-test":
            compose(c, "down")
            up(c)
        concurrency = args.concurrency if args.command == "batching-test" else 1
        count = args.n or (128 if concurrency > 1 else 11)
        if args.command == "cold-test" and count < 2:
            parser.error("cold-test needs at least 2 requests")
        summary = client.measure(c, count, concurrency, cold=args.command == "cold-test")
        if args.command == "batching-test":
            verified = summary["max_batch_for_these_requests"] > 1 and not summary["worker_errors"]
            print("BATCHING VERIFIED" if verified else "BATCHING NOT VERIFIED: no batch > 1 observed")
            return 0 if verified else 3
        if summary["warm_requests"]:
            return 0 if summary["warm_real_predictions"] == summary["warm_requests"] and \
                all(ms == 0 for ms in summary["warm_load_ms"]) else 2
        return 0 if summary["real_predictions"] == summary["requests"] else 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, TimeoutError, subprocess.CalledProcessError, OSError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)
