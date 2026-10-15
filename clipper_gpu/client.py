"""Stdlib-only management/query clients and auditable measurements."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid


def http_json(url, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError("HTTP %s: %s" % (exc.code, exc.read().decode("utf-8", "replace"))) from exc
    try:
        return json.loads(text)
    except ValueError:
        return text


def state(c):
    return http_json("http://127.0.0.1:" + c["STATS_PORT"] + "/state")


def parse_prediction(response, request_id):
    if not isinstance(response, dict) or not isinstance(response.get("default"), bool):
        raise ValueError("Missing Clipper default flag: " + repr(response))
    output = response.get("output")
    if response["default"]:
        return None
    if isinstance(output, str):
        output = json.loads(output)
    if not isinstance(output, dict) or output.get("request_id") != request_id or not output.get("label"):
        raise ValueError("Response is not this request's GPU inference: " + repr(response))
    return output


def query_one(c, raw, phase="query"):
    request_id = uuid.uuid4().hex
    wrapped = json.dumps({"request_id": request_id, "image_b64": base64.b64encode(raw).decode("ascii")})
    start = time.perf_counter()
    response = http_json("http://127.0.0.1:" + c["QUERY_PORT"] + "/" + c["APPLICATION"] + "/predict",
                         {"input": wrapped}, timeout=120)
    elapsed = (time.perf_counter() - start) * 1000
    output = parse_prediction(response, request_id)
    result = {"request_id": request_id, "phase": phase, "e2e_ms": elapsed,
              "default": response["default"], "real_prediction": output is not None,
              "client_e2e_over_slo": elapsed > float(c["SLO_MS"]),
              "image_sha256": hashlib.sha256(raw).hexdigest(), "response": response}
    if output:
        result.update(output=output)
    return result


def print_query(result):
    output = result.get("output", {})
    load_text = "%.2fms" % output["load_ms"] if "load_ms" in output else "n/a"
    print("%s e2e=%8.2fms load=%s batch=%s label=%r" % (
        "DEFAULT" if result["default"] else "OK", result["e2e_ms"], load_text,
        output.get("batch_size", "-"), output.get("label", "<SLO fallback>")), flush=True)


def completed_batch(worker_state, request_id):
    return next((event for event in worker_state.get("recent_batches", [])
                 if request_id in event["request_ids"]), None)


def wait_completed(c, request_id, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = state(c)
        event = completed_batch(snapshot, request_id)
        if event:
            return event
        if snapshot["errors"]:
            raise RuntimeError("GPU worker failed. Check worker logs/JSONL.")
        time.sleep(0.2)
    raise TimeoutError("No GPU completion event for request " + request_id)


def measure(c, count=11, concurrency=1, cold=False):
    raw = Path(c["INPUT_IMAGE"]).read_bytes()
    before = state(c)
    if cold and before["model_loaded"]:
        raise RuntimeError("Model is already resident; use cold-test to recreate only this project")
    if concurrency > 1 and not before["model_loaded"]:
        raise RuntimeError("Run cold-test first; batching-test measures a resident model")
    records = []
    if cold:
        print("Runtime initialization before requests: %s ms" % before.get("startup_ms", "unknown"))
        print("Cold definition:", before.get("cold_definition", "unspecified (legacy worker)"))
    if cold:
        result = query_one(c, raw, "cold")
        print_query(result)
        result["worker_completion"] = wait_completed(c, result["request_id"])
        records.append(result)
        print("cold actual worker completion: %.2fms; load: %.2fms" % (
            result["worker_completion"]["worker_total_ms"], result["worker_completion"]["load_ms"]))
        if result["worker_completion"].get("load_stages"):
            print("cold model preparation stages:", json.dumps(result["worker_completion"]["load_stages"]))
        count -= 1
    phase = "warm" if cold or before["model_loaded"] else "unclassified"
    if concurrency == 1:
        for _ in range(count):
            result = query_one(c, raw, phase)
            records.append(result)
            print_query(result)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            records.extend(executor.map(lambda _: query_one(c, raw, "concurrent"), range(count)))
    # Defaults can expire in the native queue without executing. Record that honestly.
    deadline = time.monotonic() + 15
    while True:
        after = state(c)
        for result in records:
            if "worker_completion" not in result:
                event = completed_batch(after, result["request_id"])
                if event:
                    result["worker_completion"] = event
        if all("worker_completion" in r for r in records) or time.monotonic() >= deadline or after["errors"]:
            break
        time.sleep(0.2)
    selected_batches = {}
    for result in records:
        event = result.get("worker_completion")
        if event:
            selected_batches[tuple(event["request_ids"])] = event
    warm = [r for r in records if r["phase"] == "warm"]
    warm_times = [r["e2e_ms"] for r in warm]
    summary = {"run_id": before["run_id"], "requests": len(records),
               "real_predictions": sum(r["real_prediction"] for r in records),
               "default_responses": sum(r["default"] for r in records),
               "worker_forward_delta": after["forward_calls"] - before["forward_calls"],
               "warmup_forward_delta": after.get("warmup_forward_calls", 0) - before.get("warmup_forward_calls", 0),
               "all_request_ids_executed": all("worker_completion" in r for r in records),
               "request_ids_without_completion": [r["request_id"] for r in records if "worker_completion" not in r],
               "worker_errors": after["errors"], "initial_model_resident": before["model_loaded"],
               "runtime_startup_ms": before.get("startup_ms"),
               "runtime_startup_stages": before.get("startup_stages"),
               "cold_definition": before.get("cold_definition", "legacy worker"),
               "warmup_queries_per_load": before.get("warmup_queries"),
               "max_batch_for_these_requests": max((e["batch_size"] for e in selected_batches.values()), default=0),
               "warm_requests": len(warm), "warm_real_predictions": sum(r["real_prediction"] for r in warm),
               "warm_load_ms": [r.get("worker_completion", {}).get("load_ms") for r in warm],
               "warm_mean_ms": statistics.mean(warm_times) if warm_times else None,
               "warm_min_ms": min(warm_times) if warm_times else None,
               "warm_max_ms": max(warm_times) if warm_times else None,
               "slo_ms": float(c["SLO_MS"]), "batch_size_setting": int(c["BATCH_SIZE"]),
               "precision": "fp32", "cache_bytes": 0, "model_revision": c["MODEL_REVISION"],
               "image_sha256": hashlib.sha256(raw).hexdigest()}
    cold_result = next((r for r in records if r["phase"] == "cold"), None)
    if cold_result is not None:
        completion = cold_result.get("worker_completion", {})
        summary.update(cold_client_e2e_ms=cold_result["e2e_ms"],
                       cold_default_response=cold_result["default"],
                       cold_worker_total_ms=completion.get("worker_total_ms"),
                       cold_load_ms=completion.get("load_ms"),
                       cold_load_stages=completion.get("load_stages", {}))
    report = {"summary": summary, "requests": records}
    path = Path(c["RESULT_DIR"]) / ("client-" + before["run_id"] + "-" + uuid.uuid4().hex[:8] + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved:", path)
    return summary
