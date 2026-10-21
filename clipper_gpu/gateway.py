"""JPEG ingress -> CPU stage -> original C++ queue with original deadline."""
import concurrent.futures
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
import urllib.error
import urllib.request

from .prepared import CpuProcessor, PIPELINE


def fallback(reason, native_submitted=False):
    return {"default": True, "output": "CLIPPER_SLO_FALLBACK", "default_explanation": reason,
            "pipeline": PIPELINE, "native_submitted": native_submitted}


def create_server(processor, application, slo_ms, address=("0.0.0.0", 8080),
                  query_base="http://query:1337", limit=0, workers=1):
    if limit < 0 or not 1 <= workers <= 32:
        raise ValueError("Invalid queue/preprocessing worker limit")
    # Zero preserves the previous deployment's absence of an admission cap.
    # A positive limit is explicitly opt-in, never a new default of 256.
    slots = threading.BoundedSemaphore(limit) if limit else None
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="clipper-cpu")
    slo_us = int(float(slo_ms) * 1000)
    if slo_us <= 0:
        raise ValueError("SLO must be positive")
    query_base = query_base.rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        def send(self, body, status=200):
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == "/health":
                return self.send({"ready": True, "pipeline": PIPELINE})
            if self.path == "/metrics":
                with urllib.request.urlopen(query_base + "/metrics", timeout=5) as response:
                    content = response.read()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404)

        def do_POST(self):
            received = time.time_ns() // 1000
            deadline = received + slo_us
            if self.path != "/" + application + "/predict":
                return self.send_error(404)
            if slots is not None and not slots.acquire(blocking=False):
                return self.send(fallback("CPU ingress outstanding request limit reached"))
            future = None
            native_submitted = False
            try:
                self.connection.settimeout(slo_us / 1e6)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 24 * 1024 * 1024:
                    raise ValueError("Invalid image request size")
                request = json.loads(self.rfile.read(length))
                payload = json.loads(request["input"])
                future = pool.submit(processor.prepare, payload, received, deadline)
                remaining = max(0, (deadline - time.time_ns() // 1000) / 1e6)
                prepared = future.result(timeout=remaining)
                body = json.dumps({"input": json.dumps(prepared), "deadline_unix_us": deadline}).encode()
                remaining = (deadline - time.time_ns() // 1000) / 1e6
                if remaining <= 0:
                    return self.send(fallback("SLO expired during CPU preparation"))
                outgoing = urllib.request.Request(query_base + "/" + application + "/predict",
                    data=body, headers={"Content-Type": "application/json"})
                # Attempted submission: transport failure cannot prove whether native accepted it.
                native_submitted = True
                with urllib.request.urlopen(outgoing, timeout=remaining) as response:
                    result = json.load(response)
                if time.time_ns() // 1000 > deadline:
                    result = fallback("Original ingress deadline exceeded", native_submitted=True)
                self.send(result)
            except (TimeoutError, concurrent.futures.TimeoutError):
                self.send(fallback("Original ingress deadline exceeded", native_submitted))
            except urllib.error.URLError as error:
                self.send(fallback("Native query unavailable/deadline expired: " + str(error), native_submitted))
            except (ValueError, KeyError, TypeError) as error:
                self.send({"error": str(error), "pipeline": PIPELINE}, status=400)
            finally:
                # A timed-out running CPU task still owns its bounded slot.
                if slots is not None:
                    if future is not None and not future.done():
                        future.add_done_callback(lambda unused: slots.release())
                    else:
                        slots.release()

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def server_close(self):
            super().server_close()
            pool.shutdown(wait=True)

    return Server(address, Handler)


def main():
    server = create_server(CpuProcessor(), os.environ["APPLICATION"], os.environ["SLO_MS"],
        limit=int(os.environ.get("MAX_QUEUE_SIZE", "0")),
        workers=int(os.environ.get("PREPROCESS_WORKERS", "1")))
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
