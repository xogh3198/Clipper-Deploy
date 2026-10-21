"""Actual local HTTP transport with fake CPU/native peers; not a GPU experiment."""
import base64
import json
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

from clipper_gpu.gateway import create_server
from clipper_gpu.prepared import PIPELINE, decode_tensor


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.messages = []
        messages = self.messages

        class Native(BaseHTTPRequestHandler):
            def do_POST(self):
                messages.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                data = json.dumps({"default": False, "output": "native prediction"}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.native = ThreadingHTTPServer(("127.0.0.1", 0), Native)
        self.thread = threading.Thread(target=self.native.serve_forever, daemon=True)
        self.thread.start()
        self.servers = []

    def tearDown(self):
        for server, thread in self.servers:
            server.shutdown()
            server.server_close()
            thread.join(3)
        self.native.shutdown()
        self.native.server_close()
        self.thread.join(3)

    def gateway(self, prepare, slo=200, limit=0):
        server = create_server(SimpleNamespace(prepare=prepare), "imagenet", slo,
            ("127.0.0.1", 0), "http://127.0.0.1:%d" % self.native.server_port, limit=limit)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.servers.append((server, thread))
        return "http://127.0.0.1:%d/imagenet/predict" % server.server_port

    def request(self, url):
        data = json.dumps({"input": json.dumps({"request_id": "one", "image_b64": "JPEG"})}).encode()
        with urllib.request.urlopen(urllib.request.Request(url, data=data,
                headers={"Content-Type": "application/json"}), timeout=3) as response:
            return json.load(response)

    def test_prepared_tensor_and_original_deadline_reach_native_frontend(self):
        received_values = []

        def prepare(payload, received, deadline):
            received_values.append((received, deadline))
            time.sleep(0.03)
            return {"pipeline": PIPELINE, "tensor_b64": "READY", "request_id": payload["request_id"]}

        reply = self.request(self.gateway(prepare))
        self.assertFalse(reply["default"])
        message = self.messages[0]
        payload = json.loads(message["input"])
        self.assertNotIn("image_b64", payload)
        self.assertEqual(payload["tensor_b64"], "READY")
        received, deadline = received_values[0]
        self.assertEqual(deadline - received, 200000)
        self.assertEqual(message["deadline_unix_us"], deadline)

    def test_timed_out_cpu_work_keeps_bounded_slot_until_it_finishes(self):
        release, started = threading.Event(), threading.Event()
        calls = []

        def prepare(*args):
            calls.append(1)
            started.set()
            release.wait(3)
            return {"tensor_b64": "READY"}

        url = self.gateway(prepare, slo=30, limit=1)
        try:
            self.assertTrue(self.request(url)["default"])
            self.assertTrue(started.is_set())
            response = self.request(url)
            self.assertIn("limit", response["default_explanation"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(self.messages, [])
        finally:
            release.set()


class TensorContractTests(unittest.TestCase):
    def payload(self):
        return {"pipeline": PIPELINE, "contract": "same", "shape": [1, 3, 1, 1],
                "tensor_b64": base64.b64encode(b"\0" * 12).decode()}

    def test_validated_cpu_tensor_decode(self):
        tensor = Mock()
        torch = SimpleNamespace(from_numpy=Mock(return_value=tensor))
        decode_tensor(self.payload(), "same", torch)
        tensor.reshape.assert_called_once_with([1, 3, 1, 1])

    def test_reject_legacy_jpeg_wrong_contract_and_bad_shape(self):
        for payload in ({"image_b64": "JPEG"}, {**self.payload(), "contract": "other"},
                        {**self.payload(), "shape": [1, 3, 99999, 99999]},
                        {**self.payload(), "tensor_b64": "AAAA"}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                decode_tensor(payload, "same", Mock())
