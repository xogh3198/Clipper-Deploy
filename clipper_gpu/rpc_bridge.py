"""Load unmodified Clipper RPC, without importing the legacy admin package.

The original metrics package launches a local Redis on import. Metrics collection
is disabled here; native frontend metrics and our worker events remain available.
No transport, queue, batching or wire format is replaced.
"""
import importlib.util
import sys
import types


def load_rpc(path="/opt/clipper/containers/python/rpc.py"):
    if "clipper_admin" in sys.modules:
        raise RuntimeError("Unexpected clipper_admin import before the isolated RPC bridge")
    admin = types.ModuleType("clipper_admin")
    admin.__path__ = []
    metrics = types.ModuleType("clipper_admin.metrics")

    def disabled(*args, **kwargs):
        raise RuntimeError("Legacy RPC metrics are intentionally disabled")

    metrics.add_metric = disabled
    metrics.report_metric = disabled
    admin.metrics = metrics
    sys.modules["clipper_admin"] = admin
    sys.modules["clipper_admin.metrics"] = metrics
    spec = importlib.util.spec_from_file_location("clipper_original_rpc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
