import importlib.metadata
import json
import os
from pathlib import Path

from .cache import MODEL_FILES


def main():
    import torch
    from huggingface_hub import snapshot_download
    from .rpc_bridge import load_rpc
    torch.set_num_threads(int(os.environ.get("CPU_THREADS", "2")))
    result = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
              "arch": torch.cuda.get_arch_list(), "cuda_test": (torch.ones(1, device="cuda") * 2).item(),
              "versions": {name: importlib.metadata.version(name) for name in
                           ("torch", "torchvision", "transformers", "pillow", "numpy", "pyzmq")}}
    path = snapshot_download(repo_id=os.environ["MODEL_ID"], revision=os.environ["MODEL_REVISION"],
                             allow_patterns=MODEL_FILES, local_files_only=True)
    if not (Path(path) / "config.json").is_file() or not list(Path(path).glob("*.safetensors")):
        raise RuntimeError("Pinned model config/weights missing; run download")
    rpc = load_rpc()
    result.update(model_id=os.environ["MODEL_ID"], revision=os.environ["MODEL_REVISION"], snapshot=path,
                  original_rpc_version=rpc.RPC_VERSION, legacy_rpc_metrics=False,
                  warning="GPU/weights/RPC import check only; not frontend, batching or SLO verification")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
