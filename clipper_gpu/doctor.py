import importlib.metadata
import json
import os
from pathlib import Path


def main():
    import torch
    from huggingface_hub import snapshot_download
    from .rpc_bridge import load_rpc
    from .prepared import CpuProcessor, PIPELINE
    torch.set_num_threads(int(os.environ.get("CPU_THREADS", "2")))
    result = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
              "arch": torch.cuda.get_arch_list(), "cuda_test": (torch.ones(1, device="cuda") * 2).item(),
              "versions": {name: importlib.metadata.version(name) for name in
                           ("torch", "torchvision", "transformers", "pillow", "numpy", "pyzmq")}}
    path = snapshot_download(repo_id=os.environ["MODEL_ID"], revision=os.environ["MODEL_REVISION"],
                             local_files_only=True)
    if not (Path(path) / "config.json").is_file() or not list(Path(path).glob("*.safetensors")):
        raise RuntimeError("Pinned model config/weights missing; run download")
    rpc = load_rpc()
    processor = CpuProcessor()
    result.update(model_id=os.environ["MODEL_ID"], revision=os.environ["MODEL_REVISION"], snapshot=path,
                  pipeline=PIPELINE, preprocess_contract=processor.contract,
                  cpu_interop_threads=torch.get_num_interop_threads(),
                  original_rpc_version=rpc.RPC_VERSION, legacy_rpc_metrics=False,
                  warning="GPU/weights/RPC import check only; not frontend, batching or SLO verification")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
