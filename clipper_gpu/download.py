"""Download pretrained weights/config; never train or load them into a GPU."""
import json
import os


def main():
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=os.environ["MODEL_ID"], revision=os.environ["MODEL_REVISION"],
                             allow_patterns=["*.json", "*.safetensors", "*.bin"])
    print(json.dumps({"model_id": os.environ["MODEL_ID"], "revision": os.environ["MODEL_REVISION"],
                      "snapshot": path, "training": False}))


if __name__ == "__main__":
    main()
