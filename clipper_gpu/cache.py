"""Use the same inference-file filter for online download and offline checks."""

# GPUModel uses use_safetensors=True. Requiring the alternative PyTorch .bin
# checkpoint makes a valid safetensors-only cache fail offline validation.
MODEL_FILES = ["*.json", "*.safetensors"]
