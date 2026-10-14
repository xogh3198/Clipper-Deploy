ARG PYTHON_IMAGE=python:3.11-slim-bookworm
FROM ${PYTHON_IMAGE}
ARG TORCH_VERSION=2.11.0+cu126
ARG TORCHVISION_VERSION=0.26.0+cu126
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}
COPY deploy/docker/requirements-gpu.txt /opt/requirements-gpu.txt
RUN python -m pip install --no-cache-dir -r /opt/requirements-gpu.txt
COPY containers/python/rpc.py /opt/clipper/containers/python/rpc.py
COPY clipper_gpu /app/clipper_gpu
# The unmodified original RPC writes this path. The target is writable by any UID.
RUN ln -s /tmp/clipper-model-is-ready.check /model_is_ready.check
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/models/hub TOKENIZERS_PARALLELISM=false
LABEL org.clipper.gpu.base-commit="9f25e3fc7f8edc891615e81c5b80d3d8aed72608"
CMD ["python", "-m", "clipper_gpu.worker"]
