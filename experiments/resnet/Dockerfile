FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.11.0+cu126 torchvision==0.26.0+cu126
COPY experiments/resnet/requirements.txt /opt/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/requirements.txt
WORKDIR /app
COPY experiments/resnet /app/experiments/resnet
COPY clipper_gpu /app/clipper_gpu
COPY containers/python/rpc.py /opt/clipper/containers/python/rpc.py
RUN ln -s /tmp/paired-clipper-is-ready.check /model_is_ready.check
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /tmp
CMD ["python", "-m", "experiments.resnet.server"]
