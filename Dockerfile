FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Minimal deps; torch/torchvision come from the base image
COPY requirements/runpod.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt

# Code
COPY . /app

# Runpod serverless handler
ENV RP_HANDLER=runpod_server
CMD ["python", "-u", "-m", "runpod_server"]