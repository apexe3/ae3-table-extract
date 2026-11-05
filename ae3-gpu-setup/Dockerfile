# AE3-TL — Open-Source Table Extraction Toolkit
# Copyright 2026 Team APEX:E3
# Licensed under the Apache License, Version 2.0; see LICENSE in the repository root.
#
# AE3-TL GPU service, built to match the benchmarked setup exactly:
#   CUDA 12.4.1 runtime, Python 3.11, onnxruntime-gpu 1.23.2,
#   cuDNN 9.27 + cuBLAS 12.9 from pip, every package pinned from requirements.txt,
#   batch size 1, one model copy per worker process.
#
# Requests must carry the page's words: there is no Tesseract in this image (see README.md).
#
# Build:  docker build -t apexe3/ae3-table-extract:gpu .
# Run:    see README.md in this folder

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="AE3-TL (gpu)" \
      org.opencontainers.image.description="Table extraction from page images: AE3-TL on ONNX Runtime + CUDA" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/apexe3/ae3-table-extract" \
      org.opencontainers.image.vendor="Team APEX:E3" \
      org.opencontainers.image.variant="gpu"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

# Tesseract is only used for pages sent without words (scans). It does not slow down
# requests that carry words or use ocr=never.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd \
 && rm -rf /var/lib/apt/lists/*

# uv and Python 3.11, as in setup_gpu.sh
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /usr/local/bin/
RUN uv python install 3.11 \
 && uv venv /opt/venv --python 3.11 \
 && chmod -R a+rX /opt/python /opt/venv

# The exact package set from the benchmarked environment (pip freeze), minus the editable line
COPY requirements.txt /tmp/requirements.txt
RUN grep -v '^-e ' /tmp/requirements.txt > /tmp/req.txt \
 && uv pip install --python /opt/venv/bin/python -r /tmp/req.txt \
 && rm /tmp/requirements.txt /tmp/req.txt

# The service itself. --no-deps stops pip pulling the CPU onnxruntime over the GPU build,
# which would run about 35x slower with no error.
COPY ae3-tl-build /tmp/ae3-tl-build
RUN uv pip install --python /opt/venv/bin/python --no-deps /tmp/ae3-tl-build \
 && rm -rf /tmp/ae3-tl-build

ENV PATH=/opt/venv/bin:$PATH \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Fail the build if the wrong onnxruntime is installed
RUN python -c "import importlib.metadata as m, onnxruntime as o, sys; \
names = sorted(d.metadata['Name'].lower() for d in m.distributions() if d.metadata['Name'].lower().startswith('onnxruntime')); \
print('onnxruntime packages:', names, o.__version__, o.get_available_providers()); \
sys.exit(0 if names == ['onnxruntime-gpu'] and 'CUDAExecutionProvider' in o.get_available_providers() else 'wrong onnxruntime build')"

# Settings from serve.sh and the benchmark; all can be overridden with -e at run time
ENV AE3_PROVIDER=cuda \
    AE3_BATCH_SIZE=1 \
    AE3_CELL_BUILDER=fast \
    AE3_WEIGHTS_DIR=/data/weights \
    AE3_COPIES=1 \
    AE3_PDF_WORKERS=1 \
    OCR_ENGINE=tesseract

COPY docker/entrypoint.sh /usr/local/bin/ae3-entrypoint
RUN chmod 755 /usr/local/bin/ae3-entrypoint \
 && useradd --system --uid 10001 --create-home --home-dir /home/ae3 ae3 \
 && mkdir -p /data/weights \
 && chown ae3:ae3 /data/weights
USER ae3
VOLUME /data/weights

EXPOSE 8000
# Each copy loads CUDA kernels on start (about a minute), so allow a long start period
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

ENTRYPOINT ["/usr/local/bin/ae3-entrypoint"]
