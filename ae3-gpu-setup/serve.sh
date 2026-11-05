#!/bin/bash
# AE3-TL — Open-Source Table Extraction Toolkit
# Copyright 2026 Team APEX:E3
# Licensed under the Apache License, Version 2.0; see LICENSE in the repository root.
#
# Starts the GPU service installed by setup_gpu.sh, with the same start-up as the Docker image:
# weights downloaded once, a check that the model really runs on CUDA, then AE3_COPIES worker
# processes on port 8000.
#
#   AE3_API_KEY=<key> AE3_COPIES=3 bash serve.sh
set -euo pipefail

VENV="${AE3_VENV:-/opt/ae3-venv}"
SITE="$VENV/lib/python3.11/site-packages/nvidia"
KIT="$(cd "$(dirname "$0")" && pwd)"

# cuDNN, cuBLAS and NVRTC come from the pip packages, not from the system
export LD_LIBRARY_PATH="$SITE/cudnn/lib:$SITE/cublas/lib:$SITE/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export PATH="$VENV/bin:$PATH"

# Benchmark settings; override any of them in the environment
export AE3_PROVIDER="${AE3_PROVIDER:-cuda}"
export AE3_BATCH_SIZE="${AE3_BATCH_SIZE:-1}"
export AE3_CELL_BUILDER="${AE3_CELL_BUILDER:-fast}"
export AE3_WEIGHTS_DIR="${AE3_WEIGHTS_DIR:-$HOME/.cache/ae3-tl/weights}"
export AE3_PDF_WORKERS="${AE3_PDF_WORKERS:-1}"
export OCR_ENGINE="${OCR_ENGINE:-none}"

# The Docker entrypoint does the rest: key check, threads per copy, weights, GPU check, uvicorn
exec bash "$KIT/docker/entrypoint.sh"
