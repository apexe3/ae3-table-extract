#!/bin/bash
# AE3-TL — Open-Source Table Extraction Toolkit
# Copyright 2026 Team APEX:E3
# Licensed under the Apache License, Version 2.0; see LICENSE in the repository root.
#
# Starts AE3-TL with AE3_COPIES model copies (one uvicorn worker each) on port 8000.
# Before starting, it downloads the weights once and proves the model really runs on CUDA,
# so a broken GPU setup stops here instead of silently running 35x slower on the CPU.
set -euo pipefail

if [ -z "${AE3_API_KEY:-}" ]; then
  echo "AE3_API_KEY is required: docker run ... -e AE3_API_KEY=<key>" >&2
  exit 1
fi

export AE3_PROVIDER="${AE3_PROVIDER:-cuda}"
COPIES="${AE3_COPIES:-1}"
CORES="$(nproc)"

# ONNX Runtime CPU threads: every core for one copy, an equal share for several copies
if [ -z "${AE3_THREADS:-}" ]; then
  if [ "$COPIES" -gt 1 ]; then
    THREADS=$(( CORES / COPIES ))
    [ "$THREADS" -lt 1 ] && THREADS=1
    export AE3_THREADS="$THREADS"
  else
    export AE3_THREADS=0
  fi
fi

echo "[ae3] copies=${COPIES} cores=${CORES} threads_per_copy=${AE3_THREADS} provider=${AE3_PROVIDER} batch=${AE3_BATCH_SIZE:-1}"

# 1. Weights: download once, before the copies start, so they do not race each other
python - <<'EOF'
import os
from pathlib import Path
from ae3_tl.weights import ensure_weights
summary = ensure_weights(Path(os.environ["AE3_WEIGHTS_DIR"]))
print("[ae3] weights ready:", summary if isinstance(summary, dict) else "ok")
EOF

# 2. GPU check: open the page model on CUDA and confirm ONNX Runtime kept the CUDA provider
if [ "${AE3_PROVIDER}" = "cuda" ] && [ "${AE3_SKIP_GPU_CHECK:-0}" != "1" ]; then
  python - <<'EOF'
import os, sys
from pathlib import Path
import onnxruntime as ort

model = next(Path(os.environ["AE3_WEIGHTS_DIR"]).glob("ae3tl_pages_*.onnx"), None)
if model is None:
    sys.exit("[ae3] page model not found in " + os.environ["AE3_WEIGHTS_DIR"])
ort.set_default_logger_severity(3)
try:
    session = ort.InferenceSession(str(model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
except Exception as error:
    sys.exit(f"[ae3] could not open the model on CUDA: {error}")
used = session.get_providers()
if not used or used[0] != "CUDAExecutionProvider":
    sys.exit("[ae3] CUDA is NOT in use (providers: %s). Check that the container was started with "
             "--gpus and that the host driver supports CUDA 12. Set AE3_SKIP_GPU_CHECK=1 to start anyway." % used)
print("[ae3] GPU check passed: model runs on CUDAExecutionProvider")
EOF
fi

# 3. Serve: one model copy per worker process, as in the benchmark
exec uvicorn ae3_tl.serve.app:app \
  --host 0.0.0.0 --port 8000 \
  --workers "${COPIES}" \
  --log-level "${AE3_LOG_LEVEL:-info}" \
  --timeout-keep-alive 30
