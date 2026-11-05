#!/bin/bash
# AE3-TL — Open-Source Table Extraction Toolkit
# Copyright 2026 Team APEX:E3
# Licensed under the Apache License, Version 2.0; see LICENSE in the repository root.
#
# Installs the GPU service without Docker, on a Linux machine with an NVIDIA GPU and a CUDA 12
# runtime (for example a RunPod or cloud GPU image). It installs exactly what the Docker image
# installs: Python 3.11 through uv, the pinned packages in requirements.txt, then the service
# with --no-deps so the CPU onnxruntime can never replace the GPU build.
#
#   bash setup_gpu.sh                    # venv at /opt/ae3-venv
#   AE3_VENV=$HOME/ae3-venv bash setup_gpu.sh
#
# Then start it with serve.sh.
set -euo pipefail

VENV="${AE3_VENV:-/opt/ae3-venv}"
KIT="$(cd "$(dirname "$0")" && pwd)"

# uv installs Python 3.11 and the packages much faster than pip
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv python install 3.11
uv venv "$VENV" --python 3.11
uv pip install --python "$VENV/bin/python" -r "$KIT/requirements.txt"
uv pip install --python "$VENV/bin/python" --no-deps "$KIT/ae3-tl-build"

# The one check that matters: the GPU build of onnxruntime, and nothing else, must be installed
"$VENV/bin/python" - <<'PY'
import importlib.metadata as m, onnxruntime as o, sys
names = sorted(d.metadata["Name"].lower() for d in m.distributions()
               if d.metadata["Name"].lower().startswith("onnxruntime"))
print("onnxruntime packages:", names, "providers:", o.get_available_providers())
sys.exit(0 if names == ["onnxruntime-gpu"] and "CUDAExecutionProvider" in o.get_available_providers()
         else "wrong onnxruntime build: reinstall with the commands in setup_gpu.sh")
PY

echo "Installed into $VENV. Start the service with:  AE3_API_KEY=<key> AE3_VENV=$VENV bash serve.sh"
