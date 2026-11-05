<!-- A3Code plan · session ses_e6192eda8ffeiwc785gHIQIY3c · updated 2026-11-03T10:07:18.843Z -->

# Plan: GPU Docker Image for AE3-TL

## Goal
Provide a GPU-accelerated Docker image (`usmanapexe3/ae3-table-extract:gpu` / `:0.2.0-gpu`) alongside the published CPU image, plus documented `docker run` commands to serve the AE3-TL HTTP API on NVIDIA GPUs.

## Findings & Decisions
1. **Base image**: `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` (linux/amd64 only).
   - Rationale: onnxruntime-gpu 1.21–1.26 PyPI wheels are built with CUDA 12.8 + cuDNN 9 (per onnxruntime.ai CUDA docs); the `cudnn-runtime` flavor includes the CUDA runtime libs (cudart, cublas, cufft, curand) + cuDNN 9 that the CUDA Execution Provider needs — no pip NVIDIA wheels or LD_LIBRARY_PATH hacks.
   - Ubuntu 22.04 ships Python 3.10, which satisfies the project's `requires-python >= 3.10`.
2. **Provider pin**: set `ENV AE3_PROVIDER=cuda` in the GPU Dockerfile.
   - Rationale: `resolve_provider("auto")` in `src/ae3_tl/backends/onnx_backend.py:80` prefers TensorRT, and the PyPI onnxruntime-gpu wheel advertises `TensorrtExecutionProvider` even though TensorRT libs aren't installed → `auto` would crash at session creation. Explicit `cuda` avoids this (mirrors the CPU image's `AE3_PROVIDER=cpu`).
3. **Package install**: `pip install "onnxruntime-gpu>=1.21,<1.27"` then `pip install .` (the project package). GPU extra exists in pyproject but the direct install with version bound is clearer inside the image.
4. **Tesseract**: `apt-get install tesseract-ocr` as in the CPU image.
5. **No code changes**: device selection (`AE3TL_ONNX_DEVICE_ID`), TF32 (`AE3_TF32`), deterministic mode (`AE3_DETERMINISTIC`), and threads (`AE3TL_ONNX_THREADS`/`AE3_THREADS`) are already runtime env vars.
6. **Tagging**: `usmanapexe3/ae3-table-extract:0.2.0-gpu` and `:gpu`. Existing CPU tags (`0.2.0`, `latest`) stay as-is.
7. **Weights**: same `/data/weights` volume pattern; HF token passed at runtime (`-e AE3_HF_TOKEN=...`) while the weights repo requires it.

## Approach / Steps
1. Create `Dockerfile.gpu`:
   - FROM `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04`
   - Install `python3-pip` + `tesseract-ocr`; `pip install "onnxruntime-gpu>=1.21,<1.27" .`
   - Copy `pyproject.toml README.md LICENSE NOTICE src/`
   - `ENV AE3_PROVIDER=cuda`, `AE3_WEIGHTS_DIR=/data/weights`, volume, EXPOSE 8000, same HEALTHCHECK, `CMD ["ae3-tl", "serve", "--host", "0.0.0.0", "--port", "8000"]`
   - OCI labels matching the CPU image + a `variant=gpu` label
2. Update `README.md`: add a "Docker" section with CPU and GPU run commands:
   - CPU: `docker run -d -p 8000:8000 -e AE3_API_KEY=... -e AE3_HF_TOKEN=... -v ae3-weights:/data/weights usmanapexe3/ae3-table-extract:latest`
   - GPU: `docker run -d --gpus all -p 8000:8000 -e AE3_API_KEY=... -e AE3_HF_TOKEN=... -v ae3-weights:/data/weights usmanapexe3/ae3-table-extract:gpu`
   - Notes: NVIDIA Container Toolkit required on host; CUDA 12.8 needs driver ≥ 570.x; provider override env (`AE3_PROVIDER`) documented.
3. Commit `Dockerfile.gpu` + README changes; push to GitHub (main).
4. Build: `docker build --platform linux/amd64 -f Dockerfile.gpu -t usmanapexe3/ae3-table-extract:0.2.0-gpu -t usmanapexe3/ae3-table-extract:gpu .`
5. Verify in-container (no GPU present locally): `docker run --rm <img> python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"` → expect `CUDAExecutionProvider` listed; check `ae3-tl --version` and tesseract present.
6. Push both GPU tags to Docker Hub.
7. Optional: re-tag/push nothing else; CPU image untouched.

## Open Questions
- **GPU inference validation**: this machine has no NVIDIA GPU; end-to-end GPU extraction (provider `cuda`, latency) must be confirmed on an NVIDIA host or CI. I can verify build + provider listing only.
- **Driver baseline**: CUDA 12.8 build requires host driver ≥ 570. If users need older drivers, a fallback variant (`nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` + `onnxruntime-gpu==1.20.*`, CUDA 12.x/cuDNN 9) can be added later.
- **TensorRT variant**: out of scope (needs separate TensorRT install and engine-build warmup); `AE3_PROVIDER=tensorrt` already supported in code if users build their own image on top.
- Confirm push of GPU tags to Docker Hub under `usmanapexe3` (same account already authenticated) — proceeding per user request to "publish".

## Status
- CPU image published: `usmanapexe3/ae3-table-extract:0.2.0` + `:latest` (digest sha256:725454…) — done, tested end-to-end (2 tables, score 0.988 via `/v1/extract`).
- `Dockerfile` + `.dockerignore` exist locally, **not yet committed** to git.
- GPU image: not started — plan above ready for BUILD mode.
