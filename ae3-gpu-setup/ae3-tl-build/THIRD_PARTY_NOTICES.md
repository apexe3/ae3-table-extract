# Third-party notices

AE3-TL is released under the Apache License 2.0 (`LICENSE.txt`). It builds on the components
below, each under its own licence. Full licence texts are in `licenses/`.

## Model and weights

`ae3tl_pages_v3_ep1_fp32_opset17.onnx` and `ae3tl_relhead_v3_ep1_fp32_opset17.onnx` are the AE3-TL
weights, released by Team APEX:E3 under the Apache License 2.0 and downloaded at start-up from the
Hugging Face repository `APEXE3HQ/ae3-tl`. Both were exported from `ae3tl_pages_v3_ep1_ema.pth`,
sha256 47000f3b785b65c4685e283e266ed67be0eccc2018abef8b183e5f903e34f48c. They were trained on
PubTables-1M and PubTables-v2 (CDLA-Permissive-2.0). No POTATR weights are included.

## Table Transformer post-processing

`src/ae3_tl/postprocess/postprocess.py` is vendored from Kensho's POTATR release, which derives it
from Microsoft's Table Transformer (https://github.com/microsoft/table-transformer). MIT, Copyright
(C) 2021 Microsoft Corporation. The only modification is that the small `Rect` class in the file
replaces the rectangle type the original imported.

## Runtime

| component | licence |
|---|---|
| ONNX Runtime | MIT |
| FastAPI, pydantic, pdfplumber, pdfminer.six | MIT |
| Starlette, Uvicorn, httpx, numpy, PDFium | BSD-3-Clause |
| Pillow | MIT-CMU |
| pypdfium2 | Apache-2.0 OR BSD-3-Clause |
| python-multipart | Apache-2.0 |
| certifi (CA bundle, unmodified) | MPL-2.0 |
| CPU image base: `python:3.11-slim-bookworm` (Debian 12) | free licences per package |
| GPU image base: `nvidia/cuda:12.4.1-runtime-ubuntu22.04` | NVIDIA Deep Learning Container License, plus Ubuntu package licences |
| cuDNN, cuBLAS, NVRTC (`nvidia-*-cu12` pip packages, GPU image) | NVIDIA Software License Agreement (redistributable) |
| Tesseract OCR 5.5.0 (tesseract-ocr, libtesseract5) | Apache-2.0 |
| Tesseract eng language model (tesseract-ocr-eng) | Apache-2.0 |
| Tesseract osd language model (tesseract-ocr-osd) | Apache-2.0 |
| Leptonica (libleptonica6) | BSD-2-Clause |
| libgomp1 (GCC OpenMP runtime, Debian) | GPL-3.0-or-later WITH GCC Runtime Library Exception 3.1 |

## GPL and LGPL

The Python packages carry no GPL or LGPL component. That claim is true for the Python packages
only: the Debian and Ubuntu bases carry LGPL system libraries and libgomp1 (GPL-3.0-or-later WITH GCC Runtime
Library Exception 3.1), all dynamically linked.

The GCC Runtime Library Exception permits linking libgomp1 into a proprietary program. Tesseract
and its language models are Apache-2.0 and Leptonica is BSD-2-Clause. The NVIDIA CUDA, cuDNN and
cuBLAS libraries in the GPU image are redistributed unmodified under NVIDIA's licences. Tesseract and the models are used unmodified through
the `tesseract` program, so their licences apply to those parts only.

No PyMuPDF, MuPDF, poppler, Ghostscript or LibreOffice is part of either image or this tree.
