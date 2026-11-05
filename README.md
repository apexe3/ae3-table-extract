# AE3-TL — Open-Source Table Extraction

Extract tables from images (PNG, JPEG, WebP, …) as structured JSON, Markdown,
HTML or CSV. AE3-TL combines a deep-learning table-structure model (ONNX
Runtime) with Tesseract OCR to reconstruct the rows, columns, headers, spans,
captions and footers of every table on a page.

**Free and open source under the Apache License 2.0 — you can use it in
commercial products, modify it and redistribute it.**
Created and maintained by **Team APEX:E3**.

## Research paper

The method behind AE3-TL — a table-structure network that predicts
row/column boxes, headers, spans, captions and footers, joined with
relation links and merged with OCR word boxes — is described in the
bundled research paper [`AE3-TL_2026.pdf`](AE3-TL_2026.pdf). Reading it
first explains *why* the pipeline looks the way it does: what the model
predicts, how the relation head links rows to captions and footers, and
how cells are assembled from model geometry plus OCR text.

---

## What's in this folder

```
ae3-table-extract/
├── README.md              ← you are here
├── AE3-TL_2026.pdf        ← the research paper
├── LICENSE                ← Apache 2.0 (open source, commercial use allowed)
├── NOTICE                 ← attribution notice (keep with redistributions)
├── requirements.txt       ← Python dependencies
├── pyproject.toml         ← installable as a package (pip install -e .)
├── Dockerfile             ← lightweight CPU image of the library in src/
├── Dockerfile.gpu         ← lightweight GPU image of the library in src/
├── .gitignore
├── examples/
│   ├── run_local.py       ← weights download + image → tables (JSON/MD/HTML/CSV)
│   ├── extract_to_md.py   ← tables JSON + source image → a Markdown report
│   ├── sample.png         ← a real page to try it on
│   └── sample_words.json  ← its OCR words, for the --words flow
├── src/
│   └── ae3_tl/            ← the table-extraction library
└── ae3-gpu-setup/         ← the production GPU service and its Docker images
    ├── README.md          ← GPU deployment guide (start here to deploy)
    ├── Dockerfile         ← apexe3/ae3-table-extract:gpu
    ├── Dockerfile.cpu     ← apexe3/ae3-table-extract:cpu
    ├── build_and_push.sh  ← builds and publishes both images
    ├── scripts/make_words.py ← PDFs or scans → page images + words files
    ├── scripts/bench.py   ← throughput and latency of a running service
    └── ae3-tl-build/      ← the service source (API, runtime, tests)
```

## How it works

```
                        ┌──────────────────────────┐
                        │   page image (PNG/JPEG)  │
                        └────────────┬─────────────┘
                                     │
              ┌──────────────────────┴──────────────────────┐
              ▼                                             ▼
┌─────────────────────────────┐              ┌──────────────────────────────┐
│  OCR — Tesseract            │              │  Model — AE3-TL ONNX         │
│                             │              │  (weights auto-downloaded    │
│  reads every word:          │              │   from Hugging Face on first │
│  text + bounding box        │              │   run, ~126 MB)              │
│  (small pages upscaled      │              │                              │
│   automatically)            │              │  predicts the structure:     │
│                             │              │  table/row/column boxes,     │
│                             │              │  headers, spans, captions,   │
│                             │              │  footers + relation links    │
└──────────────┬──────────────┘              └───────────────┬──────────────┘
               │                                             │
               │            ┌────────────────────────┐       │
               └───────────►│  Cell builder          │◄──────┘
                            │  merges OCR words with │
                            │  model boxes into      │
                            │  complete cells        │
                            └───────────┬────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │  Output                │
                            │  JSON · Markdown ·     │
                            │  HTML · CSV            │
                            └───────────┬────────────┘
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
        ┌─────────────────────────┐          ┌──────────────────────────┐
        │  tables.json            │          │  extract_to_md.py        │
        │  (run_local.py -o)      │          │  renders a Markdown      │
        │                         │          │  report, optionally      │
        │                         │          │  embedding table crops   │
        └─────────────────────────┘          └──────────────────────────┘
```

1. **OCR** — Tesseract reads every word on the page with its bounding box
   (small images are upscaled automatically first).
2. **Model** — the AE3-TL neural network (ONNX, ~126 MB) predicts the table
   structure: table/row/column boxes, captions, footers and relation links.
3. **Cell builder** — model boxes are merged with the OCR words into complete
   cells: row/column indices, header flags, row/column spans, captions.

OCR quality drives the text; the model drives the structure.

## Setup

### 1. Python 3.10+

```bash
python3 --version
```

### 2. Install the Python dependencies

```bash
cd ae3-table-extract
python3 -m pip install -r requirements.txt
```

### 3. Install the Tesseract binary (system dependency)

The OCR step calls the `tesseract` program — it is not a pip package.

| OS | Install |
|---|---|
| macOS | `brew install tesseract` |
| Debian / Ubuntu | `sudo apt install tesseract-ocr` |
| Fedora | `sudo dnf install tesseract` |
| Windows | installer from <https://github.com/UB-Mannheim/tesseract/wiki> (add its folder to `PATH`) |
| Docker | `apt-get install -y tesseract-ocr` in your image |

Check it works:

```bash
tesseract --version
```

### 4. Add `src` to `PYTHONPATH`

The examples add the bundled `src/` folder to `sys.path` themselves, so for
the command-line tools nothing more is needed. To import the library in your
own code, either set:

```bash
export PYTHONPATH=/path/to/ae3-table-extract/src
```

or install it in editable mode:

```bash
pip install -e .
```

### 5. Model weights (downloaded automatically)

On first run the ONNX weights (~126 MB) are fetched from the public Hugging
Face repository `APEXE3HQ/ae3-tl` into `./weights` next to the source tree.
No account or token is needed. To pre-download them:

```bash
python examples/run_local.py download
```

## Quick start

### Read tables from an image

A sample image ships in `examples/sample.png` — try it right away:

```bash
# JSON (printed, or written with -o)
python examples/run_local.py tables examples/sample.png -o tables.json

# Markdown table straight to the terminal
python examples/run_local.py tables page.png --format markdown

# An image on the web instead of a local file
python examples/run_local.py tables --url https://example.com/report.png -o tables.json
```

### Turn the JSON + image into a polished Markdown report

```bash
python examples/extract_to_md.py --json tables.json --image page.png -o tables.md

# Also embed a cropped screenshot of each table
python examples/extract_to_md.py --json tables.json --image page.png --pictures -o tables.md
```

### Use it from Python

```python
import sys
sys.path.insert(0, "src")  # or pip install -e . and drop this line

from ae3_tl.backends import create_engine
from ae3_tl.image import service
from ae3_tl.serve.settings import load_settings

settings = load_settings()
engine = create_engine(settings)          # loads ONNX Runtime + downloads weights once

with open("page.png", "rb") as fh:
    data = fh.read()

document = service.extract(engine, settings, data, fmt="json", rotation="auto")
with open("tables.json", "w") as fh:
    import json
    json.dump(document, fh, indent=2)
```

`service.extract` accepts `fmt="json"`, `"markdown"`, `"html"` or `"csv"`.

## Docker

The production service is published on Docker Hub as **`apexe3/ae3-table-extract`**:

| tag | for |
|---|---|
| `:gpu`, `:0.23-gpu` | NVIDIA GPUs (x86_64, CUDA 12, driver 550+). ~190 ms a page, 37.7 pages/s on half an A100 |
| `:cpu`, `:0.23-cpu` | any machine (x86_64 and ARM). Includes Tesseract, so it also reads scanned pages |

Full instructions (sizing, sharing a GPU with vLLM, security, building, troubleshooting) are in
the **[GPU deployment guide](ae3-gpu-setup/README.md)**. The short version, on a GPU server with
the NVIDIA Container Toolkit:

```bash
openssl rand -hex 24 > ~/ae3-tl.key && chmod 600 ~/ae3-tl.key

docker run -d --name ae3-tl --restart unless-stopped \
  --gpus all -p 8000:8000 --shm-size 1g \
  -e AE3_API_KEY=$(cat ~/ae3-tl.key) \
  -e AE3_COPIES=3 \
  -v ae3tl-weights:/data/weights \
  apexe3/ae3-table-extract:gpu

docker logs -f ae3-tl        # wait for "GPU check passed" and one startup line per copy
```

Then send a page **with its words** (the GPU image has no OCR). To make words files for your own
PDFs or scans, use `ae3-gpu-setup/scripts/make_words.py` (see the guide):

```bash
curl localhost:8000/ready
curl -s localhost:8000/v1/extract -H "X-API-Key: $(cat ~/ae3-tl.key)" \
  -F "file=@examples/sample.png" -F "words=@examples/sample_words.json" \
  -F "ocr=never" -F "rotation=0" -F "format=markdown"
```

Notes:

- `AE3_COPIES` is the number of model copies on the GPU, about 3 GB each.
- For scanned pages without words, run `apexe3/ae3-table-extract:cpu` instead (no `--gpus`).
  It runs Tesseract on the CPU: seconds per page rather than milliseconds.
- The `Dockerfile` and `Dockerfile.gpu` at the repo root build a lighter image of the
  library in `src/` (`ae3-tl serve`), handy for development:
  `docker build -t ae3-tl .` or `docker build -f Dockerfile.gpu -t ae3-tl-gpu .`.
- Build the GPU image on an x86_64 Linux server; on an Apple-silicon Mac it runs under
  emulation and is very slow.

## Output shape

`tables.json` looks like this (trimmed):

```json
{
  "model": {"id": "AE3-TL", "version": "0.2.0"},
  "words": 214,
  "elapsed_ms": 1830,
  "tables_total": 1,
  "tables": [
    {
      "index": 1,
      "page": 1,
      "rows": 4,
      "columns": 5,
      "score": 0.94,
      "caption": "Quarterly revenue",
      "box": [52.0, 118.0, 602.0, 388.0],
      "cells": [
        {"row": 0, "column": 0, "text": "Region", "header": true,
         "row_span": 1, "col_span": 1},
        {"row": 1, "column": 2, "text": "1,204", "row_span": 1, "col_span": 1}
      ]
    }
  ]
}
```

Every cell has `row`, `column`, `text`, optional `header`/`subheader` flags,
`row_span`/`col_span` and a confidence `score`.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `AE3_WEIGHTS_DIR` | `./weights` | where the ONNX weights live |
| `AE3_WEIGHTS_REPO` | `APEXE3HQ/ae3-tl` | Hugging Face repo to fetch weights from |
| `AE3_HF_TOKEN` / `HF_TOKEN` | unset | only needed while the weights repo is private |
| `AE3_PROVIDER` | `auto` | ONNX execution: `auto`, `cpu`, `cuda`, `tensorrt` or `coreml` |
| `AE3_THREADS` | 0 (auto) | ONNX Runtime intra-op threads |
| `OCR_ENGINE` | `tesseract` | `tesseract` or `none` (structure only, no text) |
| `AE3_ENABLE_CROP` | off | allow the model to return cell image crops |

`AE3_PROVIDER=coreml` on Apple-silicon Macs and `cuda` on NVIDIA GPUs make
large batches noticeably faster; `cpu` is fine for one image at a time.

## Using a different OCR engine

Tesseract is the built-in OCR, but it is not the only option. The model only
needs *word boxes* — a list of `{text, bbox, block_num, line_num, span_num}`
records — so any OCR that can produce them can replace it:

| Alternative | Notes |
|---|---|
| **PaddleOCR** (`pip install paddleocr`) | often stronger on non-Latin scripts and stylised text; exports word boxes in JSON |
| **EasyOCR** (`pip install easyocr`) | 80+ languages, simple Python API; returns per-word boxes |
| **Google Cloud Vision / AWS Textract / Azure Document Intelligence** | cloud APIs with high accuracy; Textract also reads tables natively |
| **Any other engine** | anything that writes a JSON/CSV/XML words file — see the accepted formats below |

Once you have word boxes, skip Tesseract entirely with `--words`:

```bash
python examples/run_local.py tables page.png --words my_ocr_words.json -o tables.json
```

Accepted words formats (auto-detected from the bytes):

- **JSON** — `[{"text": ..., "bbox": [x0, y0, x1, y1], "block_num": ..., "line_num": ..., "span_num": ...}, ...]`
- **CSV** — a header with `text` and `bbox` (`"x0,y0,x1,y1"` or four columns)
- **XML** — `<words><word bbox="x0,y0,x1,y1">text</word></words>`

Boxes must be in the pixel frame of the image you pass in. A ready-made
`examples/sample_words.json` (Tesseract output for the sample page) shows the
exact shape. Word quality is what drives cell text, so a better OCR for your
documents directly improves the extraction.

## Tips, limits & troubleshooting

- **Blank or wrong text** — OCR is the bottleneck. Higher-resolution images
  (≈150 dpi or better) read far better than small thumbnails. Very small
  images are auto-upscaled 2×–4× before OCR, which fixes many cases.
- **White text on dark backgrounds** — Tesseract struggles; try inverting
  the image first.
- **`tesseract: command not found`** — the binary is missing or not on
  `PATH`; see step 3 of Setup.
- **No tables found** — check `OCR_ENGINE` is not `none` and that the image
  actually contains a table; the model reports structure scores in the JSON.
- **Private weights repo** — pass `--token` or set `AE3_HF_TOKEN`.
- **Slow first run** — the weights download once (~126 MB) and are reused
  from disk afterwards.

## License & attribution

Released under the [Apache License 2.0](LICENSE) — free for commercial use.
Every source file carries the license header; [`NOTICE`](NOTICE) must stay
with redistributions. Third-party model weights from the `APEXE3HQ/ae3-tl`
repository are subject to their own license. Created by **Team APEX:E3**.
