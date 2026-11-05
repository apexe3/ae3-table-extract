# AE3-TL service

The HTTP service behind the AE3-TL Docker images. Send one page image, get every table back: the
cells and the text of each cell, the table, row and column boxes, the caption, the footer and a
score. The model is AE3-TL, run through ONNX Runtime on CUDA or on the CPU; no PyTorch is needed.

Text comes from a words file you supply (the fast, exact path for digital documents) or from
Tesseract (CPU image, for scanned pages). PDF input is off by default; `AE3_PDF_INPUT=1` turns it on.

This folder is the service source. To build, publish and run the images, see the deployment guide
in [`../README.md`](../README.md).

## Run it

```bash
docker run -d --name ae3-tl --gpus all -p 8000:8000 \
  -e AE3_API_KEY=$(openssl rand -hex 24) \
  -v ae3tl-weights:/data/weights \
  apexe3/ae3-table-extract:gpu
```

Use `apexe3/ae3-table-extract:cpu` without `--gpus` on a machine with no NVIDIA GPU. The weights
(about 126 MB) download into the volume on first start. The service refuses to start without
`AE3_API_KEY`.

## API keys

Every request must carry the key in `AE3_API_KEY`, as `X-API-Key: <key>` or as
`Authorization: Bearer <key>`. There is no default key and no way to turn the check off:

- `AE3_API_KEY` empty or missing: the service logs `AE3_API_KEY is required` and exits.
- A request with no key, or the wrong key, gets `401` with
  `{"error": {"code": "unauthorized", ...}}`.
- The key is compared in constant time (`hmac.compare_digest`).
- `/health`, `/ready` and `/metrics` stay open so container health checks and scrapers work.

Give each client environment its own key, so one key can be rotated without touching the others.
Keep the key out of client-side code.

## Call it

```bash
# Digital page with its words: the fast path (GPU image)
curl -s -H "X-API-Key: $AE3_API_KEY" -X POST http://localhost:8000/v1/extract \
  -F file=@page.png -F words=@page_words.json -F ocr=never -F rotation=0 -F format=markdown

# Scanned page, Tesseract reads the words (CPU image)
curl -s -H "X-API-Key: $AE3_API_KEY" -X POST http://localhost:8000/v1/extract \
  -F file=@examples/sample.png -F format=markdown

AE3_API_KEY=$AE3_API_KEY python examples/openai_example.py examples/sample.png
```

| input | how it is read |
|---|---|
| PNG, JPEG, WebP, single-page TIFF | EXIF orientation, Tesseract words |
| any of the above + words file | the supplied words, no OCR |

## Endpoints

| method | path | purpose |
|---|---|---|
| POST | `/v1/extract` | an upload, `file_id` or `file_url`; `format`, `mode`, `ocr`, `include_crops` |
| POST | `/v1/jobs` | a long request as a job |
| GET | `/v1/jobs/{id}` | job status, and the extraction once it is done |
| POST | `/v1/files` | upload once, get a `file_id` |
| POST | `/v1/chat/completions` | an `image_url` part or a `file` part, OpenAI response shape |
| POST | `/mcp` | MCP: `extract_tables`, `job_status` |
| GET | `/v1/tables/{id}/crop` | the table as a PNG, when `AE3_ENABLE_CROP=1` |
| GET | `/health`, `/ready`, `/v1/models`, `/metrics` | operations |

Limits are on MB, pixels, pages and queue depth; a full queue answers `429` with `Retry-After`.

## Response

```json
{"object": "ae3.extraction", "service": "ae3-image", "kind": "image",
 "model": {"id": "AE3-TL", "version": "v3-ep1"},
 "file_id": "82ab6fb29764", "format": "json", "mode": "full",
 "source": "ocr", "pages_total": 1, "pages_returned": 1, "tables_total": 1, "elapsed_ms": 2465.0,
 "pages": [{"page": 1, "width": 1654, "height": 2339, "source": "ocr",
   "tables": [{"id": "82ab6fb29764-t0", "index": 0,
     "box": [325.4, 1075.7, 1137.2, 1557.0],
     "model_box": [325.4, 1075.7, 1137.2, 1557.0],
     "content_box": [325.0, 1084.0, 1136.0, 1549.0],
     "label": "table", "score": 0.9881, "rows": 14, "columns": 2,
     "caption": "Table 1. Characteristics of the study population.", "footer": "",
     "cells": [{"row": 0, "column": 0, "row_span": 1, "col_span": 1,
                "box": [325.0, 1084.0, 1136.0, 1104.0], "text": "Characteristic", "header": true}],
     "source": "ocr", "body": "| Characteristic | n |"}]}]}
```

`source` is `ocr`, `provided_words` or `none`; with OCR on, each cell also carries
`ocr_confidence`. `mode=summary` drops cells and bodies and keeps ids and metadata.

### The three table boxes

| field | what it is |
|---|---|
| `model_box` | the table box AE3-TL predicts for that table, in page pixels. Crops are cut from it. |
| `box` | the table box of the response. Same value as `model_box` in this release. |
| `content_box` | the box the cell builder derives from the detected rows and columns. It hugs the text, so it can be a few pixels inside `model_box`. |

On `examples/sample.png` the model's Table 1 box is `[325.4, 1075.7, 1137.2, 1557.0]`, and the
`content_box` derived from its rows and columns is `[325.0, 1084.0, 1136.0, 1549.0]`. Those
numbers come from the shipped checkpoint; the ruled table on that page is wider than the box the
model predicts, which is a model property, not a post-processing step that narrows it.

## State, and what it means at scale

The service keeps a short-lived in-process cache of recent uploads, and jobs live in memory. So
`file_id`, jobs and crop all belong to the copy that answered. It is not stateless, and the
caveat matters as soon as there is more than one copy:

- **One copy:** a `file_id` works for `AE3_CACHE_TTL_S` (30 minutes) while the upload fits in
  `AE3_CACHE_MB` (256 MB), a job can be polled, and crop can cut from the upload.
- **Several copies (`AE3_COPIES` above 1):** any copy may answer. A request must carry the image or
  the `file_url` itself each time. Do not rely on `file_id`, on the crop URL, or on a job poll
  landing on the copy that started the job. Ask for `include_crops=true` if you need crops, so they
  come back with the extraction.
- A `file_id` sent to a copy that does not hold it gets `404 file_not_found`; a crop gets
  `404 crop_unavailable`. Both are expected with more than one copy.

`AE3_COPIES` starts that many worker processes behind port 8000, each with its own model on the GPU,
and splits the CPU cores between them (`AE3_THREADS`). See the deployment guide for how many fit.

## The cell builder

Cells come from one of two builders, chosen with `AE3_CELL_BUILDER`:

- `fast` (the default) is the vectorised builder in `src/ae3_tl/postprocess/cell_fast.py`.
- `legacy` is the per-object builder in `src/ae3_tl/postprocess/postprocess.py`, kept so a run can
  be compared against an older build.

Both give the same cells. On the 50 pinned parity pages, `tests/parity/builders.py` finds no
difference in any of the 64 tables, and post-processing takes 3.6 ms/page with `fast` against
26.25 ms/page with `legacy` (median, one thread). Any other value is refused with a `ValueError`.

## Measured

| setup | result |
|---|---|
| one page with its words, warm, one request at a time (A100, GPU image) | 190 to 210 ms per page |
| 3 copies on half of an A100 (shared with a vLLM model), 24 parallel clients, words attached | 37.7 pages/s, about 136,000 pages/hour, 0 errors in 300 |
| same, without words (structure only) | 46.5 pages/s, about 167,000 pages/hour |
| 1, 4 and 8 copies on an RTX 5090, PubTables-v2 test pages, OCR off | 29.1, 46.9 and 70.0 pages/s |
| one page image with Tesseract OCR, 200 DPI, CPU image | about 2.4 s |
| accuracy, 100 val pages, dataset images, Tesseract text | GriTS-Top 0.98508, GriTS-Con 0.85187, exact table 0.7967 |

The benchmark accuracy of AE3-TL on PubTables-v2 (with the dataset's own words) is on the
[model card](https://huggingface.co/APEXE3HQ/ae3-tl).

## Limits

- Text quality follows the source: a supplied words file is exact; Tesseract makes OCR errors.
- The GPU image has no Tesseract: send the words with every request.
- Heavy perspective is not corrected. Rotation is handled in quarter turns; tilt up to 8 degrees is
  deskewed.
- PDF input is off by default and has not been benchmarked yet. Office inputs are not read.
- The crop endpoint is off by default (`AE3_ENABLE_CROP=1` turns it on).
- Very small tables (under about 3 rows by 3 columns) and tables that span two pages can be missed
  or split.

## Licence

Apache License 2.0, see `LICENSE.txt`. Third-party components and their licences are listed in
`THIRD_PARTY_NOTICES.md`; `src/ae3_tl/postprocess/postprocess.py` and
`src/ae3_tl/postprocess/kensho_text.py` keep their MIT notices.

## Model provenance

The ONNX weights were exported from the EMA checkpoint `ae3tl_pages_v3_ep1_ema.pth`, sha256
`47000f3b785b65c4685e283e266ed67be0eccc2018abef8b183e5f903e34f48c`, the checkpoint behind the
published benchmark results. Parity evidence for the export:

- the 2,000-page PyTorch versus ONNX gate found 0 pages with a different table count, 0 pages with
  a different cell count, and 17 cells of 137,687 that differ;
- the 50-page array parity in `tests/parity/` replays 50 pages through the ONNX files and compares
  every output array to the pinned v3 arrays.
