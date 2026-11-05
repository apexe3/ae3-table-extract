# AE3-TL API

Base URL `http://localhost:8000`. Every `/v1` and `/mcp` request needs `X-API-Key: <key>` (or
`Authorization: Bearer <key>`), where the key is `AE3_API_KEY` in `.env`. `/health`, `/ready` and
`/metrics` need no key.

This release serves page images. PDF input is off: a PDF answers `415` with
`{"error": {"code": "pdf_disabled", "message": "PDF input is off in this release; send a page image"}}`.

## POST /v1/extract

Multipart form. Send exactly one of `file`, `file_id`, `file_url`.

| part | meaning |
|---|---|
| `file` | a page image (PNG, JPEG, WebP, single-page TIFF) |
| `file_id` | an id from `POST /v1/files`; valid on the copy that holds it, for the cache TTL |
| `file_url` | an http(s) URL the service fetches; needs `AE3_ALLOW_OUTBOUND=1` and a host in `AE3_FETCH_ALLOWLIST` |
| `words` | optional words file, XML, JSON or CSV, boxes in page pixels |
| `format` | `json` (default), `markdown`, `html`, `csv` |
| `mode` | `full` (default) or `summary` (no cells or bodies) |
| `ocr` | `auto` (default), `always`, `never` |
| `rotation` | `auto`, `0`, `90`, `180`, `270` |
| `deskew` | `true` (default) removes tilt up to 8 degrees |
| `include_crops` | `true` returns table and cell crops as PNG data URLs in the same response |

## Response

```json
{"object": "ae3.extraction", "kind": "image", "file_id": "82ab6fb29764",
 "source": "ocr", "format": "json", "mode": "full",
 "pages_total": 1, "pages_returned": 1, "tables_total": 1, "elapsed_ms": 2465.0,
 "pages": [{"page": 1, "width": 1654, "height": 2339, "source": "ocr",
   "tables": [{"id": "82ab6fb29764-t0",
     "box": [325.4, 1075.7, 1137.2, 1557.0],
     "model_box": [325.4, 1075.7, 1137.2, 1557.0],
     "content_box": [325.0, 1084.0, 1136.0, 1549.0],
     "score": 0.9881, "rows": 14, "columns": 2,
     "caption": "Table 1. ...", "footer": "", "source": "ocr",
     "cells": [{"row": 0, "column": 0, "row_span": 1, "col_span": 1,
                "box": [325.0, 1084.0, 1136.0, 1104.0], "text": "Characteristic", "header": true,
                "ocr_confidence": 0.93}]}]}]}
```

`source` is `ocr`, `provided_words` or `none`. `ocr_confidence` appears on cells when the page was
read with Tesseract.

### Table boxes

- `model_box`: the box AE3-TL predicts for the table, in page pixels. Crops are cut from it.
- `box`: the box of the record. Same value as `model_box` in this release.
- `content_box`: the box the cell builder derives from the detected rows and columns. It hugs the
  text, so it can sit a few pixels inside `model_box`.

### Crops

`AE3_ENABLE_CROP=1` turns on `GET /v1/tables/{table_id}/crop`. Without `box`, the endpoint cuts
`model_box` of that table from the upload the copy still holds. With `box=x0,y0,x1,y1` it cuts that
box instead. A copy that does not hold the upload answers `404 crop_unavailable`; pass `box`, or ask
for `include_crops=true` with the extraction.

## Other endpoints

- `POST /v1/jobs` same form, always a job; `GET /v1/jobs/{id}` returns status and the result.
- `POST /v1/files` multipart `file`, returns `{"id": "<sha12>"}`.
- `POST /v1/chat/completions` OpenAI shape: content parts `image_url` or `file` (data URL or URL).
- `POST /mcp` JSON-RPC: `extract_tables`, `job_status`.
- `GET /health`, `/ready`, `/v1/models`, `/metrics`.

## State, and what it means when you run more than one copy

`file_id`, jobs and crop all live in the memory of the copy that answered. They are not shared
between copies and they are lost on restart. With one copy that is invisible. With `AE3_COPIES` above 1,
a later request can land on another copy and find nothing:

- send the image, or the `file_url`, with every request;
- ask for `include_crops=true` if you need crops, so they come back with the extraction;
- poll a job on the copy that started it, or keep jobs to a single copy;
- do not call the service stateless without this caveat.

## Errors

`{"error": {"code": "...", "message": "..."}}` with 4xx. `401` for a missing or wrong key, `403`
for a URL that is not allowed, `415` for a PDF, `429` with `Retry-After` for a full queue.
