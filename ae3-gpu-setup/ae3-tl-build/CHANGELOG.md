# Changelog

## 0.2.0

First release of the image and PDF services together.

- One service for page images and PDFs. A PDF page with a text layer is never OCRed.
- Words file in POTATR form, boxes in page pixels, wins over text layer and OCR. XML is parsed
  with `defusedxml`, so entities and external references are refused.
- Jobs for PDFs over the synchronous page limit, with `GET /v1/jobs/{id}` and MCP `job_status`.
- OpenAI-shaped `POST /v1/chat/completions` with an `image_url` or a `file` part.
- Prometheus text at `/metrics`.
- MB, pixel, page and queue limits; a full queue answers 429 with Retry-After.

- Caption and footer strings are built with Kensho's `extract_text_inside_bbox` / `extract_text_from_spans`, copied unchanged from POTATR (MIT notice kept), so the shipped strings are assembled exactly as POTATR's are. Caption F1 at @con is 0.97658, from 0.93424.

### Review fixes in this build

- The API key has no default. `AE3_API_KEY` is required, the service refuses to start without it,
  `docker compose` refuses to start while it is empty, and keys are compared with
  `hmac.compare_digest`. `.env.example` ships it empty, with a comment. New README section on keys.
- The PDF service checks the same key. A request without one gets 401.
- Outbound fetches are off by default (`AE3_ALLOW_OUTBOUND=0`). With `1` the service needs
  `AE3_FETCH_ALLOWLIST` and refuses to start without it. Every hop is checked, every resolved
  address must be public, and the connection goes to the address that was checked; private,
  loopback, link-local and metadata addresses are refused, including on a redirect.
- The PDF service is published on `AE3_PDF_PORT` (8001), so it answers from outside the compose
  network.
- `compose.scale.yml` sets `AE3_THREADS` per copy (cores / copies) instead of leaving ONNX Runtime
  on every core in every copy.
- The service is documented as keeping a short-lived in-process cache. In scaled mode a request
  must carry the image or the `file_url` each time, and should ask for `include_crops=true`.
- Table records carry `model_box`, the box AE3-TL predicts, next to `box` and `content_box`, and
  crops are cut from `model_box`.
- The model is named AE3-TL everywhere: README, model card, `AE3_MODEL_NAME` default and API
  responses.
- `weights/PROVENANCE.json`, `MODEL_CARD.md` and `README.md` name the scored checkpoint:
  `ae3tl_pages_v3_ep1_ema.pth`, sha256
  `47000f3b785b65c4685e283e266ed67be0eccc2018abef8b183e5f903e34f48c`.
- `THIRD_PARTY_NOTICES.md` regenerated from this tree and this image, with Tesseract, Leptonica,
  the Tesseract language models and nginx added, and the GPL/LGPL sentence limited to Python
  packages.
- `LICENSE.txt` is marked as a draft for legal review, with the contact as a placeholder.
- Release items added: `SHA256SUMS` for the tree, `uv.lock`, the 50-page parity test under
  `tests/`, `CHANGELOG.md` and the sizing numbers in `docs/SIZING.md`.
- `tests/` now covers the key, the PDF service key, the SSRF refusals, the words XML parser, the
  table record and the parity run.

## 0.1.0

- First preview: page images only, MCP `extract_tables_from_image`, CPU image with ONNX Runtime.
