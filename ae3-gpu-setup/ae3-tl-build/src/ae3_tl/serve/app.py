# AE3-TL — Open-Source Table Extraction Toolkit
# Copyright 2026 Team APEX:E3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This software is free and open source: you may use, modify and distribute
# it, including in commercial products, under the terms of the License.
# AE3-TL is created and maintained by Team APEX:E3.

"""ae3-image: one page image or one PDF in, every table out, plus the same tool over MCP.

    POST /v1/extract            image or PDF, by upload, file_id or file_url
    POST /v1/jobs               a long PDF as a job
    GET  /v1/jobs/{job_id}      job status, and the extraction once it is done
    POST /v1/files              upload once, get a file_id
    POST /v1/chat/completions   OpenAI-shaped request with an image_url or a file part
    POST /v1/responses          OpenAI Responses API: image in, tables out
    GET  /health /ready /v1/models /metrics
    POST /mcp                   MCP: extract_tables, job_status

No request writes to disk. Uploads live in a bounded in-process cache so a file_id works across
two calls; a job lives in memory for its TTL.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
import time
import uuid

import numpy as np
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.middleware.gzip import GZipMiddleware

from ae3_tl._version import __version__
from ae3_tl.backends import create_engine
from ae3_tl.backends.onnx_backend import resolve_provider
from ae3_tl.image import decode
from ae3_tl.image import service as image_service
from ae3_tl.pdf.preprocess import page_tensors
from ae3_tl.mcp.mcp_server import McpServer
from ae3_tl.pdf.pdfpool import PdfPool
from ae3_tl.serve import extract as extract_service
from ae3_tl.serve.auth import authorize
from ae3_tl.serve.cache import ImageCache
from ae3_tl.serve.errors import ApiError
from ae3_tl.serve.fetch import fetch
from ae3_tl.serve.jobs import Job, Jobs
from ae3_tl.serve.metrics import Metrics
from ae3_tl.serve.settings import load_settings
from ae3_tl.weights import ensure_weights

STARTED_AT = time.time()
SWEEP_SECONDS = 300
IMAGE_ID = re.compile(r"^[0-9a-f]{12}$")
NUMBER = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")
log = logging.getLogger("ae3.image")


class Batcher:
    """Concurrent single-page runs into one forward pass, the way a serving engine batches.

    A request joins the queue and waits at most `wait_s` for company, up to `max_batch`
    pages. One worker thread runs the whole batch and splits the outputs back.
    """

    def __init__(self, run, max_batch: int, wait_s: float):
        self.run = run
        self.max_batch = max(1, max_batch)
        self.wait_s = max(0.0, wait_s)
        self.queue: asyncio.Queue = asyncio.Queue()
        self.task: asyncio.Task | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self._loop(), name="ae3-batcher")

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()

    async def submit(self, tensors):
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((tensors, future))
        return await future

    async def _loop(self) -> None:
        while True:
            batch = [await self.queue.get()]
            if self.max_batch > 1:
                deadline = time.monotonic() + self.wait_s
                while len(batch) < self.max_batch:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self.queue.get(), remaining))
                    except asyncio.TimeoutError:
                        break
            try:
                stacked = np.concatenate([tensors for tensors, _ in batch], axis=0)
                outputs = await asyncio.to_thread(self.run, stacked)
                for index, (_, future) in enumerate(batch):
                    if not future.done():
                        future.set_result(tuple(o[index:index + 1] for o in outputs))
            except Exception as error:                          # noqa: BLE001
                for _, future in batch:
                    if not future.done():
                        future.set_exception(error)


class Gate:
    """Admission control: a fixed number of extractions run, a bounded number wait.

    A request that finds the wait queue full gets 503 with Retry-After, so load
    is shed at the door instead of piling up. A request that waits too long for a
    slot gets 503 too.
    """

    def __init__(self, limit: int, depth: int, wait_s: float):
        self.limit = max(1, limit)
        self.depth = max(0, depth)
        self.wait_s = max(0.0, float(wait_s))
        self.slots = asyncio.Semaphore(self.limit)
        self.lock = asyncio.Lock()
        self.running = 0
        self.waiting = 0

    async def __aenter__(self) -> "Gate":
        async with self.lock:
            if self.waiting >= self.depth and self.slots.locked():
                raise ApiError(503, "queue_full",
                               f"{self.running} extractions running and {self.waiting} waiting; "
                               "retry shortly")
            self.waiting += 1
        try:
            await asyncio.wait_for(self.slots.acquire(), timeout=self.wait_s)
        except asyncio.TimeoutError:
            raise ApiError(503, "queue_timeout",
                           f"no free slot within {self.wait_s:.0f}s") from None
        finally:
            async with self.lock:
                self.waiting -= 1
        async with self.lock:
            self.running += 1
        return self

    async def __aexit__(self, *exc) -> None:
        async with self.lock:
            self.running -= 1
        self.slots.release()


async def _sweep(state) -> None:
    while True:
        await asyncio.sleep(SWEEP_SECONDS)
        await asyncio.to_thread(state.cache.sweep)
        state.jobs.sweep()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = load_settings()
    settings.require_api_key()
    app.state.settings = settings
    app.state.limiter = RateLimiter(settings.rate_limit_per_min)
    app.state.weights = ensure_weights(settings.weights_dir)
    app.state.engine = create_engine(settings)
    app.state.pool = PdfPool(settings.pdf_workers, page_timeout_s=settings.page_timeout_s)
    app.state.cache = ImageCache(settings.cache_mb * 1024 * 1024, settings.cache_ttl_s)
    app.state.jobs = Jobs(settings.job_limit, settings.job_ttl_s)
    app.state.metrics = Metrics()
    app.state.gate = Gate(settings.concurrency, settings.queue_depth,
                           settings.queue_timeout_s)
    app.state.batcher = Batcher(app.state.engine.network.run, settings.batch_max,
                                settings.batch_wait_ms / 1000.0)
    app.state.batcher.start()
    app.state.mcp = McpServer(app)
    app.state.ready = True
    app.state.warmup_ms = None
    if settings.warmup:
        started = time.perf_counter()
        try:
            await asyncio.to_thread(image_service.warm, app.state.engine)
            app.state.warmup_ms = round((time.perf_counter() - started) * 1000, 1)
        except Exception as error:                              # noqa: BLE001
            log.error("warm-up failed: %s", type(error).__name__)
    sweeper = asyncio.create_task(_sweep(app.state), name="ae3-image-sweeper")
    log.info("ae3-image ready: model=%s version=%s provider=%s ocr=%s threads=%s pdf=%s "
             "warmup_ms=%s", settings.model_name, app.state.engine.config.version,
             settings.onnx_provider, settings.ocr_engine, settings.onnx_threads,
             settings.pdf_input, app.state.warmup_ms)
    try:
        yield
    finally:
        sweeper.cancel()
        await app.state.batcher.stop()
        app.state.pool.close()


app = FastAPI(
    title="ae3-image",
    version=__version__,
    description=("Tables out of one page image or one PDF: cells and cell text, row and column "
                 "boxes, caption, footer, score. A PDF page with a text layer is never OCRed. "
                 "The same tool is served over MCP at /mcp."),
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, error: ApiError) -> JSONResponse:
    headers = {"Retry-After": "5"} if error.status == 503 else None
    return JSONResponse(status_code=error.status, headers=headers,
                        content={"error": {"type": error.code, "code": error.code,
                                           "message": error.message}})


@app.middleware("http")
async def count_requests(request: Request, call_next):
    identifier = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - started
    route = request.scope.get("route")
    label = getattr(route, "path", None) or "unmatched"
    if label != "/metrics":
        request.app.state.metrics.hit(label, response.status_code, elapsed)
        log.info("%s %s %s %s %d %.0fms", identifier,
                 request.client.host if request.client else "-",
                 request.method, label, response.status_code, elapsed * 1000)
    response.headers["X-Request-ID"] = identifier
    return response


class RateLimiter:
    """A per-key sliding window, off when the limit is zero."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        window = self.hits.setdefault(key, [])
        while window and window[0] <= now - 60.0:
            window.pop(0)
        if len(window) >= self.per_minute:
            return False
        window.append(now)
        return True


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    limiter = request.app.state.limiter
    if limiter.per_minute > 0 and (request.url.path.startswith("/v1/") or request.url.path == "/mcp"):
        key = (request.headers.get("x-api-key") or request.headers.get("authorization")
               or (request.client.host if request.client else "unknown"))
        if not limiter.allow(key):
            return JSONResponse(status_code=429,
                                content={"error": {"type": "rate_limited", "code": "rate_limited",
                                                   "message": "too many requests; retry after a minute"}},
                                headers={"Retry-After": "60"})
    return await call_next(request)


@app.middleware("http")
async def body_limit(request: Request, call_next):
    limit = request.app.state.settings.max_body_mb * 1024 * 1024
    raw = request.headers.get("content-length")
    if raw and raw.isdigit() and int(raw) > limit:
        return JSONResponse(status_code=413,
                            content={"error": {"type": "too_large", "code": "too_large",
                                               "message": f"the body is larger than "
                                                          f"{request.app.state.settings.max_body_mb} MiB"}})
    return await call_next(request)


EXTRACTION_PATHS = ("/v1/extract", "/v1/jobs", "/v1/files", "/v1/chat/completions",
                    "/v1/responses", "/mcp", "/v1/tables")


@app.middleware("http")
async def surface_gate(request: Request, call_next):
    """In infer mode only the model surface is served; extraction belongs to the caller."""
    if request.app.state.settings.surface == "infer":
        path = request.url.path
        hit = path == "/mcp" or path.startswith("/mcp/")
        if not hit:
            for prefix in EXTRACTION_PATHS:
                if path == prefix or path.startswith(prefix + "/"):
                    hit = True
                    break
        if hit:
            return JSONResponse(status_code=404,
                                content={"error": {"type": "not_found", "code": "not_found",
                                                   "message": "this deployment serves inference only: "
                                                              "POST /v1/infer with a page image"}})
    return await call_next(request)


def settings_of(request: Request):
    return request.app.state.settings


async def _read(upload: UploadFile, limit: int, label: str) -> bytes:
    chunks, total = [], 0
    while True:
        chunk = await upload.read(1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise ApiError(413, "too_large", f"the {label} is larger than {limit // 1024} KiB")
        chunks.append(chunk)
    if not total:
        raise ApiError(400, "empty_upload", f"the {label} part is empty")
    return b"".join(chunks)


async def _payload(value: str, settings) -> bytes:
    """Bytes behind a data URL or an http(s) URL, for file_url and chat parts."""
    if not value:
        raise ApiError(400, "missing_input", "empty URL or data part")
    if value.startswith("data:"):
        header, _, payload = value.partition(",")
        try:
            return base64.b64decode(payload) if "base64" in header else payload.encode()
        except ValueError as error:
            raise ApiError(400, "invalid_data_url", "the data URL does not decode") from error
    if not settings.allow_outbound:
        raise ApiError(403, "url_not_allowed", "file_url is turned off on this service")
    return await fetch(value, settings.max_fetch_mb * 1024 * 1024, settings.fetch_timeout_s,
                       allowed_domains=settings.fetch_allowlist)


async def _source(request: Request, upload, file_id: str | None, file_url: str | None) -> bytes:
    settings = settings_of(request)
    limit = max(settings.max_image_mb, settings.max_pdf_mb) * 1024 * 1024
    if upload is not None:
        return await _read(upload, limit, "file")
    if file_id:
        data = request.app.state.cache.get(file_id)
        if data is None:
            raise ApiError(404, "file_not_found", "no file with that id is cached; upload it again")
        return data
    if file_url:
        return await _payload(file_url, settings)
    raise ApiError(400, "missing_input", "send a file part, a file_id or a file_url")


def _boxes(document: dict) -> dict[str, list[float]]:
    """The model's own table box per table, so /crop can cut it without a box in the URL."""
    boxes: dict[str, list[float]] = {}
    for page in document.get("pages") or []:
        for table in page.get("tables") or []:
            box = table.get("model_box") or table.get("box")
            if box:
                boxes[table["id"]] = box
    return boxes


async def _run(request: Request, data: bytes, words_bytes: bytes | None, **options) -> dict:
    state = request.app.state
    async with state.gate:
        try:
            document = await asyncio.wait_for(
                extract_service.extract(state.engine, state.settings, state.pool,
                                        data, words_bytes, **options),
                timeout=state.settings.request_timeout_s)
        except asyncio.TimeoutError:
            raise ApiError(504, "request_timeout",
                           f"extraction exceeded {state.settings.request_timeout_s:.0f}s") from None
    state.cache.put(document["file_id"], data, _boxes(document))
    return document


async def _run_job(request: Request, job: Job, data: bytes, words_bytes, options) -> dict:
    async def progress(done: int, total: int) -> None:
        job.pages_done, job.pages_total = done, total
    return await extract_service.extract(request.app.state.engine, request.app.state.settings,
                                         request.app.state.pool, data, words_bytes,
                                         progress=progress, **options)


def _submit(request: Request, data: bytes, words_bytes, options) -> Job:
    try:
        return request.app.state.jobs.submit(
            lambda job: _run_job(request, job, data, words_bytes, options))
    except RuntimeError as error:
        raise ApiError(429, "queue_full",
                       f"{request.app.state.settings.job_limit} jobs are already running") from error


@app.get("/", include_in_schema=False)
def root(request: Request) -> dict[str, Any]:
    return {"service": "ae3-image", "model": request.app.state.settings.model_name,
            "endpoints": ["/v1/extract", "/v1/jobs", "/v1/files", "/v1/models",
                          "/v1/chat/completions", "/health", "/ready", "/metrics", "/mcp",
                          "/docs"]}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "ae3-image",
            "uptime_s": round(time.time() - STARTED_AT, 1)}


@app.get("/ready")
def ready(request: Request) -> JSONResponse:
    state = request.app.state
    return JSONResponse({"status": "ready" if state.ready else "starting",
                         "model": state.settings.model_name,
                         "version": state.engine.config.version,
                         "provider": resolve_provider(state.engine.onnx.provider),
                         "ocr_engine": state.settings.ocr_engine,
                         "pdf_input": state.settings.pdf_input,
                         "surface": state.settings.surface,
                         "weights": getattr(state, "weights", None),
                         "warmup_ms": state.warmup_ms},
                        status_code=200 if state.ready else 503)


@app.post("/v1/infer")
async def infer(request: Request, file: UploadFile = File(...),
                include_embeddings: bool = True):
    """One page image in, the model's raw outputs out. No cells, no text.

    The 300x256 embeddings are about 1.6 MB of JSON; pass include_embeddings=false
    to leave them out when the caller only needs logits, boxes and indices.
    """
    authorize(request)
    settings = settings_of(request)
    data = await _read(file, settings.max_image_mb * 1024 * 1024, "file")
    page = decode.load(data, settings.max_image_mb * 1024 * 1024, settings.max_image_pixels)
    state = request.app.state
    async with state.gate:
        logits, boxes, embeddings, indices = await state.batcher.submit(
            page_tensors([page.image]))
    body = {"object": "ae3.inference",
            "model": {"id": settings.model_name, "version": state.engine.config.version},
            "image": {"width": page.width, "height": page.height},
            "logits": logits[0].tolist(),
            "boxes": boxes[0].tolist(),
            "encoder_indices": indices[0].tolist()}
    if include_embeddings:
        body["query_embeddings"] = embeddings[0].tolist()
    return body


@app.get("/v1/models")
def models(request: Request) -> dict[str, Any]:
    authorize(request)
    settings = settings_of(request)
    return {"object": "list",
            "data": [{"id": settings.model_name, "object": "model",
                      "created": int(STARTED_AT), "owned_by": "APEX:E3",
                      "version": request.app.state.engine.config.version}]}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics(request: Request) -> PlainTextResponse:
    gate = request.app.state.gate
    text = request.app.state.metrics.render()
    text += (f"# TYPE ae3_in_flight gauge\nae3_in_flight {gate.running}\n"
             f"# TYPE ae3_queue_waiting gauge\nae3_queue_waiting {gate.waiting}\n"
             f"# TYPE ae3_concurrency_limit gauge\nae3_concurrency_limit {gate.limit}\n"
             f"# TYPE ae3_queue_depth_limit gauge\nae3_queue_depth_limit {gate.depth}\n")
    return PlainTextResponse(text, media_type="text/plain; version=0.0.4")


@app.post("/v1/files")
async def upload(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    authorize(request)
    settings = settings_of(request)
    data = await _read(file, max(settings.max_image_mb, settings.max_pdf_mb) * 1024 * 1024, "file")
    identifier = hashlib.sha256(data).hexdigest()[:12]
    request.app.state.cache.put(identifier, data)
    return {"id": identifier, "bytes": len(data), "kind": extract_service.kind_of(data),
            "expires_in_s": settings.cache_ttl_s}


@app.post("/v1/extract")
async def extract(request: Request,
                  file: UploadFile | None = File(None),
                  words: UploadFile | None = File(None),
                  file_id: str | None = Form(None),
                  file_url: str | None = Form(None),
                  format: str = Form("json"),
                  pages: str = Form(""),
                  mode: str = Form("full"),
                  ocr: str = Form("auto"),
                  rotation: str = Form("auto"),
                  deskew: bool | None = Form(None),
                  include_crops: bool = Form(False)):
    authorize(request)
    settings = settings_of(request)
    data = await _source(request, file, file_id, file_url)
    words_bytes = await _read(words, settings.max_words_mb * 1024 * 1024, "words") if words else None
    options = {"fmt": format, "pages": pages, "mode": mode, "ocr": ocr,
               "rotation": rotation, "deskew": deskew, "include_crops": include_crops}
    if extract_service.kind_of(data) == "pdf":
        total = await request.app.state.pool.page_count(data)
        selected = extract_service.count_pages(pages, total, settings.max_pages)
        if selected > settings.sync_page_limit:
            job = _submit(request, data, words_bytes, options)
            job.pages_total = selected
            return JSONResponse({"object": "ae3.job", "job_id": job.id, "status": job.status,
                                 "pages_total": total, "pages_selected": selected,
                                 "poll": f"/v1/jobs/{job.id}"}, status_code=202)
    return await _run(request, data, words_bytes, **options)


@app.post("/v1/jobs")
async def create_job(request: Request,
                     file: UploadFile | None = File(None),
                     words: UploadFile | None = File(None),
                     file_id: str | None = Form(None),
                     file_url: str | None = Form(None),
                     format: str = Form("json"),
                     pages: str = Form(""),
                     mode: str = Form("summary"),
                     ocr: str = Form("auto"),
                     rotation: str = Form("auto"),
                     deskew: bool | None = Form(None)):
    authorize(request)
    settings = settings_of(request)
    data = await _source(request, file, file_id, file_url)
    words_bytes = await _read(words, settings.max_words_mb * 1024 * 1024, "words") if words else None
    job = _submit(request, data, words_bytes,
                  {"fmt": format, "pages": pages, "mode": mode, "ocr": ocr,
                   "rotation": rotation, "deskew": deskew})
    return JSONResponse({"object": "ae3.job", "job_id": job.id, "status": job.status,
                         "poll": f"/v1/jobs/{job.id}"}, status_code=202)


@app.get("/v1/jobs/{job_id}")
def job_status(request: Request, job_id: str) -> dict[str, Any]:
    authorize(request)
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise ApiError(404, "job_not_found", "no job with that id; jobs expire")
    return job.view()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict[str, Any]:
    authorize(request)
    settings = settings_of(request)
    try:
        body = await request.json()
    except ValueError as error:
        raise ApiError(400, "invalid_json", "the body is not valid JSON") from error
    data = None
    for message in body.get("messages") or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "image_url":
                image = part.get("image_url")
                value = image if isinstance(image, str) else (image or {}).get("url")
                data = await _payload(str(value or ""), settings)
            elif kind == "input_image":
                data = await _payload(str(part.get("image_url") or ""), settings)
            elif kind in ("file", "input_file"):
                entry = part.get("file") or part.get("input_file") or {}
                value = entry.get("file_data") or entry.get("url") or entry.get("data")
                data = await _payload(str(value or ""), settings)
    if data is None:
        raise ApiError(400, "missing_input", "send an image_url part or a file part")
    fmt = str(body.get("ae3_format") or "json")
    document = await _run(request, data, None, fmt=fmt, pages="", mode="full", ocr="auto",
                          rotation="auto", deskew=None)
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:16]}", "object": "chat.completion",
            "created": int(time.time()), "model": settings.model_name,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(document)}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


async def _responses_payload(request: Request, body: dict, settings):
    """The bytes behind the first image or file part of an OpenAI Responses input."""
    items = body.get("input")
    if isinstance(items, str):
        items = [{"content": [{"type": "input_text", "text": items}]}]
    for item in items or []:
        if not isinstance(item, dict):
            continue
        parts = item.get("content")
        if not isinstance(parts, list):
            parts = [item]
        for part in parts:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "input_image":
                image = part.get("image_url")
                value = image if isinstance(image, str) else (image or {}).get("url")
                if value:
                    return await _payload(str(value), settings)
            if kind in ("input_file", "file"):
                entry = part.get("file") or part.get("input_file") or {}
                if entry.get("file_id"):
                    data = request.app.state.cache.get(str(entry["file_id"]))
                    if data is None:
                        raise ApiError(404, "file_not_found",
                                       "no file with that id is cached; upload it again")
                    return data
                value = entry.get("file_data") or entry.get("file_url") or entry.get("url") or entry.get("data")
                if value:
                    return await _payload(str(value), settings)
    raise ApiError(400, "missing_input", "send an input_image part or a file part")


@app.post("/v1/responses")
async def responses(request: Request):
    authorize(request)
    settings = settings_of(request)
    try:
        body = await request.json()
    except ValueError as error:
        raise ApiError(400, "invalid_json", "the body is not valid JSON") from error
    data = await _responses_payload(request, body, settings)
    fmt = str(body.get("ae3_format") or "json")
    document = await _run(request, data, None, fmt=fmt, pages="", mode="full", ocr="auto",
                          rotation="auto", deskew=None)
    text = json.dumps(document)
    created = int(time.time())
    rid = f"resp_{uuid.uuid4().hex[:24]}"
    mid = f"msg_{uuid.uuid4().hex[:24]}"

    def shape(status: str, content_text: str, output: list | None = None) -> dict:
        return {"id": rid, "object": "response", "created_at": created, "status": status,
                "error": None, "incomplete_details": None,
                "instructions": body.get("instructions"), "max_output_tokens": None,
                "model": settings.model_name,
                "output": output if output is not None else [{
                    "type": "message", "id": mid, "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": content_text, "annotations": []}]}],
                "parallel_tool_calls": True, "previous_response_id": None,
                "reasoning": {"effort": None, "summary": None}, "store": False,
                "temperature": 1.0, "text": {"format": {"type": "text"}},
                "tool_choice": "auto", "tools": [], "top_p": 1.0, "truncation": "disabled",
                "usage": {"input_tokens": 0, "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0},
                          "total_tokens": 0},
                "metadata": {}}

    if not body.get("stream"):
        return shape("completed", text)

    def sse(name: str, payload: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

    async def events():
        yield sse("response.created", shape("in_progress", "", output=[]))
        yield sse("response.output_item.added", {
            "type": "response.output_item.added", "output_index": 0,
            "item": {"type": "message", "id": mid, "status": "in_progress",
                     "role": "assistant", "content": []}})
        yield sse("response.content_part.added", {
            "type": "response.content_part.added", "item_id": mid, "output_index": 0,
            "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}})
        for start in range(0, len(text), 2000):
            yield sse("response.output_text.delta", {
                "type": "response.output_text.delta", "item_id": mid, "output_index": 0,
                "content_index": 0, "delta": text[start:start + 2000]})
        yield sse("response.output_text.done", {
            "type": "response.output_text.done", "item_id": mid, "output_index": 0,
            "content_index": 0, "text": text})
        yield sse("response.output_item.done", {
            "type": "response.output_item.done", "output_index": 0,
            "item": {"type": "message", "id": mid, "status": "completed", "role": "assistant",
                     "content": [{"type": "output_text", "text": text, "annotations": []}]}})
        yield sse("response.completed", shape("completed", text))

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/v1/tables/{table_id}/crop")
def crop(request: Request, table_id: str, box: str | None = None) -> Response:
    authorize(request)
    settings = settings_of(request)
    if not settings.enable_crop:
        raise ApiError(404, "crop_disabled", "the crop endpoint is off in v0.1")
    page_id = table_id.rsplit("-t", 1)[0]
    if not IMAGE_ID.match(page_id):
        raise ApiError(400, "invalid_table_id", "a table id looks like 9f2c1a7b3d40-p1-t0")
    data = request.app.state.cache.get(page_id)
    if data is None:
        raise ApiError(404, "crop_unavailable", "the uploaded file is no longer cached")
    values = None
    if box:
        numbers = [float(value) for value in NUMBER.findall(box)]
        if len(numbers) != 4:
            raise ApiError(400, "invalid_box", "box must be x0,y0,x1,y1")
        values = numbers
    else:
        values = request.app.state.cache.box(page_id, table_id)
    if values is None:
        raise ApiError(404, "crop_unavailable",
                       "the table is not in this copy's cache; send box=x0,y0,x1,y1")
    import io
    from PIL import Image
    image = Image.open(io.BytesIO(data)).convert("RGB")
    x0, y0, x1, y1 = (int(round(value)) for value in values)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.width, x1), min(image.height, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ApiError(400, "invalid_box", "the box is empty or outside the image")
    buffer = io.BytesIO()
    image.crop((x0, y0, x1, y1)).save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


@app.post("/mcp")
async def mcp_post(request: Request) -> Response:
    authorize(request)
    return await request.app.state.mcp.handle_post(request)


@app.get("/mcp")
def mcp_get() -> Response:
    return Response(status_code=405, headers={"Allow": "POST"},
                    content=b'{"error": "MCP here answers POST only"}',
                    media_type="application/json")
