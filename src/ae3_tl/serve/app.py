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

"""The HTTP API: one FastAPI app, the endpoints the README documents.

create_app() builds everything from the environment: weights download on first start, the
ONNX engine on the chosen provider, the API-key gate, the bounded queue and the Prometheus
counters. The heavy model work stays in ae3_tl.image.service; this module only moves bytes
and documents.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from ae3_tl.backends.engine_ort import EngineOrt
from ae3_tl.serve import render
from ae3_tl.serve.errors import ApiError
from ae3_tl.serve.settings import load_settings

log = logging.getLogger("ae3.serve")

MODEL_ID = "AE3-TL"


# --------------------------------------------------------------------- state


@dataclass
class _State:
    """Everything one server process owns: the engine, the queue, the counters."""

    settings: Any
    engine: EngineOrt | None = None
    provider: str = "loading"
    startup_error: str | None = None
    started: float = field(default_factory=time.time)
    queue: asyncio.Queue | None = None
    workers: list[asyncio.Task] = field(default_factory=list)
    jobs: dict[str, dict] = field(default_factory=dict)
    files: dict[str, dict] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=lambda: {
        "requests_total": 0.0,
        "extract_total": 0.0,
        "chat_total": 0.0,
        "errors_total": 0.0,
        "tables_total": 0.0,
        "rejects_total": 0.0,
    })


def create_app() -> FastAPI:
    settings = load_settings()
    settings.require_api_key()
    state = _State(settings=settings)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        state.queue = asyncio.Queue(maxsize=settings.queue_depth)
        for index in range(max(1, settings.concurrency)):
            state.workers.append(asyncio.create_task(_worker(state, index)))
        loop = asyncio.get_running_loop()
        try:
            from ae3_tl import weights as weights_module

            weights_module.ensure_weights(settings.weights_dir)
            state.provider = _provider_name(settings)
            state.engine = await loop.run_in_executor(None, EngineOrt, settings)
            if settings.warmup:
                from ae3_tl.image import service as image_service

                await loop.run_in_executor(None, image_service.warm, state.engine)
        except Exception as exc:  # noqa: BLE001 - startup failure must surface in /ready
            log.error("startup failed: %s", exc)
            state.provider = "error"
            state.startup_error = str(exc)
        yield
        for task in state.workers:
            task.cancel()

    app = FastAPI(title="AE3-TL Serve", version=_version(), lifespan=lifespan,
                  docs_url="/docs", openapi_url="/openapi.json")
    app.state.ae3 = state

    _routes(app, state)
    return app


def _version() -> str:
    from ae3_tl._version import __version__

    return __version__


def _provider_name(settings) -> str:
    from ae3_tl.backends.onnx_backend import resolve_provider

    try:
        return resolve_provider(settings.onnx_provider)
    except Exception:  # noqa: BLE001
        return settings.onnx_provider


# ------------------------------------------------------------------- routes


def _routes(app: FastAPI, state: _State) -> None:
    settings = state.settings

    @app.exception_handler(ApiError)
    async def _api_error(_request: Request, exc: ApiError):
        return _json({"error": {"code": exc.code, "message": exc.message}}, exc.status)

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception):
        log.exception("unhandled error")
        state.metrics["errors_total"] += 1
        return _json({"error": {"code": "internal", "message": str(exc)[:200]}}, 500)

    @app.middleware("http")
    async def _gate(request: Request, call_next):
        path = request.url.path
        state.metrics["requests_total"] += 1
        if path not in ("/health", "/metrics") and not path.startswith("/docs") \
                and path != "/openapi.json":
            key = request.headers.get("X-API-Key") or _bearer(request)
            if key != settings.api_key:
                state.metrics["rejects_total"] += 1
                return _json({"error": {"code": "unauthorized",
                                        "message": "missing or wrong API key"}}, 401)
        if path == "/v1/extract" and request.method == "POST":
            length = request.headers.get("content-length")
            if length and int(length) > settings.max_body_mb * 1024 * 1024:
                return _json({"error": {"code": "too_large",
                                        "message": f"body larger than {settings.max_body_mb} MiB"}},
                             413)
        return await call_next(request)

    # ------------------------------------------------------------ probes

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "ae3-image",
                "uptime_s": round(time.time() - state.started, 1)}

    @app.get("/ready")
    async def ready():
        payload: dict[str, Any] = {"status": "ready" if state.engine else "warming",
                                   "provider": state.provider}
        if state.startup_error:
            payload["status"] = "error"
            payload["error"] = state.startup_error
        return payload

    # ------------------------------------------------------------ models

    @app.get("/v1/models")
    async def models():
        version = state.engine.config.version if state.engine else _version()
        return {"object": "list", "data": [{"id": settings.model_name,
                                            "object": "model", "version": version}]}

    # ----------------------------------------------------------- extract

    @app.post("/v1/extract")
    async def extract(
        file: UploadFile | None = File(None),
        file_url: str | None = Form(None),
        file_id: str | None = Form(None),
        words: UploadFile | None = File(None),
        fmt: str = Form("json", alias="format"),
        mode: str = Form("full"),
        pages: str | None = Form(None),
        ocr: str = Form("auto"),
        rotation: str = Form("auto"),
        deskew: str | None = Form(None),
        include_crops: str = Form("false"),
    ):
        payload = await _read_payload(state, file, file_url, file_id, words)
        crops = str(include_crops).lower() in ("1", "true", "yes", "on")
        deskew_flag = None if deskew is None else str(deskew).lower() in ("1", "true", "yes", "on")
        if state.engine is None:
            raise ApiError(503, "not_ready", "the model is still warming up; try /ready")
        job = await _enqueue(state, _extract_call, payload, fmt, mode, crops,
                             rotation, deskew_flag, ocr)
        return await _await_job(state, job)

    @app.post("/v1/files")
    async def upload(file: UploadFile = File(...)):
        data = await file.read()
        _check_size(state, len(data))
        identifier = uuid.uuid4().hex[:12]
        state.files[identifier] = {"data": data, "name": file.filename or "upload",
                                   "kind": file.content_type or "application/octet-stream",
                                   "at": time.time()}
        return {"object": "file", "id": identifier, "bytes": len(data),
                "name": state.files[identifier]["name"]}

    @app.get("/v1/files/{file_id}")
    async def get_file(file_id: str):
        record = state.files.get(file_id)
        if record is None:
            raise ApiError(404, "file_not_found", f"no file {file_id}")
        return {"object": "file", "id": file_id, "bytes": len(record["data"]),
                "name": record["name"], "kind": record["kind"]}

    @app.post("/v1/jobs")
    async def jobs_submit(
        file: UploadFile | None = File(None),
        file_url: str | None = Form(None),
        file_id: str | None = Form(None),
        fmt: str = Form("json", alias="format"),
        pages: str | None = Form(None),
    ):
        payload = await _read_payload(state, file, file_url, file_id, None)
        if state.engine is None:
            raise ApiError(503, "not_ready", "the model is still warming up; try /ready")
        if not (payload["kind"] == "pdf" and settings.pdf_input):
            raise ApiError(415, "pdf_disabled",
                           "PDF input is off; start with AE3_PDF_INPUT=1")
        job_id = uuid.uuid4().hex[:12]
        state.jobs[job_id] = {"status": "queued", "pages_done": 0,
                              "pages_total": _page_count(payload, pages), "id": job_id}
        asyncio.create_task(_run_job(state, job_id, _extract_call, payload, fmt, "full",
                                     False, "auto", None, "auto"))
        return _json({"object": "ae3.job", "job_id": job_id, "status": "queued",
                      "poll": f"/v1/jobs/{job_id}"}, 202)

    @app.get("/v1/jobs/{job_id}")
    async def jobs_poll(job_id: str):
        job = state.jobs.get(job_id)
        if job is None:
            raise ApiError(404, "job_not_found", f"no job {job_id}")
        return job

    # ------------------------------------------------------------- chat

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = json.loads(await request.body() or b"{}")
        except json.JSONDecodeError:
            raise ApiError(400, "invalid_json", "the body is not JSON")
        messages = body.get("messages") or []
        if not messages:
            raise ApiError(400, "invalid_request", "messages is required")
        payload, text = _chat_payload(state, messages)
        fmt = (body.get("ae3_format") or "json").lower()
        if fmt not in render.FORMATS:
            raise ApiError(400, "invalid_format",
                           f"ae3_format must be one of {', '.join(render.FORMATS)}")
        if state.engine is None:
            raise ApiError(503, "not_ready", "the model is still warming up; try /ready")
        job = await _enqueue(state, _extract_call, payload, fmt, "full", False,
                             "auto", None, "auto")
        document = await _await_job(state, job)
        content = json.dumps(document) if fmt == "json" else _chat_text(document)
        state.metrics["chat_total"] += 1
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": settings.model_name,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # ---------------------------------------------------------- metrics

    @app.get("/metrics")
    async def metrics():
        lines = []
        counters = dict(state.metrics)
        counters["queue_depth"] = state.queue.qsize() if state.queue else 0
        counters["in_flight"] = sum(1 for job in state.jobs.values()
                                    if job["status"] == "running")
        counters["uptime_s"] = time.time() - state.started
        for name, value in sorted(counters.items()):
            lines.append(f"ae3_{name} {value}")
        return Response("\n".join(lines) + "\n", media_type="text/plain")


# ------------------------------------------------------------------ helpers


def _json(payload: dict, status: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status)


def _bearer(request: Request) -> str:
    header = request.headers.get("Authorization") or ""
    return header[7:] if header.lower().startswith("bearer ") else ""


def _check_size(state: _State, size: int) -> None:
    limit = state.settings.max_image_mb * 1024 * 1024
    if size > limit:
        raise ApiError(413, "too_large",
                       f"the file is larger than {state.settings.max_image_mb} MiB")


async def _read_payload(state: _State, file, file_url, file_id, words) -> dict:
    """Turn the three ways a page can arrive into one {data, kind, name} record."""
    if file is not None:
        data = await file.read()
        name = file.filename or "upload"
    elif file_id:
        record = state.files.get(file_id)
        if record is None:
            raise ApiError(404, "file_not_found", f"no file {file_id}")
        data, name = record["data"], record["name"]
    elif file_url:
        data, name = await _fetch_url(state, file_url)
    else:
        raise ApiError(400, "no_input", "send file, file_url or file_id")
    _check_size(state, len(data))
    payload = {"data": data, "name": name, "kind": _kind(data, name)}
    if words is not None:
        payload["words"] = await words.read()
    return payload


def _kind(data: bytes, name: str) -> str:
    if data[:5] == b"%PDF-" or name.lower().endswith(".pdf"):
        return "pdf"
    from ae3_tl.image.decode import sniff

    return sniff(data) or "image"


async def _fetch_url(state: _State, url: str) -> tuple[bytes, str]:
    import httpx

    settings = state.settings
    if not settings.allow_outbound:
        raise ApiError(403, "outbound_disabled",
                       "file_url is off; start with AE3_ALLOW_OUTBOUND=1")
    host = httpx.URL(url).host
    if host not in settings.fetch_allowlist:
        raise ApiError(403, "host_not_allowed", f"{host} is not in AE3_FETCH_ALLOWLIST")
    async with httpx.AsyncClient(follow_redirects=True,
                                 timeout=settings.fetch_timeout_s) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.content
    _check_size(state, len(data))
    return data, url.rsplit("/", 1)[-1] or "fetched"


def _chat_payload(state: _State, messages: list) -> tuple[dict | None, str]:
    """Pull the first image/PDF part out of the messages and keep the text prompt."""
    text = "Extract every table."
    for message in reversed(messages):
        content = message.get("content")
        if isinstance(content, str):
            text = content or text
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "text":
                text = part.get("text") or text
            elif part.get("type") in ("image_url", "input_image"):
                url = (part.get("image_url") or {}).get("url") or part.get("url") or ""
                if url.startswith("data:"):
                    head, _, body64 = url.partition(",")
                    payload = base64.b64decode(body64)
                    return {"data": payload, "name": "data-url",
                            "kind": _kind(payload, "data-url")}, text
                if url:
                    return None, url
            elif part.get("type") in ("file", "input_file"):
                file_part = part.get("file") or part.get("input_file") or {}
                file_data = file_part.get("file_data") or file_part.get("url") or ""
                if file_data.startswith("data:"):
                    _, _, body64 = file_data.partition(",")
                    payload = base64.b64decode(body64)
                    return {"data": payload, "name": "data-url",
                            "kind": "pdf"}, text
    return None, text


def _chat_text(document: dict) -> str:
    """Markdown when the caller asked for it: one fenced or plain block per table."""
    parts = []
    for page in document.get("pages") or [document]:
        for table in page.get("tables") or []:
            caption = table.get("caption") or ""
            body_text = table.get("body") or json.dumps(table)
            parts.append((f"{caption}\n\n{body_text}").strip())
    return "\n\n".join(parts) or "No tables found."


async def _worker(state: _State, index: int) -> None:
    while True:
        item = await state.queue.get()
        future, func, args = item
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, func, state, *args)
            loop.call_soon_threadsafe(future.set_result, result)
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(future.set_exception, exc)
        finally:
            state.queue.task_done()


async def _enqueue(state: _State, func: Callable, *args) -> asyncio.Future:
    queue = state.queue
    if queue is None:
        raise ApiError(503, "not_ready", "the server is still starting")
    if queue.qsize() >= state.settings.queue_depth:
        raise ApiError(503, "busy", "the queue is full; retry shortly")
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    await queue.put((future, func, args))
    return future


async def _await_job(state: _State, future: asyncio.Future) -> dict:
    timeout = float(getattr(state.settings, "request_timeout_s", 600.0))
    return await asyncio.wait_for(future, timeout=timeout)


def _extract_call(state: _State, payload: dict, fmt: str, mode: str, crops: bool,
                  rotation: str, deskew: bool | None, ocr_mode: str) -> dict:
    """Runs in the executor: decode, OCR, model, render."""
    from ae3_tl.image import service as image_service

    settings = state.settings
    if payload["kind"] == "pdf":
        raise ApiError(415, "pdf_disabled", "PDF input is off; start with AE3_PDF_INPUT=1")
    if ocr_mode == "off":
        settings = _without_ocr(settings)
    started = time.perf_counter()
    result = image_service.extract(
        state.engine, settings, payload["data"],
        words_bytes=payload.get("words"), fmt=fmt, include_crops=crops,
        rotation=rotation, deskew=deskew)
    state.metrics["extract_total"] += 1
    state.metrics["tables_total"] += result.get("tables_total", 0)
    document = _shape_document(state, result, payload, mode, started)
    return document


def _without_ocr(settings):
    """A frozen Settings with the OCR engine switched off, for ocr=off requests."""
    from dataclasses import replace

    return replace(settings, ocr_engine="none")


def _shape_document(state: _State, result: dict, payload: dict, mode: str,
                    started: float) -> dict:
    """image service output -> the README's ae3.extraction document shape."""
    document = dict(result)
    document["object"] = "ae3.extraction"
    document["kind"] = "image"
    document["file_id"] = payload.get("file_id") or result.get("image_id")
    document["mode"] = mode
    document["pages_total"] = 1
    document["pages"] = [{
        "page": 1,
        "width": result["image"]["width"],
        "height": result["image"]["height"],
        "tables": document.pop("tables", []),
    }]
    document["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    if mode == "summary":
        for table in document["pages"][0]["tables"]:
            table.pop("cells", None)
            table.pop("body", None)
            table.pop("crops", None)
            table.pop("rows_boxes", None)
            table.pop("columns_boxes", None)
    for drop in ("image", "transform", "source", "words"):
        document.pop(drop, None)
    return document


def _page_count(payload: dict, pages: str | None) -> int:
    if not pages or "-" not in pages:
        return 1
    try:
        start, end = pages.split("-", 1)
        return max(1, int(end) - int(start) + 1)
    except ValueError:
        return 1


async def _run_job(state: _State, job_id: str, func: Callable, *args) -> None:
    job = state.jobs[job_id]
    job["status"] = "running"
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, func, state, *args)
        job["status"] = "done"
        job["result"] = result
    except Exception as exc:  # noqa: BLE001
        job["status"] = "failed"
        job["error"] = str(exc)[:200]
