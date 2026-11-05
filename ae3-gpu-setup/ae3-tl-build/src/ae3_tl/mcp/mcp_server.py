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

"""MCP over Streamable HTTP at /mcp: extract_tables and job_status.

A POST carries one JSON-RPC 2.0 message; initialize, tools/list and tools/call are answered,
notifications get 202 with no body, GET answers 405 because no SSE stream is opened.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from ae3_tl._version import __version__
from ae3_tl.serve import extract as extract_service
from ae3_tl.serve import schemas
from ae3_tl.serve.errors import ApiError
from ae3_tl.serve.fetch import fetch

log = logging.getLogger("ae3.mcp")

PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "ae3-image", "title": "ae3-image table extraction", "version": __version__}
FORMATS = ("json", "markdown", "html", "csv")


def _ok(message_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _error(message_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def _result(payload: dict) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
            "structuredContent": payload, "isError": False}


def _failure(code: str, message: str) -> dict:
    payload = {"error": {"type": code, "message": message}}
    return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
            "structuredContent": payload, "isError": True}


class McpServer:
    """The JSON-RPC side of /mcp, with the tools bound to this FastAPI app."""

    def __init__(self, app):
        self.app = app

    @property
    def state(self):
        return self.app.state

    async def handle_post(self, request: Request) -> Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return JSONResponse(_error(None, -32700, "the body is not valid JSON"), status_code=400)
        headers = {"Mcp-Session-Id": request.headers.get("mcp-session-id") or uuid.uuid4().hex}
        if isinstance(body, list):
            answers = [answer for answer in [await self._dispatch(item) for item in body]
                       if answer is not None]
            if not answers:
                return Response(status_code=202, headers=headers)
            return JSONResponse(answers, headers=headers)
        if not isinstance(body, dict):
            return JSONResponse(_error(None, -32600, "a JSON-RPC message must be an object"),
                                status_code=400)
        answer = await self._dispatch(body)
        if answer is None:
            return Response(status_code=202, headers=headers)
        return JSONResponse(answer, headers=headers)

    async def _dispatch(self, message: dict) -> dict | None:
        method = message.get("method")
        message_id = message.get("id")
        params = message.get("params") or {}
        if method is None or method.startswith("notifications/"):
            return None
        try:
            if method == "initialize":
                requested = params.get("protocolVersion")
                version = requested if requested in SUPPORTED_VERSIONS else PROTOCOL_VERSION
                return _ok(message_id, {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": SERVER_INFO,
                    "instructions": ("Extract tables from a page image or a PDF with "
                                     "extract_tables; poll a long PDF with job_status."),
                })
            if method == "ping":
                return _ok(message_id, {})
            if method == "tools/list":
                return _ok(message_id, {"tools": [self._tool(tool) for tool in schemas.TOOLS]})
            if method == "tools/call":
                return _ok(message_id, await self._call(params))
            if method == "resources/list":
                return _ok(message_id, {"resources": []})
            if method == "prompts/list":
                return _ok(message_id, {"prompts": []})
            return _error(message_id, -32601, f"method not found: {method}")
        except ApiError as error:
            log.info("mcp tool error: code=%s", error.code)
            return _ok(message_id, _failure(error.code, error.message))
        except Exception as error:                              # noqa: BLE001
            log.error("mcp tool failed: %s", type(error).__name__)
            return _ok(message_id, _failure("internal_error", "the extraction failed"))

    @staticmethod
    def _tool(tool: dict) -> dict:
        return {"name": tool["name"],
                "title": tool["name"].replace("_", " ").title(),
                "description": tool["description"],
                "inputSchema": tool["parameters"],
                "annotations": {"readOnlyHint": tool["name"] == "job_status",
                                "destructiveHint": False, "idempotentHint": True,
                                "openWorldHint": True}}

    async def _call(self, params: dict) -> dict:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return _failure("invalid_arguments", "arguments must be an object")
        if name == "job_status":
            job_id = str(arguments.get("job_id") or "")
            if not job_id:
                return _failure("invalid_arguments", "job_id is required")
            job = self.state.jobs.get(job_id)
            if job is None:
                return _failure("job_not_found", "no job with that id; jobs expire")
            return _result(job.view())
        if name != "extract_tables":
            return _failure("unknown_tool", f"no tool named {name}")

        settings = self.state.settings
        file_url, file_id = arguments.get("file_url"), arguments.get("file_id")
        if bool(file_url) == bool(file_id):
            return _failure("invalid_arguments", "send exactly one of file_url or file_id")
        fmt = str(arguments.get("format") or "json").lower()
        if fmt not in FORMATS:
            return _failure("invalid_arguments", f"format must be one of {', '.join(FORMATS)}")
        if file_id:
            data = self.state.cache.get(str(file_id))
            if data is None:
                return _failure("file_not_found",
                                "no file with that id is cached; upload it with POST /v1/files")
        else:
            if not settings.allow_outbound:
                return _failure("url_not_allowed", "file_url is turned off on this service")
            data = await fetch(str(file_url), settings.max_fetch_mb * 1024 * 1024,
                               settings.fetch_timeout_s,
                               allowed_domains=settings.fetch_allowlist)
        words_bytes = None
        if arguments.get("words_url"):
            if not settings.allow_outbound:
                return _failure("url_not_allowed", "words_url is turned off on this service")
            words_bytes = await fetch(str(arguments["words_url"]),
                                      settings.max_words_mb * 1024 * 1024,
                                      settings.fetch_timeout_s,
                                      allowed_domains=settings.fetch_allowlist)

        options = {"fmt": fmt, "pages": "", "mode": "full", "ocr": "auto",
                   "rotation": "auto", "deskew": None}
        if extract_service.kind_of(data) == "pdf":
            total = await self.state.pool.page_count(data)
            selected = extract_service.count_pages("", total, settings.max_pages)
            if selected > settings.sync_page_limit:
                job = await self._submit(data, words_bytes, options)
                job.pages_total = selected
                return _result({"object": "ae3.job", "job_id": job.id, "status": job.status,
                                "pages_total": total, "pages_selected": selected,
                                "poll": f"/v1/jobs/{job.id}"})

        try:
            async with self.state.gate:
                document = await asyncio.wait_for(
                    extract_service.extract(self.state.engine, settings,
                                            self.state.pool, data, words_bytes,
                                            **options),
                    timeout=settings.request_timeout_s)
        except ApiError as error:
            return _failure(error.code, error.message)
        except asyncio.TimeoutError:
            return _failure("request_timeout",
                            f"extraction exceeded {settings.request_timeout_s:.0f}s")
        self.state.cache.put(document["file_id"], data)
        return _result(document)

    async def _submit(self, data: bytes, words_bytes, options):
        async def runner(job):
            async def progress(done: int, total: int) -> None:
                job.pages_done, job.pages_total = done, total
            return await extract_service.extract(self.state.engine, self.state.settings,
                                                 self.state.pool, data, words_bytes,
                                                 progress=progress, **options)
        try:
            return self.state.jobs.submit(runner)
        except RuntimeError as error:
            raise ApiError(429, "queue_full", "too many jobs are running") from error
