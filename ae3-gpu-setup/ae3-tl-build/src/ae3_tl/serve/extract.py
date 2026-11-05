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

"""One entry point for both inputs: a page image or a PDF, with words from where they are best.

A page image gets Tesseract words unless a words file is supplied. A PDF page gets its own text
layer and Tesseract only when that layer is empty, so a page with text is never OCRed.
"""

import asyncio
import hashlib
import json
import time
from typing import Any

from ae3_tl.image import decode, service as image_service
from ae3_tl.pdf.pdfpool import PdfPool
from ae3_tl.serve import render
from ae3_tl.serve.errors import ApiError

SOURCE_NAMES = {"text_layer": "pdf_text", "ocr": "ocr", "none": "none"}
SUMMARY_FIELDS = ("id", "index", "page", "box", "label", "score", "rows", "columns", "caption",
                  "footer", "source")


def kind_of(data: bytes) -> str:
    return "pdf" if data[:5] == b"%PDF-" else "image"


async def extract(engine, settings, pool: PdfPool, data: bytes, words_bytes: bytes | None = None,
                  *, fmt: str = "json", pages: str = "", mode: str = "full", ocr: str = "auto",
                  rotation: str = "auto", deskew: bool | None = None,
                  include_crops: bool = False, progress=None) -> dict[str, Any]:
    if fmt not in render.FORMATS:
        raise ApiError(400, "invalid_format", f"format must be one of {', '.join(render.FORMATS)}")
    if mode not in ("full", "summary"):
        raise ApiError(400, "invalid_mode", "mode must be full or summary")
    started = time.perf_counter()
    kind = kind_of(data)
    if kind == "pdf":
        if not settings.pdf_input:
            raise ApiError(415, "pdf_disabled",
                           "PDF input is off in this release; send a page image")
        document = await _pdf(engine, settings, pool, data, words_bytes, fmt=fmt, pages=pages,
                              mode=mode, ocr=ocr, progress=progress)
    else:
        if words_bytes is not None:
            words_document = words_bytes
        else:
            words_document = None
        single = await asyncio.to_thread(
            image_service.extract, engine, settings, data, words_document, fmt=fmt,
            include_crops=include_crops, rotation=rotation, deskew=deskew)
        tables = [_with_source(table, single["source"]) for table in single["tables"]]
        tables = _apply_mode(tables, mode)
        document = {
            "object": "ae3.extraction",
            "service": "ae3-image",
            "model": single["model"],
            "kind": "image",
            "file_id": single["image_id"],
            "format": fmt,
            "mode": mode,
            "source": single["source"],
            "pages_total": 1,
            "pages_returned": 1,
            "tables_total": len(tables),
            "elapsed_ms": None,
            "pages": [{"page": 1, "width": single["image"]["width"],
                       "height": single["image"]["height"], "source": single["source"],
                       "tables": tables}],
        }
    document["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    engine_metrics = getattr(engine, "counters", None)
    if engine_metrics is not None:
        engine_metrics(len(document["pages"]), document["tables_total"])
    return document


async def _pdf(engine, settings, pool: PdfPool, data: bytes, words_bytes, *, fmt, pages, mode, ocr,
               progress) -> dict[str, Any]:
    total = await pool.page_count(data)
    first, last = _page_range(pages, total, settings.max_pages)
    if last - first > settings.max_pages:
        raise ApiError(413, "too_many_pages",
                       f"the request asks for more than {settings.max_pages} pages")
    supplied = _words_pages(words_bytes) if words_bytes else None
    document_id = hashlib.sha256(data).hexdigest()[:12]
    page_records = []
    tables_total = 0
    async for index, image, size, words, source in pool.pages(data, settings.render_dpi, first,
                                                              last, ocr):
        if supplied is not None:
            page_words = supplied[index] if index < len(supplied) else None
            if page_words:
                words, source = page_words, "provided_words"
        prediction = await asyncio.to_thread(engine.predict, image, words, size)
        tables = []
        for table_index, table in enumerate(prediction.tables):
            record = render.table_record(
                table, table_index, f"{document_id}-p{index + 1}-t{table_index}", fmt)
            record["page"] = index + 1
            record["source"] = SOURCE_NAMES.get(source, "none")
            if record["source"] == "ocr":
                _add_ocr_confidence(record["cells"], words)
            tables.append(record)
        tables = _apply_mode(tables, mode)
        tables_total += len(tables)
        page_records.append({"page": index + 1, "width": size[0], "height": size[1],
                             "source": SOURCE_NAMES.get(source, "none"), "tables": tables})
        if progress:
            progress(len(page_records), total)
    return {
        "object": "ae3.extraction",
        "service": "ae3-image",
        "model": {"id": settings.model_name, "version": engine.config.version},
        "kind": "pdf",
        "file_id": document_id,
        "format": fmt,
        "mode": mode,
        "source": _dominant_source(page_records),
        "pages_total": total,
        "pages_returned": len(page_records),
        "tables_total": tables_total,
        "elapsed_ms": None,
        "pages": page_records,
    }


def count_pages(specification: str, total: int, limit: int) -> int:
    """How many pages a request selects, after the page cap."""
    first, last = _page_range(specification, total, limit)
    return max(0, last - first)


def _page_range(specification: str, total: int, limit: int) -> tuple[int, int]:
    text = (specification or "").strip()
    if not text:
        return 0, min(total, limit)
    try:
        first, last = ((int(value) for value in text.split("-", 1)) if "-" in text
                       else (int(text), int(text)))
    except ValueError as error:
        raise ApiError(400, "invalid_pages", "pages must be '3' or '1-5'") from error
    if first < 1 or last < first:
        raise ApiError(400, "invalid_pages", "pages must be 1-based and ascending")
    return first - 1, min(last, total, first - 1 + limit)


def _words_pages(words_bytes: bytes) -> list[list[dict]] | None:
    from ae3_tl.image import words as words_module

    document = json.loads(words_bytes)
    pages = document.get("pages") if isinstance(document, dict) else None
    if isinstance(pages, list) and pages:
        out = []
        for page in pages:
            entries = page.get("words") if isinstance(page, dict) else page
            out.append(words_module.parse(json.dumps(entries).encode()) if entries else [])
        return out
    return None


def _with_source(table: dict, source: str) -> dict:
    return {**table, "source": "provided_words" if source == "provided_words" else source}


def _apply_mode(tables: list[dict], mode: str) -> list[dict]:
    if mode == "full":
        return tables
    return [{key: table[key] for key in SUMMARY_FIELDS if key in table} for table in tables]


def _add_ocr_confidence(cells: list[dict], words: list[dict]) -> None:
    if not words or "confidence" not in words[0]:
        return
    for cell in cells:
        x0, y0, x1, y1 = cell["box"]
        inside = [word["confidence"] for word in words
                  if x0 <= (word["bbox"][0] + word["bbox"][2]) / 2 <= x1
                  and y0 <= (word["bbox"][1] + word["bbox"][3]) / 2 <= y1]
        if inside:
            cell["ocr_confidence"] = round(sum(inside) / len(inside), 3)


def _dominant_source(page_records: list[dict]) -> str:
    kinds = {page["source"] for page in page_records}
    if len(kinds) == 1:
        return kinds.pop()
    return "mixed"
