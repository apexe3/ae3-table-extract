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

"""One page image -> the extraction document: decode, turn, deskew, words, model, boxes, crops."""

import base64
import hashlib
import io
import time
from typing import Any

import numpy as np
from PIL import Image

from ae3_tl import ocr
from ae3_tl.image import decode, geometry, words as words_module
from ae3_tl.serve import render
from ae3_tl.serve.errors import ApiError

ROTATIONS = ("auto", "0", "90", "180", "270")
CROP_TYPES = ("png",)


def warm(engine) -> None:
    """One blank page through the model, so the first request does not pay for the cold start."""
    engine.predict(Image.new("RGB", (768, 990), "white"), [], (768, 990))


def extract(engine, settings, data: bytes, words_bytes: bytes | None = None, *,
            fmt: str = "json", include_crops: bool = False, rotation: str = "auto",
            deskew: bool | None = None) -> dict[str, Any]:
    """Read one page image and return the API document: tables, cells, boxes, scores, crops."""
    if fmt not in render.FORMATS:
        raise ApiError(400, "invalid_format", f"format must be one of {', '.join(render.FORMATS)}")
    rotation = "auto" if rotation is None else str(rotation).strip().lower()
    if rotation not in ROTATIONS:
        raise ApiError(400, "invalid_rotation", "rotation must be auto, 0, 90, 180 or 270")
    if words_bytes is not None and len(words_bytes) > settings.max_words_mb * 1024 * 1024:
        raise ApiError(413, "words_too_large",
                       f"the words file is larger than {settings.max_words_mb} MiB")

    started = time.perf_counter()
    page = decode.load(data, settings.max_image_mb * 1024 * 1024, settings.max_image_pixels)

    quarter = page.exif_turn
    if rotation == "auto":
        if settings.ocr_engine == "tesseract" and ocr.available():
            quarter = (quarter + ocr.orientation(page.image)) % 360
    else:
        quarter = int(rotation) % 360

    working, matrix, angle = _prepare(page.image, quarter, settings.deskew if deskew is None
                                      else bool(deskew))

    if words_bytes is not None:
        provided = words_module.parse(words_bytes)
        words_module.check_frame(provided, page.width, page.height)
        words = words_module.map_words(provided, matrix)
        source = "provided_words"
    elif settings.ocr_engine == "tesseract" and ocr.available():
        words = ocr.words_from_image(working)
        source = "ocr"
    else:
        words = []
        source = "none"

    prediction = engine.predict(working, words, working.size)
    inverse = np.linalg.inv(matrix)

    def move(box):
        return geometry.clip_box(geometry.map_box(inverse, box), page.width, page.height)

    image_id = hashlib.sha256(data).hexdigest()[:12]
    tables = []
    for index, table in enumerate(prediction.tables):
        identifier = f"{image_id}-t{index}"
        record = render.table_record(table, index, identifier, fmt, move)
        record["crop_url"] = (f"/v1/tables/{identifier}/crop?box="
                              + ",".join(f"{value:.1f}" for value in record["model_box"]))
        if include_crops:
            record["crops"] = _crops(page.image, record)
        tables.append(record)

    return {
        "object": "ae3.image.extraction",
        "service": "ae3-image",
        "model": {"id": settings.model_name, "version": engine.config.version},
        "image_id": image_id,
        "image": {"kind": page.kind, "width": page.width, "height": page.height,
                  "bytes": len(data)},
        "transform": {"rotation": quarter, "deskew_deg": round(angle, 2)},
        "source": source,
        "words": len(words),
        "format": fmt,
        "tables_total": len(tables),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "tables": tables,
    }


def _prepare(image: Image.Image, quarter: int, deskew: bool):
    """Turn and deskew the page, and return the map from upload pixels to the working image."""
    matrix = np.eye(3)
    if quarter:
        image, step = geometry.turn(image, quarter)
        matrix = step @ matrix
    angle = 0.0
    if deskew:
        angle = geometry.estimate_skew(image)
        if angle:
            image, step = geometry.turn(image, angle)
            matrix = step @ matrix
    return image, matrix, angle


def _crops(image: Image.Image, record: dict) -> dict[str, str]:
    """PNG data URLs cut from the uploaded image, at the model's table box and at every cell box."""
    crops = {}
    table_crop = _crop(image, record["model_box"])
    if table_crop:
        crops["table"] = table_crop
    cells = []
    for cell in record.get("cells") or []:
        cut = _crop(image, cell["box"])
        if cut:
            cells.append(cut)
    if cells:
        crops["cells"] = cells
    return crops


def _crop(image: Image.Image, box) -> str | None:
    x0, y0, x1, y1 = (int(round(value)) for value in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.width, x1), min(image.height, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    buffer = io.BytesIO()
    image.crop((x0, y0, x1, y1)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
