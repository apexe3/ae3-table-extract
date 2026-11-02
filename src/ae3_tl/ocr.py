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

"""Tesseract, called as a program: words and page orientation, no OCR Python binding.

A word record is {"bbox": [x0, y0, x1, y1], "text": str, "block_num": int, "line_num": int,
"span_num": int}. Boxes are in the pixel frame of the image that was passed in, which is the
frame the cell builder already uses.
"""

import csv
import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

TIMEOUT_S = 120
ROTATE = re.compile(r"^Rotate:\s*(\d+)", re.MULTILINE)
UPSCALE_BELOW = 1000
UPSCALE_FACTOR = 2


def engine() -> str:
    """OCR_ENGINE: "tesseract" to read pages with Tesseract, "none" to skip OCR."""
    name = (os.environ.get("OCR_ENGINE") or "none").strip().lower()
    if name not in ("none", "tesseract"):
        raise ValueError(f"OCR_ENGINE must be tesseract or none, got {name!r}")
    return name


def available() -> bool:
    return shutil.which("tesseract") is not None


def _run(image, *args: str) -> str:
    with tempfile.TemporaryDirectory(prefix="ae3ocr-") as folder:
        path = Path(folder) / "page.png"
        _readable(image).save(path, format="PNG")
        done = subprocess.run(["tesseract", str(path), "stdout", *args],
                              capture_output=True, text=True, timeout=TIMEOUT_S)
        if done.returncode != 0:
            raise RuntimeError(f"tesseract failed: {done.stderr.strip()[:200]}")
        return done.stdout


def _readable(image):
    """Small thumbnails upscaled before tesseract: it reads tiny text far better enlarged."""
    if min(image.size) >= UPSCALE_BELOW:
        return image
    factor = UPSCALE_FACTOR
    while image.width * factor < UPSCALE_BELOW and image.height * factor < UPSCALE_BELOW:
        factor += 1
    return image.resize((image.width * factor, image.height * factor), Image.LANCZOS)


def words_from_image(image, psm: int = 3, lang: str = "eng") -> list[dict]:
    """Every word Tesseract sees, with its box, block, line and span numbers."""
    source = _readable(image)
    rows = list(csv.reader(io.StringIO(_run(source, "--psm", str(psm), "-l", lang, "tsv")),
                           delimiter="\t"))
    if not rows:
        return []
    header = rows[0]
    try:
        index = {name: header.index(name) for name in
                 ("block_num", "par_num", "line_num", "word_num", "left", "top", "width",
                  "height", "conf", "text")}
    except ValueError:
        return []
    words = []
    scale_x = image.width / source.width
    scale_y = image.height / source.height
    for row in rows[1:]:
        if len(row) < len(header):
            continue
        text = row[index["text"]].strip()
        if not text:
            continue
        try:
            confidence = float(row[index["conf"]])
        except ValueError:
            confidence = -1.0
        if confidence < 0:
            continue
        key = (int(row[index["block_num"]]), int(row[index["par_num"]]),
               int(row[index["line_num"]]))
        left, top = int(row[index["left"]]), int(row[index["top"]])
        words.append({
            "bbox": [left * scale_x, top * scale_y,
                     (left + int(row[index["width"]])) * scale_x,
                     (top + int(row[index["height"]])) * scale_y],
            "text": text,
            "block_num": key[0] * 1000 + key[1],
            "line_num": key[2],
            "span_num": int(row[index["word_num"]]),
            "confidence": round(confidence / 100.0, 4),
        })
    words.sort(key=lambda word: (word["block_num"], word["line_num"], word["span_num"]))
    return words


def orientation(image) -> int:
    """Quarter turn clockwise (0, 90, 180 or 270) that makes the page upright, from OSD.

    Returns 0 when Tesseract's orientation model is not installed or is unsure.
    """
    try:
        report = _run(image, "--psm", "0")
    except (RuntimeError, subprocess.SubprocessError):
        return 0
    match = ROTATE.search(report)
    if not match:
        return 0
    turn = int(match.group(1)) % 360
    return turn if turn in (0, 90, 180, 270) else 0
