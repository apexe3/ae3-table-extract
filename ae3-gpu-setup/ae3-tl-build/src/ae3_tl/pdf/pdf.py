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

"""PDF -> page images (pypdfium2) and page words (pdfplumber).

Page images are rendered at a chosen DPI; words are extracted in PDF points and scaled to the
rendered pixel size, so word boxes and model boxes share one coordinate frame.
"""

import io
from typing import Iterator

import pdfplumber
import pypdfium2 as pdfium
from PIL import Image

POINTS_PER_INCH = 72.0


def page_count(pdf_bytes: bytes) -> int:
    """Number of pages in the PDF."""
    with pdfium.PdfDocument(pdf_bytes) as document:
        return len(document)


def words_from_page(page: pdfplumber.page.Page, scale: float) -> list[dict]:
    """A page's words as boxes in rendered pixels, grouped into lines.

    pdfplumber gives word boxes in PDF points; multiplying them by scale puts them in the
    coordinate frame of the rendered page image. Words are ordered top to bottom, left to right,
    and numbered by line and by position within the line.
    """
    extracted = page.extract_words(keep_blank_chars=False, use_text_flow=False,
                                   extra_attrs=["size"])
    if not extracted:
        return []
    ordered = sorted(extracted, key=lambda word: (round(word["top"], 1), word["x0"]))

    words = []
    line_num = -1
    line_top = None
    span_num = 0
    for word in ordered:
        height = max(1.0, word["bottom"] - word["top"])
        tolerance = max(1.5, 0.3 * height)
        if line_top is None or abs(word["top"] - line_top) > tolerance:
            line_num += 1
            line_top = word["top"]
            span_num = 0
        words.append({
            "bbox": [word["x0"] * scale, word["top"] * scale,
                     word["x1"] * scale, word["bottom"] * scale],
            "text": word["text"],
            "block_num": 0,
            "line_num": line_num,
            "span_num": span_num,
            "flags": 0,
        })
        span_num += 1
    return words


def iter_pages(pdf_bytes: bytes, dpi: int, first_page: int = 0, last_page: int | None = None
               ) -> Iterator[tuple[int, Image.Image, list[dict]]]:
    """Yield (page index from 0, page image, words) for the requested page range."""
    scale = dpi / POINTS_PER_INCH
    document = pdfium.PdfDocument(pdf_bytes)
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as plumber:
            total = len(document)
            end = total if last_page is None else min(last_page, total)
            for index in range(max(0, first_page), end):
                bitmap = document[index].render(scale=scale)
                image = bitmap.to_pil().convert("RGB")
                words = words_from_page(plumber.pages[index], scale)
                yield index, image, words
    finally:
        document.close()
