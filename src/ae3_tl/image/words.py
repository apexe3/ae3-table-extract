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

"""Words for one page: a POTATR-style words file, normalised and moved into the working frame.

Accepted inputs (the format is read from the bytes):
  XML   <words><word bbox="x0,y0,x1,y1">text</word></words>, as POTATR and PubTables write it
  JSON  [{"text": ..., "bbox": [x0, y0, x1, y1], "block_num": ..}, ...], or {"words": [...]},
        or {"pages": [{"width", "height", "words": [...]}]} with one page
  CSV   a header with a text column and a bbox column ("x0,y0,x1,y1" or four numbers)
"""

import csv
import io
import json
import re

import defusedxml
import defusedxml.ElementTree as ElementTree

import numpy as np

from ae3_tl.serve.errors import ApiError

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
BBOX_KEYS = ("bbox", "box", "bounds", "rect")
TEXT_KEYS = ("text", "word", "content", "value")


def _numbers(value) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        found = NUMBER.findall(value)
        if len(found) == 4:
            return [float(item) for item in found]
    return None


def _word(entry: dict) -> dict | None:
    text = None
    for key in TEXT_KEYS:
        if isinstance(entry.get(key), str):
            text = entry[key]
            break
    box = None
    for key in BBOX_KEYS:
        if key in entry:
            box = _numbers(entry[key])
            if box:
                break
    if box is None and all(axis in entry for axis in ("x0", "y0", "x1", "y1")):
        box = [float(entry[axis]) for axis in ("x0", "y0", "x1", "y1")]
    if box is None and all(axis in entry for axis in ("left", "top", "width", "height")):
        left, top = float(entry["left"]), float(entry["top"])
        box = [left, top, left + float(entry["width"]), top + float(entry["height"])]
    if not text or not text.strip() or box is None:
        return None
    if box[2] < box[0]:
        box[0], box[2] = box[2], box[0]
    if box[3] < box[1]:
        box[1], box[3] = box[3], box[1]
    return {
        "bbox": box,
        "text": text,
        "block_num": int(entry.get("block_num") or 0),
        "line_num": int(entry.get("line_num") or 0),
        "span_num": int(entry.get("span_num") or 0),
    }


def _from_json(document) -> list[dict]:
    if isinstance(document, dict):
        if isinstance(document.get("pages"), list):
            pages = document["pages"]
            if len(pages) != 1:
                raise ApiError(400, "invalid_words",
                               f"the words file holds {len(pages)} pages; send one page")
            document = pages[0]
        document = document.get("words") if isinstance(document, dict) else document
    if not isinstance(document, list):
        raise ApiError(400, "invalid_words",
                       "the words file must hold a list of words under 'words' or 'pages'")
    out = [word for word in (_word(entry) for entry in document if isinstance(entry, dict)) if word]
    return out


def _from_csv(text: str) -> list[dict]:
    delimiter = "\t" if text.split("\n", 1)[0].count("\t") >= 3 else ","
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    if not rows:
        return []
    out = []
    for row in rows:
        cleaned = {(key or "").strip().lower(): (value or "").strip() for key, value in row.items()}
        numbers = [float(value) for value in NUMBER.findall(cleaned.get("bbox", ""))]
        if len(numbers) == 4:
            cleaned["bbox"] = numbers
        word = _word(cleaned)
        if word is not None:
            out.append(word)
    return out


def _from_xml(data: bytes) -> list[dict]:
    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError as error:
        raise ApiError(400, "invalid_words", f"the words XML does not parse: {error}") from error
    except defusedxml.DefusedXmlException as error:
        raise ApiError(400, "invalid_words",
                       "the words XML asks for entities or an external reference; "
                       "plain words XML only") from error
    out = []
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1].lower()
        if tag not in ("word", "token"):
            continue
        entry = dict(element.attrib)
        entry["text"] = (element.text or "").strip()
        word = _word(entry)
        if word is not None:
            out.append(word)
    return out


def parse(data: bytes) -> list[dict]:
    """Words from a words file in XML, JSON or CSV form."""
    if not data or not data.strip():
        raise ApiError(400, "invalid_words", "the words part is empty")
    stripped = data.lstrip()
    if stripped.startswith(b"<"):
        words = _from_xml(data)
    elif stripped[:1] in (b"{", b"["):
        try:
            words = _from_json(json.loads(data))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ApiError(400, "invalid_words", f"the words JSON does not parse: {error}") from error
    else:
        try:
            words = _from_csv(data.decode("utf-8"))
        except (UnicodeDecodeError, csv.Error) as error:
            raise ApiError(400, "invalid_words", "the words file is neither XML, JSON nor CSV") from error
    if not words:
        raise ApiError(400, "invalid_words", "no word with a box and text was found")
    return _number_lines(words)


def _number_lines(words: list[dict]) -> list[dict]:
    """Give every word a line number, grouping words that share a baseline when none carries one."""
    if any(word["line_num"] for word in words):
        return words
    ordered = sorted(words, key=lambda word: ((word["bbox"][1] + word["bbox"][3]) / 2, word["bbox"][0]))
    line, baseline = 0, None
    for word in ordered:
        middle = (word["bbox"][1] + word["bbox"][3]) / 2
        height = max(1.0, word["bbox"][3] - word["bbox"][1])
        if baseline is None or abs(middle - baseline) > 0.6 * height:
            line += 1
            baseline = middle
        word["line_num"] = line
    for index, word in enumerate(ordered, start=1):
        word["span_num"] = index
    return ordered


def check_frame(words: list[dict], width: int, height: int) -> None:
    """A words file is in image pixels; refuse one that is clearly in another frame."""
    right = max(word["bbox"][2] for word in words)
    bottom = max(word["bbox"][3] for word in words)
    if right > width * 1.05 + 2 or bottom > height * 1.05 + 2:
        raise ApiError(400, "words_out_of_bounds",
                       f"the words reach {right:.0f}x{bottom:.0f} but the image is "
                       f"{width}x{height}; the boxes must be in image pixels")


def map_words(words: list[dict], matrix: np.ndarray) -> list[dict]:
    """Move word boxes from the uploaded image into the working (turned, deskewed) frame."""
    out = []
    for word in words:
        x0, y0, x1, y1 = word["bbox"]
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float).T
        moved = matrix @ np.vstack([corners, np.ones((1, 4))])
        out.append({**word,
                    "bbox": [float(moved[0].min()), float(moved[1].min()),
                             float(moved[0].max()), float(moved[1].max())]})
    return out
