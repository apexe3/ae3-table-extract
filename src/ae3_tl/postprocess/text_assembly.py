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
"""Assemble caption and footer text from OCR words.

The words a caption or footer box covers are filtered by overlap, ordered in
reading order (block, then line, then span) and joined with single spaces —
the same assembly rule the AE3-TL training pipeline used to build its labels.
"""

from __future__ import annotations

from collections.abc import Sequence


def bbox_area(bbox: Sequence[float]) -> float:
    """The area of a box, clamped to zero for inverted or empty boxes."""
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def covers(box: Sequence[float], region: Sequence[float], minimum: float = 0.5) -> bool:
    """True when at least "minimum" of the box's area lies inside the region."""
    width = min(box[2], region[2]) - max(box[0], region[0])
    height = min(box[3], region[3]) - max(box[1], region[1])
    area = bbox_area(box)
    if area <= 0:
        return False
    return max(0.0, width) * max(0.0, height) / area >= minimum


def words_inside_bbox(words: Sequence[dict], bbox: Sequence[float],
                      minimum: float = 0.5) -> list[dict]:
    """The words whose boxes overlap the target box, in reading order."""
    inside = [word for word in words if covers(word["bbox"], bbox, minimum)]
    inside.sort(key=lambda word: (word["block_num"], word["line_num"], word["span_num"]))
    return inside


def extract_text_inside_bbox(words: Sequence[dict], bbox: Sequence[float]) -> tuple[str, list[dict]]:
    """The text of a caption or footer box: its words joined with single spaces."""
    inside = words_inside_bbox(words, bbox)
    return " ".join(word["text"].strip() for word in inside), inside
