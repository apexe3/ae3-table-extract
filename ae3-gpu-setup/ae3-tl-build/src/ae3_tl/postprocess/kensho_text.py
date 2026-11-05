"""Kensho text assembly, copied unchanged from POTATR (MIT).

Source: potatr/postprocess.py, commit fdb8f57,
https://github.com/kensho-technologies/potatr
MIT License, Copyright (c) 2025-present Kensho Technologies, LLC.
Caption and footer strings are built with these functions so that AE3-TL and POTATR
are assembled the same way.
"""
from __future__ import annotations

from typing import Any, Sequence

Object = Any
BBox = list

def _bbox_area(bbox: Sequence[float]) -> float:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _bbox_intersect(a: Sequence[float], b: Sequence[float]) -> BBox:
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def overlaps(bbox1: Sequence[float], bbox2: Sequence[float], threshold: float = 0.5) -> bool:
    """Test if more than "threshold" fraction of bbox1 overlaps with bbox2."""
    area1 = _bbox_area(bbox1)
    if area1 == 0:
        return False
    return _bbox_area(_bbox_intersect(bbox1, bbox2)) / area1 >= threshold


def get_bbox_span_subset(
    spans: list[Object], bbox: Sequence[float], threshold: float = 0.5
) -> list[Object]:
    """Reduce the set of spans to those that fall within a bounding box.

    threshold: the fraction of the span that must overlap with the bbox.
    """
    span_subset = []
    for span in spans:
        if overlaps(span["bbox"], bbox, threshold):
            span_subset.append(span)
    return span_subset


def extract_text_from_spans(spans: list[Object], join_with_space: bool = True) -> str:
    """Convert a collection of page tokens/words/spans into a single text string."""

    if join_with_space:
        join_char = " "
    else:
        join_char = ""
    spans_copy = spans[:]

    if len(spans_copy) == 0:
        return ""

    spans_copy.sort(key=lambda span: span["span_num"])
    spans_copy.sort(key=lambda span: span["line_num"])
    spans_copy.sort(key=lambda span: span["block_num"])

    # Force the span at the end of every line within a block to have exactly one space
    # unless the line ends with a space or ends with a non-space followed by a hyphen
    line_texts = []
    line_span_texts = [spans_copy[0]["text"].strip()]
    for span1, span2 in zip(spans_copy[:-1], spans_copy[1:]):
        if (
            not span1["block_num"] == span2["block_num"]
            or not span1["line_num"] == span2["line_num"]
        ):
            line_text = join_char.join(line_span_texts).strip()
            if (
                len(line_text) > 0
                and not line_text[-1] == " "
                and not (len(line_text) > 1 and line_text[-1] == "-" and not line_text[-2] == " ")
            ):
                if not join_with_space:
                    line_text += " "
            line_texts.append(line_text)
            line_span_texts = [span2["text"].strip()]
        else:
            line_span_texts.append(span2["text"].strip())
    line_text = join_char.join(line_span_texts)
    line_texts.append(line_text)

    return join_char.join(line_texts).strip()


def extract_text_inside_bbox(
    spans: list[Object], bbox: Sequence[float]
) -> tuple[str, list[Object]]:
    """Extract the text inside a bounding box."""
    bbox_spans = get_bbox_span_subset(spans, bbox)
    bbox_text = extract_text_from_spans(bbox_spans)

    return bbox_text, bbox_spans
