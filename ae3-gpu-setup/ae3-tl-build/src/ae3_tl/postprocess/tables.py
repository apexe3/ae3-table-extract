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

"""Detections on one page -> the tables of the page, in the potatr-infer JSON shape."""

import html as html_escape
import os
from typing import Any, Sequence

from ae3_tl.postprocess import cell_fast as CF
from ae3_tl.postprocess import kensho_text
from ae3_tl.postprocess import postprocess as PP

TABLE_CLASSES = (0, 8)
CAPTION_CLASSES = (6, 14)
FOOTER_CLASSES = (7, 15)
STRUCTURE_CLASSES = (1, 2, 3, 4, 5, 9, 10, 11, 12, 13)
PARENT_CLASSES = (0, 6, 7, 8, 14, 15)

TABLE_LABELS = {0: "table", 8: "table rotated"}

# detector class -> Table Transformer class name, the map the cell builder expects
T2C = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 8: 0, 9: 2, 10: 1, 11: 3, 12: 4, 13: 5}
FAMILY = {0: (1, 2, 3, 4, 5), 8: (9, 10, 11, 12, 13)}

CELL_CLASS_NAMES = ["table", "table row", "table column", "table column header",
                    "table projected row header", "table spanning cell"]
CELL_CLASS_MAP = {i: n for i, n in enumerate(CELL_CLASS_NAMES)}
CELL_THRESHOLDS = {n: 0.5 for n in CELL_CLASS_NAMES}

CELL_BUILDERS = ("fast", "legacy")


def cell_builder() -> str:
    """The cell builder to use: AE3_CELL_BUILDER=fast (the default) or legacy."""
    name = (os.environ.get("AE3_CELL_BUILDER") or "fast").strip().lower()
    if name not in CELL_BUILDERS:
        raise ValueError(f"AE3_CELL_BUILDER must be fast or legacy, got {name!r}")
    return name


def pad_box(box: Sequence[float], pad: float) -> list[float]:
    return [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad]


def rotate_box(box: Sequence[float], height: float) -> list[float]:
    """Page coordinates -> the frame of the page rotated 90 degrees clockwise."""
    x0, y0, x1, y1 = box
    return [height - y1 - 1.0, x0, height - y0 - 1.0, x1]


def unrotate_box(box: Sequence[float], width: float) -> list[float]:
    """Inverse of rotate_box: that frame -> page coordinates."""
    x0, y0, x1, y1 = box
    return [y0, width - x1 - 1.0, y1, width - x0 - 1.0]


def text_in_box(box: Sequence[float], words: Sequence[dict]) -> str:
    """The text of the words at least half inside a box, top to bottom."""
    x0, y0, x1, y1 = box
    picked = []
    for word in words:
        b = word["bbox"]
        iw = min(x1, b[2]) - max(x0, b[0])
        ih = min(y1, b[3]) - max(y0, b[1])
        if iw <= 0 or ih <= 0:
            continue
        area = (b[2] - b[0]) * (b[3] - b[1])
        if area > 0 and iw * ih >= 0.5 * area:
            picked.append((b[1], b[0], word["text"]))
    picked.sort()
    return " ".join(text for _, _, text in picked)


def table_cells(table: dict, structure: Sequence[dict], words: Sequence[dict],
                pad: float) -> tuple[list[dict[str, Any]], list[float], float]:
    """Cells of one table, from its structure objects and the page words.

    The cell builder works in the table's own frame: the words and objects are moved so the padded
    table box starts at (0, 0). For a rotated table they are also rotated into reading orientation,
    and the cells are rotated back afterwards, so every cell box is in page pixels. The second
    return value is the box the builder derives from the detected rows and columns, in page pixels;
    the third is its confidence that the words slotted into the cells.

    AE3_CELL_BUILDER picks the builder: fast (the default) or legacy.
    """
    box = pad_box(table["bbox"], pad)
    ox, oy = box[0], box[1]
    width, height = box[2] - box[0], box[3] - box[1]
    rotated = table["cls"] == 8

    if cell_builder() == "fast":
        cells, content, confidence = CF.cells_for_table(structure, words, table["bbox"],
                                                        rotated, pad)
    else:
        cells, content, confidence = _legacy_cells(box, table, structure, words)

    if rotated:
        content = unrotate_box(content, width)
    content = [content[0] + ox, content[1] + oy, content[2] + ox, content[3] + oy]

    for cell in cells:
        if rotated:
            cell["bbox"] = unrotate_box(cell["bbox"], width)
        cell["bbox"] = [cell["bbox"][0] + ox, cell["bbox"][1] + oy,
                        cell["bbox"][2] + ox, cell["bbox"][3] + oy]
    return cells, [round(float(value), 2) for value in content], confidence


def _legacy_cells(box: Sequence[float], table: dict, structure: Sequence[dict],
                  words: Sequence[dict]) -> tuple[list[dict[str, Any]], list[float], float]:
    """The per-object builder in postprocess.py, in the table's frame."""
    ox, oy = box[0], box[1]
    width, height = box[2] - box[0], box[3] - box[1]
    rotated = table["cls"] == 8

    tokens = []
    for word in words:
        if PP.iob(word["bbox"], box) < 0.5:
            continue
        tokens.append({
            "bbox": [word["bbox"][0] - ox, word["bbox"][1] - oy,
                     word["bbox"][2] - ox, word["bbox"][3] - oy],
            "text": word["text"],
            "block_num": word.get("block_num", 0),
            "line_num": word.get("line_num", 0),
            "span_num": word.get("span_num", 0),
            "flags": word.get("flags", 0),
        })
    objects = [{
        "bbox": [o["bbox"][0] - ox, o["bbox"][1] - oy, o["bbox"][2] - ox, o["bbox"][3] - oy],
        "label": T2C[o["cls"]],
        "score": o["score"],
        "page_num": 0,
    } for o in structure]

    table_box = [0.0, 0.0, width, height]
    if rotated:
        for token in tokens:
            token["bbox"] = rotate_box(token["bbox"], height)
        for obj in objects:
            obj["bbox"] = rotate_box(obj["bbox"], height)
        table_box = rotate_box(table_box, height)

    table_object = {"bbox": table_box, "page_num": 0}
    _structures, cells, confidence = PP.objects_to_cells(
        table_object, objects, tokens, CELL_CLASS_MAP, CELL_THRESHOLDS)
    return cells, table_object["bbox"], confidence

def cell_records(cells: Sequence[dict]) -> list[dict[str, Any]]:
    """The public view of a cell: its rows, columns, text, header flags and page box."""
    return [{
        "row_nums": [int(v) for v in cell["row_nums"]],
        "column_nums": [int(v) for v in cell["column_nums"]],
        "cell_text": cell.get("cell_text") or "",
        "header": bool(cell.get("header", False)),
        "subheader": bool(cell.get("subheader", False)),
        "bbox": [round(float(v), 2) for v in cell["bbox"]],
    } for cell in cells]


def cells_to_html(cells: Sequence[dict]) -> str:
    """Cells -> an HTML table with colspan and rowspan, the form TEDS compares."""
    ordered = sorted(cells, key=lambda c: min(c["column_nums"]))
    ordered = sorted(ordered, key=lambda c: min(c["row_nums"]))

    out = ["<table>"]
    current_row = -1
    section = None
    row_open = False
    for cell in ordered:
        row = min(cell["row_nums"])
        if row > current_row:
            current_row = row
            if row_open:
                out.append("</tr>")
            wanted = "thead" if cell.get("header") else "tbody"
            if section != wanted:
                if section is not None:
                    out.append("</%s>" % section)
                out.append("<%s>" % wanted)
                section = wanted
            out.append("<tr>")
            row_open = True
        tag = "th" if section == "thead" else "td"
        attributes = ""
        if len(cell["column_nums"]) > 1:
            attributes += ' colspan="%d"' % len(cell["column_nums"])
        if len(cell["row_nums"]) > 1:
            attributes += ' rowspan="%d"' % len(cell["row_nums"])
        out.append("<%s%s>%s</%s>" % (tag, attributes,
                                      html_escape.escape(cell.get("cell_text") or ""), tag))
    if row_open:
        out.append("</tr>")
    if section is not None:
        out.append("</%s>" % section)
    out.append("</table>")
    return "".join(out)


def assign_structure_to_tables(tables: Sequence[dict], structure: Sequence[dict],
                               pad: float) -> dict[int, list[dict]]:
    """Each structure object goes to the table it is mostly inside, of the matching class family."""
    per_table: dict[int, list[dict]] = {i: [] for i in range(len(tables))}
    for obj in structure:
        best, chosen = 0.0, None
        for i, table in enumerate(tables):
            if obj["cls"] not in FAMILY[table["cls"]]:
                continue
            overlap = PP.iob(obj["bbox"], pad_box(table["bbox"], pad))
            if overlap > best:
                best, chosen = overlap, i
        if chosen is not None and best >= 0.5:
            per_table[chosen].append(obj)
    return per_table


LINK_RULES = ("best", "legacy")


def link_rule() -> str:
    """AE3_CAPTION_LINKING: best (the default) or legacy.

    best gives a caption or footer box the one table with the highest mean of the two relation-head
    directions, the rule the v3 checkpoint is scored with. legacy keeps the earlier per-pair 0.5
    threshold, which can link one box to several tables and leaves a box unlinked below it.
    """
    name = (os.environ.get("AE3_CAPTION_LINKING") or "best").strip().lower()
    if name not in LINK_RULES:
        raise ValueError(f"AE3_CAPTION_LINKING must be best or legacy, got {name!r}")
    return name


def link_min_score() -> float:
    """AE3_LINK_MIN_SCORE: the lowest best-table score a best link may carry (default 0.0)."""
    return float(os.environ.get("AE3_LINK_MIN_SCORE") or 0.0)


def parts_for(boxes: Sequence[dict], words: Sequence[dict],
              choices: Sequence[tuple[int, float]]) -> list[dict[str, Any]]:
    """The chosen boxes of one table, with their text, top to bottom."""
    parts = []
    for index, score in choices:
        box = boxes[index]
        text, _ = kensho_text.extract_text_inside_bbox(words, box["bbox"])
        text = text.strip()
        if not text:
            continue
        parts.append({
            "text": text,
            "bbox": [round(float(v), 2) for v in box["bbox"]],
            "score": round(float(box["score"]), 4),
            "edge_score": round(score, 4),
        })
    parts.sort(key=lambda part: part["bbox"][1])
    return parts


def linked_parts(boxes: Sequence[dict], probabilities: Sequence[Sequence[Sequence[float]]],
                 words: Sequence[dict], table_index: int, threshold: float) -> list[dict[str, Any]]:
    """The boxes the per-pair threshold links to one table, with their text, top to bottom.

    probabilities[i][j] is the pair (box i -> table j, table j -> box i) of relation-head scores.
    The threshold rule reads the first: a box belongs to the table when that score reaches the
    threshold, so one box can be linked to several tables on a page.
    """
    return parts_for(boxes, words,
                     [(i, float(probabilities[i][table_index][0])) for i in range(len(boxes))
                      if float(probabilities[i][table_index][0]) >= threshold])


def best_links(probabilities: Sequence[Sequence[Sequence[float]]],
               min_score: float) -> list[tuple[int, float]]:
    """The table each box belongs to under the v3 rule, and the score that chose it.

    The score of a box and a table is the mean of the two relation-head directions. The box goes to
    the table with the highest mean, and only when that mean reaches min_score; a box below it is
    left unlinked, as (-1, score).
    """
    links = []
    for row in probabilities:
        means = [0.5 * (float(pair[0]) + float(pair[1])) for pair in row]
        if not means:
            links.append((-1, 0.0))
            continue
        best = max(range(len(means)), key=lambda j: means[j])
        links.append((best, means[best]) if means[best] >= min_score else (-1, means[best]))
    return links


def best_parts(boxes: Sequence[dict], links: Sequence[tuple[int, float]],
               words: Sequence[dict], table_index: int) -> list[dict[str, Any]]:
    """The boxes whose best table is this one, with their text, top to bottom."""
    return parts_for(boxes, words,
                     [(i, score) for i, (table, score) in enumerate(links)
                      if table == table_index])


def page_tables(objects: Sequence[dict], words: Sequence[dict],
                caption_probabilities: Sequence[Sequence[Sequence[float]]],
                footer_probabilities: Sequence[Sequence[Sequence[float]]],
                edge_threshold: float, pad: float) -> list[dict[str, Any]]:
    """All tables on a page, ready for the API response.

    objects are the detections above the object threshold, with cls, bbox, score and qid.
    caption_probabilities[i][j] and footer_probabilities[i][j] are the pair of relation-head scores
    between box i and table j; link_rule() decides which tables the boxes go to.
    """
    tables = [o for o in objects if o["cls"] in TABLE_CLASSES]
    structure = [o for o in objects if o["cls"] in STRUCTURE_CLASSES]
    captions = [o for o in objects if o["cls"] in CAPTION_CLASSES]
    footers = [o for o in objects if o["cls"] in FOOTER_CLASSES]
    per_table = assign_structure_to_tables(tables, structure, pad)
    by_best = link_rule() == "best"
    if by_best:
        caption_links = best_links(caption_probabilities, link_min_score())
        footer_links = best_links(footer_probabilities, link_min_score())

    out = []
    for i, table in enumerate(tables):
        cells, content_box, _confidence = table_cells(table, per_table[i], words, pad)
        if by_best:
            caption_parts = best_parts(captions, caption_links, words, i)
            footer_parts = best_parts(footers, footer_links, words, i)
        else:
            caption_parts = linked_parts(captions, caption_probabilities, words, i, edge_threshold)
            footer_parts = linked_parts(footers, footer_probabilities, words, i, edge_threshold)

        out.append({
            "label": TABLE_LABELS[table["cls"]],
            "bbox": [round(float(v), 2) for v in table["bbox"]],
            "model_bbox": [round(float(v), 2) for v in table["bbox"]],
            "content_bbox": content_box,
            "score": round(float(table["score"]), 4),
            "html": cells_to_html(cells),
            "cells": cell_records(cells),
            "caption": " ".join(part["text"] for part in caption_parts),
            "footer": " ".join(part["text"] for part in footer_parts),
            "caption_parts": caption_parts,
            "footer_parts": footer_parts,
        })
    return out
