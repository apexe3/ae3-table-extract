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

"""One table as a record, and that record in markdown, html or csv.

A cell record is {"row", "column", "row_span", "col_span", "box", "text", "header"}. The
record is the only shape the API returns; the formats are renderings of it.
"""

import csv
import html
import io
from typing import Any, Callable, Sequence

FORMATS = ("json", "markdown", "html", "csv")


def shape(cells: Sequence[dict]) -> tuple[int, int]:
    rows = max((cell["row"] + cell["row_span"] for cell in cells), default=0)
    columns = max((cell["column"] + cell["col_span"] for cell in cells), default=0)
    return rows, columns


def grid(cells: Sequence[dict], rows: int, columns: int) -> list[list[str]]:
    """Plain text grid: a spanning cell puts its text at its top left corner."""
    out = [["" for _ in range(columns)] for _ in range(rows)]
    for cell in cells:
        if cell["row"] < rows and cell["column"] < columns:
            out[cell["row"]][cell["column"]] = cell.get("text") or ""
    return out


def to_markdown(cells: Sequence[dict], rows: int, columns: int) -> str:
    if not rows or not columns:
        return ""
    values = grid(cells, rows, columns)
    header = any(cell["header"] and cell["row"] == 0 for cell in cells)
    lines = []
    for index, row in enumerate(values):
        cells_text = [(value or "").replace("|", "\\|").replace("\n", "<br>") for value in row]
        lines.append("| " + " | ".join(cells_text) + " |")
        if header and index == 0:
            lines.append("|" + "|".join([" --- "] * columns) + "|")
    return "\n".join(lines)


def to_csv(cells: Sequence[dict], rows: int, columns: int) -> str:
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(grid(cells, rows, columns))
    return buffer.getvalue()


def to_html(cells: Sequence[dict], rows: int, columns: int) -> str:
    if not cells:
        return ""
    by_row: dict[int, list[dict]] = {}
    for cell in cells:
        by_row.setdefault(cell["row"], []).append(cell)
    header = any(cell["header"] for cell in cells)
    out = ["<table>"]
    for row in range(rows):
        out.append("<tr>")
        for cell in sorted(by_row.get(row, []), key=lambda item: item["column"]):
            tag = "th" if header and cell["header"] else "td"
            attributes = ""
            if cell["col_span"] > 1:
                attributes += f' colspan="{cell["col_span"]}"'
            if cell["row_span"] > 1:
                attributes += f' rowspan="{cell["row_span"]}"'
            out.append(f"<{tag}{attributes}>{html.escape(cell.get('text') or '')}</{tag}>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def body(cells: Sequence[dict], fmt: str) -> str | None:
    if fmt == "json":
        return None
    rows, columns = shape(cells)
    if fmt == "markdown":
        return to_markdown(cells, rows, columns)
    if fmt == "csv":
        return to_csv(cells, rows, columns)
    return to_html(cells, rows, columns)


def table_record(table: dict, index: int, identifier: str, fmt: str = "json",
                 move: Callable[[Sequence[float]], list[float]] | None = None) -> dict[str, Any]:
    """One table of the model output as the API record.

    box is the table box of the response, model_box is the detector's own table box of that
    table, and content_box is the box the cell builder derives from the detected rows and
    columns, which is the box that hugs the text. Crops are cut from model_box.

    move maps a box from the model's frame to the caller's frame; the image service passes the
    inverse of its turn and deskew, the PDF service passes nothing.
    """
    move = move or (lambda box: [float(value) for value in box])
    cells = []
    for cell in table.get("cells") or []:
        cells.append({
            "row": min(cell["row_nums"]),
            "column": min(cell["column_nums"]),
            "row_span": len(cell["row_nums"]),
            "col_span": len(cell["column_nums"]),
            "box": move(cell["bbox"]),
            "text": cell.get("cell_text") or "",
            "header": bool(cell.get("header")),
            "subheader": bool(cell.get("subheader")),
        })
    rows, columns = shape(cells)
    row_boxes = []
    for row in range(rows):
        boxes = [cell["box"] for cell in cells if cell["row"] == row]
        row_boxes.append(_union(boxes))
    column_boxes = []
    for column in range(columns):
        boxes = [cell["box"] for cell in cells if cell["column"] == column]
        column_boxes.append(_union(boxes))
    captions = table.get("caption_parts") or []
    footers = table.get("footer_parts") or []
    record = {
        "id": identifier,
        "index": index,
        "box": move(table["bbox"]),
        "model_box": move(table.get("model_bbox") or table["bbox"]),
        "content_box": (move(table["content_bbox"]) if table.get("content_bbox") else None),
        "label": table.get("label") or "table",
        "score": table.get("score"),
        "rows": rows,
        "columns": columns,
        "caption": (table.get("caption") or "").strip(),
        "footer": (table.get("footer") or "").strip(),
        "caption_box": move(captions[0]["bbox"]) if captions else None,
        "footer_box": move(footers[0]["bbox"]) if footers else None,
        "rows_boxes": row_boxes,
        "columns_boxes": column_boxes,
        "cells": cells,
    }
    rendered = body(cells, fmt)
    if rendered is not None:
        record["body"] = rendered
    return record


def _union(boxes: Sequence[Sequence[float]]) -> list[float] | None:
    boxes = [box for box in boxes if box]
    if not boxes:
        return None
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]
