#!/usr/bin/env python3
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
"""Turn the extraction JSON written by run_local.py plus the source image into a Markdown file.

Reads the tables from the JSON file — no service call — renders every table with
tabulate as a GitHub-flavored Markdown table, and can also cut each table region
out of the image into the report:

  python examples/extract_to_md.py --json tables.json --image page.png -o tables.md
  python examples/extract_to_md.py --json tables.json --image page.png --pictures -o tables.md

The image can also be an http(s) URL.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image
from tabulate import tabulate


def table_markdown(table: dict, page: int) -> str:
    """One table as a Markdown block: heading, grid, caption/footer when present."""
    cells = table.get("cells") or []
    rows, cols = table.get("rows", 0), table.get("columns", 0)
    if not cells or not rows or not cols:
        return f"### Table {table['index']} (page {page})\n\n_Empty_\n"

    grid = [[None] * cols for _ in range(rows)]
    spans = []
    for cell in cells:
        r, c = cell["row"], cell["column"]
        text = (cell.get("text") or "").strip().replace("\n", " ").replace("|", "\\|")
        if cell.get("header") or cell.get("subheader"):
            text = f"**{text}**" if text else ""
        rs, cs = cell.get("row_span", 1), cell.get("col_span", 1)
        if rs > 1 or cs > 1:
            spans.append((r, c, rs, cs, text))
            continue
        grid[r][c] = text

    for r, c, rs, cs, text in spans:
        if r < rows and c < cols:
            grid[r][c] = text or ""
        for rr in range(r, min(r + rs, rows)):
            for cc in range(c, min(c + cs, cols)):
                if (rr, cc) != (r, c):
                    grid[rr][cc] = ""

    header = grid[0]
    body = grid[1:] if rows > 1 else []
    parts = [f"### Table {table['index']} — page {page} "
             f"({rows}x{cols}, score {table.get('score', 0):.2f})\n"]
    if table.get("caption"):
        parts.append(f"**Caption:** {table['caption']}\n")
    parts.append(tabulate(body or [""], headers=header or [""] * cols,
                          tablefmt="github", missingval=""))
    if table.get("footer"):
        parts.append(f"\n_Footer:_ {table['footer']}")
    return "\n".join(parts) + "\n"


def crop_data_url(image: Image.Image, box: list[float]) -> str | None:
    """The table region as a PNG data URL, or None when the box is empty."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.width, x1), min(image.height, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    buffer = io.BytesIO()
    image.crop((x0, y0, x1, y1)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def document_markdown(doc: dict, source: str, image: Image.Image | None = None,
                      pictures: bool = False) -> str:
    """The whole extraction as one Markdown document."""
    lines = [f"# AE3-TL extraction — {source}\n",
             f"- model: {doc['model']['id']} {doc['model'].get('version', '')}",
             f"- tables: {doc.get('tables_total')} · "
             f"words: {doc.get('words', '?')} · "
             f"took {doc.get('elapsed_ms', 0):.0f} ms\n"]
    # run_local.py writes the image-level document: tables sit at the top level
    pages = doc.get("pages") or [{"page": 1, "tables": doc.get("tables") or []}]
    for page in pages:
        lines.append(f"## Page {page['page']}\n")
        tables = page.get("tables") or []
        if not tables:
            lines.append("_No tables detected._\n")
        for table in tables:
            lines.append(table_markdown(table, page["page"]))
            if pictures and image is not None and (table.get("model_box") or table.get("box")):
                data_url = crop_data_url(image, table.get("model_box") or table.get("box"))
                if data_url:
                    lines.append(f"![table {table['index']} crop]({data_url})\n")
    return "\n".join(lines) + "\n"


def _load_image(source: str) -> Image.Image:
    """A page image from a local path or an http(s) URL."""
    if source.startswith(("http://", "https://")):
        import httpx
        response = httpx.get(source, headers={"User-Agent": "Mozilla/5.0"},
                             timeout=60.0, follow_redirects=True)
        if response.status_code != 200:
            sys.exit(f"image fetch failed: HTTP {response.status_code} for {source}")
        data = response.content
    else:
        path = Path(source)
        if not path.exists():
            sys.exit(f"image not found: {source}")
        data = path.read_bytes()
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as error:
        sys.exit(f"could not decode image: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", required=True, type=Path,
                        help="extraction document (as written by run_local.py -o)")
    parser.add_argument("--image", help="the page image the JSON came from: a local "
                                        "path or an http(s) URL")
    parser.add_argument("--pictures", action="store_true",
                        help="embed a cropped picture of each table in the report")
    parser.add_argument("-o", "--out", default="tables.md", help="output Markdown file")
    args = parser.parse_args()

    doc = json.loads(args.json.read_text())
    image = None
    name = args.json.stem
    if args.image:
        image = _load_image(args.image)
        name = args.image.rsplit("/", 1)[-1].split("?")[0] or "image"
    elif args.pictures:
        sys.exit("--pictures needs --image")

    output = document_markdown(doc, name, image=image, pictures=args.pictures)
    Path(args.out).write_text(output)
    print(f"wrote {args.out}: {doc.get('tables_total', 0)} tables, {len(output)} chars")


if __name__ == "__main__":
    main()
