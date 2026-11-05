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

"""Cell builder parity on the pinned pages: fast against legacy, and both against expected.json.

    python tests/parity/builders.py [--fixture tests/parity] [--dpi 200] [--out result.json]

Pages render from tests/parity/pages.pdf at --dpi and take their words from the PDF text layer, as
the PDF service does. Each page runs twice, with AE3_CELL_BUILDER=fast and then legacy, and the two
extractions must agree on every table: cells, html, boxes, content box, score, caption and footer.
Both runs are also checked against expected.json: page and table counts, rows, columns, caption
and the sha256 of the markdown body. The confidence field of expected.json is not compared: it was
written by an earlier build and the shipped 0.2.0 code, legacy path included, no longer produces
it (see the note in result.json).
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from ae3_tl.backends import create_engine
from ae3_tl.pdf.pdf import iter_pages
from ae3_tl.postprocess import tables as T
from ae3_tl.serve import render
from ae3_tl.serve.settings import load_settings

HERE = Path(__file__).resolve().parent
TABLE_FIELDS = ("label", "bbox", "model_bbox", "content_bbox", "score", "html",
                "caption", "footer", "caption_parts", "footer_parts")
CELL_FIELDS = ("row_nums", "column_nums", "cell_text", "header", "subheader", "bbox")


def run_page(engine, image, words, builder, confidences):
    os.environ["AE3_CELL_BUILDER"] = builder
    recorded = []
    original = T.table_cells

    def recording(*args, **kwargs):
        cells, content, confidence = original(*args, **kwargs)
        recorded.append(confidence)
        return cells, content, confidence

    T.table_cells = recording
    try:
        prediction = engine.predict(image, words, image.size)
    finally:
        T.table_cells = original
    confidences[builder] = recorded
    return prediction


def table_differences(where, fast, legacy, confidences):
    differences = []
    if len(fast) != len(legacy):
        return [f"{where}: {len(fast)} tables fast, {len(legacy)} legacy"]
    for index, (a, b) in enumerate(zip(confidences["fast"], confidences["legacy"])):
        if a != b:
            differences.append(f"{where} table {index}: confidence {a} fast, {b} legacy")
    for index, (a, b) in enumerate(zip(fast, legacy)):
        for name in TABLE_FIELDS:
            if a[name] != b[name]:
                differences.append(f"{where} table {index}: {name} differs")
        if len(a["cells"]) != len(b["cells"]):
            differences.append(f"{where} table {index}: {len(a['cells'])} cells fast, "
                               f"{len(b['cells'])} legacy")
            continue
        for cell_index, (ca, cb) in enumerate(zip(a["cells"], b["cells"])):
            if any(ca[name] != cb[name] for name in CELL_FIELDS):
                differences.append(f"{where} table {index} cell {cell_index}: differs")
    return differences


def summary_differences(where, prediction, pinned):
    differences = []
    if len(prediction.tables) != len(pinned["tables"]):
        return [f"{where}: {len(prediction.tables)} tables, expected {len(pinned['tables'])}"]
    for index, (table, want) in enumerate(zip(prediction.tables, pinned["tables"])):
        record = render.table_record(table, index, "parity", "markdown")
        if (record["rows"], record["columns"]) != (want["rows"], want["columns"]):
            differences.append(f"{where} table {index}: shape "
                               f"{(record['rows'], record['columns'])} != "
                               f"{(want['rows'], want['columns'])}")
        if record["caption"][:120] != want["caption"]:
            differences.append(f"{where} table {index}: caption differs")
        body = hashlib.sha256(record["body"].encode("utf-8")).hexdigest()
        if body != want["markdown_sha256"]:
            differences.append(f"{where} table {index}: markdown differs")
    return differences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=HERE)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    expected = json.loads((arguments.fixture / "expected.json").read_text())
    data = (arguments.fixture / expected["fixture"]).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected["fixture_sha256"]:
        print(f"FAIL fixture bytes changed: {digest} != {expected['fixture_sha256']}")
        return 1

    engine = create_engine(load_settings())
    rows, differences = [], []
    pages = tables = cells = mismatched_tables = 0
    for index, image, words in iter_pages(data, arguments.dpi):
        confidences = {}
        fast = run_page(engine, image, words, "fast", confidences)
        legacy = run_page(engine, image, words, "legacy", confidences)
        where = f"page {index + 1}"
        page_differences = (table_differences(where, fast.tables, legacy.tables, confidences)
                            + summary_differences(where, fast, expected["per_page"][index]))
        differences.extend(page_differences)
        pinned_tables = len(expected["per_page"][index]["tables"])
        pages += 1
        tables += len(fast.tables)
        cells += sum(len(table["cells"]) for table in fast.tables)
        mismatched_tables += len(page_differences)
        rows.append({"page": index + 1, "tables": len(fast.tables),
                     "cells": sum(len(table["cells"]) for table in fast.tables),
                     "differences": len(page_differences)})
        print(f"{where} tables={len(fast.tables)} (expected {pinned_tables}) "
              f"cells={rows[-1]['cells']} differences={len(page_differences)}", flush=True)

    for line in differences[:20]:
        print("  " + line)
    result = {"pages": pages, "tables": tables, "cells": cells,
              "tables_with_differences": mismatched_tables,
              "note": "expected.json confidence is from an earlier build and is not compared",
              "pages_with_differences": [row["page"] for row in rows if row["differences"]],
              "per_page": rows}
    if arguments.out:
        arguments.out.write_text(json.dumps(result, indent=1))
    ok = pages == expected["pages"] and tables == expected["tables"] and not differences
    print(f"RESULT pages={pages}/{expected['pages']} tables={tables}/{expected['tables']} "
          f"cells={cells} differing_tables={mismatched_tables} "
          f"{'PARITY OK' if ok else 'PARITY FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
