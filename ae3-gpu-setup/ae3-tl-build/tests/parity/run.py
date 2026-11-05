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

"""50-page parity: replay each pinned page and compare arrays, tables, cells, boxes and scores.

    python tests/parity/run.py --fixture tests/parity --model DIR [--pdf-dir DIR] [--dpi 200]

The fixture holds pages.pdf, expected.json and SOURCES.json. --model holds ids.txt, words/<id>.json
and arrays/ort_cpu__<id>.npz, the pinned ONNX outputs of the v3 checkpoint.

Each page is rendered at --dpi from the source PDF (--pdf-dir, page <id>), then:

1. the shipped ONNX files run on that page and every output array is compared to the pin;
2. the extraction is built from the pinned arrays and the page words: the reference;
3. the extraction is built from the rendered page and the same words: the run.

Then the two extractions are compared: tables, cell records, identical cells, the largest table
box delta and the largest score delta. Exit code 0 only when the arrays match within 1e-4 and
every cell record is identical.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from ae3_tl.backends import create_engine
from ae3_tl.pdf.preprocess import page_tensors
from ae3_tl.serve.settings import load_settings

POINTS_PER_INCH = 72.0
FIELDS = ("logits", "boxes", "embeddings", "indices")


def render(pdf_path: Path, page_number: int, dpi: int):
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        return document[page_number - 1].render(scale=dpi / POINTS_PER_INCH).to_pil().convert("RGB")
    finally:
        document.close()


def cell_key(cell: dict) -> tuple:
    """The identity of a cell record, in either the post-processing or the API shape."""
    if "row_nums" in cell:
        return (tuple(cell["row_nums"]), tuple(cell["column_nums"]), cell.get("cell_text") or "",
                bool(cell.get("header")))
    return (cell["row"], cell["column"], cell["row_span"], cell["col_span"], cell["text"])


def compare(run_tables: list, reference_tables: list) -> dict:
    run_cells = [cell for table in run_tables for cell in table["cells"]]
    ref_cells = [cell for table in reference_tables for cell in table["cells"]]
    identical = sum(1 for a, b in zip(run_cells, ref_cells) if cell_key(a) == cell_key(b))
    box_delta = 0.0
    score_delta = 0.0
    for a, b in zip(run_tables, reference_tables):
        score_delta = max(score_delta, abs(float(a["score"]) - float(b["score"])))
        for index in range(4):
            box_delta = max(box_delta, abs(float(a["bbox"][index]) - float(b["bbox"][index])))
        for ca, cb in zip(a["cells"], b["cells"]):
            for index in range(4):
                box_delta = max(box_delta,
                                abs(float(ca["bbox"][index]) - float(cb["bbox"][index])))
    return {"tables": len(run_tables), "cells": len(run_cells), "identical": identical,
            "max_box": round(box_delta, 4), "max_score": round(score_delta, 6)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=None,
                        help="directory of the pinned page images, <id>.jpg")
    parser.add_argument("--pdf-dir", type=Path, default=None,
                        help="directory of the source PDFs, <pmcid>.pdf, used without --images")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--ids", type=Path, default=None,
                        help="file of page ids, default <model>/ids.txt")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    if arguments.weights is not None:
        import os
        os.environ["AE3_WEIGHTS_DIR"] = str(arguments.weights)
    engine = create_engine(load_settings())
    ids_path = arguments.ids or (arguments.model / "ids.txt")
    ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    pdf_dir = arguments.pdf_dir or arguments.model

    rows = []
    totals = {"pages": 0, "tables": 0, "cells": 0, "identical": 0,
              "max_box": 0.0, "max_score": 0.0, "max_array": 0.0}
    bad = []
    for identifier in ids:
        if arguments.images is not None:
            image = Image.open(arguments.images / f"{identifier}.jpg").convert("RGB")
        else:
            pmcid, _, page = identifier.partition("_page_")
            image = render(pdf_dir / f"{pmcid}.pdf", int(page), arguments.dpi)
        words = json.loads((arguments.model / "words" / f"{identifier}.json").read_text())
        pinned = np.load(arguments.model / "arrays" / f"ort_cpu__{identifier}.npz")

        arrays = engine.network.run(page_tensors([image]))
        array_delta = 0.0
        for name, value in zip(FIELDS, arrays):
            mine, pin = np.asarray(value), np.asarray(pinned[name])
            while mine.ndim > pin.ndim and mine.shape[0] == 1:
                mine = mine[0]
            if mine.shape != pin.shape:
                array_delta = float("inf")
                break
            array_delta = max(array_delta,
                              float(np.abs(mine.astype(np.float64) - pin.astype(np.float64)).max()))

        saved = engine.network.run
        engine.network.run = lambda _batch: tuple(pinned[name][None] for name in FIELDS)
        try:
            reference = engine.predict(image, words, image.size)
        finally:
            engine.network.run = saved
        run = engine.predict(image, words, image.size)

        row = {"id": identifier, "arrays": array_delta, **compare(run.tables, reference.tables)}
        rows.append(row)
        totals["pages"] += 1
        totals["tables"] += row["tables"]
        totals["cells"] += row["cells"]
        totals["identical"] += row["identical"]
        totals["max_box"] = max(totals["max_box"], row["max_box"])
        totals["max_score"] = max(totals["max_score"], row["max_score"])
        totals["max_array"] = max(totals["max_array"], array_delta)
        if row["identical"] != row["cells"] or array_delta > 1e-4:
            bad.append(row)
        print(f"{identifier} tables={row['tables']} cells={row['cells']} "
              f"identical={row['identical']} max_box={row['max_box']} "
              f"max_score={row['max_score']} arrays={array_delta:.2e}", flush=True)

    result = {"pages": totals["pages"], "tables": totals["tables"], "cells": totals["cells"],
              "identical": totals["identical"], "max_box_delta": totals["max_box"],
              "max_score_delta": totals["max_score"], "max_array_delta": totals["max_array"],
              "pages_with_differences": [row["id"] for row in bad], "per_page": rows}
    if arguments.out:
        arguments.out.write_text(json.dumps(result, indent=1))
    print(f"RESULT pages={totals['pages']} tables={totals['tables']} cells={totals['cells']} "
          f"identical={totals['identical']} max_box_delta={totals['max_box']:.3f} "
          f"max_score_delta={totals['max_score']:.6f} "
          f"max_array_delta={totals['max_array']:.3e} "
          f"pages_with_differences={len(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
