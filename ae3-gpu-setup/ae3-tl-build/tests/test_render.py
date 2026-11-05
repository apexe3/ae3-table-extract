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

"""The table record: box, model_box, content_box, and the crops cut from the model box.

    python tests/test_render.py
"""

import sys

from ae3_tl.serve import render

TABLE = {
    "bbox": [100.0, 200.0, 400.0, 500.0],
    "model_bbox": [100.0, 200.0, 400.0, 500.0],
    "content_bbox": [102.0, 208.0, 398.0, 492.0],
    "label": "table",
    "score": 0.97,
    "caption": "Table 1.",
    "footer": "",
    "caption_parts": [],
    "footer_parts": [],
    "cells": [
        {"row_nums": [0], "column_nums": [0], "cell_text": "a", "header": True,
         "subheader": False, "bbox": [102.0, 208.0, 250.0, 240.0]},
        {"row_nums": [0], "column_nums": [1], "cell_text": "b", "header": True,
         "subheader": False, "bbox": [250.0, 208.0, 398.0, 240.0]},
    ],
}


def main() -> int:
    record = render.table_record(TABLE, 0, "9f2c1a7b3d40-t0", "markdown")
    assert record["box"] == [100.0, 200.0, 400.0, 500.0], record["box"]
    assert record["model_box"] == [100.0, 200.0, 400.0, 500.0], record["model_box"]
    assert record["content_box"] == [102.0, 208.0, 398.0, 492.0], record["content_box"]
    assert record["rows"] == 1 and record["columns"] == 2, (record["rows"], record["columns"])
    assert record["body"].startswith("| a | b |"), record["body"]
    print("record:", record["box"], record["model_box"], record["content_box"])

    moved = render.table_record(TABLE, 0, "id", "json", lambda box: [value * 2 for value in box])
    assert moved["model_box"] == [200.0, 400.0, 800.0, 1000.0], moved["model_box"]
    assert moved["content_box"] == [204.0, 416.0, 796.0, 984.0], moved["content_box"]
    print("moved:", moved["model_box"])
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
