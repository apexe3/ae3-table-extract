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

"""One page's result, shared by the PyTorch and ONNX backends."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PagePrediction:
    """One page's tables, plus the caption and footer detections the relation head scored."""

    page_size: tuple[int, int]
    tables: list[dict[str, Any]]
    captions: list[dict[str, Any]] = field(default_factory=list)
    footers: list[dict[str, Any]] = field(default_factory=list)
    objects: int = 0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.page_size[0],
            "height": self.page_size[1],
            "tables": self.tables,
            "objects": self.objects,
            "timing_ms": {"inference": round(self.inference_ms, 1),
                          "postprocess": round(self.postprocess_ms, 1)},
        }
