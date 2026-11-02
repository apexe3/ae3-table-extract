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

"""The model card: architecture, classes, thresholds and weights, with no torch import.

The CPU image ships onnxruntime only, so everything the service needs to describe its model has to
import without torch. app/service/model.py re-exports ModelConfig for the PyTorch path.
"""

import json
from pathlib import Path


class ModelConfig:
    """The model card in weights/ae3_tl.json: architecture, classes, thresholds, weights."""

    def __init__(self, path: Path):
        doc = json.loads(Path(path).read_text())
        self.raw = doc
        self.name = doc["name"]
        self.version = doc["version"]
        self.classes = doc["classes"]
        self.thresholds = doc["thresholds"]
        self.weights = doc["weights"]
        self.relation_head = doc.get("relation_head", {"hidden_dim": 256, "mid": 512})
        self.sections = doc["config"]
        self.input_size = tuple(doc.get("input", {}).get("size", [768, 768]))

    @property
    def num_classes(self) -> int:
        return int(self.raw.get("num_classes") or len(self.classes))

    def weights_path(self, weights_dir: Path, override: Path | None = None) -> Path:
        if override is not None:
            return Path(override)
        return Path(weights_dir) / self.weights["file"]
