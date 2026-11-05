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

"""Where the weights live: ./weights next to the tree, or AE3_WEIGHTS_DIR."""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_WEIGHTS = PACKAGE_DIR.parents[1] / "weights"


def default_weights_dir() -> Path:
    """AE3_WEIGHTS_DIR, then ./weights of the unzipped tree, then ./weights of the cwd."""
    explicit = os.environ.get("AE3_WEIGHTS_DIR")
    candidates = [Path(explicit)] if explicit else []
    candidates += [SOURCE_WEIGHTS, Path.cwd() / "weights"]
    for candidate in candidates:
        if (candidate / "ae3_tl.json").is_file():
            return candidate
    return SOURCE_WEIGHTS
