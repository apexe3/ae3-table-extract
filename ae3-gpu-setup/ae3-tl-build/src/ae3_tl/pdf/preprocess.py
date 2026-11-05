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

"""Page image -> the network's input tensor, in numpy.

The PyTorch path in app/service/engine.py resizes with PIL's bilinear filter, converts to float32
and divides by 255. This does the same arithmetic on the same PIL image, so both backends are fed
the same tensor; tests/onnx_gates.py checks that they are bit-identical on the parity pages.
"""

import numpy as np
from PIL import Image

INPUT_SIZE = (768, 768)


def page_tensor(image: Image.Image) -> np.ndarray:
    """One page as (1, 3, 768, 768) float32 in 0..1."""
    resized = image.convert("RGB")
    if resized.size != INPUT_SIZE:
        resized = resized.resize(INPUT_SIZE, Image.BILINEAR)
    array = np.asarray(resized)
    array = np.ascontiguousarray(array.transpose(2, 0, 1)[None])
    return array.astype(np.float32) / np.float32(255.0)


def page_tensors(images) -> np.ndarray:
    """A batch of pages as (n, 3, 768, 768) float32."""
    return np.concatenate([page_tensor(image) for image in images], axis=0)
