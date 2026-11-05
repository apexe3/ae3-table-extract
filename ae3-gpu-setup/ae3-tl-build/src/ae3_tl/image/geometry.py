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

"""Quarter turns, deskew, and the affine map between the working image and the upload.

Boxes from the model are in the working image (turned and deskewed). A response box is in the
uploaded image, so every box goes through the inverse of the transform that was applied.
"""

import math

import numpy as np
from PIL import Image

MAX_SKEW_DEG = 8.0
MIN_SKEW_DEG = 0.3
SCORE_WIDTH = 500


def rotation_matrix(angle_deg: float, size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    """PIL rotate(angle_deg, expand=True) as a 3x3 forward map, plus the output size."""
    width, height = size
    theta = math.radians(angle_deg)
    cos, sin = math.cos(theta), math.sin(theta)
    rotation = np.array([[cos, sin], [-sin, cos]], dtype=float)
    corners = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]]) @ rotation.T
    low, high = corners.min(axis=0), corners.max(axis=0)
    matrix = np.eye(3)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = -low
    out_size = (int(round(high[0] - low[0])), int(round(high[1] - low[1])))
    return matrix, out_size


def turn(image: Image.Image, angle_deg: float) -> tuple[Image.Image, np.ndarray]:
    """Rotate a page, expanding the canvas so nothing is cut. Returns the image and the map."""
    angle = angle_deg % 360.0
    if abs(angle) < 1e-6:
        return image, np.eye(3)
    if abs(angle - round(angle / 90.0) * 90.0) < 1e-6:
        angle = round(angle / 90.0) * 90.0
    matrix, size = rotation_matrix(angle, image.size)
    rotated = image.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor="white")
    if rotated.size != size:
        rotated = rotated.resize(size, Image.BILINEAR)
    return rotated, matrix


def map_points(matrix: np.ndarray, points) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    flat = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (matrix @ flat.T).T[:, :2]


def map_box(matrix: np.ndarray, box) -> list[float]:
    corners = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
    mapped = map_points(matrix, corners)
    return [float(mapped[:, 0].min()), float(mapped[:, 1].min()),
            float(mapped[:, 0].max()), float(mapped[:, 1].max())]


def clip_box(box, width: int, height: int) -> list[float]:
    return [round(max(0.0, min(float(box[0]), width)), 1),
            round(max(0.0, min(float(box[1]), height)), 1),
            round(max(0.0, min(float(box[2]), width)), 1),
            round(max(0.0, min(float(box[3]), height)), 1)]


def union_box(boxes) -> list[float] | None:
    boxes = [box for box in boxes if box]
    if not boxes:
        return None
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]


def _score(binary: np.ndarray, slope: float, height: int) -> float:
    """Variance of the row histogram after shearing the page by slope, per pixel column."""
    columns = np.arange(binary.shape[1]) - binary.shape[1] / 2.0
    shift = np.round(columns * slope).astype(np.int32)
    rows = np.arange(binary.shape[0], dtype=np.int32)[:, None] + shift[None, :]
    useful = (rows >= 0) & (rows < height)
    histogram = np.bincount(rows[useful & binary], minlength=height + 4).astype(np.float64)
    return float(histogram.var())


def estimate_skew(image: Image.Image) -> float:
    """The angle to pass to turn() so the text lines come out level, in degrees."""
    grey = image.convert("L")
    scale = min(1.0, SCORE_WIDTH / max(1, grey.width))
    if scale < 1.0:
        grey = grey.resize((max(1, int(grey.width * scale)), max(1, int(grey.height * scale))),
                           Image.BILINEAR)
    binary = (np.asarray(grey) < 200)
    if binary.sum() < 200:
        return 0.0
    limit = math.tan(math.radians(MAX_SKEW_DEG))
    coarse = np.arange(-limit, limit + 1e-9, 0.01)
    best = max(coarse, key=lambda slope: _score(binary, float(slope), grey.height))
    fine = np.arange(best - 0.01, best + 0.01 + 1e-9, 0.002)
    best = max(fine, key=lambda slope: _score(binary, float(slope), grey.height))
    angle = -math.degrees(math.atan(float(best)))
    return 0.0 if abs(angle) < MIN_SKEW_DEG else angle
