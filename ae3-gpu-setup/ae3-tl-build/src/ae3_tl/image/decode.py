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

"""Upload bytes -> one RGB page image. The format is read from the bytes, not the file name."""

import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from ae3_tl.serve.errors import ApiError

KINDS = ("png", "jpeg", "webp", "tiff")
EXIF_TURNS = {3: 180, 6: 90, 8: 270}


def sniff(data: bytes) -> str | None:
    """The image format from the first bytes, or None."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    return None


@dataclass
class Page:
    """One decoded page, in display orientation, with the EXIF quarter turn that was applied."""

    image: Image.Image
    kind: str
    exif_turn: int
    width: int
    height: int


def load(data: bytes, max_bytes: int, max_pixels: int) -> Page:
    if not data:
        raise ApiError(400, "empty_image", "the image part is empty")
    if len(data) > max_bytes:
        raise ApiError(413, "image_too_large",
                       f"the image is {len(data) // 1024} KiB; the limit is {max_bytes // 1024} KiB")
    kind = sniff(data)
    if kind is None:
        raise ApiError(415, "unsupported_image_type",
                       "the bytes are not PNG, JPEG, WebP or TIFF")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            n_frames = int(getattr(probe, "n_frames", 1))
            if kind == "tiff" and n_frames > 1:
                raise ApiError(400, "multi_page_tiff",
                               f"the TIFF holds {n_frames} pages; send one page per request")
            exif_turn = EXIF_TURNS.get(int(probe.getexif().get(274, 1) or 1), 0)
            probe.load()
            image = ImageOps.exif_transpose(probe).convert("RGB")
    except Image.DecompressionBombError as error:
        raise ApiError(413, "image_too_large", "the image is too large to decode") from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ApiError(415, "broken_image",
                       f"the {kind} image could not be decoded") from error
    if image.width * image.height > max_pixels:
        raise ApiError(413, "image_too_large",
                       f"the page is {image.width}x{image.height} pixels; the limit is "
                       f"{max_pixels} pixels")
    return Page(image=image, kind=kind, exif_turn=exif_turn,
                width=image.width, height=image.height)
