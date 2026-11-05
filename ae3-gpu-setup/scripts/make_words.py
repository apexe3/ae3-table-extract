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

"""Make the words files that the GPU service needs: page image + words JSON, ready to send.

The GPU image has no OCR, so every request carries the page's words. This script makes them with
the service's own code, so the boxes are exactly in the frame the service expects:

  * a PDF with a text layer: each page is rendered to page_N.png and its words are read from the
    PDF itself (exact text, no OCR, fast);
  * a scanned page or photo: Tesseract reads the words (this is how examples/sample_words.json
    was made).

Easiest to run inside the CPU image, which has the service code and Tesseract:

    docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" -w /work \\
      --entrypoint python apexe3/ae3-table-extract:cpu \\
      scripts/make_words.py report.pdf --out pages/

    # a folder of scanned pages
    ... scripts/make_words.py scans/ --out pages/

Each page becomes pages/<name>.png and pages/<name>.json. Send both, with ocr=never and
rotation=0, or point scripts/bench.py at the folder: --images pages/ --words pages/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def from_pdf(path: Path, out: Path, dpi: int, ocr_empty: bool, lang: str) -> int:
    """Render each page and take its words from the PDF's text layer."""
    from ae3_tl.pdf.pdf import iter_pages

    count = 0
    for index, image, words in iter_pages(path.read_bytes(), dpi):
        stem = f"{path.stem}_p{index + 1}"
        if not words and ocr_empty:
            words = ocr(image, lang)
            note = "no text layer, read with Tesseract"
        else:
            note = "text layer" if words else "NO TEXT LAYER: scanned page? use --ocr-empty"
        image.save(out / f"{stem}.png")
        (out / f"{stem}.json").write_text(json.dumps(words))
        print(f"{stem}: {len(words)} words ({note})")
        count += 1
    return count


def ocr(image: Image.Image, lang: str) -> list[dict]:
    from ae3_tl import ocr as tesseract

    if not tesseract.available():
        sys.exit("Tesseract is not installed here: run this in the CPU image (see --help)")
    return tesseract.words_from_image(image, lang=lang)


def from_image(path: Path, out: Path, lang: str) -> None:
    """Read an image's words with Tesseract; copy the image next to its words."""
    image = Image.open(path).convert("RGB")
    words = ocr(image, lang)
    image.save(out / f"{path.stem}.png")
    (out / f"{path.stem}.json").write_text(json.dumps(words))
    print(f"{path.stem}: {len(words)} words (Tesseract)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    parser.add_argument("inputs", nargs="+", type=Path, help="PDFs, images, or folders of them")
    parser.add_argument("--out", type=Path, default=Path("pages"), help="output folder (default: pages)")
    parser.add_argument("--dpi", type=int, default=200, help="PDF render resolution (default: 200)")
    parser.add_argument("--ocr-empty", action="store_true",
                        help="read PDF pages that have no text layer with Tesseract")
    parser.add_argument("--lang", default="eng", help="Tesseract language (default: eng)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for item in args.inputs:
        files += sorted(p for p in item.rglob("*") if p.is_file()) if item.is_dir() else [item]

    done = 0
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            done += from_pdf(path, args.out, args.dpi, args.ocr_empty, args.lang)
        elif suffix in IMAGE_SUFFIXES:
            from_image(path, args.out, args.lang)
            done += 1
    if not done:
        sys.exit("no PDFs or images found")
    print(f"{done} pages written to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
