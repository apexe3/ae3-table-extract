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
"""Download the AE3-TL ONNX weights from Hugging Face, then read tables from one image.

Everything runs in-process: ONNX Runtime + tesseract, no server needed.

  python examples/run_local.py download                    # weights into ./weights
  python examples/run_local.py download --token hf_xxx     # only needed for a private repo
  python examples/run_local.py tables image.png            # print tables as JSON
  python examples/run_local.py tables image.png --format markdown
  python examples/run_local.py tables --url https://example.com/page.jpg -o tables.json
  python examples/run_local.py tables image.png --words ocr.json   # any OCR, no Tesseract

The first `tables` run also downloads the weights automatically when they are
missing, so the explicit `download` step is optional.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import httpx

REPO = "APEXE3HQ/ae3-tl"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/main/weights"
FILES = (
    "ae3_tl.json",
    "ae3tl_pages_v3_ep1_fp32_opset17.onnx",
    "ae3tl_relhead_v3_ep1_fp32_opset17.onnx",
)
HERE = Path(__file__).resolve().parent
WEIGHTS_DIR = Path(os.environ.get("AE3_WEIGHTS_DIR") or HERE.parent / "weights")
SRC_DIR = HERE.parent / "src"


def _progress_bar(done: int, total: int, width: int = 32) -> str:
    """A text loading bar: [####      ] 45.2/124.2 MB  36.4%."""
    fraction = min(done / total, 1.0) if total else 0.0
    filled = int(fraction * width)
    bar = "#" * filled + " " * (width - filled)
    return f"[{bar}] {done / 1e6:7.1f}/{total / 1e6:.1f} MB {fraction * 100:5.1f}%"


def download(token: str | None = None, dest: Path = WEIGHTS_DIR) -> None:
    """Fetch each weight file that is missing; skip the ones already on disk."""
    token = token or os.environ.get("AE3_HF_TOKEN") or os.environ.get("HF_TOKEN") or None
    dest = Path(dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for name in FILES:
        target = dest / name
        if target.is_file() and target.stat().st_size > 0:
            print(f"have  {name} ({target.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
            continue
        url = f"{BASE_URL}/{name}"
        print(f"get   {name}", file=sys.stderr, flush=True)
        with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                          timeout=httpx.Timeout(30.0, read=600.0)) as response:
            if response.status_code != 200:
                sys.exit(f"HTTP {response.status_code} for {url}\n"
                         f"set --token or AE3_HF_TOKEN if the repository is private")
            total = int(response.headers.get("content-length") or 0)
            partial = target.with_suffix(target.suffix + ".part")
            done = 0
            with open(partial, "wb") as handle:
                for chunk in response.iter_bytes(1 << 18):
                    handle.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {_progress_bar(done, total)}", end="", file=sys.stderr, flush=True)
                    elif done % (10 << 20) < (1 << 18):
                        print(f"\r  {done / 1e6:7.1f} MB", end="", file=sys.stderr, flush=True)
            if total:
                print(f"\r  {_progress_bar(done, total)}", file=sys.stderr)
            else:
                print(file=sys.stderr)
        partial.replace(target)
        print(f"saved {name} -> {target} ({target.stat().st_size / 1e6:.1f} MB)",
              file=sys.stderr)
    print(f"weights ready in {dest}", file=sys.stderr)


def tables(image: str | None, url: str | None, fmt: str, out: Path | None,
           crops: bool, words: Path | None = None, token: str | None = None) -> dict:
    """One image in, the table document out: weights downloaded on first use."""
    download(token)
    sys.path.insert(0, str(SRC_DIR))

    from ae3_tl.backends import create_engine
    from ae3_tl.image import service
    from ae3_tl.serve.settings import load_settings

    settings = load_settings()
    if words is not None:
        settings = replace(settings, ocr_engine="none")
    engine = create_engine(settings)
    if url:
        data = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60.0,
                         follow_redirects=True).raise_for_status().content
    else:
        data = Path(str(image)).read_bytes()
    document = service.extract(engine, settings, data, fmt=fmt, include_crops=crops,
                               rotation="auto",
                               words_bytes=words.read_bytes() if words else None)
    body = _body(document, fmt)
    if out:
        out.write_text(body)
    else:
        print(body)
    return document


def _body(document: dict, fmt: str) -> str:
    """JSON pretty-printed, or the service-rendered markdown/html/csv text."""
    if fmt == "json":
        import json
        return json.dumps(document, indent=2)
    tables = document.get("tables") or []
    return "\n\n".join(filter(None, (t.get("body", "") for t in tables)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    pull = commands.add_parser("download", help="fetch the weights from Hugging Face")
    pull.add_argument("--token", default=None, help="HF token for a private repository")
    pull.add_argument("--dir", type=Path, default=WEIGHTS_DIR, help="target directory")

    read = commands.add_parser("tables", help="read the tables from one image")
    source = read.add_mutually_exclusive_group(required=True)
    source.add_argument("image", nargs="?", help="path to a page image")
    source.add_argument("--url", help="http(s) URL of the page image")
    read.add_argument("--format", default="json", choices=("json", "markdown", "html", "csv"))
    read.add_argument("--crops", action="store_true", help="embed cell crops as data URLs")
    read.add_argument("--words", type=Path, default=None,
                      help="word boxes from any OCR (JSON, CSV or XML) instead of "
                           "running Tesseract")
    read.add_argument("-o", "--out", type=Path, default=None, help="write to this file")

    args = parser.parse_args()
    if args.command == "download":
        download(args.token, args.dir)
    else:
        tables(args.image, args.url, args.format, args.out, args.crops, args.words)


if __name__ == "__main__":
    main()
