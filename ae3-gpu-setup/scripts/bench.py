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

"""Throughput and latency of a running AE3-TL service.

Sends page images to POST /v1/extract from several threads at once and reports pages per second,
pages per hour, client and server latency, and every HTTP status. Standard library only, so it
runs with any Python 3.9+ on the server or on a laptop.

    export AE3_API_KEY=<key>

    # one page with its words, sent 300 times by 24 clients (the realistic fast path)
    python scripts/bench.py --images examples/sample.png --words examples/sample_words.json \
        --count 300 --concurrency 24

    # a folder of pages, words matched by file name (page_1.png <-> page_1.json)
    python scripts/bench.py --images pages/ --words words/ --concurrency 24

    # one request at a time, to read single-page latency
    python scripts/bench.py --images examples/sample.png --words examples/sample_words.json \
        --count 5 --concurrency 1

Run it on the GPU server itself (--url http://localhost:8000) to measure the service; run it from
elsewhere to include the network. The service's own processing time is `server ms` below; the
rest of `client ms` is upload and network.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle, islice
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
WORDS_SUFFIXES = (".json", ".xml", ".csv")


def find_images(path: Path) -> list[Path]:
    """One image, or every image in a folder (recursively), sorted."""
    if path.is_file():
        return [path]
    images = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        sys.exit(f"no images found in {path}")
    return images


def words_for(image: Path, words: Path | None) -> Path | None:
    """The words file for an image: the file given, or the same stem in the words folder."""
    if words is None:
        return None
    if words.is_file():
        return words
    for suffix in WORDS_SUFFIXES:
        candidate = words / (image.stem + suffix)
        if candidate.is_file():
            return candidate
    return None


def multipart(fields: dict[str, str], files: dict[str, Path]) -> tuple[bytes, str]:
    """Encode form fields and files as multipart/form-data."""
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, path in files.items():
        kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'
                     f"Content-Type: {kind}\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def send(url: str, key: str, image: Path, words: Path | None, args) -> dict:
    """POST one page and return its status, timings and table count."""
    fields = {"format": args.format, "rotation": args.rotation, "ocr": args.ocr or ("never" if words else "auto")}
    files = {"file": image} | ({"words": words} if words else {})
    body, content_type = multipart(fields, files)
    request = urllib.request.Request(f"{url}/v1/extract", data=body, method="POST",
                                     headers={"X-API-Key": key, "Content-Type": content_type})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            payload = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        payload, status = error.read(), error.code
    except Exception as error:  # connection refused, timeout, ...
        return {"page": image.name, "status": f"error: {type(error).__name__}",
                "client_ms": (time.perf_counter() - started) * 1000}
    result = {"page": image.name, "status": status, "client_ms": (time.perf_counter() - started) * 1000}
    if status == 200 and args.format == "json":
        document = json.loads(payload)
        result["server_ms"] = document.get("elapsed_ms")
        result["tables"] = document.get("tables_total")
    return result


def percentile(values: list[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * (len(ordered) - 1)))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=os.environ.get("AE3_URL", "http://localhost:8000"),
                        help="service address (default: $AE3_URL or http://localhost:8000)")
    parser.add_argument("--images", type=Path, required=True, help="an image, or a folder of images")
    parser.add_argument("--words", type=Path, help="a words file, or a folder of words files named like the images")
    parser.add_argument("--count", type=int, help="pages to send (default: every image once); images repeat if needed")
    parser.add_argument("--concurrency", type=int, default=16, help="parallel clients (default: 16)")
    parser.add_argument("--ocr", choices=("auto", "always", "never"), help="default: never with words, auto without")
    parser.add_argument("--rotation", default="0", help="0 skips the rotation check (default); auto runs it")
    parser.add_argument("--format", default="json", choices=("json", "markdown", "html", "csv"))
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds per request")
    parser.add_argument("--out", type=Path, help="write one JSON line per page here")
    args = parser.parse_args()

    key = os.environ.get("AE3_API_KEY")
    if not key:
        sys.exit("set AE3_API_KEY to the service's key")

    images = find_images(args.images)
    count = args.count or len(images)
    jobs = [(image, words_for(image, args.words)) for image in islice(cycle(images), count)]
    url = args.url.rstrip("/")
    print(f"{count} pages, {args.concurrency} clients, {url}, "
          f"{sum(1 for _, w in jobs if w)} with words", flush=True)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(lambda job: send(url, key, job[0], job[1], args), jobs))
    wall = time.perf_counter() - started

    if args.out:
        args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))

    statuses = Counter(str(row["status"]) for row in rows)
    ok = [row for row in rows if row["status"] == 200]
    client = [row["client_ms"] for row in ok]
    server = [row["server_ms"] for row in ok if row.get("server_ms") is not None]
    print(f"status       {dict(statuses)}")
    print(f"wall         {wall:.2f} s")
    print(f"throughput   {len(ok) / wall:.1f} pages/s, {len(ok) / wall * 3600:,.0f} pages/hour"
          + ("" if len(ok) == count else f"  (successful pages only: {len(ok)} of {count})"))
    if client:
        print(f"client ms    median {statistics.median(client):.0f}, p95 {percentile(client, 0.95):.0f}")
    if server:
        print(f"server ms    median {statistics.median(server):.0f}, p95 {percentile(server, 0.95):.0f}")
    return 0 if len(ok) == count else 1


if __name__ == "__main__":
    sys.exit(main())
