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

"""Model weights from Hugging Face when they are not already on disk.

The image carries no weights. At start-up the service downloads whatever is missing
from the model repository named in AE3_WEIGHTS_REPO (default APEXE3HQ/ae3-tl).

The token is optional: AE3_HF_TOKEN or HF_TOKEN is used when set, which is only
needed while the repository is private. A public repository needs no token.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import httpx

log = logging.getLogger("ae3.weights")

DEFAULT_REPO = "APEXE3HQ/ae3-tl"
FILES = (
    "ae3_tl.json",
    "ae3tl_pages_v3_ep1_fp32_opset17.onnx",
    "ae3tl_relhead_v3_ep1_fp32_opset17.onnx",
)


def ensure_weights(weights_dir: Path) -> dict:
    """Download any missing weight file; return what is on disk now."""
    started = time.time()
    repo = (os.environ.get("AE3_WEIGHTS_REPO") or DEFAULT_REPO).strip()
    token = (os.environ.get("AE3_HF_TOKEN") or os.environ.get("HF_TOKEN") or "").strip()
    missing = [name for name in FILES if not _present(weights_dir / name)]
    downloaded: list[str] = []
    if missing:
        weights_dir.mkdir(parents=True, exist_ok=True)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        log.info("weights: %d of %d files missing in %s; fetching from %s%s",
                 len(missing), len(FILES), weights_dir, repo,
                 "" if token else " (no token: the repository must be public)")
        for name in missing:
            _fetch(repo, name, weights_dir / name, headers)
            downloaded.append(name)
    total = sum((weights_dir / name).stat().st_size for name in FILES if _present(weights_dir / name))
    seconds = round(time.time() - started, 1)
    log.info("weights: ready, %d files, %.1f MB, %s in %ss",
             len(FILES), total / 1e6, "downloaded" if downloaded else "already on disk", seconds)
    return {"dir": str(weights_dir), "repo": repo, "files": len(FILES),
            "mb": round(total / 1e6, 1), "downloaded": downloaded, "seconds": seconds}


def _present(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _fetch(repo: str, name: str, target: Path, headers: dict) -> None:
    url = f"https://huggingface.co/{repo}/resolve/main/weights/{name}"
    partial = target.with_name(target.name + ".part")
    log.info("weights: downloading %s from %s", name, repo)
    with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                      timeout=httpx.Timeout(30.0, read=300.0)) as response:
        if response.status_code != 200:
            raise RuntimeError(
                f"cannot download weights/{name} from {repo}: HTTP {response.status_code}. "
                "Set AE3_HF_TOKEN if the repository is private, or point AE3_WEIGHTS_DIR "
                "at a directory that already holds the weights.")
        with open(partial, "wb") as handle:
            for chunk in response.iter_bytes(1 << 20):
                handle.write(chunk)
    partial.replace(target)
    size = target.stat().st_size
    log.info("weights: %s ready (%.1f MB)", name, size / 1e6)
    return size
