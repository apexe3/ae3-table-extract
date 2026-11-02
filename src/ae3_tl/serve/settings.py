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

"""Runtime settings, read from the environment."""

import os
from dataclasses import dataclass
from pathlib import Path

from ae3_tl.paths import default_weights_dir

DEFAULT_MODEL_NAME = "AE3-TL"


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _surface() -> str:
    """full serves every endpoint; infer serves the model-provider surface only."""
    value = (os.environ.get("AE3_SURFACE") or "full").strip().lower()
    if value not in ("full", "infer"):
        raise ValueError(f"AE3_SURFACE must be full or infer, got {value!r}")
    return value


def _csv(name: str) -> tuple[str, ...]:
    return tuple(item.strip().lower()
                 for item in (os.environ.get(name) or "").split(",")
                 if item.strip())


@dataclass(frozen=True)
class Settings:
    weights_dir: Path
    model_config_file: Path
    onnx_provider: str
    onnx_threads: int
    batch_size: int
    model_name: str
    surface: str

    pdf_input: bool
    enable_crop: bool
    max_image_mb: int
    max_image_pixels: int
    max_words_mb: int
    max_pdf_mb: int
    max_pages: int
    sync_page_limit: int
    ocr_engine: str
    deskew: bool
    render_dpi: int
    pdf_workers: int
    page_timeout_s: float

    concurrency: int
    batch_max: int
    batch_wait_ms: float
    queue_depth: int
    queue_timeout_s: float
    request_timeout_s: float
    max_body_mb: int
    job_limit: int
    job_ttl_s: float
    api_key: str
    rate_limit_per_min: int

    cache_mb: int
    cache_ttl_s: float
    allow_outbound: bool
    fetch_allowlist: tuple[str, ...]
    max_fetch_mb: int
    fetch_timeout_s: float
    warmup: bool

    def require_api_key(self) -> None:
        """Both services call this at start: no key, no service."""
        if not self.api_key:
            raise ValueError("AE3_API_KEY is required: set it in .env or the environment")


def load_settings() -> Settings:
    weights_dir = Path(os.environ.get("AE3_WEIGHTS_DIR") or default_weights_dir())
    provider = (os.environ.get("AE3_ONNX_PROVIDER")
                or os.environ.get("AE3_PROVIDER")
                or "auto").strip().lower()
    if provider not in ("auto", "cpu", "cuda", "tensorrt", "coreml"):
        raise ValueError(f"AE3_PROVIDER must be auto, cpu, cuda, tensorrt or coreml, got {provider!r}")
    ocr_engine = (os.environ.get("OCR_ENGINE") or "tesseract").strip().lower()
    if ocr_engine not in ("tesseract", "none"):
        raise ValueError(f"OCR_ENGINE must be tesseract or none, got {ocr_engine!r}")
    allow_outbound = _flag("AE3_ALLOW_OUTBOUND", False)
    allowlist = _csv("AE3_FETCH_ALLOWLIST")
    if allow_outbound and not allowlist:
        raise ValueError("AE3_ALLOW_OUTBOUND=1 needs AE3_FETCH_ALLOWLIST: a comma list of hosts")
    return Settings(
        weights_dir=weights_dir,
        model_config_file=weights_dir / "ae3_tl.json",
        onnx_provider=provider,
        onnx_threads=_int("AE3_THREADS", 0),
        batch_size=_int("AE3_BATCH_SIZE", 1),
        model_name=(os.environ.get("AE3_MODEL_NAME") or DEFAULT_MODEL_NAME).strip(),
        surface=_surface(),
        pdf_input=_flag("AE3_PDF_INPUT", False),
        enable_crop=_flag("AE3_ENABLE_CROP", False),
        max_image_mb=_int("AE3_MAX_IMAGE_MB", 25),
        max_image_pixels=_int("AE3_MAX_IMAGE_PIXELS", 60_000_000),
        max_words_mb=_int("AE3_MAX_WORDS_MB", 20),
        max_pdf_mb=_int("AE3_MAX_PDF_MB", 50),
        max_pages=_int("AE3_MAX_PAGES", 100),
        sync_page_limit=_int("AE3_SYNC_PAGE_LIMIT", 20),
        ocr_engine=ocr_engine,
        deskew=_flag("AE3_DESKEW", True),
        render_dpi=_int("AE3_RENDER_DPI", 200),
        pdf_workers=_int("AE3_PDF_WORKERS", max(1, min(4, (os.cpu_count() or 2) - 2))),
        page_timeout_s=_float("AE3_PAGE_TIMEOUT_S", 60.0),
        concurrency=_int("AE3_CONCURRENCY", max(2, min(8, os.cpu_count() or 2))),
        batch_max=_int("AE3_BATCH_MAX", 1),
        batch_wait_ms=_float("AE3_BATCH_WAIT_MS", 10.0),
        queue_depth=_int("AE3_QUEUE_DEPTH", 64),
        queue_timeout_s=_float("AE3_QUEUE_TIMEOUT_S", 120.0),
        request_timeout_s=_float("AE3_REQUEST_TIMEOUT_S", 600.0),
        max_body_mb=_int("AE3_MAX_BODY_MB", 64),
        job_limit=_int("AE3_JOB_LIMIT", 16),
        job_ttl_s=_float("AE3_JOB_TTL_S", 1800.0),
        api_key=(os.environ.get("AE3_API_KEY") or "").strip(),
        rate_limit_per_min=_int("AE3_RATE_LIMIT_PER_MIN", 0),
        cache_mb=_int("AE3_CACHE_MB", 256),
        cache_ttl_s=_float("AE3_CACHE_TTL_S", 1800.0),
        allow_outbound=allow_outbound,
        fetch_allowlist=allowlist,
        max_fetch_mb=_int("AE3_MAX_FETCH_MB", 50),
        fetch_timeout_s=_float("AE3_FETCH_TIMEOUT_S", 60.0),
        warmup=_flag("AE3_WARMUP", True),
    )
