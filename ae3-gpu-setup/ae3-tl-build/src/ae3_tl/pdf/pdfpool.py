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

"""PDF rendering and word extraction in a pool of worker processes.

Rendering a page and reading its words is CPU work that does not need the model, so it runs in
warm worker processes. The parent process only decodes the page image and feeds the model, and
workers open each PDF once per chunk. Page images travel as PNG bytes with compression turned
down: lossless, so the model sees the same pixels as the single-process path.
"""

import asyncio
import io
import signal
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import pdfplumber
import pypdfium2 as pdfium
from PIL import Image

from ae3_tl import ocr
from ae3_tl.pdf.pdf import POINTS_PER_INCH, words_from_page

# Pages per worker task: small enough that the first pages reach the model quickly.
CHUNK_PAGES = 4
PNG_COMPRESS_LEVEL = 1


def _encode(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
    return buffer.getvalue()


class PageTimeout(Exception):
    """One page took longer than AE3TL_PAGE_TIMEOUT_S to render or read."""


def _alarm(_signum, _frame):
    raise PageTimeout("a page took longer than the page timeout")


def _render_chunk(args: tuple[bytes, int, tuple[int, ...], str, float]
                  ) -> list[tuple[int, bytes, tuple[int, int], list[dict], str]]:
    """Render one chunk: (page index, PNG bytes, (width, height), words, words source).

    Each page runs under a per-page alarm, so a pathological page cannot hold a worker
    forever. OCR is used only for pages that need it: never, only when the text layer is empty
    (auto), or for every page (always).
    """
    pdf_bytes, dpi, indices, ocr_mode, page_timeout = args
    scale = dpi / POINTS_PER_INCH
    use_ocr = ocr_mode in ("auto", "always") and ocr.available()
    out = []
    document = pdfium.PdfDocument(pdf_bytes)
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as plumber:
            for index in indices:
                if page_timeout > 0:
                    signal.signal(signal.SIGALRM, _alarm)
                    signal.setitimer(signal.ITIMER_REAL, page_timeout)
                try:
                    image = document[index].render(scale=scale).to_pil().convert("RGB")
                    words = words_from_page(plumber.pages[index], scale)
                    source = "text_layer" if words else "none"
                    if use_ocr and (ocr_mode == "always" or not words):
                        ocr_words = ocr.words_from_image(image)
                        if ocr_words:
                            words, source = ocr_words, "ocr"
                finally:
                    if page_timeout > 0:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                out.append((index, _encode(image), image.size, words, source))
    finally:
        document.close()
    return out


def _count(pdf_bytes: bytes) -> int:
    with pdfium.PdfDocument(pdf_bytes) as document:
        return len(document)


def _warm() -> int:
    return 1


def decode(payload: bytes) -> Image.Image:
    """PNG bytes -> page image."""
    image = Image.open(io.BytesIO(payload))
    image.load()
    return image.convert("RGB")


class PdfPool:
    """Warm worker processes that render pages and extract their words."""

    def __init__(self, workers: int, chunk_pages: int = CHUNK_PAGES, page_timeout_s: float = 60.0):
        self.workers = max(1, workers)
        self.chunk_pages = max(1, chunk_pages)
        self.page_timeout_s = float(page_timeout_s)
        self._new_pool()

    async def page_count(self, pdf_bytes: bytes) -> int:
        return await asyncio.wrap_future(self._pool.submit(_count, pdf_bytes))

    def chunks(self, total: int, first: int, last: int | None) -> list[tuple[int, ...]]:
        end = total if last is None else min(last, total)
        indices = range(max(0, first), end)
        return [tuple(indices[start:start + self.chunk_pages])
                for start in range(0, len(indices), self.chunk_pages)]

    async def pages(self, pdf_bytes: bytes, dpi: int, first: int, last: int | None,
                    ocr_mode: str = "never"):
        """Yield (page index, page image, page size in the words frame, words, source) in order.

        Only a couple of chunks per document are in flight, so several concurrent documents do
        not each hold every rendered page of their document in memory. A chunk that overruns its
        deadline means a worker is stuck: the pool is recreated, which kills the workers.
        """
        total = await self.page_count(pdf_bytes)
        chunks = iter(self.chunks(total, first, last))
        queue = deque()
        for _ in range(max(2, self.workers * 2)):
            chunk = next(chunks, None)
            if chunk is None:
                break
            queue.append(self._submit(pdf_bytes, dpi, chunk, ocr_mode))
        while queue:
            future = queue.popleft()
            try:
                rendered = await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=self.page_timeout_s * self.chunk_pages + 30.0)
            except (asyncio.TimeoutError, TimeoutError) as error:
                self.recycle()
                raise PageTimeout("a page could not be rendered in time") from error
            except PageTimeout:
                raise
            for index, payload, size, words, source in rendered:
                yield index, decode(payload), size, words, source
            chunk = next(chunks, None)
            if chunk is not None:
                queue.append(self._submit(pdf_bytes, dpi, chunk, ocr_mode))

    def _submit(self, pdf_bytes: bytes, dpi: int, chunk: tuple[int, ...], ocr_mode: str):
        return self._pool.submit(_render_chunk,
                                 (pdf_bytes, dpi, chunk, ocr_mode, self.page_timeout_s))

    def _new_pool(self) -> None:
        self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=None)
        for _ in range(self.workers):
            self._pool.submit(_warm).result()

    def recycle(self) -> None:
        """Kill the worker processes and start a fresh pool, after a stuck PDF task."""
        processes = list(getattr(self._pool, "_processes", {}).values())
        self._pool.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            try:
                if process.is_alive():
                    process.kill()
            except (AttributeError, ValueError):
                pass
        self._new_pool()

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
