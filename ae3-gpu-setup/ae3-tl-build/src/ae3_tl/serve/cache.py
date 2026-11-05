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

"""A bounded in-process cache of recent uploads, so /crop can cut from the original image.

Nothing is written to disk. A restarted container starts empty, and several copies do not share
the cache; the crop URL carries the box, so any copy can answer a crop for an image it holds.
"""

import threading
import time
from collections import OrderedDict


class ImageCache:
    def __init__(self, max_bytes: int, ttl_s: float):
        self.max_bytes = max(1, max_bytes)
        self.ttl_s = float(ttl_s)
        self._items: "OrderedDict[str, tuple[float, bytes, dict]]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def put(self, key: str, data: bytes, boxes: dict | None = None) -> None:
        with self._lock:
            self._drop(key)
            self._items[key] = (time.time(), data, dict(boxes or {}))
            self._bytes += len(data)
            while self._bytes > self.max_bytes and len(self._items) > 1:
                _key, (_stamp, old, _boxes) = self._items.popitem(last=False)
                self._bytes -= len(old)

    def get(self, key: str) -> bytes | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            self._items.move_to_end(key)
            return item[1]

    def box(self, key: str, table_id: str) -> list[float] | None:
        with self._lock:
            item = self._items.get(key)
            return None if item is None else item[2].get(table_id)

    def sweep(self) -> None:
        cutoff = time.time() - self.ttl_s
        with self._lock:
            for key in [key for key, (stamp, _data, _boxes) in self._items.items()
                        if stamp < cutoff]:
                self._drop(key)

    def stats(self) -> dict:
        with self._lock:
            return {"images": len(self._items), "bytes": self._bytes}

    def _drop(self, key: str) -> None:
        item = self._items.pop(key, None)
        if item is not None:
            self._bytes -= len(item[1])
