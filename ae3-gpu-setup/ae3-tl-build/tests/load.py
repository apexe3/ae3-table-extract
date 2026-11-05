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

"""Concurrent page images through a running service: pages per second and error count.

    python tests/load.py 16 8 [http://localhost:8000]
"""

import concurrent.futures
import os
import sys
import time

import httpx

KEY = {"X-API-Key": os.environ.get("AE3_API_KEY") or ""}


def main() -> int:
    requests = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    clients = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    base = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8000"
    body = open("examples/sample.png", "rb").read()

    def one(_index: int) -> int:
        response = httpx.post(f"{base}/v1/extract", headers=KEY,
                              files={"file": ("page.png", body, "image/png")},
                              data={"format": "json"}, timeout=300.0)
        return response.status_code

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
        codes = list(pool.map(one, range(requests)))
    elapsed = time.perf_counter() - started
    errors = sum(1 for code in codes if code != 200)
    print(f"{requests} pages, {clients} clients: {elapsed:.2f}s, "
          f"{requests / elapsed:.3f} pages/s, {errors} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
