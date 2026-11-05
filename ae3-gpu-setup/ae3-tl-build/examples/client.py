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

"""Read one page image with the HTTP API and print each table as markdown.

    python examples/client.py examples/sample.png [http://localhost:8000]
"""

import json
import os
import sys

import httpx


def api_key() -> str:
    """AE3_API_KEY, or a clear exit: there is no default key."""
    key = (os.environ.get("AE3_API_KEY") or "").strip()
    if not key:
        sys.exit("set AE3_API_KEY to the key the service was started with")
    return key


HEADERS = {"X-API-Key": api_key()}


def extract(base: str, path: str, fmt: str = "markdown") -> dict:
    with open(path, "rb") as handle:
        response = httpx.post(f"{base}/v1/extract", headers=HEADERS,
                              files={"image": (path, handle, "application/octet-stream")},
                              data={"format": fmt}, timeout=120.0)
    response.raise_for_status()
    return response.json()


def main() -> int:
    image = sys.argv[1] if len(sys.argv) > 1 else "examples/sample.png"
    base = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    document = extract(base, image)
    print(f"{document['image']['width']}x{document['image']['height']} "
          f"{document['source']}, {document['tables_total']} tables, "
          f"{document['elapsed_ms']} ms")
    for table in document["tables"]:
        print()
        print(f"table {table['index']} score {table['score']} "
              f"{table['rows']}x{table['columns']} box {table['box']}")
        if table["caption"]:
            print(table["caption"])
        print(table.get("body", json.dumps(table["cells"])[:200]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
