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

"""Ask the OpenAI-shaped endpoint for the tables of a page image or a PDF.

    python examples/openai_example.py examples/sample.png [http://localhost:8000]
"""

import base64
import json
import os
import sys
from pathlib import Path

import httpx



def api_key() -> str:
    """AE3_API_KEY, or a clear exit: there is no default key."""
    key = (os.environ.get("AE3_API_KEY") or "").strip()
    if not key:
        sys.exit("set AE3_API_KEY to the key the service was started with")
    return key


KEY = {"X-API-Key": api_key()}


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/sample.png")
    base = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    data = base64.b64encode(path.read_bytes()).decode()
    mime = "application/pdf" if path.suffix.lower() == ".pdf" else "image/png"
    part = ({"type": "file", "file": {"filename": path.name,
                                      "file_data": f"data:{mime};base64,{data}"}}
            if mime == "application/pdf" else
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
    body = {"model": "AE3-TL", "ae3_format": "markdown",
            "messages": [{"role": "user",
                          "content": [{"type": "text", "text": "Extract the tables."}, part]}]}
    response = httpx.post(f"{base}/v1/chat/completions", json=body, headers=KEY, timeout=180.0)
    response.raise_for_status()
    answer = response.json()
    content = answer["choices"][0]["message"]["content"]
    document = json.loads(content) if isinstance(content, str) else content
    print(f"{path.name}: {answer['model']}, {document['kind']}, "
          f"{document['pages_total']} pages, {document['tables_total']} tables")
    for page in document.get("pages", [])[:1]:
        for table in page["tables"][:1]:
            print(table.get("body") or json.dumps(table["cells"])[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
