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

"""Call the MCP endpoint: list the tools, then extract one page image.

    python examples/mcp_call.py examples/sample.png [http://localhost:8000]
"""

import json
import os
import sys

import httpx

TOOL = "extract_tables"


def call(base: str, session: httpx.Client, message: dict) -> dict:
    response = session.post(f"{base}/mcp", json=message)
    response.raise_for_status()
    return response.json()


def main() -> int:
    image = sys.argv[1] if len(sys.argv) > 1 else "examples/sample.png"
    base = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    key = (os.environ.get("AE3_API_KEY") or "").strip()
    if not key:
        sys.exit("set AE3_API_KEY to the key the service was started with")
    headers = {"X-API-Key": key}
    with httpx.Client(timeout=120.0, headers=headers) as session:
        print(json.dumps(call(base, session, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                              "params": {"protocolVersion": "2025-06-18"}})
                         ["result"]["serverInfo"], indent=2))
        tools = call(base, session, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        print("tools:", [tool["name"] for tool in tools["result"]["tools"]])
        with open(image, "rb") as handle:
            upload = session.post(f"{base}/v1/files",
                                  files={"file": (image, handle, "application/octet-stream")})
        upload.raise_for_status()
        file_id = upload.json()["id"]
        print("file_id:", file_id)
        answer = call(base, session, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                      "params": {"name": TOOL,
                                                 "arguments": {"file_id": file_id,
                                                               "format": "markdown"}}})
        document = answer["result"]["structuredContent"]
        print(f"{document['tables_total']} tables, source {document['source']}, "
              f"{document['elapsed_ms']} ms")
        for table in document["tables"]:
            print()
            print(table["caption"] or f"table {table['index']}")
            print(table["body"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
