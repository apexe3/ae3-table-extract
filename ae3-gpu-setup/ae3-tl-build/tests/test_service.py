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

"""End to end against a running service: key, extract, boxes, crop, MCP.

    python tests/test_service.py [http://localhost:8000] [examples/sample.png]

AE3_API_KEY must hold the key the service was started with.
"""

import os
import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
IMAGE = sys.argv[2] if len(sys.argv) > 2 else "examples/sample.png"
KEY = os.environ["AE3_API_KEY"]
HEADERS = {"X-API-Key": KEY}


def main() -> int:
    with httpx.Client(timeout=180.0) as client:
        assert client.get(f"{BASE}/health").json()["status"] == "ok"

        refused = client.post(f"{BASE}/v1/extract", files={"file": ("page.png", open(IMAGE, "rb").read(), "image/png")})
        assert refused.status_code == 401, refused.text
        print("no key: 401", refused.json()["error"]["code"])

        wrong = client.get(f"{BASE}/v1/models", headers={"X-API-Key": KEY + "x"})
        assert wrong.status_code == 401, wrong.text
        print("wrong key: 401")

        ready = client.get(f"{BASE}/ready", headers=HEADERS)
        assert ready.status_code == 200, ready.text
        assert ready.json()["model"] == "AE3-TL", ready.text
        print("ready:", ready.json()["model"], ready.json()["version"])

        bearer = client.get(f"{BASE}/v1/models", headers={"Authorization": f"Bearer {KEY}"})
        assert bearer.status_code == 200, bearer.text
        assert bearer.json()["data"][0]["id"] == "AE3-TL", bearer.text
        print("models:", bearer.json()["data"][0]["id"])

        with open(IMAGE, "rb") as handle:
            document = client.post(f"{BASE}/v1/extract", headers=HEADERS,
                                   files={"file": ("page.png", handle, "image/png")},
                                   data={"format": "json", "include_crops": "true"}).json()
        assert "error" not in document, document
        assert document["tables_total"], "no table came back"
        page = document["pages"][0]
        assert page["source"] in ("ocr", "provided_words", "none"), page["source"]
        table = page["tables"][0]
        assert table["model_box"] and len(table["model_box"]) == 4
        assert table["content_box"] and len(table["content_box"]) == 4
        assert table["box"] == table["model_box"], (table["box"], table["model_box"])
        assert table["crops"]["table"].startswith("data:image/png;base64,")
        print(f"extract: {document['tables_total']} tables, model_box {table['model_box']}, "
              f"content_box {table['content_box']}, source {page['source']}")

        crop = client.get(f"{BASE}{table['crop_url']}", headers=HEADERS)
        assert crop.status_code == 200, crop.text
        assert crop.content.startswith(b"\x89PNG")
        print(f"crop: {len(crop.content)} bytes")

        listed = client.post(f"{BASE}/mcp", headers=HEADERS,
                             json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).json()
        names = [tool["name"] for tool in listed["result"]["tools"]]
        assert names == ["extract_tables", "job_status"], names
        with open(IMAGE, "rb") as handle:
            file_id = client.post(f"{BASE}/v1/files", headers=HEADERS,
                                  files={"file": ("page.png", handle, "image/png")}).json()["id"]
        called = client.post(f"{BASE}/mcp", headers=HEADERS, json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "extract_tables", "arguments": {"file_id": file_id,
                                                               "format": "markdown"}}}).json()
        assert called["result"]["isError"] is False, called
        mcp_document = called["result"]["structuredContent"]
        assert mcp_document["tables_total"]
        print(f"mcp: {names[0]} returned {mcp_document['tables_total']} tables from file_id")
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
