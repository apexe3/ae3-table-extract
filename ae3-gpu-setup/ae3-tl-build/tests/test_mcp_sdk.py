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

"""MCP over Streamable HTTP, driven by the official MCP Python SDK client.

    python tests/test_mcp_sdk.py [http://localhost:8000] [examples/sample.png]

Runs initialize, tools/list and tools/call (extract_tables) on the sample PNG, and prints the
transcript and the tool result. Needs the `mcp` package; the leisure of a throwaway venv is
described in tests/README.md.
"""

import asyncio
import json
import os
import sys

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

try:                       # mcp 2.x drives the transport with httpx2
    import httpx2 as transport_httpx
except ImportError:        # older mcp versions use httpx
    import httpx as transport_httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
IMAGE = sys.argv[2] if len(sys.argv) > 2 else "examples/sample.png"
KEY = os.environ["AE3_API_KEY"]


async def main() -> int:
    async with httpx.AsyncClient(timeout=180.0) as http:
        with open(IMAGE, "rb") as handle:
            upload = await http.post(f"{BASE}/v1/files", headers={"X-API-Key": KEY},
                                     files={"file": (IMAGE, handle, "image/png")})
        upload.raise_for_status()
        file_id = upload.json()["id"]
    print(f"upload: file_id {file_id}")

    async with transport_httpx.AsyncClient(timeout=180.0,
                                           headers={"X-API-Key": KEY}) as transport_client:
        async with streamable_http_client(f"{BASE}/mcp",
                                          http_client=transport_client) as (read, write):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                print(f"initialize: server {initialized.server_info.name} "
                      f"{initialized.server_info.version}, "
                      f"protocol {initialized.protocol_version}")
                tools = await session.list_tools()
                print(f"tools/list: {[tool.name for tool in tools.tools]}")
                result = await session.call_tool(
                    "extract_tables", {"file_id": file_id, "format": "markdown"})
                print(f"tools/call: is_error={result.is_error}")
                text = result.content[0].text if result.content else ""
                document = json.loads(text)
                print(f"tool result: tables_total={document['tables_total']} "
                      f"source={document['source']} elapsed_ms={document['elapsed_ms']}")
                table = document["pages"][0]["tables"][0]
                print(f"first table: {table['rows']}x{table['columns']} "
                      f"score={table['score']} model_box={table['model_box']}")
                print("body:")
                print(table["body"])
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
