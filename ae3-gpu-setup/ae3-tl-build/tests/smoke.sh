#!/bin/sh
# AE3-TL — Open-Source Table Extraction Toolkit
# Copyright 2026 Team APEX:E3
# Licensed under the Apache License, Version 2.0; see LICENSE in the repository root.
#
# End to end against a running service: health, ready, a PNG, a PDF, a scanned PDF, MCP.
set -u
BASE=${BASE:-http://localhost:8000}
KEY=${AE3_API_KEY:?set AE3_API_KEY to the key the service was started with}
H="X-API-Key: $KEY"
status=0

check() { if [ "$1" = "$2" ]; then echo "ok   $3"; else echo "FAIL $3 (got $1, want $2)"; status=1; fi; }

check "$(curl -s -o /dev/null -w '%{http_code}' -H "$H" $BASE/health)" 200 "health"
check "$(curl -s -o /dev/null -w '%{http_code}' -H "$H" $BASE/ready)" 200 "ready"
check "$(curl -s -o /dev/null -w '%{http_code}' -H "$H" -X POST $BASE/v1/extract -F file=@examples/sample.png)" 200 "png"
check "$(curl -s -o /dev/null -w '%{http_code}' -H "$H" -X POST $BASE/v1/extract -F file=@examples/sample.pdf -F pages=3)" 200 "pdf"
check "$(curl -s -o /dev/null -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' $BASE/mcp)" 200 "mcp tools/list"

curl -s -H "$H" -X POST $BASE/v1/extract -F file=@examples/sample.png -F format=json \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); t=d["pages"][0]["tables"]; print("     tables:", len(t), "cells:", len(t[0]["cells"]) if t else 0)'

exit $status
