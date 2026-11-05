#!/bin/sh
# AE3-TL — Open-Source Table Extraction Toolkit
# Copyright 2026 Team APEX:E3
# Licensed under the Apache License, Version 2.0; see LICENSE in the repository root.
#
# One page image in, the tables out. Start the service first (see ../README.md).
set -e
BASE=${BASE:-http://localhost:8000}
IMAGE=${1:-examples/sample.png}
if [ -z "${AE3_API_KEY:-}" ]; then
  echo "set AE3_API_KEY to the key the service was started with" >&2
  exit 1
fi
curl -s -X POST -H "X-API-Key: $AE3_API_KEY" "$BASE/v1/extract" \
  -F "file=@$IMAGE" -F format=json -F include_crops=false | python -m json.tool
