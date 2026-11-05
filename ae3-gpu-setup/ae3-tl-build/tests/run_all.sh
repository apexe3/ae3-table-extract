#!/bin/sh
# AE3-TL — Open-Source Table Extraction Toolkit
# Copyright 2026 Team APEX:E3
# Licensed under the Apache License, Version 2.0; see LICENSE in the repository root.
#
# Every test for the service. Runs from this folder, with the service already up, e.g.
#
#   docker run -d -p 8000:8000 -e AE3_API_KEY=<key> -e AE3_PDF_INPUT=1 apexe3/ae3-table-extract:cpu
#   AE3_API_KEY=<key> sh tests/run_all.sh
#
# The parity test needs AE3_PARITY_FIXTURE; it says "skipped" without it.
set -u
cd "$(dirname "$0")/.." || exit 1
KEY=${AE3_API_KEY:?set AE3_API_KEY to the key the service was started with}
BASE=${BASE:-http://localhost:8000}
PDF_BASE=${PDF_BASE:-http://localhost:8001}
IMAGE=${IMAGE:-ae3-image:0.2.0}
status=0

run() {
  name=$1
  shift
  if "$@"; then echo "PASS $name"; else echo "FAIL $name"; status=1; fi
}

inside() {
  docker run --rm -v "$PWD:/repo:ro" --entrypoint python "$IMAGE" "$@"
}

echo "== unit and protocol tests, inside the image"
run "ssrf" inside /repo/tests/test_ssrf.py
run "words xml (defusedxml)" inside /repo/tests/test_words.py
run "generated response shapes" inside /repo/tests/test_render.py

echo "== service tests, from this machine"
export AE3_API_KEY=$KEY
run "image service" python3 tests/test_service.py "$BASE" "${1:-examples/sample.png}"
run "pdf service" python3 tests/test_pdf_service.py "$PDF_BASE" examples/sample.pdf
run "shell smoke" sh tests/smoke.sh

echo "== parity"
run "50-page parity" python3 tests/test_parity.py

exit $status
