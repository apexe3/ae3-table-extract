#!/bin/sh
# AE3-TL — Open-Source Table Extraction Toolkit
# Copyright 2026 Team APEX:E3
# Licensed under the Apache License, Version 2.0; see LICENSE in the repository root.
#
# Fails if a copyleft PDF renderer appears in the tree, in the image, or on import.
#   sh tests/licence_gate.sh [image]
set -u
cd "$(dirname "$0")/.." || exit 1
IMAGE=${1:-apexe3/ae3-table-extract:gpu}
PATTERN='pymupdf|mupdf|fitz'
STATUS=0

echo "== 1. tree"
if grep -rIinE "$PATTERN" \
     --exclude=THIRD_PARTY_NOTICES.md --exclude=LICENSE.txt --exclude=licence_gate.sh \
     --exclude-dir=licenses --exclude-dir=.git . ; then
  echo "FAIL: the pattern is in the tree"; STATUS=1
else
  echo "clean: no match in code, requirements, lock files or Dockerfile"
fi

echo "== 2. packages in the image ($IMAGE)"
if docker run --rm --entrypoint python "$IMAGE" \
     -c "import importlib.metadata as m; print('\\n'.join(sorted(d.metadata['Name'] or '' for d in m.distributions())))" \
     | grep -iE "$PATTERN"; then
  echo "FAIL: the package is installed in the image"; STATUS=1
else
  echo "clean: no such package in the image"
fi

echo "== 3. import in the image"
if docker run --rm --entrypoint python "$IMAGE" -c "import fitz" 2>/dev/null; then
  echo "FAIL: import fitz succeeded"; STATUS=1
else
  echo "clean: import fitz fails, as it should"
fi

exit $STATUS
