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

"""50-page parity as a test: runs tests/parity/run.py on the fixture named by AE3_PARITY_FIXTURE.

    AE3_PARITY_FIXTURE=/p python tests/test_parity.py

The fixture is not in this zip: the pages are PubTables-v2 images. Without AE3_PARITY_FIXTURE the
test says so and exits 0, so an unconfigured checkout still passes.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    fixture = os.environ.get("AE3_PARITY_FIXTURE")
    if not fixture:
        print("skipped: set AE3_PARITY_FIXTURE to the directory with ids.txt, images/ and arrays/")
        return 0
    command = [sys.executable, str(HERE / "parity" / "run.py"), "--fixture", fixture]
    weights = os.environ.get("AE3_WEIGHTS_DIR") or str(HERE.parent / "weights")
    command += ["--weights", weights]
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
