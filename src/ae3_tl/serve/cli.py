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

"""The ae3-tl command line: serve starts the HTTP API, download fetches the weights."""

import argparse
import json
import sys

import uvicorn

from ae3_tl._version import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ae3-tl", description="AE3-TL table extraction toolkit")
    parser.add_argument("--version", action="version", version=f"ae3-tl {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="start the HTTP API server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workers", type=int, default=1,
                       help="uvicorn workers; >1 loads the model in every worker")

    download = subparsers.add_parser("download", help="download the model weights and exit")
    download.add_argument("--dir", default=None,
                          help="target directory (default AE3_WEIGHTS_DIR or ./weights)")

    args = parser.parse_args(argv)

    if args.command == "download":
        from pathlib import Path

        from ae3_tl.paths import default_weights_dir
        from ae3_tl.weights import ensure_weights

        target = Path(args.dir) if args.dir else default_weights_dir()
        summary = ensure_weights(target)
        print(json.dumps(summary, indent=2))
        return 0

    from ae3_tl.serve.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
