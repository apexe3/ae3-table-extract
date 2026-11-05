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

"""ae3-tl: serve the image service, or read one image."""

import argparse
import json
import sys


def _serve(module: str, host: str, port: int, log_level: str) -> None:
    import uvicorn

    uvicorn.run(module, host=host, port=port, log_level=log_level)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ae3-tl", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, target, default_port in (("serve", "ae3_tl.serve.app:app", 8000),):
        command = commands.add_parser(name)
        command.add_argument("--host", default="0.0.0.0")
        command.add_argument("--port", type=int, default=default_port)
        command.add_argument("--log-level", default="info")
        command.set_defaults(target=target)
    extract = commands.add_parser("extract", help="read one page image and print the document")
    extract.add_argument("image")
    extract.add_argument("--words")
    extract.add_argument("--format", default="json", choices=("json", "markdown", "html", "csv"))
    extract.add_argument("--include-crops", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "serve":
        _serve(args.target, args.host, args.port, args.log_level)
        return 0

    from ae3_tl.backends import create_engine
    from ae3_tl.image import service
    from ae3_tl.serve.settings import load_settings

    settings = load_settings()
    engine = create_engine(settings)
    words = open(args.words, "rb").read() if args.words else None
    document = service.extract(engine, settings, open(args.image, "rb").read(), words,
                               fmt=args.format, include_crops=args.include_crops)
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
