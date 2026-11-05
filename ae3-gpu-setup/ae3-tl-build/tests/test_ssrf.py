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

"""SSRF refusals, one case per attack shape.

    python tests/test_ssrf.py

7 of the 8 cases need no network: the address check refuses them before any connection. The
allowed-host case and the redirect case run a local HTTP server and patch the address check so
that the first loopback hop is reachable; the redirect into 127.0.0.1 must still be refused.
"""

import asyncio
import http.server
import sys
import threading

from ae3_tl.serve import fetch as fetch_module
from ae3_tl.serve.errors import ApiError
from ae3_tl.serve.fetch import fetch

LIMIT = 1024 * 1024
TIMEOUT = 5.0


def refused(url: str, allowlist: tuple[str, ...], status: int = 403) -> str:
    try:
        asyncio.run(fetch(url, LIMIT, TIMEOUT, allowed_domains=allowlist))
    except ApiError as error:
        assert error.status == status, (url, error.status, error.message)
        return f"{error.status} {error.code}: {url}"
    raise AssertionError(f"{url} was not refused")


class Server(http.server.BaseHTTPRequestHandler):
    def do_GET(self):                                        # noqa: N802
        if self.path == "/ok":
            body = b"page"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(302)
        self.send_header("Location", self.headers.get("X-Target", "http://127.0.0.1:9/secret"))
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *_args):
        pass


def main() -> int:
    print("loopback:", refused("http://127.0.0.1:9/page.png", ("127.0.0.1",)))
    print("private 10.x:", refused("http://10.1.2.3/page.png", ("10.1.2.3",)))
    print("metadata:",
          refused("http://169.254.169.254/latest/meta-data/", ("169.254.169.254",)))
    print("ipv6 loopback:", refused("http://[::1]:9/page.png", ("::1",)))
    print("decimal ip:", refused("http://2130706433:9/page.png", ("2130706433",)))
    print("not on allowlist:", refused("http://example.com/page.png", ("other.example",)))

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    original = fetch_module.blocked
    try:
        seen = []

        def allow_first_loopback(address):
            if address.is_loopback and not seen:
                seen.append(address)
                return False
            return original(address)

        fetch_module.blocked = allow_first_loopback
        body = asyncio.run(fetch(f"http://127.0.0.1:{port}/ok", LIMIT, TIMEOUT,
                                 allowed_domains=("127.0.0.1",)))
        assert body == b"page", body
        print("allowed host: fetched", len(body), "bytes")

        try:
            asyncio.run(fetch(f"http://127.0.0.1:{port}/go", LIMIT, TIMEOUT,
                              allowed_domains=("127.0.0.1",)))
        except ApiError as error:
            assert error.status == 403, (error.status, error.message)
            print("redirect into loopback: refused, 403", error.code)
        else:
            raise AssertionError("a redirect into 127.0.0.1 was followed")
    finally:
        fetch_module.blocked = original
        httpd.shutdown()
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
