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

"""The one API key check both services use.

The key comes from AE3_API_KEY. A request offers it as X-API-Key or as a Bearer token. The
comparison is constant time, so a wrong key of the right length cannot be guessed by timing.
"""

import hmac

from ae3_tl.serve.errors import ApiError


def offered(request) -> str:
    """The key a request carries, or an empty string."""
    value = request.headers.get("x-api-key") or ""
    if not value:
        header = request.headers.get("authorization") or ""
        if header[:7].strip().lower() == "bearer":
            value = header[7:].strip()
    return value


def authorize(request) -> None:
    """Raise 401 unless the request carries the configured key."""
    key = request.app.state.settings.api_key
    value = offered(request)
    if not key or not value or not hmac.compare_digest(value.encode(), key.encode()):
        raise ApiError(401, "unauthorized", "a valid API key is required (X-API-Key or Bearer)")
