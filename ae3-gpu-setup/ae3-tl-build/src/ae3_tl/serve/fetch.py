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

"""Fetch a file_url or words_url: http and https only, an allowlist, and a checked address.

Every hop of a redirect is checked again. A host is resolved once, every address it resolves to
must be public, and the connection is made to an address that was checked, with the original host
kept in the Host header and in the TLS server name. Redirects are followed here, one hop at a
time, so a public host cannot bounce the request into a private one.
"""

import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import httpx

from ae3_tl.serve.errors import ApiError

ALLOWED_SCHEMES = ("http", "https")
REDIRECTS = (301, 302, 303, 307, 308)
MAX_REDIRECTS = 5
METADATA_NAMES = {"metadata.google.internal", "metadata.goog", "instance-data"}


def blocked(address: ipaddress._BaseAddress) -> bool:
    """True for an address a request must never reach."""
    return (address.is_private or address.is_loopback or address.is_link_local
            or address.is_multicast or address.is_reserved or address.is_unspecified)


def allowed(host: str, allowlist: tuple[str, ...]) -> bool:
    """Exact host, or a suffix entry written as .example.com or *.example.com."""
    for entry in allowlist:
        entry = entry.strip().lower()
        if entry.startswith("*."):
            entry = entry[1:]
        if host == entry or (entry.startswith(".") and host.endswith(entry)):
            return True
    return False


def numeric(host: str) -> str | None:
    """A host written as one decimal number, as in http://2130706433/ , which is 127.0.0.1."""
    if not host.isdigit() or len(host) > 10:
        return None
    try:
        return str(ipaddress.IPv4Address(int(host)))
    except ValueError:
        return None


def resolve(host: str) -> list[str]:
    """Every address a host resolves to, as text."""
    address = numeric(host)
    if address is not None:
        return [address]
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as error:
        raise ApiError(400, "fetch_failed", f"{host} does not resolve") from error
    addresses = []
    for info in infos:
        address = info[4][0].split("%", 1)[0]
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise ApiError(400, "fetch_failed", f"{host} does not resolve")
    return addresses


def hop(url: str, allowlist: tuple[str, ...]) -> tuple[str, str, int, str]:
    """Check one URL. Returns (scheme, host, port, address), or raises ApiError."""
    parts = urlsplit(url)
    if parts.scheme not in ALLOWED_SCHEMES or not parts.hostname:
        raise ApiError(400, "invalid_url", "file_url must be an http or https URL")
    host = parts.hostname.lower().rstrip(".")
    if host in METADATA_NAMES:
        raise ApiError(403, "url_not_allowed", f"{host} is not allowed")
    if not allowed(host, allowlist):
        raise ApiError(403, "url_not_allowed", f"{host} is not in AE3_FETCH_ALLOWLIST")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    address = None
    for candidate in resolve(host):
        if blocked(ipaddress.ip_address(candidate)):
            raise ApiError(403, "url_not_allowed",
                           f"{host} resolves to a non-public address")
        address = address or candidate
    return parts.scheme, host, port, address


def pinned(scheme: str, address: str, port: int, parts) -> str:
    """The same request, addressed to the checked IP instead of the name."""
    host = f"[{address}]" if ":" in address else address
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    return f"{scheme}://{host}:{port}{path}"


async def fetch(url: str, max_bytes: int, timeout_s: float, *,
                allowed_domains: tuple[str, ...] = ()) -> bytes:
    """The bytes behind a URL, or an ApiError when it is not allowed, too big or unreadable.

    allowed_domains is the AE3_FETCH_ALLOWLIST: exact hosts, or a dot-suffix entry such as
    .example.com. It is keyword only, so a caller cannot pass a domain list into another
    parameter by position.
    """
    allowlist = allowed_domains
    if not allowlist:
        raise ApiError(403, "url_not_allowed", "file_url needs AE3_FETCH_ALLOWLIST")
    current = url
    async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=False) as client:
        for _ in range(MAX_REDIRECTS + 1):
            parts = urlsplit(current)
            scheme, host, port, address = hop(current, allowlist)
            target = pinned(scheme, address, port, parts)
            headers = {"Host": host if port in (80, 443) else f"{host}:{port}"}
            try:
                async with client.stream("GET", target, headers=headers,
                                         extensions={"sni_hostname": host}) as response:
                    if response.status_code in REDIRECTS:
                        location = response.headers.get("location") or ""
                        if not location:
                            raise ApiError(400, "fetch_failed",
                                           "a redirect answered without a location")
                        current = urljoin(current, location)
                        continue
                    if response.status_code != 200:
                        raise ApiError(400, "fetch_failed",
                                       f"the URL answered {response.status_code}")
                    chunks, total = [], 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > max_bytes:
                            raise ApiError(413, "file_too_large",
                                           f"the file behind the URL is larger than "
                                           f"{max_bytes // (1024 * 1024)} MiB")
                        chunks.append(chunk)
                    return b"".join(chunks)
            except httpx.HTTPError as error:
                raise ApiError(400, "fetch_failed",
                               f"the URL could not be read: {type(error).__name__}") from error
    raise ApiError(400, "fetch_failed", f"more than {MAX_REDIRECTS} redirects")
