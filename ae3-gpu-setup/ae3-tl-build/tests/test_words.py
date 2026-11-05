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

"""Words files: plain XML parses, and an XML that asks for entities or an external reference does not.

    python tests/test_words.py
"""

import sys

from ae3_tl.image import words
from ae3_tl.serve.errors import ApiError

PLAIN = b"""<words><word bbox="10,20,60,40">Cell</word><word bbox="70,20,120,40">text</word></words>"""

BILLION_LAUGHS = b"""<?xml version="1.0"?>
<!DOCTYPE lolz [
 <!ENTITY lol "lol">
 <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
 <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
]>
<words><word bbox="0,0,10,10">&lol3;</word></words>"""

EXTERNAL = b"""<?xml version="1.0"?>
<!DOCTYPE words [<!ENTITY secret SYSTEM "file:///etc/passwd">]>
<words><word bbox="0,0,10,10">&secret;</word></words>"""


def refused(document: bytes, label: str) -> None:
    try:
        words.parse(document)
    except ApiError as error:
        assert error.status == 400, (label, error.status)
        print(f"{label}: refused, {error.code}")
        return
    raise AssertionError(f"{label} was parsed")


def main() -> int:
    parsed = words.parse(PLAIN)
    assert [word["text"] for word in parsed] == ["Cell", "text"], parsed
    assert parsed[0]["bbox"] == [10.0, 20.0, 60.0, 40.0], parsed[0]
    print(f"plain xml: {len(parsed)} words with boxes")
    refused(BILLION_LAUGHS, "entity expansion")
    refused(EXTERNAL, "external entity")
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
