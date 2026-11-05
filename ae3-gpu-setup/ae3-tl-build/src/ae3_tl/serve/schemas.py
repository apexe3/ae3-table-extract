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

"""The MCP tool surface, built from one object each."""

EXTRACT_TABLES = {
    "name": "extract_tables",
    "description": (
        "Extract every table of one page image or one PDF: per table a box, rows, columns, cells "
        "with cell text, caption, footer, score and id; text sources are pdf_text, ocr or "
        "provided_words. A PDF page with a text layer is never OCRed. PDFs over the synchronous "
        "page limit answer with a job_id to poll with job_status. Use format=json for cells, or "
        "markdown, html or csv for a rendered body."),
    "parameters": {
        "type": "object",
        "properties": {
            "file_url": {"type": "string", "format": "uri",
                         "description": "http(s) URL of the page image or PDF."},
            "file_id": {"type": "string", "pattern": "^[0-9a-f]{12}$",
                        "description": "Id returned by POST /v1/files."},
            "words_url": {"type": "string", "format": "uri",
                          "description": ("Optional POTATR words file (XML or JSON), boxes in "
                                          "page pixels. Replaces the text layer and OCR.")},
            "format": {"type": "string", "enum": ["json", "markdown", "html", "csv"],
                       "default": "json",
                       "description": "Body format of each table in its 'body' field."},
        },
        "oneOf": [{"required": ["file_url"]}, {"required": ["file_id"]}],
        "additionalProperties": False,
    },
}

JOB_STATUS = {
    "name": "job_status",
    "description": ("Poll a job started by extract_tables for a long PDF. Returns queued, "
                    "running, done or error, page progress, and the extraction once it is done."),
    "parameters": {
        "type": "object",
        "properties": {"job_id": {"type": "string", "description": "A job id."}},
        "required": ["job_id"],
        "additionalProperties": False,
    },
}

TOOLS = [EXTRACT_TABLES, JOB_STATUS]
