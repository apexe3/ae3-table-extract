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

"""Long PDFs as jobs: an in-process store with a TTL. Nothing is written to disk."""

import asyncio
import time
import uuid
from typing import Awaitable, Callable


class Job:
    def __init__(self, identifier: str):
        self.id = identifier
        self.status = "queued"
        self.pages_total = 0
        self.pages_done = 0
        self.result: dict | None = None
        self.error: str | None = None
        self.created = time.time()

    def view(self, include_result: bool = True) -> dict:
        body = {"job_id": self.id, "status": self.status,
                "pages_total": self.pages_total, "pages_done": self.pages_done}
        if self.error:
            body["error"] = self.error
        if include_result and self.result is not None:
            body["result"] = self.result
        return body


class Jobs:
    def __init__(self, limit: int, ttl_s: float):
        self.limit = max(1, limit)
        self.ttl_s = float(ttl_s)
        self._jobs: dict[str, Job] = {}
        self._running = 0

    def get(self, identifier: str) -> Job | None:
        job = self._jobs.get(identifier)
        if job is not None and time.time() - job.created > self.ttl_s:
            self._jobs.pop(identifier, None)
            return None
        return job

    def submit(self, runner: Callable[[Job], Awaitable[dict]]) -> Job:
        self.sweep()
        if self._running >= self.limit:
            raise RuntimeError("too many jobs")
        job = Job(uuid.uuid4().hex[:16])
        self._jobs[job.id] = job
        asyncio.create_task(self._run(job, runner))
        return job

    async def _run(self, job: Job, runner) -> None:
        self._running += 1
        job.status = "running"
        try:
            job.result = await runner(job)
            job.status = "done"
        except Exception as error:                              # noqa: BLE001
            job.error = type(error).__name__
            job.status = "error"
        finally:
            self._running -= 1

    def sweep(self) -> None:
        cutoff = time.time() - self.ttl_s
        for identifier in [key for key, job in self._jobs.items() if job.created < cutoff]:
            self._jobs.pop(identifier, None)
