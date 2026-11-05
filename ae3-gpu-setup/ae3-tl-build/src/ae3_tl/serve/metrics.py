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

"""Request counts and timings in the Prometheus text format. Counts only, no content."""

import threading
from collections import Counter

BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.requests = Counter()
        self.errors = Counter()
        self.pages = 0
        self.tables = 0
        self.durations = []

    def hit(self, route: str, status: int, seconds: float) -> None:
        with self._lock:
            self.requests[(route, status)] += 1
            if status >= 400:
                self.errors[status] += 1
            self.durations.append((route, seconds))

    def count_pages(self, pages: int, tables: int) -> None:
        with self._lock:
            self.pages += pages
            self.tables += tables

    def render(self) -> str:
        lines = ["# TYPE ae3_requests_total counter"]
        with self._lock:
            for (route, status), value in sorted(self.requests.items()):
                lines.append(f'ae3_requests_total{{route="{route}",status="{status}"}} {value}')
            lines.append("# TYPE ae3_errors_total counter")
            for status, value in sorted(self.errors.items()):
                lines.append(f'ae3_errors_total{{status="{status}"}} {value}')
            lines.append("# TYPE ae3_pages_total counter")
            lines.append(f"ae3_pages_total {self.pages}")
            lines.append("# TYPE ae3_tables_total counter")
            lines.append(f"ae3_tables_total {self.tables}")
            lines.append("# TYPE ae3_request_seconds histogram")
            by_route: dict[str, list[float]] = {}
            for route, seconds in self.durations:
                by_route.setdefault(route, []).append(seconds)
            for route, values in sorted(by_route.items()):
                values = values[-5000:]
                for bound in BUCKETS:
                    count = sum(1 for value in values if value <= bound)
                    lines.append(f'ae3_request_seconds_bucket{{route="{route}",le="{bound:g}"}} {count}')
                lines.append(f'ae3_request_seconds_bucket{{route="{route}",le="+Inf"}} {len(values)}')
                lines.append(f'ae3_request_seconds_sum{{route="{route}"}} {sum(values):.4f}')
                lines.append(f'ae3_request_seconds_count{{route="{route}"}} {len(values)}')
        return "\n".join(lines) + "\n"
