"""Shared retry policy for active evaluation pipeline and metric judges."""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")

_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def call_with_retry(
    fn: Callable[[], T], max_retries: int = 5, base_delay: float = 1.0
) -> T:
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - retry boundary is intentional
            status_code = getattr(exc, "status_code", None)
            is_retryable = status_code in _RETRYABLE_STATUS_CODES or status_code is None
            attempt += 1
            if attempt > max_retries or not is_retryable:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, base_delay)
            time.sleep(delay)
