# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Retry logic with exponential backoff for OneCollector exporter."""

import random
import threading
from time import time
from typing import Callable, Optional

from .event_source import event_source
from .transport import HttpJsonPostTransport


class RetryHandler:
    """Handles retry logic with exponential backoff and jitter."""

    def __init__(self, max_retries: int = 6, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def execute_with_retry(
        self,
        operation: Callable[[], tuple[bool, Optional[int]]],
        deadline_sec: float,
        shutdown_event: threading.Event,
    ) -> bool:
        for retry_num in range(self.max_retries):
            remaining_time = deadline_sec - time()
            if remaining_time <= 0:
                return False

            try:
                success, status_code = operation()
                if success:
                    return True
                if not HttpJsonPostTransport.is_retryable(status_code):
                    return False
            except Exception as ex:
                event_source.export_exception_thrown("RetryHandler", ex)
                if retry_num + 1 == self.max_retries:
                    return False

            if retry_num + 1 == self.max_retries:
                return False

            backoff = min(self.base_delay * (2**retry_num), self.max_delay)
            backoff *= random.uniform(0.8, 1.2)
            remaining_time = deadline_sec - time()
            wait_time = min(backoff, remaining_time)
            if wait_time <= 0:
                return False
            if shutdown_event.wait(wait_time):
                return False

        return False
