# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Payload builder for batching telemetry items."""


class PayloadBuilder:
    """Builds payloads respecting size and item count limits."""

    NEWLINE_SEPARATOR = b"\n"

    def __init__(self, max_size_bytes: int, max_items: int):
        self.max_size_bytes = max_size_bytes
        self.max_items = max_items
        self.reset()

    def reset(self) -> None:
        self.items: list[bytes] = []
        self.current_size = 0

    def can_add(self, item_bytes: bytes) -> bool:
        if self.max_items != -1 and len(self.items) >= self.max_items:
            return False
        if self.max_size_bytes != -1:
            separator_size = len(self.NEWLINE_SEPARATOR) if self.items else 0
            new_size = self.current_size + len(item_bytes) + separator_size
            if new_size > self.max_size_bytes:
                return False
        return True

    def add(self, item_bytes: bytes) -> None:
        self.items.append(item_bytes)
        self.current_size += len(item_bytes)
        if len(self.items) > 1:
            self.current_size += len(self.NEWLINE_SEPARATOR)

    def build(self) -> bytes:
        if not self.items:
            return b""
        return self.NEWLINE_SEPARATOR.join(self.items)

    @property
    def item_count(self) -> int:
        return len(self.items)

    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0
