# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Resolve URI model sources to local paths."""

from .llmman import resolve_model


def resolve_input(value):
    """Resolve a URI source (e.g. oci://...) to a local path, or pass `value` through.

    A scheme is required, since a bare registry/name:tag looks like a Hugging Face repo id.
    """
    # Map each scheme to a function that takes the URI and returns a local path
    resolvers = {"oci": resolve_model}

    text = str(value or "")
    if "://" not in text:
        return value
    resolver = resolvers.get(text.split("://", 1)[0].lower())
    return resolver(value) if resolver else value
