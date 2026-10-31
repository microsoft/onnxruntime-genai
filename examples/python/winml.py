# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import traceback

import onnxruntime_genai as og
from windowsml import EpCatalog


def register_execution_providers() -> list[str]:
    """Discover available EPs via the WinML catalog and register them with OGA."""
    registered = []
    with EpCatalog() as catalog:
        for provider in catalog.find_all_providers():
            provider.ensure_ready()
            if provider.library_path == "":
                continue
            try:
                og.register_execution_provider_library(provider.name, provider.library_path)
                registered.append(provider.name)
            except Exception as e:
                print(f"Failed to register execution provider {provider.name}: {e}", file=sys.stderr)
                traceback.print_exc()
    return registered
