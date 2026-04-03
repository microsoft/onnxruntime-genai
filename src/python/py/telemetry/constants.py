# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""OneCollector connection string for ONNX Runtime GenAI telemetry.

This is a base64-encoded connection string to prevent accidental exposure.
GenAI uses its own instrumentation key, separate from Olive and Foundry Local.
"""

# TODO: Replace with GenAI-specific instrumentation key before production release.
# Using Olive's key temporarily for development/testing.
CONNECTION_STRING = "SW5zdHJ1bWVudGF0aW9uS2V5PTlkNWRkYWVjNjFlMjQ1NjdiNzg4YTIwYWVhMzI0NjMxLTcyMzdkN2M2LWVlNjEtNGNmZC1iYjdiLTU5MDNhOTcyYzJlNC03MDQ3"
