# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""OneCollector connection string for ONNX Runtime GenAI telemetry.

Base64 keeps the connection string out of plain-text source searches; it is not
encryption and does not make the instrumentation key secret.
GenAI uses its own instrumentation key, separate from Olive and Foundry Local.
"""

CONNECTION_STRING = "SW5zdHJ1bWVudGF0aW9uS2V5PTlkNWRkYWVjNjFlMjQ1NjdiNzg4YTIwYWVhMzI0NjMxLWQyMTZmODZmLTQ4NzQtNDU5Yi1hMzM1LWIzYTliODBhY2FkNi03MzI3"
