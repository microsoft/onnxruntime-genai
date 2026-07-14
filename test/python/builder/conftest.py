# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

# Mock heavy ML dependencies so builder unit tests can run without them installed.
# These tests exercise pure graph-construction logic and do not need torch or
# transformers at runtime.

import sys
import types

_MOCK_MODULES = [
    "torch",
    "torch.nn",
    "transformers",
    "transformers.models",
    "transformers.models.mistral",
    "transformers.models.mistral.modeling_mistral",
]

for _mod in _MOCK_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# Provide the handful of torch attributes actually referenced at module import time.
_torch = sys.modules["torch"]
if not hasattr(_torch, "nn"):
    _torch.nn = sys.modules["torch.nn"]
if not hasattr(_torch.nn, "Linear"):
    _torch.nn.Linear = object
if not hasattr(_torch.nn, "Parameter"):
    _torch.nn.Parameter = object
if not hasattr(_torch, "float8_e4m3fn"):
    _torch.float8_e4m3fn = None
if not hasattr(_torch, "bfloat16"):
    _torch.bfloat16 = None

# Provide transformers symbols imported at the top of base.py.
_transformers = sys.modules["transformers"]
for _sym in (
    "AutoConfig",
    "AutoModelForCausalLM",
    "AutoModelForSpeechSeq2Seq",
    "AutoTokenizer",
    "GenerationConfig",
    "Mistral3ForConditionalGeneration",
):
    if not hasattr(_transformers, _sym):
        setattr(_transformers, _sym, object)

# Provide onnx_ir.tensor_adapters stubs (TorchTensor / to_torch_dtype) so that
# base.py's top-level import succeeds when torch is mocked.
import onnx_ir as _ir  # noqa: E402 – installed via onnx-ir

if not hasattr(_ir, "tensor_adapters"):
    _ta = types.ModuleType("onnx_ir.tensor_adapters")
    _ta.TorchTensor = object
    _ta.to_torch_dtype = lambda *a, **kw: None
    sys.modules["onnx_ir.tensor_adapters"] = _ta
    _ir.tensor_adapters = _ta
else:
    # onnx_ir is real but may lack TorchTensor when torch is mocked.
    _ta = _ir.tensor_adapters
    if not hasattr(_ta, "TorchTensor"):
        _ta.TorchTensor = object
    if not hasattr(_ta, "to_torch_dtype"):
        _ta.to_torch_dtype = lambda *a, **kw: None
