# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Shared helpers for the model-builder unit tests.

The builder modules import each other as ``models.builders.<name>`` and pull in ``onnx_ir``,
``onnxruntime.quantization`` and ``transformers``. ``load_builder_module`` loads one of them from
source without installing the ``onnxruntime_genai`` package, stubbing whichever of those
dependencies is missing so the pure-Python parts of the builder can still be exercised.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"


def _module_available(module_name):
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def stub_missing_builder_dependencies():
    if not _module_available("onnx_ir"):
        onnx_ir = types.ModuleType("onnx_ir")
        onnx_ir.DataType = types.SimpleNamespace(INT4=object(), FLOAT=object(), FLOAT16=object(), BFLOAT16=object())
        tensor_adapters = types.ModuleType("onnx_ir.tensor_adapters")
        tensor_adapters.TorchTensor = object
        tensor_adapters.to_torch_dtype = lambda dtype: dtype
        sys.modules["onnx_ir"] = onnx_ir
        sys.modules["onnx_ir.tensor_adapters"] = tensor_adapters

    if not _module_available("onnxruntime.quantization.matmul_nbits_quantizer"):
        # Prefer the real onnxruntime package when it is installed; only fabricate a
        # top-level stub when the package truly isn't available. This avoids shadowing a
        # real onnxruntime wheel (which would break other tests in the session) and only
        # supplies the specific submodule the builder needs.
        if _module_available("onnxruntime"):
            import onnxruntime  # noqa: PLC0415
        else:
            onnxruntime = sys.modules.setdefault("onnxruntime", types.ModuleType("onnxruntime"))
        quantization = getattr(onnxruntime, "quantization", None)
        if quantization is None:
            quantization = types.ModuleType("onnxruntime.quantization")
        matmul_nbits_quantizer = types.ModuleType("onnxruntime.quantization.matmul_nbits_quantizer")
        for class_name in (
            "KQuantWeightOnlyQuantConfig",
            "MatMulNBitsQuantizer",
            "QuantFormat",
            "RTNWeightOnlyQuantConfig",
        ):
            setattr(matmul_nbits_quantizer, class_name, type(class_name, (), {}))
        onnxruntime.quantization = quantization
        quantization.matmul_nbits_quantizer = matmul_nbits_quantizer
        sys.modules["onnxruntime.quantization"] = quantization
        sys.modules["onnxruntime.quantization.matmul_nbits_quantizer"] = matmul_nbits_quantizer

    if not _module_available("tqdm"):
        tqdm_module = types.ModuleType("tqdm")
        tqdm_module.tqdm = lambda iterable=None, *args, **kwargs: iterable
        sys.modules["tqdm"] = tqdm_module

    if not _module_available("transformers"):
        transformers = types.ModuleType("transformers")
        for class_name in (
            "AutoConfig",
            "AutoModelForCausalLM",
            "AutoModelForSpeechSeq2Seq",
            "AutoTokenizer",
            "GenerationConfig",
        ):
            setattr(transformers, class_name, type(class_name, (), {}))
        sys.modules["transformers"] = transformers


stub_missing_builder_dependencies()


def _register_builders_package():
    if str(BUILDERS_DIR.parent) not in sys.path:
        sys.path.insert(0, str(BUILDERS_DIR.parent))
    sys.modules.setdefault("models", types.ModuleType("models"))
    builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
    builders_package.__path__ = [str(BUILDERS_DIR)]


def load_builder_module(module_name):
    """Load ``src/python/py/models/builders/<module_name>.py`` as ``models.builders.<module_name>``."""
    _register_builders_package()
    qualified_name = f"models.builders.{module_name}"
    if qualified_name in sys.modules:
        return sys.modules[qualified_name]
    spec = importlib.util.spec_from_file_location(qualified_name, BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module
