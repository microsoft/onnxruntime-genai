# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Shared helpers for the model-builder unit tests.

The builder modules import each other as ``models.builders.<name>`` and pull in ``onnx_ir``,
``onnxruntime.quantization`` and ``transformers``. ``load_builder_module`` loads one of them from
source without installing the ``onnxruntime_genai`` package, stubbing whichever of those
dependencies is missing so the pure-Python parts of the builder can still be exercised.
"""

from __future__ import annotations

import functools
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
    sys.modules["models"].__path__ = [str(BUILDERS_DIR.parent)]
    builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
    builders_package.__path__ = [str(BUILDERS_DIR)]
    sys.modules["models"].builders = builders_package


def load_builder_module(module_name):
    """Load ``src/python/py/models/builders/<module_name>.py`` as ``models.builders.<module_name>``."""
    _register_builders_package()
    qualified_name = f"models.builders.{module_name}"
    if qualified_name in sys.modules:
        setattr(sys.modules["models.builders"], module_name, sys.modules[qualified_name])
        return sys.modules[qualified_name]
    module_path = BUILDERS_DIR.joinpath(*module_name.split("."))
    module_path = module_path / "__init__.py" if module_path.is_dir() else module_path.with_suffix(".py")
    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    setattr(sys.modules["models.builders"], module_name, module)
    return module


@functools.cache
def ort_runs_tensor_scatter():
    """Whether the installed onnxruntime has a CPU kernel for the opset-24 TensorScatter node.

    The Nemotron Parse decoder updates its KV cache with TensorScatter. Some CI jobs pin an older
    onnxruntime: the DirectML job's onnxruntime-directml 1.25.0.dev knows the opset-24 schema but has no
    CPU kernel for it. This loads a one-node model once, so a load error in the model under test still
    fails that test instead of skipping it. Without the Python onnxruntime package, assume support.
    """
    try:
        import onnx  # noqa: PLC0415
        import onnxruntime  # noqa: PLC0415
    except ImportError:
        return True

    def value(name, elem_type, shape):
        return onnx.helper.make_tensor_value_info(name, elem_type, shape)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("TensorScatter", ["past", "updates", "write_indices"], ["present"], axis=1)],
        "tensor_scatter_probe",
        [
            value("past", onnx.TensorProto.FLOAT, [1, 4, 2]),
            value("updates", onnx.TensorProto.FLOAT, [1, 1, 2]),
            value("write_indices", onnx.TensorProto.INT64, [1]),
        ],
        [value("present", onnx.TensorProto.FLOAT, [1, 4, 2])],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 24)], ir_version=10)
    try:
        onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as error:
        message = str(error)
        if (
            "Could not find an implementation" in message
            or "Kernel not found" in message
            or "is not a registered function/op" in message
            or "Current official support for domain ai.onnx is till opset" in message
        ):
            return False
        raise
    return True


@functools.cache
def ort_runs_causal_conv_with_state():
    """Whether the installed onnxruntime can load the CausalConvWithState node the LFM2 builder writes.

    Tests that run an LFM2 export in onnxruntime need this. Some CI jobs pin an older onnxruntime: the
    DirectML job's onnxruntime-directml 1.25.0.dev does not have the op, and 1.25.1 to 1.28 do not know
    its state_window attribute. This loads a one-node model once, so a load error in the model under test
    still fails that test instead of skipping it.
    """
    import onnx  # noqa: PLC0415
    import onnxruntime  # noqa: PLC0415

    def float_value(name, shape):
        return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)

    node = onnx.helper.make_node(
        "CausalConvWithState",
        ["x", "weight", "", "past"],
        ["y", "present"],
        domain="com.microsoft",
        ndim=1,
        activation="none",
        state_window=0,
    )
    graph = onnx.helper.make_graph(
        [node],
        "causal_conv_with_state_probe",
        [float_value("x", [1, 4, 3]), float_value("weight", [4, 1, 3]), float_value("past", [1, 4, 2])],
        [float_value("y", [1, 4, 3]), float_value("present", [1, 4, 2])],
    )
    opsets = [onnx.helper.make_opsetid("", 21), onnx.helper.make_opsetid("com.microsoft", 1)]
    model = onnx.helper.make_model(graph, opset_imports=opsets, ir_version=10)
    try:
        onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as error:
        message = str(error)
        if "is not a registered function/op" in message or "Unrecognized attribute: state_window" in message:
            return False
        raise
    return True


@functools.cache
def ort_runs_engram_op(op_type):
    """Probe Engram operators missing from the older ORT wheel pinned by DirectML CI."""
    import onnx  # noqa: PLC0415
    import onnxruntime  # noqa: PLC0415

    def value(name, dtype, shape):
        return onnx.helper.make_tensor_value_info(name, dtype, shape)

    if op_type == "EngramGate":
        inputs = [
            value("key", onnx.TensorProto.FLOAT, [1, 2, 2, 4]),
            value("query", onnx.TensorProto.FLOAT, [1, 2, 2, 4]),
            value("value", onnx.TensorProto.FLOAT, [1, 2, 4]),
            value("key_scale", onnx.TensorProto.FLOAT, [2, 4]),
            value("query_scale", onnx.TensorProto.FLOAT, [2, 4]),
        ]
        outputs = [value("gated", onnx.TensorProto.FLOAT, [1, 2, 2, 4])]
        attributes = {"epsilon": 1e-5}
    elif op_type == "NGramHashMapping":
        inputs = [
            value("tokens", onnx.TensorProto.INT64, [1, 2]),
            value("multipliers", onnx.TensorProto.INT64, [2]),
            value("vocab_sizes", onnx.TensorProto.INT64, [1]),
            value("past_tokens", onnx.TensorProto.INT64, [1, 1]),
            value("offsets", onnx.TensorProto.INT64, [1]),
            value("eos", onnx.TensorProto.INT64, []),
        ]
        outputs = [
            value("hashes", onnx.TensorProto.INT64, [1, 2, 1]),
            value("present_tokens", onnx.TensorProto.INT64, [1, 1]),
        ]
        attributes = {"max_ngram_size": 2, "n_head_per_ngram": 1, "pad_id": 0, "reset_on_eos": 1}
    else:
        raise ValueError(f"Unsupported Engram operator probe: {op_type}")

    node = onnx.helper.make_node(
        op_type,
        [item.name for item in inputs],
        [item.name for item in outputs],
        domain="com.microsoft",
        **attributes,
    )
    graph = onnx.helper.make_graph([node], f"{op_type}_probe", inputs, outputs)
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 21), onnx.helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    try:
        onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except onnxruntime.capi.onnxruntime_pybind11_state.Fail as error:
        if f"com.microsoft:{op_type}" in str(error) and "is not a registered function/op" in str(error):
            return False
        raise
    return True
