import ast
import contextlib
import io
import json
import os
import re
import shutil
import unittest
import warnings
from collections.abc import Callable
from typing import Any, Union

import numpy as np
import onnx
import torch
import transformers


class PvVersion:
    """Simple version of packaging.version.Version."""

    def to_int(self, i: str) -> int:
        if i[0] == "0" and len(i) != 1:
            raise ValueError(f"{self!r} is not a valid version")
        return int(i)

    def __init__(self, version: str):
        self.version = version
        self.t_version = tuple(
            self.to_int(i) for i in re.split(r"[.+]", version) if not i.startswith(("dev", "rc", "post", "cpu", "cu"))
        )

    def __repr__(self) -> str:
        "usual"
        return f"Version({self.version!r})"

    def __eq__(self, other) -> bool:
        """=="""
        return self.version == other.version

    def __ge__(self, other) -> bool:
        """Lesser than."""
        assert isinstance(other, PvVersion), f"Unexpected type {type(other)}"
        return self.t_version >= other.t_version

    def __gt__(self, other) -> bool:
        """Lesser than."""
        assert isinstance(other, PvVersion), f"Unexpected type {type(other)}"
        return self.t_version > other.t_version

    def __le__(self, other) -> bool:
        """Lesser than."""
        assert isinstance(other, PvVersion), f"Unexpected type {type(other)}"
        return self.t_version <= other.t_version

    def __lt__(self, other) -> bool:
        """Lesser than."""
        assert isinstance(other, PvVersion), f"Unexpected type {type(other)}"
        return self.t_version < other.t_version


def requires_yobx(version: str = "", msg: str = "") -> Callable:
    try:
        import yobx
    except ImportError:
        return unittest.skip(msg or "yobx not installed")

    if not hasattr(yobx, "__version__"):
        return unittest.skip(msg or "yobx not properly installed")

    if not version:
        return lambda x: x

    if PvVersion(yobx.__version__) < PvVersion(version):
        msg = f"onnx_ir version {yobx.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def requires_transformers(version: str = "", msg: str = "") -> Callable:
    try:
        import transformers
    except ImportError:
        return unittest.skip(msg or "transformers not installed")

    if not hasattr(transformers, "__version__"):
        return unittest.skip(msg or "transformers not properly installed")

    if not version:
        return lambda x: x

    if PvVersion(transformers.__version__) < PvVersion(version):
        msg = msg or f"transformers version {transformers.__version__} < {version}"
        return unittest.skip(msg)
    return lambda x: x


def has_cuda() -> bool:
    """Returns ``torch.cuda.device_count() > 0``."""
    if not has_torch():
        return False
    import torch

    return torch.cuda.device_count() > 0


def has_torch(version: str = "") -> bool:
    "Returns True if torch transformers is available and recent enough."
    try:
        import torch
    except (ImportError, AttributeError):
        return False
    if not hasattr(torch, "__version__") or os.environ.get("NOTORCH", "0") == "1":
        return False
    if not version:
        return True
    return PvVersion(torch.__version__) >= PvVersion(version)


def requires_cuda(version: str = "", msg: str = "", memory: int = 0):
    """
    Skips a test if cuda is not available.

    :param version: minimum version
    :param msg: to overwrite the message
    :param memory: minimum number of Gb to run the test
    """
    if not has_torch():
        return unittest.skip(msg or "cuda not installed")

    import torch

    if torch.cuda.device_count() == 0:
        msg = msg or "only runs on CUDA but torch does not have it"
        return unittest.skip(msg or "cuda not installed")

    if version:
        if PvVersion(torch.version.cuda) < PvVersion(version):
            msg = msg or f"CUDA older than {version}"
        return unittest.skip(msg or f"cuda not recent enough {torch.version.cuda} < {version}")

    if memory:
        m = torch.cuda.get_device_properties(0).total_memory / 2**30
        if m < memory:
            msg = msg or f"available memory is not enough {m} < {memory} (Gb)"
            return unittest.skip(msg)

    return lambda x: x


def ignore_warnings(warns: list[Warning]) -> Callable:
    """
    Catches warnings.

    :param warns:   warnings to ignore
    """

    def wrapper(fct):
        if warns is None:
            raise AssertionError(f"warns cannot be None for '{fct}'.")

        def call_f(self):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warns)
                return fct(self)

        return call_f

    return wrapper


def hide_stdout(f: Callable | None = None) -> Callable:
    """
    Catches warnings, hides standard output.
    The function may be disabled by setting ``UNHIDE=1``
    before running the unit test.

    :param f: the function is called with the stdout as an argument
    """

    def wrapper(fct):
        def call_f(self):
            if os.environ.get("UNHIDE", "") in (1, "1", "True", "true"):
                fct(self)
                return
            st = io.StringIO()
            with contextlib.redirect_stdout(st), warnings.catch_warnings():
                warnings.simplefilter("ignore", (UserWarning, DeprecationWarning))
                try:
                    fct(self)
                except AssertionError as e:  # pragma: no cover
                    if "torch is not recent enough, file" in str(e):
                        raise unittest.SkipTest(str(e))  # noqa: B904
                    raise
            if f is not None:
                f(st.getvalue())
            return None

        try:  # noqa: SIM105
            call_f.__name__ = fct.__name__
        except AttributeError:  # pragma: no cover
            pass
        return call_f

    return wrapper


def _msg(msg: Callable[[], str] | str, add_bracket: bool = True) -> str:
    if add_bracket:
        m = _msg(msg, add_bracket=False)
        if m:
            if "\n" in m:
                return f"\n----\n{m}\n---\n"
            return f" ({m})"
        return ""
    if callable(msg):
        return msg()
    return msg or ""


def long_test(msg: Callable[[], str] | str | None = None) -> Callable:
    """Skips a unit test if it runs on :epkg:`azure pipeline` on :epkg:`Windows`."""
    if os.environ.get("LONGTEST", "0") in ("0", 0, False, "False", "false"):
        msg = f"Skipped (set LONGTEST=1 to run it. {_msg(msg)}"
        return unittest.skip(msg)
    return lambda x: x


class ExtTestCase(unittest.TestCase):
    _warns = []
    _do_clean = os.environ.get("DOCLEAN", "") in (1, "1", "True", "true")

    def shortDescription(self):
        # To remove annoying display on the screen every time verbosity is enabled.
        return None

    def clean_dir(self, path: str):
        """Removes a directory and all its contents if it exists."""
        if os.path.exists(path):
            shutil.rmtree(path)

    def get_dirs(self, prefix: str, clean: bool = True) -> tuple[str]:
        output_dir = f"dump_models/{prefix}/output"
        cache_dir = os.path.expanduser(f"~/.cache/modelbuilder/{prefix}")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        if clean or self._do_clean:
            self.addCleanup(self.clean_dir, f"dump_models/{prefix}/output_dir")
        return output_dir, cache_dir

    def get_model_dir(self, prefix: str, clean: bool = False) -> tuple[str]:
        model_dir = f"dump_models/{prefix}/checkpoint"
        os.makedirs(model_dir, exist_ok=True)
        if clean or self._do_clean:
            self.addCleanup(self.clean_dir, f"dump_models/{prefix}/checkpoint")
        return model_dir

    def assertExists(self, name):
        if not os.path.exists(name):
            raise AssertionError(f"File or folder {name!r} does not exists.")

    def assertEqualArray(self, expected: np.ndarray, value: np.ndarray, atol: float = 0, rtol: float = 0):
        self.assertEqual(expected.dtype, value.dtype)
        self.assertEqual(expected.shape, value.shape)
        np.testing.assert_allclose(expected, value, atol=atol, rtol=rtol)

    def assertAlmostEqual(self, expected: np.ndarray, value: np.ndarray, atol: float = 0, rtol: float = 0):
        if not isinstance(expected, np.ndarray):
            expected = np.array(expected)
        if not isinstance(value, np.ndarray):
            value = np.array(value).astype(expected.dtype)
        self.assertEqualArray(expected, value, atol=atol, rtol=rtol)

    def assertRaise(self, fct: Callable, exc_type: Exception, msg: str | None = None):
        try:
            fct()
        except exc_type as e:
            if not isinstance(e, exc_type):
                raise AssertionError(f"Unexpected exception {type(e)!r}.") from e
            if msg is None:
                return
            if msg not in str(e):
                raise AssertionError(f"Unexpected error message {e!r}.") from e
            return
        raise AssertionError("No exception was raised.")

    def assertEmpty(self, value: Any):
        if value is None:
            return
        if len(value) == 0:
            return
        raise AssertionError(f"value is not empty: {value!r}.")

    def assertNotEmpty(self, value: Any):
        if value is None:
            raise AssertionError(f"value is empty: {value!r}.")
        if isinstance(value, (list, dict, tuple, set)):
            if len(value) == 0:
                raise AssertionError(f"value is empty: {value!r}.")

    def assertStartsWith(self, prefix: str, full: str):
        if not full.startswith(prefix):
            raise AssertionError(f"prefix={prefix!r} does not start string {full!r}.")

    def check_ort(self, onx: Union["onnx.ModelProto", str]) -> "onnxruntime.InferenceSession":  # noqa: F821  # noqa: F821
        return self._check_with_ort(onx, cpu=True)

    def _check_with_ort(
        self, proto: Union["onnx.ModelProto", str], cpu: bool = False
    ) -> "onnxruntime.InferenceSession":  # noqa: F821
        from onnxruntime import InferenceSession, get_available_providers

        providers = ["CPUExecutionProvider"]
        if not cpu and "CUDAExecutionProvider" in get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        return InferenceSession(
            proto.SerializeToString() if hasattr(proto, "SerializeToString") else proto, providers=providers
        )

    def get_numpy_discrepancy(self, tensor_a, tensor_b):
        return get_numpy_discrepancy(tensor_a, tensor_b)

    def get_pytorch_discrepancy(self, tensor_a, tensor_b):
        return get_pytorch_discrepancy(tensor_a, tensor_b)

    def first_token_diff(self, expected, values):
        return first_token_diff(expected, values)

    def log_results(self, data: dict[str, Any]):
        stat_folder = "stats"
        os.makedirs(stat_folder, exist_ok=True)
        json_path = os.path.join(stat_folder, "end2end_results.json")
        serializable = {k: _make_json_serializable(v) for k, v in data.items()}
        with open(json_path, "a") as f:
            f.write(json.dumps(serializable) + "\n")
        results = _read_results(json_path)
        df = results_to_dataframe(results)
        md = df.to_markdown()
        with open(os.path.join(stat_folder, "end2end_results.md"), "w") as f:
            f.write(md + "\n")
        df.to_excel(os.path.join(stat_folder, "end2end_results.xlsx"))

    def make_dummy_text_inputs(
        self,
        np_dtype,
        provider: str,
        num_hidden_layers: int,
        num_key_value_heads: int,
        head_size: int,
        vocab_size: int,
        batch_size: int = 1,
        seq_len: int = 5,
        past_length: int = 0,
        inputs_embeds_dim: int = -1,
    ):
        import torch
        import transformers

        if inputs_embeds_dim > 0:
            onnx_feed = {
                "attention_mask": np.random.randint(0, 1, (batch_size, seq_len + past_length), dtype=np.int64),
                "inputs_embeds": np.random.randn(batch_size, seq_len, inputs_embeds_dim, 3072).astype(dtype=np_dtype),
            }
        else:
            onnx_feed = {
                "input_ids": np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int64),
                "attention_mask": np.random.randint(0, 1, (batch_size, seq_len + past_length), dtype=np.int64),
            }
        torch_feed = {k: torch.from_numpy(v).to(provider) for k, v in onnx_feed.items()}
        cache = []
        for i in range(num_hidden_layers):
            onnx_feed[f"past_key_values.{i}.key"] = np.random.randn(
                batch_size, num_key_value_heads, past_length, head_size
            ).astype(np_dtype)
            onnx_feed[f"past_key_values.{i}.value"] = np.random.randn(
                batch_size, num_key_value_heads, past_length, head_size
            ).astype(np_dtype)
            cache.append(
                (
                    torch.from_numpy(onnx_feed[f"past_key_values.{i}.key"]).to(provider),
                    torch.from_numpy(onnx_feed[f"past_key_values.{i}.value"]).to(provider),
                )
            )

        dc = transformers.cache_utils.DynamicCache()
        for i, lay in enumerate(cache):
            dc.update(lay[0], lay[1], layer_idx=i)
        torch_feed["past_key_values"] = dc
        return onnx_feed, torch_feed

    def get_input_np_dtype(self, precision):
        return get_input_np_dtype(precision)

    def get_input_torch_dtype(self, precision):
        return get_input_torch_dtype(precision)

    def fill_with_empty_cache(self, onnx_feed, session, provider, batch_size=1):
        return fill_with_empty_cache(onnx_feed, session, provider=provider, batch_size=batch_size)


def get_input_np_dtype(precision):
    if precision == "bf16":
        import ml_dtypes

        return ml_dtypes.bfloat16
    return {"int4": np.float32, "fp16": np.float16, "fp32": np.float32}[precision]


def get_input_torch_dtype(precision):
    import torch

    return {"int4": torch.float32, "fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[precision]


def ort_dtype_to_onnx_dtype(stype: str):
    return {
        "tensor(bfloat16)": onnx.TensorProto.BFLOAT16,
        "tensor(float)": onnx.TensorProto.FLOAT,
        "tensor(float16)": onnx.TensorProto.FLOAT16,
        "tensor(int64)": onnx.TensorProto.INT64,
    }[stype]


def ort_dtype_to_torch_dtype(stype: str):
    import torch

    return {
        "tensor(bfloat16)": torch.bfloat16,
        "tensor(float)": torch.float32,
        "tensor(float16)": torch.float16,
        "tensor(int64)": torch.int64,
    }[stype]


def ort_dtype_to_np_dtype(stype: str):
    import ml_dtypes

    return {
        "tensor(bfloat16)": ml_dtypes.bfloat16,
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
    }[stype]


def onnx_dtype_to_torch_dtype(stype: str):
    import torch

    return {
        onnx.TensorProto.BFLOAT16: torch.bfloat16,
        onnx.TensorProto.FLOAT: torch.float32,
        onnx.TensorProto.FLOAT16: torch.float16,
        onnx.TensorProto.INT64: torch.int64,
    }[stype]


def edit_distance(str1, str2) -> int:
    m, n = len(str1), len(str2)

    # Create a DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i  # Deletions
    for j in range(n + 1):
        dp[0][j] = j  # Insertions

    # Fill the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost
            )  # Deletion  # Insertion  # Substitution

    return dp[m][n]


def first_token_diff(expected: list[int], values: list[int]) -> dict[str, Any]:
    delta_length = len(values) - len(expected)
    first_diff = None
    for i, (a, b) in enumerate(zip(expected, values)):
        if a != b:
            if first_diff is None:
                first_diff = i
                break
    total_diff = edit_distance(expected, values)
    return dict(first_diff=first_diff, delta_length=delta_length, expected_length=len(expected), total_diff=total_diff)


def get_numpy_discrepancy(array_a, array_b):
    """
    Computes discrepancy metrics between two NumPy arrays.
    """
    # 1. Ensure inputs are numpy arrays and same shape
    a = np.asanyarray(array_a)
    b = np.asanyarray(array_b)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # 2. Calculate Absolute Difference |a - b|
    diff = np.nan_to_num(np.abs(a - b))

    # 3. Maximum Discrepancy (Max Error)
    max_disc = np.max(diff)

    # 4. Count Mismatches above thresholds
    # We use np.sum on a boolean mask
    mismatches_01 = np.sum(diff > 0.1)
    mismatches_001 = np.sum(diff > 0.01)

    # 5. Average Absolute Discrepancy (Mean Absolute Error)
    avg_disc = np.mean(diff)

    n = np.prod(a.shape)
    data = {
        "max_abs_err": float(max_disc) if not np.isnan(max_disc) else np.inf,
        "%_gt_0.1": mismatches_01 / n,
        "%_gt_0.01": mismatches_001 / n,
        "avg_abs_discrepancy": float(avg_disc),
        "shape": tuple(int(i) for i in a.shape),
        "dtype": a.dtype,
        "dnan": float(np.isnan(b).sum() - np.isnan(a).sum()) / n,
    }

    if len(array_a.shape) == 3:
        a_token = int(np.argmax(array_a[0, -1, :]))
        b_token = int(np.argmax(array_b[0, -1, :]))
        data["next_token"] = "OK" if a_token == b_token else "FAIL"
        data["next_token_id_tch"] = a_token
        data["next_token_id_ort"] = b_token
    return data


def get_pytorch_discrepancy(tensor_a, tensor_b):
    """
    Computes discrepancy metrics between two PyTorch tensors.
    """
    # 1. Ensure tensors are on the same device and detached from any graph
    # We move to CPU for the final scalar extraction (.item())
    import torch

    a = tensor_a.detach()
    b = tensor_b.to(a.device).detach()

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # 2. Absolute Difference |a - b|
    diff = torch.abs(a - b)

    # 3. Maximum Discrepancy
    max_disc = torch.max(diff).item()

    # 4. Mismatches (Count elements above thresholds)
    # Using .sum() on a boolean mask
    mismatches_01 = torch.sum(diff > 0.1).item()
    mismatches_001 = torch.sum(diff > 0.01).item()

    # 5. Average Absolute Discrepancy (Mean Absolute Error)
    avg_disc = torch.mean(diff).item()

    n = torch.prod(a.shape)
    return {
        "max_abs_err": float(max_disc),
        "%_gt_0.1": mismatches_01 / n,
        "%_gt_0.01": mismatches_001 / n,
        "avg_abs_discrepancy": float(avg_disc),
        "shape": tuple(int(i) for i in a.shape),
        "dtype": a.dtype,
    }


def _make_json_serializable(value: Any) -> Any:
    """Convert a value to a JSON-serializable form."""
    if isinstance(value, np.dtype):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def _read_results(json_path: str) -> list[dict[str, Any]]:
    """Read all newline-delimited JSON records from *json_path*.

    Lines that cannot be parsed (legacy ``str(dict)`` format) are loaded with
    :func:`ast.literal_eval` as a fallback; any value that is still
    un-evaluable is stored as the raw string.
    """
    results: list[dict[str, Any]] = []
    with open(json_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                try:
                    results.append(ast.literal_eval(line))
                except Exception:
                    # unable to interpret as a literal, is it really json format?
                    results.append(line)
    return results


def torch_dtype_to_ort_element_type(dtype):
    """Map a :class:`torch.dtype` to the corresponding ORT TensorProto element type int."""
    import torch
    from onnx import TensorProto

    return {
        torch.float32: TensorProto.FLOAT,
        torch.float16: TensorProto.FLOAT16,
        torch.bfloat16: TensorProto.BFLOAT16,
        torch.int64: TensorProto.INT64,
        torch.int32: TensorProto.INT32,
        torch.bool: TensorProto.BOOL,
    }[dtype]


def _ort_io_binding_helper(sess, input_tensors, output_tensors, device="cuda:0"):
    """Run an ORT session using IOBinding so that bfloat16 tensors work.

    Follows the pattern from the onnxruntime-genai test suite.
    Inputs may be numpy arrays (including ml_dtypes.bfloat16) or CUDA
    :class:`torch.Tensor` objects; they are moved to *device* if needed.
    Output tensors must be pre-allocated :class:`torch.Tensor` objects on
    *device*; they are filled **in-place** by ORT.

    :param sess: :class:`onnxruntime.InferenceSession`
    :param input_tensors: ``dict[str, np.ndarray | torch.Tensor]``
    :param output_tensors: ``dict[str, torch.Tensor]`` – pre-allocated on *device*
    :param device: CUDA device string, e.g. ``"cuda:0"``
    """
    import ml_dtypes
    import torch

    parts = device.split(":")
    ort_device = parts[0]
    ort_device_id = int(parts[1]) if len(parts) > 1 else 0

    bind = sess.io_binding()
    torch_refs = []  # keep tensors alive for the duration of run_with_iobinding
    ort_input_type = {i.name: ort_dtype_to_onnx_dtype(i.type) for i in sess.get_inputs()}
    ort_output_type = {i.name: ort_dtype_to_onnx_dtype(i.type) for i in sess.get_outputs()}

    for name, value in input_tensors.items():
        if isinstance(value, np.ndarray):
            if value.dtype == ml_dtypes.bfloat16:
                # NumPy has no native bf16; reinterpret the bits as int16
                # (same 16-bit width) so torch can create a bfloat16 view.
                arr_c = np.ascontiguousarray(value)
                t = torch.from_numpy(arr_c.view(np.int16)).view(torch.bfloat16).to(device)
            else:
                t = torch.from_numpy(np.ascontiguousarray(value)).to(device)
        else:
            t = value.to(device) if value.device.type != ort_device else value
            t = t.contiguous()
        torch_refs.append(t)
        bind.bind_input(name, ort_device, ort_device_id, ort_input_type[name], list(t.shape), t.data_ptr())

    for name, tensor in output_tensors.items():
        t = tensor.contiguous()
        bind.bind_output(name, ort_device, ort_device_id, ort_output_type[name], list(t.shape), t.data_ptr())

    sess.run_with_iobinding(bind)


def results_to_dataframe(results: list[dict[str, Any]]) -> str:
    """Convert a list of result dictionaries to a DataFrame."""
    import pandas

    if not results:
        return pandas.DataFrame()

    # Collect all keys in insertion order (across all rows).
    df = pandas.DataFrame(results)
    if "%_gt_0.1" in df.columns:
        df["%>0.1"] = df["%_gt_0.1"]
    if "%_gt_0.01" in df.columns:
        df["%>0.01"] = df["%_gt_0.01"]
    for c in [
        "genai_text",
        "expected_text",
        "expected_length",
        "delta_length",
        "dtype",
        "shape",
        "%_gt_0.1",
        "%_gt_0.01",
    ]:
        if c in df.columns:
            df = df.drop(c, axis=1)
    index = [c for c in ["model_id", "experiment", "precision", "provider", "input_type"] if c in df.columns]
    df = df.set_index(index).reset_index(drop=False).fillna("")
    return df


def run_session_or_io_binding(
    use_iobinding: bool,
    precision: str,
    provider: str,
    feed: dict[str, np.ndarray],
    sess: "onnxruntime.InferenceSession",  # noqa: F821
    vocab_size: int,
    results: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, "torch.Tensor"], np.ndarray]:
    import ml_dtypes
    import torch

    device = f"{provider}:0" if use_iobinding else provider
    # Build torch input tensors on the CUDA device.
    inputs_without_cache = [k for k in feed if not k.startswith("past_key_value")]
    num_hidden_layers = (len(feed) - len(inputs_without_cache)) // 2
    first_input = "input_ids" if "input_ids" in feed else "inputs_embeds"
    batch_size = feed[first_input].shape[0]
    seq_len = feed[first_input].shape[1]
    num_key_value_heads = feed["past_key_values.0.key"].shape[1]
    head_size = feed["past_key_values.0.key"].shape[-1]
    onnx_input_names = [i.name for i in sess.get_inputs()]
    onnx_output_names = [i.name for i in sess.get_outputs()]
    onnx_output_dtypes = {i.name: ort_dtype_to_onnx_dtype(i.type) for i in sess.get_outputs()}
    step = "prefill" if results is None else "decode"

    if use_iobinding:
        # For bf16 on CUDA, ORT Python bindings cannot pass bfloat16 tensors
        # through NumPy, so we use IOBinding with pre-allocated torch CUDA
        # tensors instead (following the onnxruntime-genai test pattern).
        if step == "prefill":
            torch_feed = {
                k: (torch.from_numpy(feed[k]).to(device) if isinstance(feed[k], np.ndarray) else feed[k].to(device))
                for k in inputs_without_cache
            }
            for i in range(num_hidden_layers):
                torch_feed[f"past_key_values.{i}.key"] = torch.empty(
                    batch_size, num_key_value_heads, 0, head_size, dtype=get_input_torch_dtype(precision), device=device
                )
                torch_feed[f"past_key_values.{i}.value"] = torch.empty(
                    batch_size, num_key_value_heads, 0, head_size, dtype=get_input_torch_dtype(precision), device=device
                )
            torch_feed = {k: v for k, v in torch_feed.items() if k in onnx_input_names}

            # Pre-allocate output tensors.
            # The builder upcasts bf16 logits to float32 for accuracy.
            ort_prefill_logits = torch.empty(
                batch_size,
                seq_len,
                vocab_size,
                dtype=onnx_dtype_to_torch_dtype(onnx_output_dtypes[onnx_output_names[0]]),
                device=device,
            )
            torch_outputs = {"logits": ort_prefill_logits}
            for i in range(num_hidden_layers):
                torch_outputs[f"present.{i}.key"] = torch.empty(
                    batch_size,
                    num_key_value_heads,
                    seq_len,
                    head_size,
                    dtype=get_input_torch_dtype(precision),
                    device=device,
                )
                torch_outputs[f"present.{i}.value"] = torch.empty(
                    batch_size,
                    num_key_value_heads,
                    seq_len,
                    head_size,
                    dtype=get_input_torch_dtype(precision),
                    device=device,
                )
            _ort_io_binding_helper(sess, torch_feed, torch_outputs, device)
            # Extract float32 logits as numpy; keep KV cache as torch tensors
            # so they can be fed directly into the decode io_binding call.
            if ort_prefill_logits.dtype == torch.bfloat16:
                ort_logits_np = ort_prefill_logits.detach().cpu().to(torch.float32).numpy().astype(ml_dtypes.bfloat16)
            else:
                ort_logits_np = ort_prefill_logits.detach().cpu().numpy()
        else:
            # The KV cache from prefill is already on CUDA as torch tensors.
            torch_feed = {
                k: (torch.from_numpy(feed[k]).to(device) if isinstance(feed[k], np.ndarray) else feed[k].to(device))
                for k in feed
            }
            past_kv_len = results["present.0.key"].shape[2]
            ort_decode_logits = torch.empty(
                batch_size,
                1,
                vocab_size,
                dtype=onnx_dtype_to_torch_dtype(onnx_output_dtypes[onnx_output_names[0]]),
                device=device,
            )
            torch_outputs = {"logits": ort_decode_logits}
            for i in range(num_hidden_layers):
                torch_outputs[f"present.{i}.key"] = torch.empty(
                    batch_size,
                    num_key_value_heads,
                    past_kv_len + 1,
                    head_size,
                    dtype=get_input_torch_dtype(precision),
                    device=device,
                )
                torch_outputs[f"present.{i}.value"] = torch.empty(
                    batch_size,
                    num_key_value_heads,
                    past_kv_len + 1,
                    head_size,
                    dtype=get_input_torch_dtype(precision),
                    device=device,
                )
            _ort_io_binding_helper(sess, torch_feed, torch_outputs, device)
            ort_logits_np = ort_decode_logits.detach().cpu().numpy()

        results = {"logits": ort_logits_np}
        for i in range(num_hidden_layers):
            results[f"present.{i}.key"] = torch_outputs[f"present.{i}.key"]
            results[f"present.{i}.value"] = torch_outputs[f"present.{i}.value"]
    else:
        outputs = sess.run(None, feed)
        results = dict(zip(onnx_output_names, outputs))
        ort_logits_np = outputs[0]

    return results, ort_logits_np


def fill_with_empty_cache(onnx_feed, session, provider, batch_size=1):
    for inp in session.get_inputs():
        if inp.name in onnx_feed:
            continue
        shape = list(inp.shape)
        assert len(shape) == 4, (
            f"issue with shape={shape}, name={inp.name!r}, type={inp.type}, available={list(onnx_feed)}"
        )
        shape[2] = 0
        shape[0] = batch_size
        dtype = ort_dtype_to_np_dtype(inp.type)
        onnx_feed[inp.name] = np.empty(tuple(shape), dtype=dtype)


def _flatten_key_value_cache(
    cache: transformers.cache_utils.DynamicCache,
) -> tuple[list[Any], torch.utils._pytree.Context]:
    keys = [lay.keys for lay in cache.layers]
    values = [lay.values for lay in cache.layers]
    flat = list(itertools.chain.from_iterable(zip(keys, values)))
    unique = set(type(lay) for lay in cache.layers)
    assert unique == {transformers.cache_utils.DynamicLayer}, f"Not implemented for layers type {unique}"
    keys = list(itertools.chain.from_iterable((f"key_{i}", f"value_{i}") for i in range(len(cache.layers))))
    return flat, keys


def _flatten_with_keys_cache(
    cache: transformers.cache_utils.DynamicCache,
) -> tuple[list[tuple[torch.utils._pytree.MappingKey, Any]], torch.utils._pytree.Context]:
    values, context = _flatten_key_value_cache(cache)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


def _unflatten_cache(
    values: list[Any], context: torch.utils._pytree.Context, output_type=None
) -> transformers.cache_utils.DynamicCache:
    """Restores a cache from python objects."""
    expected = list(itertools.chain.from_iterable((f"key_{i}", f"value_{i}") for i in range(len(values) // 2)))
    assert expected == context, f"Does not seem to be a dynamic cache {expected} != {context}"
    res = transformers.cache_utils.DynamicCache()
    for i in range(len(values) // 2):
        res.update(values[i * 2], values[i * 2 + i], layer_idx=i)
    assert output_type is None or isinstance(res, output_type), (
        f"Type mismatch between {output_type} (expected) and {type(res)}"
    )
    return res


def registers_dynamic_cache():
    cls = transformers.cache_utils.DynamicCache
    torch.utils._pytree.register_pytree_node(
        cls,
        _flatten_key_value_cache,
        _unflatten_cache,
        serialized_type_name=f"{cls.__module__}.{cls.__name__}",
        flatten_with_keys_fn=_flatten_with_keys_cache,
    )


# ---------------------------------------------------------------------------
# OnnxRuntime type-string → numpy dtype mapping
# ---------------------------------------------------------------------------

_ORT_TYPE_TO_NUMPY: dict[str, type] = {
    "tensor(float)": np.float32,
    "tensor(float32)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(float64)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint64)": np.uint64,
    "tensor(uint32)": np.uint32,
    "tensor(uint16)": np.uint16,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


def _ort_type_to_numpy_dtype(ort_type: str) -> type:
    """Converts an OnnxRuntime type string (e.g. ``"tensor(float)"``) to a NumPy dtype.

    :param ort_type: type string returned by ``NodeArg.type``
    :return: corresponding NumPy dtype
    :raises ValueError: when the type is unknown
    """
    try:
        return _ORT_TYPE_TO_NUMPY[ort_type]
    except KeyError:
        if "bfloat16" in ort_type:
            try:
                import ml_dtypes  # type: ignore[import]

                return ml_dtypes.bfloat16
            except ImportError:
                pass
        raise ValueError(
            f"Unknown OnnxRuntime type string {ort_type!r}. Known types: {sorted(_ORT_TYPE_TO_NUMPY)}"
        ) from None


def _get_dim(i: int, s: str | int | None, batch: int = 1) -> int:
    """Returns a concrete integer dimension from a symbolic or integer shape element.

    :param i: position of the dimension (0 = batch)
    :param s: dimension value (``int``, ``str``, or ``None`` for dynamic dims)
    :param batch: batch size to use for the batch dimension
    :return: concrete integer dimension
    """
    if isinstance(s, int):
        return s
    # None or string symbolic dim
    if i == 0:
        return batch
    # Everything else (cache length, sequence length) starts empty.
    return 0


# Inputs that are never treated as KV-cache slots.
_KNOWN_NON_CACHE: frozenset[str] = frozenset(
    {"input_ids", "attention_mask", "position_ids", "token_type_ids", "cache_position"}
)


def _make_empty_cache(
    batch: int, cache_names: list[str], cache_shapes: list[tuple], cache_types: list[str]
) -> dict[str, np.ndarray]:
    """Creates zero-filled KV-cache arrays for the first generation step.

    :param batch: batch size
    :param cache_names: names of the KV-cache inputs
    :param cache_shapes: ORT input shapes for those inputs
    :param cache_types: ORT type strings for those inputs
    :return: dict ``{name: zero ndarray}``
    """
    feeds: dict[str, np.ndarray] = {}
    for name, shape, ort_type in zip(cache_names, cache_shapes, cache_types):
        new_shape = tuple(_get_dim(i, s, batch=batch) for i, s in enumerate(shape))
        if not new_shape or new_shape[0] <= 0:
            raise ValueError(f"new_shape={new_shape} cannot have a null batch size, name={name!r}, shape={shape}")
        dtype = _ort_type_to_numpy_dtype(ort_type)
        feeds[name] = np.zeros(new_shape, dtype=dtype)
    return feeds


def onnx_generate(
    model_or_session: Union[str, onnx.ModelProto, "onnxruntime.InferenceSession"],  # noqa: F821
    input_ids: np.ndarray,
    attention_mask: np.ndarray | None = None,
    eos_token_id: int | None = None,
    max_new_tokens: int = 20,
    do_sample: bool = False,
    return_session: bool = False,
    verbose: int = 0,
) -> np.ndarray | tuple:
    """
    Performs auto-regressive token generation using an exported ONNX model
    and :class:`onnxruntime.InferenceSession`.

    The function mimics the ``generate`` method of HuggingFace *transformers*
    models.  It calls the ONNX forward pass in a loop, appending the most
    likely next token at each step (greedy decoding by default), and feeds the
    updated *past key/value* tensors back into the model on each subsequent
    call.

    Models that do **not** expose past-key-value inputs/outputs are also
    supported: in that case the full ``input_ids`` sequence is fed on every
    step (simpler but less efficient).

    :param model_or_session: path to an ``.onnx`` file, a
        :class:`onnx.ModelProto` loaded into memory, or an already-created
        :class:`onnxruntime.InferenceSession`.
    :param input_ids: initial prompt token IDs, integer array of shape
        ``[batch, seq_len]``.
    :param attention_mask: optional attention mask of shape
        ``[batch, seq_len]``.  When *None*, an all-ones mask matching
        ``input_ids`` is created automatically.
    :param eos_token_id: when set, generation stops as soon as *all* batch
        items have produced this token.
    :param max_new_tokens: upper bound on the number of tokens to generate
        (not counting the original ``input_ids``).
    :param do_sample: when *True* sample the next token from the softmax
        distribution; when *False* (default) use greedy argmax.
    :param return_session: when *True* return a 3-tuple
        ``(tokens, session, last_feeds)`` instead of just the tokens.
    :param verbose: verbosity level (0 = silent).
    :return: integer array of shape ``[batch, seq_len + generated_tokens]``
        containing the original prompt followed by the generated tokens.
        When ``return_session=True``, returns a 3-tuple
        ``(tokens, session, last_feeds)``.

    Example with a tiny synthetic ONNX decoder (no KV cache)::

        TINT64 = onnx.TensorProto.INT64
        TFLOAT = onnx.TensorProto.FLOAT
        VOCAB  = 8

        # A minimal "LM head": always returns the same logits so that the
        # argmax always picks token 3.
        fixed_logits = np.zeros((1, 1, VOCAB), dtype=np.float32)
        fixed_logits[0, 0, 3] = 10.0   # token 3 always wins

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "Constant",
                        [],
                        ["logits"],
                        value=onh.from_array(fixed_logits),
                    ),
                ],
                "tiny_lm",
                [oh.make_tensor_value_info("input_ids", TINT64, [1, None])],
                [oh.make_tensor_value_info("logits", TFLOAT, [1, 1, VOCAB])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        prompt = np.array([[1, 2]], dtype=np.int64)
        tokens = onnx_generate(model, prompt, max_new_tokens=3, eos_token_id=3)
        # tokens == [[1, 2, 3]]  (stops after the first EOS token)

    .. note::
        When the ONNX model exposes *past key/value* inputs, the function
        automatically creates zero-filled tensors for the initial call and
        feeds back the corresponding outputs on every subsequent step.  The
        KV-cache heuristic treats any input whose name is **not** in
        ``{input_ids, attention_mask, position_ids, token_type_ids,
        cache_position}`` as a KV-cache slot.  Present-key/value outputs are
        mapped back to past-key/value inputs by position (i.e. ``outputs[1]``
        → ``cache_inputs[0]``, etc.).
    """
    from onnxruntime import InferenceSession

    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be a 2-D array [batch, seq_len], got shape {input_ids.shape}")
    input_ids = np.asarray(input_ids, dtype=np.int64)

    if not isinstance(model_or_session, InferenceSession):
        providers = ["CPUExecutionProvider"]
        if isinstance(model_or_session, onnx.ModelProto):
            session_input = model_or_session.SerializeToString()
        else:
            session_input = model_or_session
        session = InferenceSession(session_input, providers=providers)
    else:
        session = model_or_session

    input_meta = session.get_inputs()
    input_names: list[str] = [m.name for m in input_meta]
    input_shapes = [m.shape for m in input_meta]
    input_types: list[str] = [m.type for m in input_meta]

    has_position_ids = "position_ids" in input_names
    has_cache_position = "cache_position" in input_names
    has_attention_mask = "attention_mask" in input_names

    batch_size = input_ids.shape[0]

    # Heuristic KV-cache detection: any input not in the set of known
    # "meta" inputs is treated as a past-key/value slot.
    cache_names = [n for n in input_names if n not in _KNOWN_NON_CACHE]
    cache_shapes = [input_shapes[input_names.index(n)] for n in cache_names]
    cache_types = [input_types[input_names.index(n)] for n in cache_names]

    # Build the initial attention mask (all ones if not provided).
    if attention_mask is None:
        attention_mask = np.ones(input_ids.shape, dtype=np.int64)
    else:
        attention_mask = np.asarray(attention_mask, dtype=np.int64)

    # Bootstrap zero-filled KV-cache arrays.
    empty_cache = _make_empty_cache(batch_size, cache_names, cache_shapes, cache_types)

    # ------------------------------------------------------------------ #
    # Prefill step                                                         #
    # ------------------------------------------------------------------ #
    feeds: dict[str, np.ndarray] = {"input_ids": input_ids}
    if has_attention_mask:
        feeds["attention_mask"] = attention_mask
    feeds.update(empty_cache)

    if has_position_ids:
        seq_len = input_ids.shape[1]
        if seq_len <= 0:
            raise ValueError(f"unexpected value for input_ids shape={input_ids.shape}")
        feeds["position_ids"] = np.tile(np.arange(seq_len, dtype=np.int64), (batch_size, 1))

    if has_cache_position:
        past_len = (
            next(iter(empty_cache.values())).shape[2]
            if empty_cache and next(iter(empty_cache.values())).ndim > 2
            else 0
        )
        feeds["cache_position"] = np.arange(past_len, input_ids.shape[1] + past_len, dtype=np.int64)

    if verbose:
        print(f"[onnx_generate] prefill feeds: {list(feeds)}")

    outputs = session.run(None, feeds)

    # ------------------------------------------------------------------ #
    # Decode loop                                                          #
    # ------------------------------------------------------------------ #
    last_position = 0
    # Per-batch EOS tracking so that the loop terminates only when *all*
    # sequences have finished.
    eos_found = np.zeros(batch_size, dtype=np.bool_)

    for step in range(max_new_tokens):
        next_token_logits = outputs[0][:, -1, :]  # [batch, vocab]

        if do_sample:
            # Sample from the probability distribution over the vocabulary.
            def _softmax(x: np.ndarray) -> np.ndarray:
                e = np.exp(x - np.max(x, axis=-1, keepdims=True))
                return e / e.sum(axis=-1, keepdims=True)

            probs = _softmax(next_token_logits)
            next_token_id = np.array(
                [np.random.choice(probs.shape[-1], p=probs[b]) for b in range(batch_size)], dtype=np.int64
            ).reshape(batch_size, 1)
        else:
            # Greedy decoding: take the argmax token.
            next_token_id = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)  # [batch, 1]

        # Update per-batch EOS flags.
        if eos_token_id is not None:
            eos_found |= next_token_id[:, 0] == eos_token_id

        input_ids = np.concatenate([input_ids, next_token_id], axis=-1)

        if verbose:
            print(f"[onnx_generate] step {step}: next_token_id={next_token_id.tolist()}")

        # Stop once every sequence in the batch has produced EOS.
        if eos_token_id is not None and eos_found.all():
            break

        # Extend the attention mask by one column of ones for the new token.
        attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=attention_mask.dtype)], axis=-1)

        # Build feeds for the next decode step.
        if not cache_names:
            # No KV cache: feed the full growing sequence.
            feeds = {"input_ids": input_ids}
            if has_attention_mask:
                feeds["attention_mask"] = attention_mask
        else:
            # KV cache: feed only the single new token; map present outputs
            # back to past inputs by position.
            feeds = {"input_ids": next_token_id}
            if has_attention_mask:
                feeds["attention_mask"] = attention_mask
            for j, name in enumerate(cache_names):
                if 1 + j < len(outputs):
                    feeds[name] = outputs[1 + j]

        if has_position_ids or has_cache_position:
            last_position = input_ids.shape[1] - 1

        if has_position_ids:
            feeds["position_ids"] = np.full((batch_size, 1), last_position, dtype=np.int64)

        if has_cache_position:
            feeds["cache_position"] = np.arange(last_position, last_position + 1, dtype=np.int64)

        outputs = session.run(None, feeds)

    if return_session:
        return input_ids, session, feeds
    return input_ids
