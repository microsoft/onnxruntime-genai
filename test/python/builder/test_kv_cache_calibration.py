# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the KV-cache scale calibration helper.

``kv_cache_calibration.py`` produces the per-layer symmetric KV scales that the model
builder consumes through ``extra_options["kv_cache_scale_file"]``. The end-to-end
``calibrate_kv_scales`` entry point needs onnxruntime, transformers and a real baseline
model, so these tests exercise the module's pure helpers instead:

* ``_get_quant_type_max`` -- quant-type to qmax mapping,
* ``_pair_envelope`` -- the rotation-invariant post-RoPE K envelope,
* ``_tokenize_corpus`` -- calibration-sequence construction (fake tokenizer),
* ``_detect_kv_shape`` -- KV geometry auto-detection (fake session / genai_config.json).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

QUANTIZATION_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "quantization"


def _load_calibration_module():
    sys.path.insert(0, str(QUANTIZATION_DIR.parent))
    models_package = sys.modules.setdefault("models", types.ModuleType("models"))
    models_package.__path__ = [str(QUANTIZATION_DIR.parent)]
    module_name = "models.quantization.kv_cache_calibration"
    spec = importlib.util.spec_from_file_location(module_name, QUANTIZATION_DIR / "kv_cache_calibration.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


kvc = _load_calibration_module()


@pytest.mark.parametrize("cache_dtype", ["FLOAT16", "BFLOAT16"])
def test_run_calibration_batch_preserves_floating_cache(tmp_path, cache_dtype):
    ir = pytest.importorskip("onnx_ir")
    ort = pytest.importorskip("onnxruntime")
    dtype = getattr(ir.DataType, cache_dtype)
    cache_input = ir.Value(name="past_key_values.0.key", type=ir.TensorType(dtype), shape=ir.Shape([1, 1, 1, 4]))
    cache_output = ir.Value(name="present.0.key", type=ir.TensorType(dtype), shape=ir.Shape([1, 1, 1, 4]))
    graph = ir.Graph(
        inputs=[cache_input],
        outputs=[cache_output],
        nodes=[ir.Node("", "Identity", inputs=[cache_input], outputs=[cache_output])],
        opset_imports={"": 20},
    )
    model_path = tmp_path / "cache.onnx"
    ir.save(ir.Model(graph, ir_version=10), model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    expected = np.asarray([1.0, -2.5, 0.0, 3.75], dtype=np.float32).reshape(1, 1, 1, 4)
    values = (expected.view(np.uint32) >> 16).astype(np.uint16) if cache_dtype == "BFLOAT16" else expected.astype(np.float16)
    actual = kvc.run_calibration_batch(session, [cache_output.name], {cache_input.name: values})
    np.testing.assert_array_equal(actual[0], expected)


@pytest.mark.parametrize("cache_dtype", ["FLOAT16", "BFLOAT16"])
def test_run_calibration_batch_aliases_paged_caches(tmp_path, cache_dtype):
    ir = pytest.importorskip("onnx_ir")
    ort = pytest.importorskip("onnxruntime")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDA paged attention is unavailable")
    dtype = getattr(ir.DataType, cache_dtype)

    def value(name, element_type, shape):
        return ir.Value(name=name, type=ir.TensorType(element_type), shape=ir.Shape(shape))

    query = value("qkv", dtype, [3, 1024])
    past_keys = value("past_keys_0", dtype, [1, 256, 1, 128])
    past_values = value("past_values_0", dtype, [1, 256, 1, 128])
    cumulative = value("cumulative_sequence_lengths", ir.DataType.INT32, [2])
    past_lengths = value("past_sequence_lengths", ir.DataType.INT32, [1])
    block_table = value("block_table", ir.DataType.INT32, [1, 1])
    metadata = value("attention_metadata", ir.DataType.INT32, [3])
    attention = value("attention", dtype, [3, 768])
    present_keys = value("present_keys_0", dtype, [1, 256, 1, 128])
    present_values = value("present_values_0", dtype, [1, 256, 1, 128])
    node = ir.node(
        "PagedAttention",
        inputs=[query, None, None, past_keys, past_values, cumulative, past_lengths, block_table, *([None] * 8), metadata],
        outputs=[attention, present_keys, present_values],
        domain="com.microsoft",
        attributes={"num_heads": 6, "kv_num_heads": 1},
        name="attention",
    )
    graph = ir.Graph(
        inputs=[query, past_keys, past_values, cumulative, past_lengths, block_table, metadata],
        outputs=[present_keys, present_values], nodes=[node], opset_imports={"": 20, "com.microsoft": 1},
    )
    model_path = tmp_path / "paged.onnx"
    ir.save(ir.Model(graph, ir_version=10), model_path)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=["CUDAExecutionProvider"])
    packed = np.concatenate([np.ones((3, 768)), np.full((3, 128), 2), np.full((3, 128), 3)], axis=1).astype(np.float32)
    feeds = {
        "qkv": (packed.view(np.uint32) >> 16).astype(np.uint16) if cache_dtype == "BFLOAT16" else packed.astype(np.float16),
        "past_keys_0": np.zeros((1, 256, 1, 128), dtype=np.uint16 if cache_dtype == "BFLOAT16" else np.float16),
        "past_values_0": np.zeros((1, 256, 1, 128), dtype=np.uint16 if cache_dtype == "BFLOAT16" else np.float16),
        "cumulative_sequence_lengths": np.asarray([0, 3], dtype=np.int32),
        "past_sequence_lengths": np.asarray([0], dtype=np.int32),
        "block_table": np.asarray([[0]], dtype=np.int32),
        "attention_metadata": np.asarray([3, 3, 1], dtype=np.int32),
    }
    outputs = kvc.run_calibration_batch(session, ["present_keys_0", "present_values_0"], feeds)
    np.testing.assert_array_equal(outputs[0].reshape(256, 128)[:3], np.full((3, 128), 2))
    np.testing.assert_array_equal(outputs[1].reshape(256, 128)[:3], np.full((3, 128), 3))


# ---------------------------------------------------------------------------
# _get_quant_type_max
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quant_type,expected_qmax",
    [
        ("int8", 128.0),
        ("int8_per_channel", 128.0),
        ("int4", 8.0),
        ("int4_per_tensor", 8.0),
        ("fp8", 448.0),
        ("fp8_per_channel", 448.0),
    ],
)
def test_get_quant_type_max_maps_prefix_to_qmax(quant_type, expected_qmax):
    assert kvc._get_quant_type_max(quant_type) == expected_qmax


@pytest.mark.parametrize("quant_type", ["int6", "int16", "fp16", "", "bf16_per_channel"])
def test_get_quant_type_max_rejects_unsupported(quant_type):
    with pytest.raises(ValueError, match="Unsupported kv_cache quant_type"):
        kvc._get_quant_type_max(quant_type)


# ---------------------------------------------------------------------------
# _pair_envelope
# ---------------------------------------------------------------------------


def _rope_rotate(x, num_kv_heads, head_size, theta):
    """Apply a RoPE rotation of angle ``theta`` to every channel pair (d, d + head_size/2)."""
    half = head_size // 2
    pairs = x.reshape(x.shape[0], num_kv_heads, head_size).astype(np.float32)
    lo = pairs[:, :, :half]
    hi = pairs[:, :, half:]
    rot_lo = lo * np.cos(theta) - hi * np.sin(theta)
    rot_hi = lo * np.sin(theta) + hi * np.cos(theta)
    return np.concatenate([rot_lo, rot_hi], axis=-1).reshape(x.shape[0], -1)


def test_pair_envelope_shape_and_duplication():
    num_kv_heads, head_size = 2, 8
    channels = num_kv_heads * head_size
    x = np.arange(3 * channels, dtype=np.float32).reshape(3, channels)

    env = kvc._pair_envelope(x, num_kv_heads, head_size)

    assert env.shape == x.shape
    # Both channels of a pair must carry the same pair norm.
    half = head_size // 2
    env_h = env.reshape(3, num_kv_heads, head_size)
    np.testing.assert_allclose(env_h[:, :, :half], env_h[:, :, half:])


def test_pair_envelope_matches_manual_norm():
    num_kv_heads, head_size = 1, 4
    # pair (0,2) and (1,3): norms sqrt(3^2+4^2)=5 and sqrt(6^2+8^2)=10
    x = np.array([[3.0, 6.0, 4.0, 8.0]], dtype=np.float32)

    env = kvc._pair_envelope(x, num_kv_heads, head_size)

    np.testing.assert_allclose(env, np.array([[5.0, 10.0, 5.0, 10.0]]), rtol=1e-6)


def test_pair_envelope_is_rotation_invariant():
    num_kv_heads, head_size = 4, 16
    channels = num_kv_heads * head_size
    rng = np.random.default_rng(0)
    x = rng.standard_normal((7, channels)).astype(np.float32)

    baseline = kvc._pair_envelope(x, num_kv_heads, head_size)
    for theta in (0.3, 1.0, 2.5, np.pi):
        rotated = _rope_rotate(x, num_kv_heads, head_size, theta)
        env = kvc._pair_envelope(rotated, num_kv_heads, head_size)
        np.testing.assert_allclose(env, baseline, rtol=1e-4, atol=1e-4)


def test_pair_envelope_upper_bounds_every_position():
    # The envelope must be >= |post-RoPE value| at any rotation angle.
    num_kv_heads, head_size = 2, 8
    channels = num_kv_heads * head_size
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, channels)).astype(np.float32)

    env = kvc._pair_envelope(x, num_kv_heads, head_size)
    for theta in np.linspace(0, 2 * np.pi, 13):
        rotated = _rope_rotate(x, num_kv_heads, head_size, theta)
        assert np.all(np.abs(rotated) <= env + 1e-4)


# ---------------------------------------------------------------------------
# _tokenize_corpus
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Deterministic tokenizer: encodes text to a fixed increasing token stream."""

    def __init__(self, tokens_per_call=100):
        self._tokens = list(range(tokens_per_call))
        self.calls = 0

    def encode(self, text):
        self.calls += 1
        return list(self._tokens)


def test_tokenize_corpus_produces_requested_sequences():
    tok = _FakeTokenizer()
    seqs = kvc._tokenize_corpus(tok, num_seqs=4, target_seq=32)

    assert len(seqs) == 4
    for seq in seqs:
        assert seq.shape == (1, 32)
        assert seq.dtype == np.int64


def test_tokenize_corpus_windows_are_non_overlapping_and_contiguous():
    tok = _FakeTokenizer(tokens_per_call=50)
    num_seqs, target_seq = 3, 20
    seqs = kvc._tokenize_corpus(tok, num_seqs=num_seqs, target_seq=target_seq)

    flat = np.concatenate([s.reshape(-1) for s in seqs])
    # Rebuild the same underlying stream the function slices from.
    stream = []
    while len(stream) < num_seqs * target_seq:
        stream.extend(range(50))
    np.testing.assert_array_equal(flat, np.asarray(stream[: num_seqs * target_seq]))


class _RecordingTokenizer:
    """Tokenizer that records the text it was asked to encode."""

    def __init__(self):
        self.encoded_text = None

    def encode(self, text):
        self.encoded_text = text
        return list(range(200))


def test_tokenize_corpus_uses_custom_corpus():
    tok = _RecordingTokenizer()
    corpus = ["custom passage one", "custom passage two"]

    kvc._tokenize_corpus(tok, num_seqs=1, target_seq=10, corpus=corpus)

    assert tok.encoded_text == "custom passage one\n\ncustom passage two"


def test_tokenize_corpus_falls_back_to_builtin_when_corpus_empty():
    tok = _RecordingTokenizer()

    kvc._tokenize_corpus(tok, num_seqs=1, target_seq=10, corpus=[])

    assert tok.encoded_text == "\n\n".join(kvc.CORPUS)


class _RotationAwareTokenizer:
    """Tokenizer that records every text it encoded and hashes it into tokens."""

    def __init__(self, tokens_per_call=30):
        self.encoded_texts = []
        self._tokens_per_call = tokens_per_call

    def encode(self, text):
        self.encoded_texts.append(text)
        return list(range(len(self.encoded_texts) * 1000, len(self.encoded_texts) * 1000 + self._tokens_per_call))


def test_tokenize_corpus_rotates_passage_order_when_repeating():
    tok = _RotationAwareTokenizer()
    corpus = ["alpha", "beta", "gamma"]

    kvc._tokenize_corpus(tok, num_seqs=3, target_seq=30, corpus=corpus)

    # 90 tokens needed, 30 produced per encode -> three passes, each with a rotated order.
    assert tok.encoded_texts == ["alpha\n\nbeta\n\ngamma", "beta\n\ngamma\n\nalpha", "gamma\n\nalpha\n\nbeta"]


def test_tokenize_corpus_warns_when_corpus_is_too_short(caplog):
    tok = _RotationAwareTokenizer()

    with caplog.at_level("WARNING"):
        kvc._tokenize_corpus(tok, num_seqs=2, target_seq=30, corpus=["alpha", "beta"])

    assert "Calibration corpus holds 30 tokens but 60 are needed" in caplog.text


@pytest.mark.parametrize(
    "num_seqs,target_seq,error",
    [(0, 10, "num_seqs must be positive"), (1, 0, "target_seq must be positive")],
)
def test_tokenize_corpus_rejects_non_positive_dimensions(num_seqs, target_seq, error):
    with pytest.raises(ValueError, match=error):
        kvc._tokenize_corpus(_FakeTokenizer(), num_seqs=num_seqs, target_seq=target_seq)


def test_tokenize_corpus_rejects_empty_token_stream():
    with pytest.raises(ValueError, match="Tokenizer produced no token IDs"):
        kvc._tokenize_corpus(_FakeTokenizer(tokens_per_call=0), num_seqs=1, target_seq=10)


# ---------------------------------------------------------------------------
# _subsample
# ---------------------------------------------------------------------------


def test_subsample_returns_input_when_under_budget():
    x = np.arange(12, dtype=np.float32).reshape(6, 2)

    assert kvc._subsample(x, 6) is x
    assert kvc._subsample(x, 100) is x


def test_subsample_spans_first_and_last_rows():
    x = np.arange(100, dtype=np.float32).reshape(100, 1)

    sampled = kvc._subsample(x, 5)

    assert sampled.shape == (5, 1)
    assert sampled[0, 0] == 0.0
    assert sampled[-1, 0] == 99.0
    # Evenly strided, so no clustering at the front like a prefix slice would produce.
    assert np.all(np.diff(sampled[:, 0]) > 1.0)


def test_subsample_rejects_non_positive_budget():
    with pytest.raises(ValueError, match="budget must be positive"):
        kvc._subsample(np.zeros((4, 2), dtype=np.float32), 0)


# ---------------------------------------------------------------------------
# _mse_threshold
# ---------------------------------------------------------------------------


def test_mse_threshold_never_exceeds_amax():
    rng = np.random.default_rng(3)
    samples = rng.standard_normal((256, 8)).astype(np.float32)
    amax = np.abs(samples).max(axis=0)

    thr = kvc._mse_threshold(samples, amax, qmax=8.0, qneg=-8.0, qpos=7.0)

    assert thr.shape == amax.shape
    assert np.all(thr <= amax + 1e-6)
    assert np.all(thr >= 0.2 * amax - 1e-6)


def test_mse_threshold_beats_minmax_on_the_same_data():
    rng = np.random.default_rng(4)
    samples = rng.standard_normal((512, 4)).astype(np.float32)
    samples[0] *= 40.0  # heavy-tailed channelwise outlier row
    amax = np.abs(samples).max(axis=0)
    qmax, qneg, qpos = 8.0, -8.0, 7.0

    def _err(thr):
        scale = thr / qmax
        deq = np.clip(np.rint(samples / scale), qneg, qpos) * scale
        return np.mean((deq - samples) ** 2, axis=0)

    thr = kvc._mse_threshold(samples, amax, qmax=qmax, qneg=qneg, qpos=qpos)

    assert np.all(_err(thr) <= _err(amax) + 1e-9)


# ---------------------------------------------------------------------------
# _load_corpus_file
# ---------------------------------------------------------------------------


def test_load_corpus_file_reads_json_array(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(["  passage one  ", "passage two", "  "]))

    assert kvc._load_corpus_file(str(path)) == ["passage one", "passage two"]


def test_load_corpus_file_json_extension_is_case_insensitive(tmp_path):
    path = tmp_path / "corpus.JSON"
    path.write_text(json.dumps(["passage one", "passage two"]))

    assert kvc._load_corpus_file(str(path)) == ["passage one", "passage two"]


def test_load_corpus_file_rejects_non_array_json(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({"passages": ["one", "two"]}))

    with pytest.raises(ValueError, match="must contain a JSON array"):
        kvc._load_corpus_file(str(path))


def test_load_corpus_file_treats_json_content_in_text_file_as_plain_text(tmp_path):
    # A .txt file whose body is a JSON array must be handled by extension, not content sniffing.
    path = tmp_path / "corpus.txt"
    path.write_text('["not", "parsed"]')

    assert kvc._load_corpus_file(str(path)) == ['["not", "parsed"]']


def test_load_corpus_file_reads_blank_line_separated_text(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("first passage\nwith two lines\n\nsecond passage\n\n\n")

    assert kvc._load_corpus_file(str(path)) == ["first passage\nwith two lines", "second passage"]


def test_load_corpus_file_raises_on_empty(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("\n\n   \n\n")

    with pytest.raises(ValueError, match="no passages"):
        kvc._load_corpus_file(str(path))


# ---------------------------------------------------------------------------
# _detect_kv_shape
# ---------------------------------------------------------------------------


class _FakeInput:
    def __init__(self, name, shape, ort_type="tensor(float16)"):
        self.name = name
        self.shape = shape
        self.type = ort_type


class _FakeSession:
    def __init__(self, inputs, outputs=None, run=None, metadata=None):
        self._inputs = inputs
        self._outputs = outputs or []
        self._run = run
        self.metadata = metadata or {}

    def get_modelmeta(self):
        return types.SimpleNamespace(custom_metadata_map=self.metadata)

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feeds):
        return self._run(output_names, feeds)


def test_detect_kv_shape_reads_concrete_input_dims():
    sess = _FakeSession(
        [
            _FakeInput("input_ids", ["batch", "seq"]),
            _FakeInput("past_key_values.0.key", ["batch", 8, "past_seq", 64]),
        ]
    )

    num_kv_heads, head_size = kvc._detect_kv_shape(sess, "/nonexistent/model.onnx")

    assert (num_kv_heads, head_size) == (8, 64)


def test_detect_kv_shape_reads_paged_cache_dims():
    session = _FakeSession([
        _FakeInput("cumulative_sequence_lengths", [2], "tensor(int32)"),
        _FakeInput("past_keys_0", ["blocks", 256, 8, 128], "tensor(bfloat16)"),
    ])
    assert kvc._detect_kv_shape(session, "/nonexistent/model.onnx") == (8, 128)


@pytest.mark.parametrize("block_dimension", [2, "block_size"])
@pytest.mark.parametrize("position_planes", [None, 3])
@pytest.mark.parametrize("cache_names", [("past_keys_3", "past_values_3"), ("past_key_values.3.key", "past_key_values.3.value")])
def test_calibrate_paged_hybrid_model_resolves_input_shapes(
    tmp_path, monkeypatch, block_dimension, position_planes, cache_names
):
    ir = pytest.importorskip("onnx_ir")
    tokenizers = pytest.importorskip("tokenizers")

    def value(name, dtype, shape):
        return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))

    cache_shape = ["num_blocks", block_dimension, 1, 4]
    keys = value(cache_names[0], ir.DataType.FLOAT, cache_shape)
    values = value(cache_names[1], ir.DataType.FLOAT, cache_shape)
    positions = value("position_ids", ir.DataType.INT64, [3, "tokens"] if position_planes else ["tokens"])
    inputs = [
        value("input_ids", ir.DataType.INT64, ["tokens"]),
        value("cumulative_sequence_lengths", ir.DataType.INT32, [2]),
        value("past_sequence_lengths", ir.DataType.INT32, [1]),
        value("block_table", ir.DataType.INT32, [1, "num_blocks"]),
        positions,
        keys,
        values,
        value("past_key_values.0.conv_state", ir.DataType.FLOAT16, ["batch", 8, 3]),
        value("past_key_values.0.recurrent_state", ir.DataType.FLOAT, ["batch", 2, 4, 4]),
    ]
    present_keys = value("present.3.key", ir.DataType.FLOAT, cache_shape)
    present_values = value("present.3.value", ir.DataType.FLOAT, cache_shape)
    key_delta = value("key_delta", ir.DataType.FLOAT, [])
    value_delta = value("value_delta", ir.DataType.FLOAT, [])
    graph = ir.Graph(
        inputs=inputs,
        outputs=[present_keys, present_values],
        nodes=[
            ir.node("Constant", inputs=[], outputs=[key_delta], attributes={"value": ir.tensor(np.float32(2))}),
            ir.node("Constant", inputs=[], outputs=[value_delta], attributes={"value": ir.tensor(np.float32(4))}),
            ir.node("Add", inputs=[keys, key_delta], outputs=[present_keys]),
            ir.node("Add", inputs=[values, value_delta], outputs=[present_values]),
        ],
        opset_imports={"": 20},
    )
    model_path = tmp_path / "model.onnx"
    ir.save(ir.Model(graph, ir_version=10), model_path)
    (tmp_path / "genai_config.json").write_text(json.dumps({"engine": {"dynamic_batching": {"block_size": 2}}}))
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    run_batch = kvc.run_calibration_batch

    def checked_run(session, output_names, feeds):
        assert feeds[cache_names[0]].shape == feeds[cache_names[1]].shape == (2, 2, 1, 4)
        np.testing.assert_array_equal(feeds["past_key_values.0.conv_state"], np.zeros((1, 8, 3), dtype=np.float16))
        np.testing.assert_array_equal(feeds["past_key_values.0.recurrent_state"], np.zeros((1, 2, 4, 4), dtype=np.float32))
        expected_positions = np.arange(3) if position_planes is None else np.tile(np.arange(3), (3, 1))
        np.testing.assert_array_equal(feeds["position_ids"], expected_positions)
        return run_batch(session, output_names, feeds)

    monkeypatch.setattr(kvc, "run_calibration_batch", checked_run)
    output_path = tmp_path / "scales.json"
    kvc.calibrate_kv_scales(
        str(model_path), str(tmp_path), str(output_path), method="minmax", target_seq=3, num_seqs=1,
        k_rotary_envelope=False, corpus=["a b c"],
    )
    result = json.loads(output_path.read_text())
    assert result["layer_ids"] == [3]
    np.testing.assert_allclose(result["scales"]["k_scales"], [[2 / 128] * 4])
    np.testing.assert_allclose(result["scales"]["v_scales"], [[4 / 128] * 4])


def test_calibrate_paged_cache_ignores_unused_slots(tmp_path, monkeypatch):
    inputs = [
        _FakeInput("input_ids", ["tokens"], "tensor(int64)"),
        _FakeInput("cumulative_sequence_lengths", [2], "tensor(int32)"),
        _FakeInput("past_sequence_lengths", [1], "tensor(int32)"),
        _FakeInput("block_table", [1, "blocks"], "tensor(int32)"),
        _FakeInput("block_table_windowed", [1, "blocks"], "tensor(int32)"),
        _FakeInput("attention_metadata", [3], "tensor(int32)"),
        _FakeInput("past_keys_3", ["blocks", 2, 1, 4], "tensor(float)"),
        _FakeInput("past_values_3", ["blocks", 2, 1, 4], "tensor(float)"),
    ]
    outputs = [types.SimpleNamespace(name=name) for name in ("present_keys_3", "present_values_3")]

    def run(output_names, feeds):
        assert output_names == ["present_keys_3", "present_values_3"]
        np.testing.assert_array_equal(feeds["cumulative_sequence_lengths"], [0, 3])
        np.testing.assert_array_equal(feeds["past_sequence_lengths"], [0])
        np.testing.assert_array_equal(feeds["attention_metadata"], [3, 3, 1])
        np.testing.assert_array_equal(feeds["block_table"], [[0, 1]])
        np.testing.assert_array_equal(feeds["block_table_windowed"], [[0, 1]])
        assert feeds["input_ids"].shape == (3,)
        assert feeds["past_keys_3"].shape == (2, 2, 1, 4)
        keys = np.full((2, 2, 1, 4), 2.0, dtype=np.float32)
        values = keys * 2
        keys.reshape(4, 4)[3] = 1000
        values.reshape(4, 4)[3] = 1000
        return [keys, values]

    session = _FakeSession(inputs, outputs, run)
    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(
        SessionOptions=types.SimpleNamespace,
        InferenceSession=lambda *args, **kwargs: session,
        get_available_providers=lambda: ["CPUExecutionProvider"],
    ))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda path: _FakeTokenizer()),
    ))
    monkeypatch.setattr(kvc, "run_calibration_batch", lambda session, names, feeds: session.run(names, feeds))
    output_path = tmp_path / "scales.json"
    kvc.calibrate_kv_scales(
        "model.onnx", "tokenizer", str(output_path), method="minmax", target_seq=3, num_seqs=1,
        k_rotary_envelope=False,
    )
    result = json.loads(output_path.read_text())
    assert result["layer_ids"] == [3]
    np.testing.assert_allclose(result["scales"]["k_scales"], [[2 / 128] * 4])
    np.testing.assert_allclose(result["scales"]["v_scales"], [[4 / 128] * 4])


def test_detect_kv_shape_falls_back_to_genai_config(tmp_path):
    (tmp_path / "genai_config.json").write_text(
        json.dumps({"model": {"decoder": {"num_key_value_heads": 4, "head_size": 128}}})
    )
    model_path = tmp_path / "model.onnx"
    # Symbolic last dim (kv_cache_dim) forces the config fallback for head_size.
    sess = _FakeSession([_FakeInput("past_key_values.0.key", ["batch", 4, "past_seq", "kv_cache_dim"])])

    num_kv_heads, head_size = kvc._detect_kv_shape(sess, str(model_path))

    assert (num_kv_heads, head_size) == (4, 128)


def test_detect_kv_shape_raises_when_undetectable(tmp_path):
    model_path = tmp_path / "model.onnx"  # no genai_config.json alongside
    sess = _FakeSession([_FakeInput("past_key_values.0.key", ["batch", "heads", "past_seq", "kv_cache_dim"])])

    with pytest.raises(ValueError, match="Could not infer num_kv_heads/head_size"):
        kvc._detect_kv_shape(sess, str(model_path))


@pytest.mark.parametrize("rope_context_input", [None, "rope_context_length", "custom_context_length"])
def test_calibrate_kv_scales_feeds_model_metadata_and_writes_scales(tmp_path, monkeypatch, rope_context_input):
    target_seq = 4
    inputs = [
        _FakeInput("input_ids", ["batch", "sequence"], "tensor(int64)"),
        _FakeInput("attention_mask", ["batch", "total_sequence"], "tensor(int64)"),
        _FakeInput("position_ids", ["batch", "sequence"], "tensor(int64)"),
        _FakeInput("past_key_values.3.key", ["batch", 1, "past_sequence", 4], "tensor(float)"),
        _FakeInput("past_key_values.3.value", ["batch", 1, "past_sequence", 4], "tensor(float)"),
        _FakeInput("past_key_values.1.conv_state", ["batch", 8, 3], "tensor(float16)"),
        _FakeInput("past_key_values.1.recurrent_state", ["batch", 2, 4, 4], "tensor(float)"),
    ]
    metadata = {}
    if rope_context_input is not None:
        inputs.append(_FakeInput(rope_context_input, [], "tensor(int64)"))
        metadata["com.microsoft.genai.rope_context_length_input"] = rope_context_input
    outputs = [
        types.SimpleNamespace(name="present.3.key"),
        types.SimpleNamespace(name="present.3.value"),
    ]

    def run(output_names, feeds):
        assert output_names == ["present.3.key", "present.3.value"]
        if rope_context_input is not None:
            assert feeds[rope_context_input].shape == ()
            assert feeds[rope_context_input].dtype == np.int64
            assert feeds[rope_context_input] == target_seq
        np.testing.assert_array_equal(feeds["position_ids"], np.arange(target_seq).reshape(1, target_seq))
        assert feeds["past_key_values.3.key"].dtype == np.float32
        assert feeds["past_key_values.3.value"].dtype == np.float32
        assert feeds["past_key_values.3.key"].shape == (1, 1, 0, 4)
        assert feeds["past_key_values.3.value"].shape == (1, 1, 0, 4)
        np.testing.assert_array_equal(feeds["past_key_values.1.conv_state"], np.zeros((1, 8, 3), dtype=np.float16))
        np.testing.assert_array_equal(
            feeds["past_key_values.1.recurrent_state"], np.zeros((1, 2, 4, 4), dtype=np.float32)
        )
        shape = (1, 1, target_seq, 4)
        return [np.full(shape, 2.0, dtype=np.float32), np.full(shape, 4.0, dtype=np.float32)]

    session = _FakeSession(inputs, outputs, run, metadata)
    session_options = types.SimpleNamespace(log_severity_level=None)
    requested_providers = []

    def make_session(*args, **kwargs):
        requested_providers.extend(kwargs["providers"])
        return session

    ort = types.SimpleNamespace(
        SessionOptions=lambda: session_options,
        InferenceSession=make_session,
        get_available_providers=lambda: ["CPUExecutionProvider"],
    )
    tokenizer = _FakeTokenizer(tokens_per_call=target_seq * 2)
    transformers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda path: tokenizer))
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    output_path = tmp_path / "scales.json"

    result = kvc.calibrate_kv_scales(
        model_path="model.onnx",
        tokenizer_path="tokenizer",
        out_json=str(output_path),
        quant_type="int8_per_channel",
        method="minmax",
        target_seq=target_seq,
        num_seqs=1,
        num_layers=1,
        num_kv_heads=1,
        head_size=4,
        k_rotary_envelope=False,
    )

    assert result == str(output_path.resolve())
    # Unavailable providers must be filtered out so CPU-only onnxruntime installs work.
    assert requested_providers == ["CPUExecutionProvider"]
    scale_data = json.loads(output_path.read_text())
    assert scale_data["layer_ids"] == [3]
    assert scale_data["qmax"] == 128.0
    scales = scale_data["scales"]
    np.testing.assert_allclose(scales["k_scales"], [[2.0 / 128.0] * 4])
    np.testing.assert_allclose(scales["v_scales"], [[4.0 / 128.0] * 4])


@pytest.mark.parametrize("target_seq", [7, 8, 9])
def test_calibrate_longrope_fixture_uses_sample_context(tmp_path, monkeypatch, target_seq):
    tokenizer = _FakeTokenizer(tokens_per_call=target_seq)
    transformers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda path: tokenizer))
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    model_path = Path(__file__).parents[2] / "models" / "longrope-context" / "model.onnx"
    output_path = tmp_path / "scales.json"

    kvc.calibrate_kv_scales(
        model_path=str(model_path),
        tokenizer_path="tokenizer",
        out_json=str(output_path),
        method="minmax",
        target_seq=target_seq,
        num_seqs=1,
        k_rotary_envelope=False,
    )

    scales = json.loads(output_path.read_text())["scales"]
    expected_scale = (target_seq - 1) * (2 if target_seq > 8 else 1) / 128.0
    np.testing.assert_allclose(scales["k_scales"], [[expected_scale]])
    np.testing.assert_allclose(scales["v_scales"], [[expected_scale]])
