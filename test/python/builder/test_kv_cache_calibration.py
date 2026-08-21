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

MODELS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models"


def _load_calibration_module():
    sys.modules.setdefault("models", types.ModuleType("models"))
    spec = importlib.util.spec_from_file_location("models.kv_cache_calibration", MODELS_DIR / "kv_cache_calibration.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["models.kv_cache_calibration"] = module
    spec.loader.exec_module(module)
    return module


kvc = _load_calibration_module()


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
    def __init__(self, inputs, outputs=None, run=None):
        self._inputs = inputs
        self._outputs = outputs or []
        self._run = run

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


def test_calibrate_kv_scales_feeds_model_metadata_and_writes_scales(tmp_path, monkeypatch):
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
    outputs = [
        types.SimpleNamespace(name="present.3.key"),
        types.SimpleNamespace(name="present.3.value"),
    ]

    def run(output_names, feeds):
        assert output_names == ["present.3.key", "present.3.value"]
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

    session = _FakeSession(inputs, outputs, run)
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
    scales = scale_data["scales"]
    np.testing.assert_allclose(scales["k_scales"], [[2.0 / 128.0] * 4])
    np.testing.assert_allclose(scales["v_scales"], [[4.0 / 128.0] * 4])
