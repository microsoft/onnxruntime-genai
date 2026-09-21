# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the LFM2-Audio model builder wiring.

LFM2-Audio checkpoints (LFM2-Audio-1.5B, LFM2.5-Audio-1.5B) have no transformers model class: their
config.json nests the LFM2 decoder config under "lfm" and the checkpoint stores the decoder weights
under the "lfm." prefix next to the audio encoder, depthformer and audio embeddings. The builder has
to read that nested config, load only the decoder with its logits tied to the token embeddings, and
label the export `lfm2_audio` (a pipeline stage) or `lfm2_audio_text` (a text-only decoder).
"""

from __future__ import annotations

import json
import sys
import types

import pytest
from _builder_test_utils import load_builder_module
from test_lfm2_vl import builder_module

base_module = load_builder_module("base")
lfm2_module = load_builder_module("lfm2")
LFM2AudioModel = lfm2_module.LFM2AudioModel

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
save_file = pytest.importorskip("safetensors.torch").save_file
# `_builder_test_utils` substitutes a stub for transformers when it is missing, which has none of the
# LFM2 classes these tests build a checkpoint from.
pytestmark = pytest.mark.skipif(
    not hasattr(transformers, "Lfm2ForCausalLM"), reason="transformers does not provide the LFM2 model classes"
)

DECODER_CONFIG = {
    "architectures": ["Lfm2ForCausalLM"],
    "block_auto_adjust_ff_dim": True,
    "block_ff_dim": 96,
    "block_ffn_dim_multiplier": 1.0,
    "block_multiple_of": 16,
    "conv_L_cache": 3,
    "conv_bias": False,
    "hidden_size": 32,
    "intermediate_size": 96,
    "layer_types": ["conv", "full_attention", "conv", "full_attention"],
    "model_type": "lfm2",
    "norm_eps": 1e-05,
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "num_key_value_heads": 2,
    "rope_theta": 1000000,
    "vocab_size": 512,
}


def _write_checkpoint(path, decoder_weights=None, shards=1):
    """An LFM2-Audio checkpoint directory: nested config.json plus lfm.* and audio tensors."""
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["Lfm2AudioForConditionalGeneration"],
        "codebooks": 8,
        "preprocessor": {"sample_rate": 16000, "features": 128, "n_fft": 512},
        "encoder": {"feat_in": 128, "d_model": 512, "n_layers": 17},
        "lfm": DECODER_CONFIG,
        "depthformer": {"layers": 6, "dim": 1024, "tie": True},
    }
    (path / "config.json").write_text(json.dumps(config))

    if decoder_weights is None:
        torch.manual_seed(0)
        model = transformers.Lfm2ForCausalLM(transformers.Lfm2Config(**DECODER_CONFIG))
        decoder_weights = {
            "lfm." + name[len("model.") :]: tensor
            for name, tensor in model.state_dict().items()
            if name.startswith("model.")
        }
    tensors = dict(decoder_weights)
    tensors["conformer.pre_encode.out.weight"] = torch.zeros(4, 4)
    tensors["audio_adapter.model.1.weight"] = torch.zeros(32, 8)
    tensors["audio_embedding.embedding.weight"] = torch.zeros(16, 32)
    tensors["depth_linear.weight"] = torch.zeros(8, 32)
    tensors["codebook_offsets"] = torch.arange(8) * 2049
    tensors = {name: tensor.contiguous() for name, tensor in tensors.items()}
    if shards == 1:
        save_file(tensors, str(path / "model.safetensors"))
    else:
        # What `save_pretrained` writes once a checkpoint is over its shard size: several files and
        # an index naming which tensor lives in which. An fp32 fine-tune of this model reaches it.
        names = sorted(tensors)
        weight_map = {}
        for shard in range(shards):
            filename = f"model-{shard + 1:05d}-of-{shards:05d}.safetensors"
            part = {name: tensors[name] for name in names[shard::shards]}
            save_file(part, str(path / filename))
            weight_map.update(dict.fromkeys(part, filename))
        (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return decoder_weights


def test_lfm2_audio_load_config_reads_the_nested_decoder_config(tmp_path):
    _write_checkpoint(tmp_path)

    config = LFM2AudioModel.load_config(str(tmp_path))

    assert config.architectures == ["Lfm2AudioForConditionalGeneration"]
    assert config._name_or_path == str(tmp_path)
    assert config.hidden_size == DECODER_CONFIG["hidden_size"]
    assert config.layer_types == DECODER_CONFIG["layer_types"]
    assert config.conv_L_cache == DECODER_CONFIG["conv_L_cache"]
    assert config.tie_word_embeddings


def test_lfm2_audio_load_config_leaves_other_checkpoints_to_autoconfig(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(DECODER_CONFIG))
    assert LFM2AudioModel.load_config(str(tmp_path)) is None
    assert LFM2AudioModel.load_config(str(tmp_path / "missing")) is None


def _audio_builder(tmp_path, extra_options=None, model_name_or_path=None):
    model = LFM2AudioModel.__new__(LFM2AudioModel)
    model.decoder_config = LFM2AudioModel.load_config(str(tmp_path))
    model.quant_type = None
    model.cache_dir = str(tmp_path / "cache")
    model.hf_token = True
    model.extra_options = extra_options or {}
    model.model_name_or_path = model_name_or_path or str(tmp_path)
    return model


def test_lfm2_audio_load_weights_loads_the_decoder_with_tied_logits(tmp_path):
    decoder_weights = _write_checkpoint(tmp_path)

    loaded = _audio_builder(tmp_path).load_weights(str(tmp_path))

    assert type(loaded).__name__ == "Lfm2ForCausalLM"
    torch.testing.assert_close(
        loaded.model.layers[0].conv.in_proj.weight, decoder_weights["lfm.layers.0.conv.in_proj.weight"]
    )
    torch.testing.assert_close(loaded.model.embedding_norm.weight, decoder_weights["lfm.embedding_norm.weight"])
    assert loaded.lm_head.weight.data_ptr() == loaded.model.embed_tokens.weight.data_ptr()
    assert not any("conformer" in name or "depth" in name or "audio" in name for name, _ in loaded.named_parameters())
    assert not any(parameter.is_meta for parameter in loaded.parameters())


@pytest.mark.parametrize("shards", [2, 3])
def test_lfm2_audio_load_weights_reads_a_sharded_checkpoint(tmp_path, shards):
    # These checkpoints are one file today, but `save_pretrained` shards anything over its size
    # limit, which an fp32 fine-tune of this model exceeds. The tensors have to be gathered from
    # every shard the index names, not just the first.
    decoder_weights = _write_checkpoint(tmp_path, shards=shards)
    assert not (tmp_path / "model.safetensors").exists(), "the sharded layout has no single file"

    loaded = _audio_builder(tmp_path).load_weights(str(tmp_path))

    torch.testing.assert_close(
        loaded.model.layers[0].conv.in_proj.weight, decoder_weights["lfm.layers.0.conv.in_proj.weight"]
    )
    torch.testing.assert_close(loaded.model.embedding_norm.weight, decoder_weights["lfm.embedding_norm.weight"])
    assert not any(parameter.is_meta for parameter in loaded.parameters())


def test_lfm2_audio_load_weights_rejects_a_checkpoint_missing_decoder_tensors(tmp_path):
    torch.manual_seed(0)
    model = transformers.Lfm2ForCausalLM(transformers.Lfm2Config(**DECODER_CONFIG))
    decoder_weights = {
        "lfm." + name[len("model.") :]: tensor
        for name, tensor in model.state_dict().items()
        if name.startswith("model.")
    }
    del decoder_weights["lfm.layers.1.self_attn.q_proj.weight"]
    _write_checkpoint(tmp_path, decoder_weights)

    with pytest.raises(
        ValueError, match=r"does not match its config: missing .*'model\.layers\.1\.self_attn\.q_proj\.weight'"
    ):
        _audio_builder(tmp_path).load_weights(str(tmp_path))


def test_lfm2_audio_load_weights_downloads_the_checkpoint_named_by_the_config(monkeypatch, tmp_path):
    # `-m <repo id>` leaves the input path empty; the checkpoint to fetch is then the one the config
    # was read from. Passing the empty path straight to the Hub asks it for a repository called "".
    _write_checkpoint(tmp_path)
    asked = []

    def fake_snapshot_download(repo_id, **kwargs):
        asked.append((repo_id, kwargs))
        return str(tmp_path)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    builder = _audio_builder(tmp_path, model_name_or_path="LiquidAI/LFM2.5-Audio-1.5B")
    loaded = builder.load_weights("")

    assert type(loaded).__name__ == "Lfm2ForCausalLM"
    assert asked[0][0] == "LiquidAI/LFM2.5-Audio-1.5B"
    assert asked[0][1]["cache_dir"] == builder.cache_dir


def test_lfm2_audio_load_weights_prefers_a_local_input_path(monkeypatch, tmp_path):
    _write_checkpoint(tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError("a local checkpoint directory must not be downloaded")

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = fail
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    loaded = _audio_builder(tmp_path, model_name_or_path="LiquidAI/LFM2.5-Audio-1.5B").load_weights(str(tmp_path))

    assert type(loaded).__name__ == "Lfm2ForCausalLM"


def test_lfm2_audio_load_weights_applies_the_lora_adapter(monkeypatch, tmp_path):
    # `load_weights` is overridden for these checkpoints, so it has to apply `adapter_path` itself;
    # returning the bare decoder would drop the adapter without a word.
    _write_checkpoint(tmp_path)
    wrapped = []

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, adapter_path, **kwargs):
            wrapped.append((model, adapter_path, kwargs))
            return "adapted"

    peft = types.ModuleType("peft")
    peft.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", peft)

    builder = _audio_builder(tmp_path, {"adapter_path": "some/adapter"})
    loaded = builder.load_weights(str(tmp_path))

    assert loaded == "adapted"
    model, adapter_path, kwargs = wrapped[0]
    assert type(model).__name__ == "Lfm2ForCausalLM"
    assert adapter_path == "some/adapter"
    assert kwargs == {"cache_dir": builder.cache_dir, "token": True}


def test_lfm2_audio_load_weights_without_an_adapter_returns_the_decoder(tmp_path):
    _write_checkpoint(tmp_path)
    loaded = _audio_builder(tmp_path).load_weights(str(tmp_path))
    assert type(loaded).__name__ == "Lfm2ForCausalLM"


@pytest.mark.parametrize(
    "existing,expected",
    [(7, [7, 128]), ([7], [7, 128]), ([7, 2], [7, 2, 128]), ([7, 128], [7, 128])],
    ids=["scalar", "list", "several", "already-there"],
)
def test_lfm2_audio_stops_when_the_model_starts_speaking(existing, expected):
    # The model answers in speech too. Everything after <|audio_start|> is audio codes meant for the
    # depthformer, which this runtime does not have, so generation has to end there rather than read
    # those positions off the text head.
    builder = LFM2AudioModel.__new__(LFM2AudioModel)
    builder.layer_types = ["conv", "full_attention"]
    builder.conv_L_cache = 3
    genai_config = {"model": {"decoder": {}, "eos_token_id": existing}}

    builder.update_genai_config(genai_config)

    assert genai_config["model"]["eos_token_id"] == expected
    # The LFM2 decoder settings the base class writes are still there.
    assert genai_config["model"]["decoder"]["layer_types"] == ["conv", "full_attention"]
    assert genai_config["model"]["decoder"]["conv_cache_size"] == 2


@pytest.mark.parametrize("exclude_embeds,expected_type", [(True, "lfm2_audio"), (False, "lfm2_audio_text")])
def test_lfm2_audio_architecture_labels_the_export(monkeypatch, tmp_path, exclude_embeds, expected_type):
    captured = {}

    class FakeLFM2AudioModel:
        model_type = "lfm2"

        def __init__(self, *args):
            captured["args"] = args
            self.exclude_embeds = exclude_embeds

        def make_genai_config(self, *args):
            captured["model_type"] = self.model_type

        def save_processing(self, *args):
            pass

    config = types.SimpleNamespace(architectures=["Lfm2AudioForConditionalGeneration"])
    monkeypatch.setattr(builder_module, "LFM2AudioModel", FakeLFM2AudioModel)

    builder_module.create_model(
        "LiquidAI/LFM2.5-Audio-1.5B",
        str(tmp_path / "input"),
        str(tmp_path / "output"),
        "int4",
        "cpu",
        str(tmp_path / "cache"),
        config_only=True,
        hf_details={"extra_kwargs": {}, "hf_name": "LiquidAI/LFM2.5-Audio-1.5B", "hf_config": config},
    )

    assert captured["args"][0] is config
    assert captured["model_type"] == expected_type
