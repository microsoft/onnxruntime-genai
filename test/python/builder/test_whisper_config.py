# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from types import SimpleNamespace

import pytest
from _builder_test_utils import load_builder_module

whisper_module = load_builder_module("whisper")
WhisperModel = whisper_module.WhisperModel


class FakeTokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def get_vocab(self):
        return self.vocabulary


def create_model():
    model = WhisperModel.__new__(WhisperModel)
    model.decoder = SimpleNamespace(model_name_or_path="test-whisper")
    model.hf_token = "token"
    model.hf_remote = False
    model.vocab_size = 8
    return model


def valid_vocabulary():
    return {
        "<|notimestamps|>": 4,
        "<|0.00|>": 5,
        "<|0.02|>": 6,
        "<|0.04|>": 7,
    }


def set_tokenizer_and_generation_config(monkeypatch, vocabulary, max_initial_timestamp_index=1):
    tokenizer = FakeTokenizer(vocabulary)

    def load_tokenizer(*args, **kwargs):
        return tokenizer

    def load_generation_config(*args, **kwargs):
        return SimpleNamespace(max_initial_timestamp_index=max_initial_timestamp_index)

    monkeypatch.setattr(whisper_module.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(whisper_module.GenerationConfig, "from_pretrained", load_generation_config)


def test_resolve_timestamp_metadata_uses_tokenizer_and_generation_config(monkeypatch):
    model = create_model()
    set_tokenizer_and_generation_config(monkeypatch, valid_vocabulary())

    assert model.resolve_timestamp_metadata({"cache_dir": "cache"}) == (4, 5, 1)


@pytest.mark.parametrize(
    "vocabulary",
    [
        {"<|0.00|>": 5},
        {"<|notimestamps|>": 4},
    ],
)
def test_resolve_timestamp_metadata_omits_missing_tokens(monkeypatch, vocabulary):
    model = create_model()
    set_tokenizer_and_generation_config(monkeypatch, vocabulary)

    assert model.resolve_timestamp_metadata({}) is None


@pytest.mark.parametrize(
    "no_timestamps_token_id,timestamp_begin_token_id",
    [
        (-1, 5),
        (5, 5),
        (6, 5),
        (4, 8),
    ],
)
def test_resolve_timestamp_metadata_rejects_invalid_order(
    monkeypatch, no_timestamps_token_id, timestamp_begin_token_id
):
    model = create_model()
    set_tokenizer_and_generation_config(
        monkeypatch,
        {
            **valid_vocabulary(),
            "<|notimestamps|>": no_timestamps_token_id,
            "<|0.00|>": timestamp_begin_token_id,
        },
    )

    with pytest.raises(ValueError, match="timestamp token IDs must satisfy"):
        model.resolve_timestamp_metadata({})


def test_resolve_timestamp_metadata_rejects_non_timestamp_suffix(monkeypatch):
    model = create_model()
    vocabulary = valid_vocabulary()
    vocabulary["ordinary-token"] = 7
    set_tokenizer_and_generation_config(monkeypatch, vocabulary)

    with pytest.raises(ValueError, match="contiguous 0.02-second suffix"):
        model.resolve_timestamp_metadata({})


def test_resolve_timestamp_metadata_defaults_initial_cap(monkeypatch):
    model = create_model()
    tokenizer = FakeTokenizer(valid_vocabulary())
    monkeypatch.setattr(
        whisper_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )

    def fail_generation_config(*args, **kwargs):
        raise OSError("missing generation config")

    monkeypatch.setattr(
        whisper_module.GenerationConfig,
        "from_pretrained",
        fail_generation_config,
    )

    model.vocab_size = 60
    vocabulary = {
        "<|notimestamps|>": 4,
        **{
            f"<|{(index * 2) // 100}.{(index * 2) % 100:02d}|>": 5 + index
            for index in range(55)
        },
    }
    tokenizer.vocabulary = vocabulary

    assert model.resolve_timestamp_metadata({}) == (4, 5, 50)
