# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import json

import pytest

from benchmark_qwen_context import make_review_prompt
from run_context_sweep import file_hash, save_table


class CharacterTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt):
        assert add_generation_prompt
        messages = json.loads(messages)
        assert [message["role"] for message in messages] == ["system", "user"]
        return messages[0]["content"] + messages[1]["content"] + "\nassistant\n"

    def encode(self, text):
        return list(text.encode("ascii"))


@pytest.mark.parametrize("length", [8192, 16384, 32768, 65536, 131072])
def test_prompt_preserves_generation_suffix_and_length(length):
    tokenizer = CharacterTokenizer()
    tokens = make_review_prompt(tokenizer, length)
    assert len(tokens) == length
    assert bytes(tokens.tolist()).endswith(b"\nassistant\n")
    assert (tokens == make_review_prompt(tokenizer, length)).all()


def test_streaming_hash(tmp_path):
    path = tmp_path / "binary"
    data = b"x" * (1024 * 1024 + 7)
    path.write_bytes(data)
    assert file_hash(path) == hashlib.sha256(data).hexdigest()


def test_table_reports_failures(tmp_path):
    save_table([
        {"status": "failed", "requested_prompt": 131072, "mtp": 7, "peak_mib": 100000},
    ], tmp_path)
    table = (tmp_path / "TABLE.md").read_text()
    assert "| 131072 | - | 7 |" in table
    assert "failed; see log" in table
    assert "| completed |" not in table


def test_table_formats_completed_measurements(tmp_path):
    save_table([{
        "status": "completed", "requested_prompt": 8192, "prompt_length": 8190,
        "mtp": 3, "ttft_ms": 1226.31, "decode_tps": 129.52, "e2e_tps": 99.00,
        "mtp_acceptance": 71.27, "first_draft_acceptance": 75.00,
        "proposal_yield": 53.75, "within_target": False, "peak_mib": 94340,
    }], tmp_path)
    assert (
        "| 8192 | 8190 | 3 | 1226.31 | 129.52 | 99.00 | 71.27% | 75.00% "
        "| 53.75% | no | 94340 | completed |"
    ) in (tmp_path / "TABLE.md").read_text()
