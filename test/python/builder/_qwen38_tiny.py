# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""A synthetic, internally consistent Qwen3.8 Flash NVFP4 checkpoint for exercising the real decoder builder.

Three linear-attention layers followed by one QSA layer, one PLE layer, two hyper-connection streams and
NVFP4 routed experts. Every dense tensor is a seeded random bf16 value so exports are reproducible, and all
non-expert projections stay dense exactly as in the official checkpoint. ``dims`` overrides any entry of
``TINY`` to reproduce the official checkpoint's tensor shapes.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

TINY = {
    "hidden": 64,
    "hc_count": 2,
    "hc_lowrank": 8,
    "heads": 4,
    "kv_heads": 2,
    "head_dim": 16,
    "linear_key_heads": 2,
    "linear_value_heads": 8,
    "linear_head_dim": 16,
    "indexer_heads": 2,
    "indexer_head_dim": 16,
    "num_experts": 8,
    "experts_per_token": 2,
    "moe_intermediate": 32,
    "vocab": 128,
    "num_layers": 4,
    "ple_layer": 1,
    "heads_per_ngram": 2,
    "ngram_rows": 8,
}


def make_config(d):
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "quantization_config": {
            "config_groups": {
                "group_0": {
                    "input_activations": {"dynamic": False, "group_size": 16, "num_bits": 4, "type": "float"},
                    "targets": ["Linear"],
                    "weights": {"dynamic": False, "group_size": 16, "num_bits": 4, "type": "float"},
                }
            },
            "ignore": [
                "model.embed_tokens",
                "mtp.*",
                "*.self_attn.*",
                "*.linear_attn.*",
                "*.mlp.gate*",
                "*.mlp.shared_expert.*",
                "*.mlp.shared_expert_gate*",
                "*hyper_connection*",
                "*.ple.*",
                "model.language_model.embed_tokens",
                "lm_head",
            ],
            "quant_algo": "NVFP4",
            "quant_method": "modelopt",
        },
        "text_config": {
            "model_type": "qwen4_exp_text",
            "vocab_size": d["vocab"],
            "hidden_size": d["hidden"],
            "num_hidden_layers": d["num_layers"],
            "num_attention_heads": d["heads"],
            "num_key_value_heads": d["kv_heads"],
            "head_dim": d["head_dim"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": d["linear_head_dim"],
            "linear_value_head_dim": d["linear_head_dim"],
            "linear_num_key_heads": d["linear_key_heads"],
            "linear_num_value_heads": d["linear_value_heads"],
            "moe_intermediate_size": d["moe_intermediate"],
            "shared_expert_intermediate_size": d["moe_intermediate"],
            "num_experts": d["num_experts"],
            "num_experts_per_tok": d["experts_per_token"],
            "hc_count": d["hc_count"],
            "hc_lowrank": d["hc_lowrank"],
            "ple_layer_ids": [d["ple_layer"] + 1],
            "ple_embed_dim": d["hidden"],
            "ple_conv_kernel_size": 4,
            "ngram_size": 3,
            "heads_per_ngram": d["heads_per_ngram"],
            "split_ngram_parts": 2,
            "mtp_num_hidden_layers": 0,
            "indexer_n_heads": d["indexer_heads"],
            "indexer_kv_heads": 1,
            "indexer_head_dim": d["indexer_head_dim"],
            "indexer_budget": 8,
            "indexer_compress_ratio": 4,
            "output_gate_type": "sigmoid",
            "max_position_embeddings": 256,
            "eos_token_id": 0,
            "bos_token_id": 0,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [1, 1, 0],
                "mrope_interleaved": True,
            },
        },
    }


def write_checkpoint(path, seed=0, dims=None):
    """Write the config, weights, tokenizer and preprocessor files of the synthetic checkpoint under ``path``."""
    d = {**TINY, **(dims or {})}
    hidden = d["hidden"]
    hc_hidden = d["hc_count"] * hidden
    num_layers = d["num_layers"]
    ngram_heads = 2 * d["heads_per_ngram"]
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    tensors = {}

    def rand(*shape, scale=0.1):
        return (torch.randn(*shape, generator=generator) * scale).to(torch.bfloat16)

    def dense(prefix, rows, columns):
        tensors[f"{prefix}.weight"] = rand(rows, columns)

    def norm(prefix, size):
        tensors[f"{prefix}.weight"] = rand(size, scale=0.05)

    def hyper(prefix, combine=True):
        norm(f"{prefix}.hc_norm", hc_hidden)
        dense(f"{prefix}.input_mix_weight_down", d["hc_lowrank"], hc_hidden)
        dense(f"{prefix}.input_mix_weight_up", hc_hidden, d["hc_lowrank"])
        if combine:
            dense(f"{prefix}.block_inject_weight", d["hc_count"], hc_hidden)

    def nvfp4_expert(prefix, rows, columns):
        codes = torch.randint(0, 256, (rows, columns // 2), generator=generator, dtype=torch.int32)
        tensors[f"{prefix}.weight"] = codes.to(torch.uint8)
        scales = torch.randint(0x30, 0x40, (rows, columns // 16), generator=generator, dtype=torch.int32)
        tensors[f"{prefix}.weight_scale"] = scales.to(torch.uint8).view(torch.float8_e4m3fn)
        tensors[f"{prefix}.weight_scale_2"] = torch.tensor(0.01, dtype=torch.float32)

    def layer(layer_id):
        prefix = f"model.language_model.layers.{layer_id}"
        if layer_id < num_layers - 1:
            key_dim = d["linear_key_heads"] * d["linear_head_dim"]
            value_dim = d["linear_value_heads"] * d["linear_head_dim"]
            attn = f"{prefix}.linear_attn"
            dense(f"{attn}.in_proj_qkv", 2 * key_dim + value_dim, hidden)
            dense(f"{attn}.in_proj_z", value_dim, hidden)
            dense(f"{attn}.in_proj_a", d["linear_value_heads"], hidden)
            dense(f"{attn}.in_proj_b", d["linear_value_heads"], hidden)
            dense(f"{attn}.out_proj", hidden, value_dim)
            tensors[f"{attn}.norm.weight"] = torch.ones(d["linear_head_dim"], dtype=torch.bfloat16)
            tensors[f"{attn}.conv1d.weight"] = rand(2 * key_dim + value_dim, 1, 4)
            tensors[f"{attn}.A_log"] = torch.log(torch.linspace(0.5, 2.0, d["linear_value_heads"])).to(torch.bfloat16)
            tensors[f"{attn}.dt_bias"] = torch.linspace(0.5, 1.5, d["linear_value_heads"]).to(torch.bfloat16)
        else:
            attn = f"{prefix}.self_attn"
            dense(f"{attn}.q_proj", 2 * d["heads"] * d["head_dim"], hidden)
            dense(f"{attn}.k_proj", d["kv_heads"] * d["head_dim"], hidden)
            dense(f"{attn}.v_proj", d["kv_heads"] * d["head_dim"], hidden)
            dense(f"{attn}.o_proj", hidden, d["heads"] * d["head_dim"])
            norm(f"{attn}.q_norm", d["head_dim"])
            norm(f"{attn}.k_norm", d["head_dim"])
            dense(f"{attn}.indexer.index_qk_proj", (d["indexer_heads"] + 1) * d["indexer_head_dim"], hidden)
            norm(f"{attn}.indexer.q_layernorm", d["indexer_head_dim"])
            norm(f"{attn}.indexer.k_layernorm", d["indexer_head_dim"])
        hyper(f"{prefix}.attn_hyper_connection")
        hyper(f"{prefix}.mlp_hyper_connection")
        dense(f"{prefix}.mlp.gate", d["num_experts"], hidden)
        dense(f"{prefix}.mlp.shared_expert_gate", 1, hidden)
        dense(f"{prefix}.mlp.shared_expert.gate_proj", d["moe_intermediate"], hidden)
        dense(f"{prefix}.mlp.shared_expert.up_proj", d["moe_intermediate"], hidden)
        dense(f"{prefix}.mlp.shared_expert.down_proj", hidden, d["moe_intermediate"])
        for expert_id in range(d["num_experts"]):
            expert = f"{prefix}.mlp.experts.{expert_id}"
            nvfp4_expert(f"{expert}.gate_proj", d["moe_intermediate"], hidden)
            nvfp4_expert(f"{expert}.up_proj", d["moe_intermediate"], hidden)
            nvfp4_expert(f"{expert}.down_proj", hidden, d["moe_intermediate"])

    ple = f"model.language_model.layers.{d['ple_layer']}.ple"
    dense(f"{ple}.key_proj", hc_hidden, hidden)
    dense(f"{ple}.value_proj", hidden, hidden)
    for name in ("norm_key", "norm_query", "norm_conv"):
        norm(f"{ple}.{name}", hc_hidden)
    tensors[f"{ple}.conv1d.weight"] = rand(hc_hidden, 1, 4)
    embedding = f"{ple}.ple_embedding"
    rows = d["ngram_rows"]
    tensors[f"{embedding}.layer_multipliers"] = torch.tensor([3, 5, 7], dtype=torch.int64)
    tensors[f"{embedding}.ngram_heads_vocab_sizes"] = torch.full((ngram_heads,), rows, dtype=torch.int64)
    tensors[f"{embedding}.ngram_heads_offsets"] = torch.arange(ngram_heads, dtype=torch.int64) * rows
    for shard_id in range(2):
        codes = torch.randint(0x30, 0x40, (ngram_heads * rows // 2, hidden // ngram_heads), generator=generator)
        tensors[f"{embedding}.ngram_embedding.shard_{shard_id}.weight"] = codes.to(torch.uint8).view(
            torch.float8_e4m3fn
        )
    tensors[f"{embedding}.ngram_embedding.weight_scale"] = torch.tensor([0.25], dtype=torch.bfloat16)

    dense("model.language_model.embed_tokens", d["vocab"], hidden)
    dense("lm_head", d["vocab"], hidden)
    hyper("model.language_model.hyper_connection_mixer", combine=False)
    for layer_id in range(num_layers):
        layer(layer_id)

    (path / "config.json").write_text(json.dumps(make_config(d)))
    save_file(tensors, path / "model.safetensors")
    write_tokenizer(path, d["vocab"])
    (path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "size": {"shortest_edge": 4096, "longest_edge": 65536},
                "patch_size": 16,
                "merge_size": 2,
                "temporal_patch_size": 2,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
            }
        )
    )
    return path


def write_tokenizer(path, vocab_size):
    vocab = {f"t{index}": index for index in range(vocab_size - 1)}
    vocab["<unk>"] = vocab_size - 1
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>").save_pretrained(path)
