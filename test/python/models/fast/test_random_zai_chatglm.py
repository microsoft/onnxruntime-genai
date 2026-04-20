# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os
import unittest

import numpy as np
from ext_test_case import ModelBuilderTestCase, hide_stdout, requires_cuda, run_session_or_io_binding

ZAI_CHATGLM_MODEL_NAME = "zai-org/chatglm3-6b"

# Minimal ChatGLM model source code saved to the model directory so that
# AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True) can load
# the model completely offline during fast tests.
_MODELING_CODE = """\
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from transformers import GenerationMixin
except ImportError:
    from transformers.generation.utils import GenerationMixin

class ChatGLMConfig(PretrainedConfig):
    model_type = "chatglm"

    def __init__(
        self,
        num_layers=1,
        seq_length=256,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=8,
        multi_query_group_num=2,
        vocab_size=2000,
        rms_norm_eps=1e-5,
        add_qkv_bias=True,
        **kwargs,
    ):
        self.num_layers = num_layers
        # transformers.GenerationMixin expects num_hidden_layers
        self.num_hidden_layers = num_layers
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_attention_heads = num_attention_heads
        self.multi_query_group_num = multi_query_group_num
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.add_qkv_bias = add_qkv_bias
        super().__init__(**kwargs)


def _build_rope_cache(seq_len, head_dim, base=10000.0):
    \"\"\"Build cos/sin caches matching the ONNX builder output for ChatGLM.

    ChatGLM uses partial_rotary_factor=0.5 so rotary_dim = head_dim // 2.
    The ONNX builder stores caches of shape [seq_len, rotary_dim // 2].
    \"\"\"
    rotary_dim = head_dim // 2  # partial_rotary_factor = 0.5
    # inv_freq has rotary_dim // 2 elements: 1/theta^(2k/rotary_dim)
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [seq_len, rotary_dim // 2]
    return freqs.cos(), freqs.sin()


def _apply_rotary_interleaved(x_rot, cos, sin):
    \"\"\"Apply interleaved RoPE to x_rot.

    x_rot : [B, S, H, rotary_dim]
    cos, sin : [S, rotary_dim // 2]  (half the number of frequencies)
    Returns tensor of same shape as x_rot.
    \"\"\"
    x1 = x_rot[..., 0::2]  # even indices  [B, S, H, rotary_dim // 2]
    x2 = x_rot[..., 1::2]  # odd indices   [B, S, H, rotary_dim // 2]
    c = cos.unsqueeze(0).unsqueeze(2)  # [1, S, 1, rotary_dim // 2]
    s = sin.unsqueeze(0).unsqueeze(2)  # [1, S, 1, rotary_dim // 2]
    out = torch.empty_like(x_rot)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def _apply_partial_rope(q, k, cos_cache, sin_cache, rotary_dim, past_len):
    \"\"\"Apply partial interleaved RoPE to q and k at the correct positions.

    q, k : [B, S, H, head_dim]
    cos_cache, sin_cache : [max_seq_len, rotary_dim // 2]
    Positions used: past_len .. past_len + S - 1
    \"\"\"
    S = q.shape[1]
    cos = cos_cache[past_len : past_len + S]  # [S, rotary_dim // 2]
    sin = sin_cache[past_len : past_len + S]

    q_rot = _apply_rotary_interleaved(q[..., :rotary_dim].clone(), cos, sin)
    q = torch.cat([q_rot, q[..., rotary_dim:]], dim=-1)

    k_rot = _apply_rotary_interleaved(k[..., :rotary_dim].clone(), cos, sin)
    k = torch.cat([k_rot, k[..., rotary_dim:]], dim=-1)
    return q, k


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.multi_query_group_num
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_dim = self.head_dim // 2  # partial_rotary_factor = 0.5

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        self.query_key_value = nn.Linear(
            config.hidden_size, q_size + 2 * kv_size, bias=config.add_qkv_bias
        )
        self.dense = nn.Linear(q_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, cos_cache, sin_cache, past_key_value=None):
        B, S, _ = hidden_states.shape
        past_len = 0 if past_key_value is None else past_key_value[0].shape[1]

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        qkv = self.query_key_value(hidden_states)
        q = qkv[..., :q_size].view(B, S, self.num_heads, self.head_dim)
        k = qkv[..., q_size : q_size + kv_size].view(B, S, self.num_kv_heads, self.head_dim)
        v = qkv[..., q_size + kv_size :].view(B, S, self.num_kv_heads, self.head_dim)

        q, k = _apply_partial_rope(q, k, cos_cache, sin_cache, self.rotary_dim, past_len)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        present = (k, v)

        groups = self.num_heads // self.num_kv_heads
        k_exp = k.repeat_interleave(groups, dim=2)
        v_exp = v.repeat_interleave(groups, dim=2)

        # [B, H, S_q, head_dim] and [B, H, S_kv, head_dim]
        q_t = q.transpose(1, 2)
        k_t = k_exp.transpose(1, 2)
        v_t = v_exp.transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q_t, k_t.transpose(-2, -1)) / scale  # [B, H, S_q, S_kv]

        S_q = q_t.shape[2]
        S_kv = k_t.shape[2]
        if S_q > 1:
            # Causal mask: only attend to positions <= current position
            mask = torch.ones(S_q, S_kv, dtype=torch.bool, device=hidden_states.device)
            mask = torch.tril(mask)
            # Past tokens (columns before S_kv - S_q) are always visible
            attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn.float(), dim=-1).to(q_t.dtype)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        out = torch.matmul(attn_weights, v_t)  # [B, H, S_q, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        return self.dense(out), present


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # dense_h_to_4h: packed gate+up projection (split by make_mlp_unpacked_regular)
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size, config.ffn_hidden_size * 2, bias=False
        )
        # dense_4h_to_h: down projection (mapped to down_proj by ChatGLMModel.make_mlp)
        self.dense_4h_to_h = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        gate_up = self.dense_h_to_4h(x)
        half = gate_up.shape[-1] // 2
        h = F.silu(gate_up[..., :half]) * gate_up[..., half:]
        return self.dense_4h_to_h(h)


class GLMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Use RMSNorm to match the ONNX builder's SimplifiedLayerNormalization
        self.input_layernorm = torch.nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attention = SelfAttention(config)
        self.post_attention_layernorm = torch.nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = MLP(config)

    def forward(self, hidden_states, cos_cache, sin_cache, past_key_value=None):
        residual = hidden_states
        attn_out, present = self.self_attention(
            self.input_layernorm(hidden_states), cos_cache, sin_cache, past_key_value
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class GLMTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([GLMBlock(config) for _ in range(config.num_layers)])
        # Use RMSNorm to match the ONNX builder's SimplifiedLayerNormalization
        self.final_layernorm = torch.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Embedding2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)


class ChatGLMTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding2D(config)
        self.encoder = GLMTransformer(config)


class ChatGLMForConditionalGeneration(GenerationMixin, PreTrainedModel):
    config_class = ChatGLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = ChatGLMTransformer(config)
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        head_dim = config.hidden_size // config.num_attention_heads
        cos_cache, sin_cache = _build_rope_cache(config.seq_length, head_dim)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)
        # Required by transformers >= 5.x (post_init sets all_tied_weights_keys, etc.)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=True,
        **kwargs,
    ):
        if inputs_embeds is None:
            hidden_states = self.transformer.embedding.word_embeddings(input_ids)
        else:
            hidden_states = inputs_embeds

        new_pkv = []
        for i, layer in enumerate(self.transformer.encoder.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, present = layer(
                hidden_states, self.cos_cache, self.sin_cache, past_kv
            )
            if use_cache:
                new_pkv.append(present)

        hidden_states = self.transformer.encoder.final_layernorm(hidden_states)
        logits = self.output_layer(hidden_states)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=tuple(new_pkv) if use_cache else None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
"""


def _save_mini_zai_chatglm(model_dir, num_layers=1):
    """Save a tiny ChatGLM model with random weights to *model_dir*.

    Uses the ``ChatGLMModel`` architecture string (the standard HuggingFace
    variant of ChatGLM-3, as opposed to the quantized ``ChatGLMForConditionalGeneration``
    variant used by THUDM/chatglm3-6b).

    Writes the model source file, a compatible config.json (with auto_map),
    a safetensors checkpoint and a minimal tokenizer so that every downstream
    call that hits the directory (AutoConfig, AutoModel, AutoTokenizer) works
    fully offline.
    """
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    # ------------------------------------------------------------------
    # 1. Write model source to model_dir so trust_remote_code can load it
    # ------------------------------------------------------------------
    modeling_path = os.path.join(model_dir, "modeling_chatglm.py")
    with open(modeling_path, "w") as f:
        f.write(_MODELING_CODE)

    # ------------------------------------------------------------------
    # 2. Instantiate config + model by importing the file we just wrote
    # ------------------------------------------------------------------
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_zai_chatglm_mini", modeling_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_zai_chatglm_mini"] = mod
    spec.loader.exec_module(mod)
    ChatGLMConfig = mod.ChatGLMConfig
    ChatGLMForConditionalGeneration = mod.ChatGLMForConditionalGeneration

    # Use "ChatGLMModel" architecture – the standard HuggingFace identifier
    # for ChatGLM-3 (builder.py dispatches both "ChatGLMForConditionalGeneration"
    # and "ChatGLMModel" to the same ChatGLMModel builder).
    config = ChatGLMConfig(
        architectures=["ChatGLMModel"],
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        num_layers=num_layers,
        seq_length=256,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=8,
        multi_query_group_num=2,
        vocab_size=2000,
        rms_norm_eps=1e-5,
        add_qkv_bias=True,
    )

    torch.manual_seed(42)
    model = ChatGLMForConditionalGeneration(config)
    model.eval()

    # ------------------------------------------------------------------
    # 3. Save weights + config.json via save_pretrained
    # ------------------------------------------------------------------
    model.save_pretrained(model_dir)

    # ------------------------------------------------------------------
    # 4. Patch config.json with auto_map so AutoConfig / AutoModel work
    # ------------------------------------------------------------------
    config_json_path = os.path.join(model_dir, "config.json")
    with open(config_json_path) as f:
        cfg_dict = json.load(f)
    cfg_dict["auto_map"] = {
        "AutoConfig": "modeling_chatglm.ChatGLMConfig",
        "AutoModelForCausalLM": "modeling_chatglm.ChatGLMForConditionalGeneration",
    }
    with open(config_json_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # ------------------------------------------------------------------
    # 5. Save a minimal tokenizer
    # ------------------------------------------------------------------
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(model_dir)

    return model, config


class TestZaiChatGLM(ModelBuilderTestCase):
    def common_fast_zai_chatglm_random_weights(self, precision, provider):
        """Prefill + single-step decode: compare PyTorch logits vs ONNX logits."""
        import torch
        from models.builder import create_model

        num_hidden_layers = 1
        basename = f"test_discrepancy_zai_chatglm_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        model, config = _save_mini_zai_chatglm(model_dir, num_layers=num_hidden_layers)
        model.eval()

        create_model(
            model_name=ZAI_CHATGLM_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
        )

        log_data = dict(
            precision=precision,
            model_id=ZAI_CHATGLM_MODEL_NAME,
            experiment="forward",
            provider=provider,
            test=basename,
            input_type="text",
            kind="fast",
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self.check_ort(onnx_path)

        batch_size = 1
        seq_len = 5
        head_size = config.hidden_size // config.num_attention_heads

        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        onnx_input_names = [i.name for i in sess.get_inputs()]

        prefill_results = None
        with self.subTest(step="prefill"):
            prefill_feed = {
                "input_ids": input_ids.numpy().astype(np.int64),
                "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
            }
            for i in range(num_hidden_layers):
                prefill_feed[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, config.multi_query_group_num, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
                prefill_feed[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, config.multi_query_group_num, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
            prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}

            prefill_results, ort_logits_np = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=prefill_feed,
                sess=sess,
                vocab_size=config.vocab_size,
            )

            with torch.no_grad():
                pt_prefill = model(input_ids)

            np_prefill = pt_prefill.logits.detach().cpu().numpy()
            disc = self.get_numpy_discrepancy(np_prefill, ort_logits_np)
            self.log_results({"step": "prefill", **disc, **log_data})
            atol = {"fp16": 3e-2, "bf16": 3e-2, "fp32": 1e-2, "int4": 0.5}
            self.assertEqual(np_prefill.shape, ort_logits_np.shape)
            np.testing.assert_allclose(np_prefill, ort_logits_np, atol=atol[precision], rtol=1e-3)

        with self.subTest(step="decode"):
            if prefill_results is None:
                raise unittest.SkipTest("prefill failed")
            next_token = int(np.argmax(prefill_results["logits"][0, -1, :]))

            decode_feed = {
                "input_ids": np.array([[next_token]], dtype=np.int64),
                "attention_mask": np.ones((batch_size, seq_len + 1), dtype=np.int64),
                "position_ids": np.array([[seq_len]], dtype=np.int64),
            }
            for i in range(num_hidden_layers):
                decode_feed[f"past_key_values.{i}.key"] = prefill_results[f"present.{i}.key"]
                decode_feed[f"past_key_values.{i}.value"] = prefill_results[f"present.{i}.value"]
            decode_feed = {k: v for k, v in decode_feed.items() if k in onnx_input_names}

            _, onnx_decode_logits = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=decode_feed,
                sess=sess,
                vocab_size=config.vocab_size,
                results=prefill_results,
            )

            with torch.no_grad():
                pt_past_kv = pt_prefill.past_key_values
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long)
                pt_decode = model(next_token_tensor, past_key_values=pt_past_kv)
                pt_decode_logits = pt_decode.logits.detach().cpu().numpy()

            disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
            self.log_results({"step": "decode", **disc, **log_data})
            atol = {"fp16": 3e-2, "bf16": 3e-2, "fp32": 1e-2, "int4": 0.5}
            rtol = {"fp16": 10, "bf16": 10, "fp32": 1e-2, "int4": 10000}
            np.testing.assert_allclose(pt_decode_logits, onnx_decode_logits, atol=atol[precision], rtol=rtol[precision])

    def common_zai_chatglm_greedy_generation(self, precision, provider):
        """End-to-end greedy generation: compare PyTorch token sequence vs ONNX."""
        import torch
        from models.builder import create_model

        num_hidden_layers = 1
        basename = f"test_generation_zai_chatglm_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model, config = _save_mini_zai_chatglm(model_dir, num_layers=num_hidden_layers)
        model.eval()

        create_model(
            model_name=ZAI_CHATGLM_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self.check_ort(onnx_path)

        input_names = {inp.name for inp in sess.get_inputs()}

        batch_size = 1
        head_size = config.hidden_size // config.num_attention_heads
        max_new_tokens = 10

        torch.manual_seed(0)
        prompt_ids = torch.randint(3, config.vocab_size, (batch_size, 5))

        # Manual greedy generation with PyTorch (avoids transformers cache API changes)
        with torch.no_grad():
            pt_current_ids = prompt_ids.clone()
            pt_past_key_values = None
            pt_tokens = prompt_ids[0].tolist()
            for _ in range(max_new_tokens):
                pt_out = model(pt_current_ids, past_key_values=pt_past_key_values)
                next_tok = int(torch.argmax(pt_out.logits[0, -1, :]).item())
                pt_tokens.append(next_tok)
                pt_past_key_values = pt_out.past_key_values
                pt_current_ids = torch.tensor([[next_tok]], dtype=torch.long)
                if next_tok == config.eos_token_id:
                    break

        current_ids = prompt_ids.numpy().astype(np.int64)

        past_kv = {}
        for i in range(num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, config.multi_query_group_num, 0, head_size), dtype=self.get_input_np_dtype(precision)
            )
            past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, config.multi_query_group_num, 0, head_size), dtype=self.get_input_np_dtype(precision)
            )

        onnx_tokens = current_ids[0].tolist()
        results = None
        for _ in range(max_new_tokens):
            past_len = past_kv["past_key_values.0.key"].shape[2]
            cur_len = current_ids.shape[1]

            feed = {
                "input_ids": current_ids,
                "attention_mask": np.ones((batch_size, past_len + cur_len), dtype=np.int64),
                "position_ids": np.arange(past_len, past_len + cur_len, dtype=np.int64).reshape(batch_size, cur_len),
            }
            for i in range(num_hidden_layers):
                feed[f"past_key_values.{i}.key"] = past_kv[f"past_key_values.{i}.key"]
                feed[f"past_key_values.{i}.value"] = past_kv[f"past_key_values.{i}.value"]
            feed = {k: v for k, v in feed.items() if k in input_names}

            results, _ = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=feed,
                sess=sess,
                vocab_size=config.vocab_size,
                results=results,
            )

            next_token = int(np.argmax(results["logits"][0, -1, :]))
            onnx_tokens.append(next_token)

            for i in range(num_hidden_layers):
                past_kv[f"past_key_values.{i}.key"] = results[f"present.{i}.key"]
                past_kv[f"past_key_values.{i}.value"] = results[f"present.{i}.value"]

            current_ids = np.array([[next_token]], dtype=np.int64)

            if next_token == config.eos_token_id:
                break

        diff = self.first_token_diff(pt_tokens, onnx_tokens)
        diff.update(
            dict(
                precision=precision,
                model_id=ZAI_CHATGLM_MODEL_NAME,
                experiment="generate",
                provider=provider,
                test=basename,
                input_type="text",
                kind="fast",
            )
        )
        self.log_results(diff)
        if precision in ("fp16", "bf16"):
            pt_tokens = pt_tokens[:5]
            onnx_tokens = onnx_tokens[:5]
        self.assertEqual(pt_tokens, onnx_tokens)

    @hide_stdout()
    def test_fast_discrepancy_zai_chatglm_fp32_cpu(self):
        self.common_fast_zai_chatglm_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_zai_chatglm_fp16_cpu(self):
        self.common_fast_zai_chatglm_random_weights("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_zai_chatglm_fp16_cuda(self):
        self.common_fast_zai_chatglm_random_weights("fp16", "cuda")

    @hide_stdout()
    def test_zai_chatglm_fp32_cpu_greedy_generation(self):
        self.common_zai_chatglm_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_zai_chatglm_fp16_cpu_greedy_generation(self):
        self.common_zai_chatglm_greedy_generation("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_zai_chatglm_fp16_cuda_greedy_generation(self):
        self.common_zai_chatglm_greedy_generation("fp16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
