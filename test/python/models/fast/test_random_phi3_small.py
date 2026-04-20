# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import importlib.util
import os
import textwrap
import unittest

import numpy as np
from model_builder_test_case import ModelBuilderTestCase, hide_stdout, requires_cuda, run_session_or_io_binding

PHI3_SMALL_MODEL_NAME = "microsoft/Phi-3-small-8k-instruct"

# Self-contained PyTorch implementation of Phi3SmallForCausalLM.
# Written to the model directory during tests so that
# AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
# can load it via the auto_map field in config.json.
_MODELING_CODE = textwrap.dedent("""\
    import math

    import torch
    import torch.nn.functional as F
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast


    class Phi3SmallConfig(PretrainedConfig):
        model_type = "phi3_small"

        def __init__(
            self,
            vocab_size=32064,
            hidden_size=512,
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=4,
            ffn_hidden_size=256,
            hidden_act="gegelu",
            max_position_embeddings=256,
            original_max_position_embeddings=256,
            rms_norm_eps=1e-5,
            rope_embedding_base=10000.0,
            mup_embedding_multiplier=1.0,
            mup_width_multiplier=1.0,
            mup_attn_multiplier=1.0,
            mup_use_scaling=False,
            gegelu_limit=20.0,
            dense_attention_every_n_layers=1,
            blocksparse_block_size=64,
            blocksparse_homo_head_pattern=True,
            blocksparse_num_local_blocks=2,
            blocksparse_triton_kernel_block_size=64,
            blocksparse_vert_stride=2,
            partial_rotary_factor=1.0,
            bos_token_id=1,
            eos_token_id=2,
            **kwargs,
        ):
            super().__init__(
                bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs
            )
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_key_value_heads = num_key_value_heads
            self.ffn_hidden_size = ffn_hidden_size
            self.hidden_act = hidden_act
            self.max_position_embeddings = max_position_embeddings
            self.original_max_position_embeddings = original_max_position_embeddings
            self.rms_norm_eps = rms_norm_eps
            self.rope_embedding_base = rope_embedding_base
            self.mup_embedding_multiplier = mup_embedding_multiplier
            self.mup_width_multiplier = mup_width_multiplier
            self.mup_attn_multiplier = mup_attn_multiplier
            self.mup_use_scaling = mup_use_scaling
            self.gegelu_limit = gegelu_limit
            self.dense_attention_every_n_layers = dense_attention_every_n_layers
            self.blocksparse_block_size = blocksparse_block_size
            self.blocksparse_homo_head_pattern = blocksparse_homo_head_pattern
            self.blocksparse_num_local_blocks = blocksparse_num_local_blocks
            self.blocksparse_triton_kernel_block_size = blocksparse_triton_kernel_block_size
            self.blocksparse_vert_stride = blocksparse_vert_stride
            self.partial_rotary_factor = partial_rotary_factor


    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)


    def _apply_rotary_emb(x, cos, sin):
        return (x * cos) + (_rotate_half(x) * sin)


    class _Phi3SmallLayerNorm(torch.nn.Module):
        \"\"\"Full LayerNorm with weight and bias (simple=False in the ONNX builder).\"\"\"

        def __init__(self, hidden_size, eps=1e-5):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
            self.eps = eps

        def forward(self, x):
            return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


    class _Phi3SmallMLP(torch.nn.Module):
        \"\"\"gegelu MLP matching Phi3SmallModel.make_mlp_proj in the ONNX builder.

        up_proj maps hidden -> 2*ffn_hidden.  The output is split into even
        (gate) and odd (linear) slices along the last dimension.  The gate
        slice goes through a clamp + QuickGelu path; the linear slice goes
        through a symmetric clamp + add-1 path.  Their product is fed into
        down_proj.
        \"\"\"

        def __init__(self, config):
            super().__init__()
            self.up_proj = torch.nn.Linear(
                config.hidden_size, 2 * config.ffn_hidden_size, bias=True
            )
            self.down_proj = torch.nn.Linear(
                config.ffn_hidden_size, config.hidden_size, bias=True
            )
            self.clamp_limit = config.gegelu_limit

        def forward(self, x):
            up = self.up_proj(x)
            gate = up[..., ::2]
            linear = up[..., 1::2]
            # Cast to float32, then clamp while preserving infinities
            gate_f32 = gate.float()
            linear_f32 = linear.float()
            gate_clipped = gate_f32.clamp(max=self.clamp_limit)
            gate_f32 = torch.where(gate_f32.isinf(), gate_f32, gate_clipped)
            linear_clipped = linear_f32.clamp(-self.clamp_limit, self.clamp_limit)
            linear_f32 = torch.where(linear_f32.isinf(), linear_f32, linear_clipped)
            # QuickGelu: sigmoid(1.702 * x) * x
            gate_f32 = gate_f32 * torch.sigmoid(1.702 * gate_f32)
            # Linear path: add 1
            linear_f32 = linear_f32 + 1.0
            result = (gate_f32 * linear_f32).to(self.down_proj.weight.dtype)
            return self.down_proj(result)


    class _Phi3SmallAttention(torch.nn.Module):
        \"\"\"Attention with combined QKV projection matching Phi3SmallModel.make_attention.

        The combined query_key_value weight has shape
        [num_kv_heads * (q_groups + 2) * head_size, hidden_size], laid out as
        [Q_groups | K | V] in the num_kv_heads dimension.  After splitting and
        reshaping we obtain separate q_proj, k_proj, v_proj tensors, exactly
        as the ONNX builder does in make_attention before calling
        super().make_attention().
        \"\"\"

        def __init__(self, config):
            super().__init__()
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = config.num_key_value_heads
            self.head_size = config.hidden_size // config.num_attention_heads
            self.q_groups = self.num_heads // self.num_kv_heads
            self.rope_theta = config.rope_embedding_base
            self.partial_rotary_factor = config.partial_rotary_factor

            qkv_out = self.num_kv_heads * (self.q_groups + 2) * self.head_size
            self.query_key_value = torch.nn.Linear(
                config.hidden_size, qkv_out, bias=True
            )
            self.o_proj = torch.nn.Linear(
                self.num_heads * self.head_size, config.hidden_size, bias=False
            )
            if config.mup_use_scaling:
                self.attn_scale = config.mup_attn_multiplier / self.head_size
            else:
                self.attn_scale = 1.0 / math.sqrt(self.head_size)

        def _make_rope(self, total_len, device, dtype):
            rotary_dim = int(self.partial_rotary_factor * self.head_size)
            if rotary_dim == 0:
                rotary_dim = self.head_size
            inv_freq = 1.0 / (
                self.rope_theta
                ** (
                    torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
                    / rotary_dim
                )
            )
            t = torch.arange(total_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(dtype), emb.sin().to(dtype)

        def forward(self, hidden, past_key_values=None):
            B, S, _ = hidden.shape
            past_len = past_key_values[0].shape[2] if past_key_values is not None else 0

            qkv = self.query_key_value(hidden)

            # The combined QKV weight uses an interleaved layout per kv-head:
            #   [kv_head_0: q_group_0 | q_group_1 | K | V |
            #    kv_head_1: q_group_0 | q_group_1 | K | V | ...]
            # This matches the view used in Phi3SmallModel.make_attention in
            # the ONNX builder (phi.py).  A simple [Q | K | V] slice would
            # produce incorrect results because Q, K and V values are
            # interleaved with each other.
            qkv_view = qkv.view(B, S, self.num_kv_heads, self.q_groups + 2, self.head_size)
            # Q: first q_groups entries per kv-head -> [B, S, num_kv, q_groups, D]
            q = (
                qkv_view[:, :, :, : self.q_groups, :]
                .contiguous()
                .view(B, S, self.num_heads, self.head_size)
                .transpose(1, 2)
            )
            # K: entry at index q_groups per kv-head -> [B, S, num_kv, D]
            k = qkv_view[:, :, :, self.q_groups, :].transpose(1, 2)
            # V: entry at index q_groups+1 per kv-head -> [B, S, num_kv, D]
            v = qkv_view[:, :, :, self.q_groups + 1, :].transpose(1, 2)

            # Apply rotary embeddings for the current positions only
            cos_all, sin_all = self._make_rope(past_len + S, hidden.device, hidden.dtype)
            cos = cos_all[past_len : past_len + S].unsqueeze(0).unsqueeze(0)
            sin = sin_all[past_len : past_len + S].unsqueeze(0).unsqueeze(0)

            rotary_dim = int(self.partial_rotary_factor * self.head_size)
            if rotary_dim == 0 or rotary_dim == self.head_size:
                q = _apply_rotary_emb(q, cos, sin)
                k = _apply_rotary_emb(k, cos, sin)
            else:
                q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
                k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
                q = torch.cat([_apply_rotary_emb(q_rot, cos, sin), q_pass], dim=-1)
                k = torch.cat([_apply_rotary_emb(k_rot, cos, sin), k_pass], dim=-1)

            # Concatenate with past KV cache
            if past_key_values is not None:
                k = torch.cat([past_key_values[0], k], dim=2)
                v = torch.cat([past_key_values[1], v], dim=2)

            # Store un-expanded K/V for the KV cache
            k_cache = k
            v_cache = v

            # GQA: repeat K/V for each query group
            if self.q_groups > 1:
                k = k.repeat_interleave(self.q_groups, dim=1)
                v = v.repeat_interleave(self.q_groups, dim=1)

            attn = F.scaled_dot_product_attention(
                q, k, v, scale=self.attn_scale, is_causal=(past_len == 0 and S > 1)
            )
            attn = attn.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_size)
            return self.o_proj(attn), (k_cache, v_cache)


    class Phi3SmallDecoderLayer(torch.nn.Module):
        \"\"\"Decoder layer whose class name ends with DecoderLayer for base.is_layer().\"\"\"

        def __init__(self, config):
            super().__init__()
            self.input_layernorm = _Phi3SmallLayerNorm(config.hidden_size, config.rms_norm_eps)
            self.post_attention_layernorm = _Phi3SmallLayerNorm(
                config.hidden_size, config.rms_norm_eps
            )
            self.self_attn = _Phi3SmallAttention(config)
            self.mlp = _Phi3SmallMLP(config)

        def forward(self, hidden, past_key_values=None):
            normed = self.input_layernorm(hidden)
            attn_out, new_pkv = self.self_attn(normed, past_key_values)
            hidden = hidden + attn_out
            normed2 = self.post_attention_layernorm(hidden)
            mlp_out = self.mlp(normed2)
            hidden = hidden + mlp_out
            return hidden, new_pkv


    class _Phi3SmallInnerModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
            self.layers = torch.nn.ModuleList(
                [Phi3SmallDecoderLayer(config) for _ in range(config.num_hidden_layers)]
            )
            self.norm = _Phi3SmallLayerNorm(config.hidden_size, config.rms_norm_eps)


    class Phi3SmallForCausalLM(PreTrainedModel):
        config_class = Phi3SmallConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = _Phi3SmallInnerModel(config)
            self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def forward(
            self, input_ids=None, past_key_values=None, attention_mask=None, **kwargs
        ):
            hidden = self.model.embed_tokens(input_ids).float()
            hidden = hidden * self.config.mup_embedding_multiplier
            new_past_kvs = []
            for i, layer in enumerate(self.model.layers):
                pkv = past_key_values[i] if past_key_values is not None else None
                hidden, new_pkv = layer(hidden, pkv)
                new_past_kvs.append(new_pkv)
            hidden = self.model.norm(hidden)
            logits = self.lm_head(hidden.to(self.lm_head.weight.dtype))
            logits = logits * (1.0 / self.config.mup_width_multiplier)
            return CausalLMOutputWithPast(logits=logits, past_key_values=new_past_kvs)
    """)


def _write_modeling_file(model_dir):
    """Write the custom modeling code to ``model_dir`` so that
    AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True) can
    load Phi3SmallForCausalLM via the auto_map in config.json."""
    path = os.path.join(model_dir, "modeling_phi3_small.py")
    with open(path, "w") as fh:
        fh.write(_MODELING_CODE)
    return path


def _load_phi3_small_class(modeling_path):
    """Dynamically import Phi3SmallForCausalLM from ``modeling_path``.

    The module is also registered in ``sys.modules`` so that transformers'
    ``_can_set_experts_implementation`` (which looks up the class module via
    ``sys.modules[cls.__module__]``) can find it.
    """
    import sys

    spec = importlib.util.spec_from_file_location("modeling_phi3_small", modeling_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modeling_phi3_small"] = mod
    spec.loader.exec_module(mod)
    return mod.Phi3SmallForCausalLM, mod.Phi3SmallConfig


class TestPhi3Small(ModelBuilderTestCase):
    def _make_config(self, num_hidden_layers=1):
        """Return a minimal Phi3SmallConfig dict for fast offline tests.

        Dimensions are chosen to be small while satisfying all shape
        constraints required by Phi3SmallModel:
          - hidden_size == num_attention_heads * head_size
          - dense_attention_every_n_layers == 1  (all layers use dense GQA)
          - max_position_embeddings == original_max_position_embeddings
            (non-LongRoPE path)
        """
        return dict(
            vocab_size=32064,
            hidden_size=128,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=8,
            num_key_value_heads=4,
            ffn_hidden_size=64,
            hidden_act="gegelu",
            max_position_embeddings=256,
            original_max_position_embeddings=256,
            rms_norm_eps=1e-5,
            rope_embedding_base=10000.0,
            mup_embedding_multiplier=10.0,
            mup_width_multiplier=8.0,
            mup_attn_multiplier=1.0,
            mup_use_scaling=True,
            gegelu_limit=20.0,
            dense_attention_every_n_layers=1,
            blocksparse_block_size=32,
            blocksparse_homo_head_pattern=True,
            blocksparse_num_local_blocks=2,
            blocksparse_triton_kernel_block_size=32,
            blocksparse_vert_stride=2,
            partial_rotary_factor=1.0,
            bos_token_id=1,
            eos_token_id=2,
        )

    def _prepare_model_dir(self, basename, num_hidden_layers=1):
        """Create the model directory, write the custom modeling file, save
        the config JSON with auto_map, instantiate the model, save its
        weights, and return (model_dir, output_dir, cache_dir, pt_model,
        config_obj) for use in the test."""
        import json

        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast

        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        # Write custom modeling code so that from_pretrained can load it
        modeling_path = _write_modeling_file(model_dir)

        # Load the class from the file we just wrote
        Phi3SmallForCausalLM, Phi3SmallConfig = _load_phi3_small_class(modeling_path)

        cfg_kwargs = self._make_config(num_hidden_layers=num_hidden_layers)
        config_obj = Phi3SmallConfig(**cfg_kwargs)

        # Instantiate and save the model weights + config (without auto_map yet).
        # safe_serialization=False avoids the "shared tensors" error that some
        # transformers versions raise when lm_head.weight is tied to
        # embed_tokens.weight via post_init()/tie_weights() but the config
        # has no _tied_weights_keys attribute set.
        model = Phi3SmallForCausalLM(config_obj)
        model.eval()
        model.save_pretrained(model_dir, safe_serialization=False)

        # model.save_pretrained overwrites config.json without auto_map, so we
        # must patch it in afterwards so that AutoConfig / AutoModelForCausalLM
        # can resolve the custom classes from the local modeling file.
        config_json_path = os.path.join(model_dir, "config.json")
        with open(config_json_path) as fh:
            cfg_dict = json.load(fh)
        cfg_dict["auto_map"] = {
            "AutoConfig": "modeling_phi3_small.Phi3SmallConfig",
            "AutoModelForCausalLM": "modeling_phi3_small.Phi3SmallForCausalLM",
        }
        with open(config_json_path, "w") as fh:
            json.dump(cfg_dict, fh, indent=2)

        # Minimal tokenizer (only BOS/EOS/UNK needed)
        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        return model_dir, output_dir, cache_dir, model, config_obj

    def common_fast_phi3_small_random_weights(self, precision, provider):
        import torch
        from models.builder import create_model

        num_hidden_layers = 1
        basename = f"test_discrepancies_phi3_small_{precision}_{provider}"

        torch.manual_seed(0)
        model_dir, output_dir, cache_dir, model, config_obj = self._prepare_model_dir(
            basename, num_hidden_layers=num_hidden_layers
        )
        model = model.to(provider)

        create_model(
            model_name=PHI3_SMALL_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        log_data = dict(
            precision=precision,
            model_id=PHI3_SMALL_MODEL_NAME,
            experiment="forward",
            provider=provider,
            test=basename,
            input_type="text",
            kind="fast",
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")

        batch_size = 1
        seq_len = 5
        head_size = config_obj.hidden_size // config_obj.num_attention_heads

        torch.manual_seed(1)
        input_ids = torch.randint(0, config_obj.vocab_size, (batch_size, seq_len)).to(provider)
        onnx_input_names = [i.name for i in sess.get_inputs()]

        prefill_results = None
        with self.subTest(step="prefill"):
            prefill_feed = {
                "input_ids": input_ids.cpu().numpy().astype(np.int64),
                "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
            }
            for i in range(num_hidden_layers):
                prefill_feed[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, config_obj.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
                prefill_feed[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, config_obj.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
            prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}

            prefill_results, ort_logits_np = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=prefill_feed,
                sess=sess,
                vocab_size=config_obj.vocab_size,
            )

            with torch.no_grad():
                pt_prefill = model(input_ids)

            np_prefill = pt_prefill.logits.detach().cpu().numpy()
            disc = self.get_numpy_discrepancy(np_prefill, ort_logits_np)
            self.log_results({"step": "prefill", **disc, **log_data})
            # gegelu and mup scaling introduce moderate numerical differences
            atol = {"fp16": 5e-2, "bf16": 5e-2, "fp32": 1e-2, "int4": 0.5}
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

            prefill_results, onnx_decode_logits = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=decode_feed,
                sess=sess,
                vocab_size=config_obj.vocab_size,
                results=prefill_results,
            )

            with torch.no_grad():
                pt_past_kv = pt_prefill.past_key_values
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(provider)
                pt_decode = model(next_token_tensor, past_key_values=pt_past_kv)
                pt_decode_logits = pt_decode.logits.detach().cpu().numpy()

            disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
            self.log_results({"step": "decode", **disc, **log_data})
            atol = {"fp16": 5e-2, "bf16": 5e-2, "fp32": 1e-2, "int4": 0.5}
            rtol = {"fp16": 10, "bf16": 10, "fp32": 1e-2, "int4": 10000}
            np.testing.assert_allclose(pt_decode_logits, onnx_decode_logits, atol=atol[precision], rtol=rtol[precision])

    def common_phi3_small_greedy_generation(self, precision, provider):
        import torch
        from models.builder import create_model

        num_hidden_layers = 1
        basename = f"test_generation_phi3_small_{precision}_{provider}"

        torch.manual_seed(42)
        model_dir, output_dir, cache_dir, model, config_obj = self._prepare_model_dir(
            basename, num_hidden_layers=num_hidden_layers
        )
        model = model.to(provider)

        create_model(
            model_name=PHI3_SMALL_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")

        input_names = {inp.name for inp in sess.get_inputs()}

        batch_size = 1
        head_size = config_obj.hidden_size // config_obj.num_attention_heads
        max_new_tokens = 10

        torch.manual_seed(0)
        prompt_ids = torch.randint(3, config_obj.vocab_size, (batch_size, 5)).to(provider)

        # Greedy generation with the PyTorch model (manual loop, since
        # PreTrainedModel.generate is not available in transformers >= 5.x).
        pt_past_kvs = None
        pt_tokens = prompt_ids[0].tolist()
        current_pt_ids = prompt_ids
        with torch.no_grad():
            for _ in range(max_new_tokens):
                pt_out = model(current_pt_ids, past_key_values=pt_past_kvs)
                next_tok = int(pt_out.logits[0, -1, :].argmax())
                pt_tokens.append(next_tok)
                pt_past_kvs = pt_out.past_key_values
                current_pt_ids = torch.tensor([[next_tok]], dtype=torch.long).to(provider)
                if next_tok == config_obj.eos_token_id:
                    break

        current_ids = prompt_ids.detach().cpu().numpy().astype(np.int64)

        past_kv = {}
        for i in range(num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, config_obj.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
            )
            past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, config_obj.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
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
                vocab_size=config_obj.vocab_size,
                results=results,
            )

            next_token = int(np.argmax(results["logits"][0, -1, :]))
            onnx_tokens.append(next_token)

            for i in range(num_hidden_layers):
                past_kv[f"past_key_values.{i}.key"] = results[f"present.{i}.key"]
                past_kv[f"past_key_values.{i}.value"] = results[f"present.{i}.value"]

            current_ids = np.array([[next_token]], dtype=np.int64)

            if next_token == config_obj.eos_token_id:
                break

        diff = self.first_token_diff(pt_tokens, onnx_tokens)
        diff.update(
            dict(
                precision=precision,
                model_id=PHI3_SMALL_MODEL_NAME,
                experiment="generate",
                provider=provider,
                test=basename,
                input_type="text",
                kind="fast",
            )
        )
        self.log_results(diff)
        if precision in ("fp16", "bf16"):
            pt_tokens = pt_tokens[:-5]
            onnx_tokens = onnx_tokens[:-5]
        self.assertEqual(pt_tokens, onnx_tokens)

    @hide_stdout()
    def test_fast_discrepancy_phi3_small_fp32_cpu(self):
        self.common_fast_phi3_small_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_phi3_small_fp16_cpu(self):
        self.common_fast_phi3_small_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_phi3_small_int4_cpu(self):
        self.common_fast_phi3_small_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_phi3_small_fp16_cuda(self):
        self.common_fast_phi3_small_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_phi3_small_bf16_cuda(self):
        self.common_fast_phi3_small_random_weights("bf16", "cuda")

    @hide_stdout()
    def test_phi3_small_fp32_cpu_greedy_generation(self):
        self.common_phi3_small_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_phi3_small_fp16_cpu_greedy_generation(self):
        self.common_phi3_small_greedy_generation("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_phi3_small_fp16_cuda_greedy_generation(self):
        self.common_phi3_small_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_phi3_small_bf16_cuda_greedy_generation(self):
        self.common_phi3_small_greedy_generation("bf16", "cuda")

    # ------------------------------------------------------------------
    # LongRoPE variant (Phi3SmallLongRoPEModel)
    # ------------------------------------------------------------------

    def common_fast_phi3_small_longrope_random_weights(self, precision, provider):
        """Build Phi3SmallLongRoPEModel ONNX and compare against PyTorch.

        Uses the direct builder instantiation approach (like the Phi4MM test)
        to bypass AutoConfig.from_pretrained LongRoPE-validation in newer
        transformers versions, which would fail because Phi3SmallConfig sets
        attributes after super().__init__().

        Identity rope_scaling factors ([1.0]*N, mscale=1.0) are used so that
        the ONNX LongRoPE rotary-embedding caches match the PyTorch standard
        RoPE exactly, enabling a numerical comparison of logits.
        """
        import torch
        from models.builder import set_io_dtype, set_onnx_dtype
        from models.builders.phi import Phi3SmallLongRoPEModel

        num_hidden_layers = 1
        basename = f"test_discrepancies_phi3_small_longrope_{precision}_{provider}"
        output_dir, cache_dir = self.get_dirs(basename)
        model_dir = self.get_model_dir(basename)

        # Write and load the custom modeling code so that Phi3SmallForCausalLM
        # and Phi3SmallConfig are available without network access.
        modeling_path = _write_modeling_file(model_dir)
        Phi3SmallForCausalLM, Phi3SmallConfig = _load_phi3_small_class(modeling_path)

        cfg_kwargs = self._make_config(num_hidden_layers=num_hidden_layers)
        config_obj = Phi3SmallConfig(**cfg_kwargs)

        # head_size = hidden_size // num_attention_heads = 128 // 8 = 16
        # factors_len = rotary_dim // 2 = head_size * partial_rotary_factor // 2 = 8
        head_size = config_obj.hidden_size // config_obj.num_attention_heads
        factors_len = head_size // 2

        # Set LongRoPE attributes directly to bypass transformers v5 validation
        # (validation runs inside PretrainedConfig.__init__ before subclass
        # attributes are set, so we set them as post-construction attributes).
        config_obj.architectures = ["Phi3SmallForCausalLM"]
        config_obj.max_position_embeddings = 512
        config_obj.rope_scaling = {
            "type": "longrope",
            "short_factor": [1.0] * factors_len,
            "long_factor": [1.0] * factors_len,
            "short_mscale": 1.0,
            "long_mscale": 1.0,
        }

        # Create PyTorch reference model (weights are not mutated by the builder).
        torch.manual_seed(0)
        pt_model = Phi3SmallForCausalLM(config_obj)
        pt_model.eval().to(provider)

        batch_size = 1
        seq_len = 5

        torch.manual_seed(1)
        input_ids = torch.randint(0, config_obj.vocab_size, (batch_size, seq_len)).to(provider)
        with torch.no_grad():
            pt_prefill = pt_model(input_ids)
        np_pt_prefill = pt_prefill.logits.detach().cpu().numpy()

        # Build ONNX using a fresh model instance (builder mutates model during
        # weight extraction in make_attention / make_mlp_proj).
        extra_options = {}
        io_dtype = set_io_dtype(precision, provider, extra_options)
        onnx_dtype = set_onnx_dtype(precision, extra_options)

        torch.manual_seed(0)
        model_for_onnx = Phi3SmallForCausalLM(config_obj)
        model_for_onnx.eval()

        class _Phi3SmallLongRoPEWithSyntheticWeights(Phi3SmallLongRoPEModel):
            def __init__(self, *args, synthetic_weights=None, **kwargs):
                super().__init__(*args, **kwargs)
                self._synthetic_weights = synthetic_weights

            def load_weights(self, _input_path):
                return self._synthetic_weights

        onnx_builder = _Phi3SmallLongRoPEWithSyntheticWeights(
            config_obj, io_dtype, onnx_dtype, provider, cache_dir, extra_options, synthetic_weights=model_for_onnx
        )
        onnx_builder.make_model(cache_dir)
        onnx_builder.save_model(output_dir)

        log_data = dict(
            precision=precision,
            model_id=PHI3_SMALL_MODEL_NAME,
            experiment="forward",
            provider=provider,
            test=basename,
            input_type="text",
            kind="fast",
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")
        onnx_input_names = [i.name for i in sess.get_inputs()]

        prefill_results = None
        with self.subTest(step="prefill"):
            prefill_feed = {
                "input_ids": input_ids.cpu().numpy().astype(np.int64),
                "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
            }
            for i in range(num_hidden_layers):
                prefill_feed[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, config_obj.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
                prefill_feed[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, config_obj.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
            prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}

            prefill_results, ort_logits_np = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=prefill_feed,
                sess=sess,
                vocab_size=config_obj.vocab_size,
            )

            disc = self.get_numpy_discrepancy(np_pt_prefill, ort_logits_np)
            self.log_results({"step": "prefill", **disc, **log_data})
            atol = {"fp16": 5e-2, "bf16": 5e-2, "fp32": 1e-2, "int4": 0.5}
            np.testing.assert_allclose(np_pt_prefill, ort_logits_np, atol=atol[precision], rtol=1e-3)

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

            prefill_results, onnx_decode_logits = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=decode_feed,
                sess=sess,
                vocab_size=config_obj.vocab_size,
                results=prefill_results,
            )

            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(provider)
            with torch.no_grad():
                pt_past_kv = pt_prefill.past_key_values
                pt_decode = pt_model(next_token_tensor, past_key_values=pt_past_kv)
                pt_decode_logits = pt_decode.logits.detach().cpu().numpy()

            disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
            self.log_results({"step": "decode", **disc, **log_data})
            atol = {"fp16": 5e-2, "bf16": 5e-2, "fp32": 1e-2, "int4": 0.5}
            rtol = {"fp16": 10, "bf16": 10, "fp32": 1e-2, "int4": 10000}
            np.testing.assert_allclose(pt_decode_logits, onnx_decode_logits, atol=atol[precision], rtol=rtol[precision])

    @hide_stdout()
    def test_fast_discrepancy_phi3_small_longrope_fp32_cpu(self):
        self.common_fast_phi3_small_longrope_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_phi3_small_longrope_fp16_cpu(self):
        self.common_fast_phi3_small_longrope_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_phi3_small_longrope_int4_cpu(self):
        self.common_fast_phi3_small_longrope_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_phi3_small_longrope_fp16_cuda(self):
        self.common_fast_phi3_small_longrope_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_phi3_small_longrope_bf16_cuda(self):
        self.common_fast_phi3_small_longrope_random_weights("bf16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
