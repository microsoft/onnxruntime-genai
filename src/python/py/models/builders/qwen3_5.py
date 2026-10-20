import copy
import json
import os

import onnx_ir as ir
from builder_config import serialize_genai_config

from .base import Model
from .expansions.qwen3_5 import Qwen35, Qwen35MoE
from .expansions.mtp import Qwen35MTP
from .mtp import MTPModel


class Qwen35TextModel(Qwen35, Model):
    def validate_gated_delta_net_options(self, state_window, ep):
        if self.use_paged_attention and ep not in {"cuda", "webgpu"}:
            raise ValueError("Paged GatedDeltaNet exports require the CUDA or WebGPU execution provider")
        if ep == "webgpu":
            supported_dtypes = (
                (ir.DataType.FLOAT16,) if self.use_paged_attention else (ir.DataType.FLOAT16, ir.DataType.FLOAT)
            )
            if self.io_dtype not in supported_dtypes:
                raise ValueError(
                    "GatedDeltaNet exports on WebGPU require FP16 for paged attention "
                    "or FP16/FP32 for non-paged attention"
                )
        if state_window == 1:
            raise ValueError("GatedDeltaNet state_window must be 0 or at least 2")
        if state_window > 9:
            raise ValueError("GatedDeltaNet state_window must not exceed 9 (state_update_capacity <= 8)")
        if state_window and self.context_length_attrs["state_update_capacity"] not in (0, state_window - 1):
            raise ValueError("state_update_capacity must equal state_window - 1 when both are specified")
        if ep not in {"cuda", "webgpu"} and self.context_length_attrs["state_update_capacity"]:
            raise ValueError("Compact recurrent state capture requires the CUDA or WebGPU execution provider")

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        if "linear_attn_op" in extra_options:
            raise ValueError("linear_attn_op is selected automatically by execution provider; remove this option. Dense CPU/DML/TRT-RTX exports retain LinearAttention.")
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.configure_gated_delta_net_io()

        # OffsetRMSNorm: Qwen3.5 uses (1 + weight) * RMSNorm(x).
        # Pre-bake the +1 into the weight initializer so the base class's
        # SkipSimplifiedLayerNormalization can be used directly.
        self.layernorm_attrs["add_offset"] = 1

        # Qwen-3.5 uses interleaved, partial MRoPE for both text and multimodal inputs.
        self.rope_attrs["mrope_layout"] = 1
        self.rope_attrs["cast"]["use_fp32"] = True
        self.rope_attrs["cast"]["root_input"] = True
        self.rope_attrs["cast"]["output_0"] = True

    def configure_gated_delta_net_io(self):
        """Declare every linear-attention graph binding, so emitters never respell a name or shape."""
        linear_layers = [
            layer_id for layer_id, layer_type in enumerate(self.layer_types) if layer_type == "linear_attention"
        ]
        if not linear_layers:
            # Without a linear-attention layer nothing below is emitted, so the capacity has no bindings.
            self.context_length_attrs["state_update_capacity"] = 0
            return

        self.validate_gated_delta_net_options(
            self.context_length_attrs["state_window"],
            self.ep,
        )
        if self.context_length_attrs["state_window"] and self.ep in {"cuda", "webgpu"}:
            self.context_length_attrs["state_update_capacity"] = self.context_length_attrs["state_window"] - 1

        if self.use_paged_attention:
            conv_shape = ["batch_size", self.linear_conv_dim, self.linear_conv_kernel_dim - 1]
            self.input_shapes["past.conv"] = conv_shape
            self.output_shapes["present.conv"] = conv_shape

        recurrent_shape = [
            "batch_size",
            self.linear_num_value_heads,
            self.linear_value_head_dim,
            self.linear_key_head_dim,
        ]
        if self.ep in {"cuda", "webgpu"}:
            self.input_types["past.recurrent"] = ir.DataType.FLOAT
            self.input_shapes["past.recurrent"] = recurrent_shape
            self.output_types["present.recurrent"] = ir.DataType.FLOAT
            self.output_shapes["present.recurrent"] = recurrent_shape

        capacity = self.context_length_attrs["state_update_capacity"]
        if not capacity:
            return

        self.input_names["state_update.capture_count"] = "state_update_capture_count"
        self.input_types["state_update.capture_count"] = ir.DataType.INT32
        self.input_shapes["state_update.capture_count"] = ["batch_size"]
        self.input_names["state_update.active"] = "state_update_active"
        self.input_types["state_update.active"] = ir.DataType.INT32
        self.input_shapes["state_update.active"] = [1]

        if self.use_paged_attention:
            self.output_names["state_update.conv_value"] = {
                layer_id: f"state_update.{layer_id}.conv_value" for layer_id in linear_layers
            }
            self.output_types["state_update.conv_value"] = self.io_dtype
            self.output_shapes["state_update.conv_value"] = ["batch_size", capacity, self.linear_conv_dim]

        # One capsule packs each captured token's decay gates, key row, and value row back to back.
        capsule_width = capacity * (
            self.linear_num_value_heads
            + self.linear_num_key_heads * self.linear_key_head_dim
            + self.linear_num_value_heads * self.linear_value_head_dim
        )
        self.output_names["state_update.recurrent_capsule"] = {
            layer_id: f"state_update.{layer_id}.recurrent_capsule" for layer_id in linear_layers
        }
        self.output_types["state_update.recurrent_capsule"] = ir.DataType.FLOAT
        self.output_shapes["state_update.recurrent_capsule"] = ["batch_size", capsule_width]

    def make_inputs_and_outputs(self):
        # Qwen-3.5 uses 3D position_ids
        self.input_shapes["position_ids"] = (
            [3, "num_tokens"] if self.use_paged_attention else [3, "batch_size", "sequence_length"]
        )
        super().make_inputs_and_outputs()

    def is_packed_attn_supported(self):
        return False

    def make_attention_init(self, config):
        # Set QK norm before the base class selects packed or paged attention paths.
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init(config)
        if self.use_paged_attention:
            self.attention_attrs["use_packed_matmul"] = self.is_packed_matmul_supported()

    def is_qkv_projection_packable(self, layer_id, attention):
        v_name = f"/model/layers.{layer_id}/attn/v_proj/MatMul"
        return v_name not in self.int4_customized_weight_config and super().is_qkv_projection_packable(
            layer_id, attention
        )

    def is_fused_rope_supported(self):
        # Qwen-3.5 applies MRoPE manually before attention, not fused in the op
        return False

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        """Dispatch to full attention or GatedDeltaNet based on layer type."""
        if self.layer_types[layer_id] == "linear_attention":
            self.make_qwen_gated_delta_net(layer_id, attention, root_input)
        else:
            super().make_attention(layer_id, attention, root_input, **kwargs)

    def get_attn_module(self, layer_id, layer):
        return layer.linear_attn if self.layer_types[layer_id] == "linear_attention" else layer.self_attn


    def make_decoder_state_groups(self, inputs, outputs):
        if not self.use_paged_attention:
            return []

        full_attention_layers = [
            layer_id
            for layer_id, layer_type in enumerate(self.layer_types)
            if layer_type in {"full_attention", "sliding_attention"}
        ]
        conv_layers = [
            layer_id
            for layer_id, layer_type in enumerate(self.layer_types)
            if layer_type in {"conv", "linear_attention"}
        ]
        linear_attention_layers = [
            layer_id for layer_id, layer_type in enumerate(self.layer_types) if layer_type == "linear_attention"
        ]
        state_groups = []
        if full_attention_layers:
            state_groups.append(self.make_paged_key_value_state_group(full_attention_layers))
        if not linear_attention_layers:
            return state_groups

        state_update_capacity = (
            self.context_length_attrs["state_update_capacity"] if "state_update_capture_count" in inputs else 0
        )

        for state_name, layer_ids in (
            ("conv", conv_layers),
            ("recurrent", linear_attention_layers),
        ):
            group = {
                "kind": f"fixed_{state_name}",
                "layer_ids": layer_ids,
            }
            if state_update_capacity:
                state_update = {
                    "capacity": state_update_capacity,
                }
                if state_name == "recurrent":
                    state_update["key_head_count"] = self.linear_num_key_heads
                group["state_update"] = state_update
            state_groups.append(group)

        return state_groups


class Qwen35MoETextModel(Qwen35MoE, Qwen35TextModel):
    """Qwen3.5 MoE hybrid model builder.

    Extends ``Qwen35TextModel`` with Mixture-of-Experts MLP layers.
    Each decoder layer replaces the dense MLP with:
    - A router that selects top-k experts from ``num_experts`` candidates
    - Packed routed expert weights (gate_up_proj + down_proj)
    - A shared expert (always-active) with its own gating signal

    The attention side (GatedDeltaNet linear + gated full) is inherited
    unchanged from the parent class.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # MoE attributes specific to Qwen-3.5 MoE
        self.moe_attrs["activation_type"] = "swiglu"
        self.moe_attrs["swiglu_fusion"] = 1
        self.moe_attrs["normalize_routing_weights"] = True

        self.moe_intermediate_size = getattr(config, "moe_intermediate_size", 512)
        self.shared_expert_intermediate_size = getattr(
            config, "shared_expert_intermediate_size", self.moe_intermediate_size
        )

    def get_moe_module(self, layer_id, layer):
        return layer.mlp


class Qwen35MoEModel(MTPModel):
    """Composite Qwen3.5 MoE builder for the decoder and optional MTP graph."""

    # Extra options naming a block drafter. The Engine drives one drafter per model, so any of
    # these supersedes the MTP head rather than shipping beside it.
    block_drafter_options = ("dflash2_path", "dspark_path")

    def requested_block_drafter(self, extra_options):
        requested = [name for name in self.block_drafter_options if extra_options.get(name)]
        if len(requested) > 1:
            raise ValueError("Block drafter options are mutually exclusive: " + ", ".join(requested) + ".")
        return requested[0] if requested else None

    def get_decoder_model_class(self):
        return Qwen35MoETextModel

    def get_mtp_model_class(self):
        return Qwen35MTPModel

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__()
        decoder_options = self.make_mtp_init(config, extra_options)
        self.decoder = self.get_decoder_model_class()(
            copy.deepcopy(config), io_dtype, onnx_dtype, ep, cache_dir, decoder_options
        )
        self.mtp = None
        if self.mtp_attrs["build"]:
            self.make_mtp_model(config, io_dtype, onnx_dtype, ep, cache_dir, decoder_options)

        self.dflash2 = None
        self.dflash2_shared_initializers = []
        self.make_dflash2_init(io_dtype, extra_options)

        self.dspark = None
        self.dspark_shared_initializers = []
        self.make_dspark_init(io_dtype, extra_options)

        self.vocab_size = self.decoder.vocab_size
        self.hf_token = self.decoder.hf_token
        self.hf_remote = self.decoder.hf_remote
        self.context_length = self.decoder.context_length
        self.exclude_embeds = self.decoder.exclude_embeds
        self.model_type = self.decoder.model_type

    def make_mtp_init(self, config, extra_options):
        decoder_options = super().make_mtp_init(config, extra_options)
        text_config = getattr(config, "text_config", config)
        num_mtp_layers = getattr(text_config, "mtp_num_hidden_layers", None)
        if num_mtp_layers is None:
            num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 0)
        self.mtp_attrs["build"] = (num_mtp_layers or 0) > 0
        self.mtp_attrs["shared_initializer_names"] = {"model.embed_tokens.weight"}
        # `_` catches the quantized table's `weight_Q4` / `weight_scales` pair.
        self.mtp_attrs["shared_initializer_prefixes"] = ("lm_head.MatMul.", "model.embed_tokens.weight_")

        block_drafter = self.requested_block_drafter(extra_options)
        if self.mtp_attrs["build"] and block_drafter:
            print(f"Skipping the MTP head: {block_drafter} supersedes it.")
            self.mtp_attrs["build"] = False

        if self.mtp_attrs["build"] and extra_options.get("exclude_mtp", False):
            print("Skipping the MTP head: exclude_mtp is set.")
            self.mtp_attrs["build"] = False

        quant_method = getattr(config, "quantization_config", {}).get("quant_method", "")
        if self.mtp_attrs["build"] and quant_method == "quark":
            print("Skipping the MTP head: Quark pre-quantized checkpoints do not include MTP weights.")
            self.mtp_attrs["build"] = False

        if not self.mtp_attrs["build"]:
            return decoder_options

        incompatible_options = [
            option for option in ("exclude_lm_head", "prune_lm_head") if extra_options.get(option, False)
        ]
        if incompatible_options:
            raise ValueError("Qwen3.5 MTP export cannot be combined with " + ", ".join(incompatible_options) + ".")
        decoder_options["include_hidden_states"] = True
        return decoder_options

    def make_mtp_model(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.mtp_attrs["io_dtype"] = io_dtype
        self.mtp_attrs["onnx_dtype"] = onnx_dtype
        self.mtp_attrs["extra_options"] = copy.deepcopy(extra_options)
        self.resolve_mtp_model_config(extra_options)

        mtp_options = self.mtp_attrs["extra_options"]
        self.drop_unusable_mtp_kv_scales(mtp_options)
        mtp_options["exclude_embeds"] = False
        mtp_options["filename"] = "mtp.onnx"
        mtp_options.pop("include_hidden_states", None)
        mtp_options.pop("exclude_lm_head", None)
        # The head is one layer deep and drafts for itself, so it never taps the target's
        # residual streams; inheriting the target's tap set would fail its layer-range check.
        mtp_options.pop("aux_hidden_state_layers", None)
        # A block drafter reads the target's aux hidden states, so it is never nested in the head.
        mtp_options.pop("dflash2_path", None)
        mtp_options.pop("dflash2_num_draft_tokens", None)
        mtp_options.pop("dspark_path", None)
        mtp_options.pop("dspark_num_draft_tokens", None)
        mtp_options.pop("dspark_top_k", None)
        self.mtp = self.get_mtp_model_class()(
            copy.deepcopy(config),
            self.mtp_attrs["io_dtype"],
            self.mtp_attrs["onnx_dtype"],
            ep,
            cache_dir,
            mtp_options,
        )

    def drop_unusable_mtp_kv_scales(self, mtp_options):
        scale_file = mtp_options.get("kv_cache_scale_file")
        if not scale_file:
            return
        try:
            with open(scale_file, encoding="utf-8") as handle:
                has_mtp_section = "mtp" in json.load(handle)
        except (OSError, ValueError):
            return
        if not has_mtp_section:
            mtp_options.pop("kv_cache_quant_scheme", None)
            mtp_options.pop("kv_cache_scale_file", None)

    def make_model(self, input_path):
        self.decoder.make_model(input_path)
        if self.mtp is not None:
            print("Building MTP (multi-token prediction) head -> mtp.onnx")
            self.mtp.make_model(input_path)
        self.make_dflash2_model(input_path)
        self.make_dspark_model(input_path)

    def save_model(self, output_dir):
        self.decoder.save_model(output_dir)
        if self.mtp is not None:
            self.mtp.save_model(output_dir)
            self.mtp_attrs["shared_initializers"] = self.share_initializers(
                output_dir, self.decoder.filename, self.mtp.filename
            )
        self.save_dflash2_model(output_dir)
        self.save_dspark_model(output_dir)

    def make_genai_config(self, config, extra_kwargs, out_dir):
        self.decoder.model_type = self.model_type
        self.decoder.make_genai_config(config, extra_kwargs, out_dir)
        if self.mtp is not None:
            self.add_mtp_to_genai_config(out_dir)
        if self.dflash2 is not None:
            self.add_dflash2_to_genai_config(out_dir)
        if self.dspark is not None:
            self.add_dspark_to_genai_config(out_dir)
        if self.dflash2 is not None or self.dspark is not None:
            self.make_block_drafter_search_defaults(out_dir)

    def make_block_drafter_search_defaults(self, out_dir):
        """Ship greedy search defaults alongside a block drafter.

        ``Engine::PrepareDflash2Feeds`` sets ``wants_drafts = greedy && ...``, so a checkpoint
        whose ``generation_config.json`` asks for sampling would decode with zero drafts and no
        error. Only ``do_sample`` is cleared; ``top_k``/``top_p``/``temperature`` stay as the
        checkpoint declared them, so a caller who opts back into sampling per turn still gets them.
        """
        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as config_file:
            genai_config = json.load(config_file)

        if not genai_config["search"].get("do_sample", False):
            return
        genai_config["search"]["do_sample"] = False
        with open(config_path, "w") as config_file:
            config_file.write(serialize_genai_config(genai_config))
        print("Set search.do_sample to false: a block drafter only proposes drafts for greedy turns.")

    def add_mtp_to_genai_config(self, out_dir):
        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as config_file:
            genai_config = json.load(config_file)

        decoder_outputs = genai_config["model"]["decoder"].setdefault("outputs", {})
        decoder_outputs.setdefault("hidden_states", "hidden_states")
        genai_config["model"]["mtp"] = {
            "enabled": True,
            "filename": "mtp.onnx",
            "num_hidden_layers": 1,
            "num_key_value_heads": self.decoder.num_kv_heads,
            "head_size": self.decoder.head_size,
            "main_hidden_states": "hidden_states",
            "inputs": {
                "input_ids": "input_ids",
                "hidden_states": "hidden_states",
                "attention_mask": "attention_mask",
                "position_ids": "position_ids",
                "past_key_names": "past_key_values.%d.key",
                "past_value_names": "past_key_values.%d.value",
            },
            "outputs": {
                "logits": "logits",
                "hidden_states": "hidden_states_out",
                "present_key_names": "present.%d.key",
                "present_value_names": "present.%d.value",
            },
        }
        dynamic_batching = genai_config.get("engine", {}).get("dynamic_batching")
        if dynamic_batching is not None:
            dynamic_batching.setdefault("prefix_caching", False)
        self.add_shared_initializers_to_genai_config(genai_config)

        with open(config_path, "w") as config_file:
            config_file.write(serialize_genai_config(genai_config))
        print("Added 'mtp' section to genai_config.json")

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        self.decoder.save_processing(model_name_or_path, extra_kwargs, out_dir)

    def require_specforge_aux_taps(self, target_layer_ids, drafter_name):
        if not target_layer_ids:
            raise ValueError(f"The {drafter_name} checkpoint must define at least one target_layer_ids entry.")

        aux_layers = [layer_id + 1 for layer_id in target_layer_ids]
        untappable = [layer_id - 1 for layer_id in aux_layers if not 1 <= layer_id < self.decoder.num_layers]
        if untappable:
            raise ValueError(
                f"The {drafter_name} checkpoint targets decoder layers {untappable}, whose outputs the exporter "
                f"cannot expose; target_layer_ids must lie in [0, {self.decoder.num_layers - 1})."
            )

        expected = ",".join(str(layer_id) for layer_id in aux_layers)
        actual = ",".join(str(layer_id) for layer_id in self.decoder.aux_hidden_state_layers)
        if actual != expected:
            raise ValueError(
                f"The {drafter_name} drafter needs aux_hidden_state_layers={expected} on the main model, "
                f"got '{actual}'."
            )

    def block_drafter_precision(self, extra_options, option_name):
        precision = str(extra_options.get(option_name, "bf16")).lower()
        allowed = {"bf16", "int4", "int8"}
        if extra_options.get("_drafter_quant_config") is not None:
            allowed.add("int2")
        if precision not in allowed:
            raise ValueError(f"{option_name} must be one of {sorted(allowed)}, got '{precision}'.")
        return precision

    def block_drafter_quant(self, precision, quant_config=None):
        """Resolve weight-only quantization for a block-drafter body, or ``None`` to keep it dense."""
        if precision == "bf16":
            return None
        bits = int(precision.removeprefix("int"))
        block_size = int(
            quant_config.weights.block_size
            if quant_config is not None
            else self.decoder.quant_attrs["matmul_block_size"]
        )
        prepack = self.decoder.matmul_attrs["weights_prepacked"] if self.decoder.ep == "cuda" and bits != 2 else 0
        return {"bits": bits, "block_size": block_size, "prepack": prepack}

    def block_drafter_lm_head_quant(self):
        """Resolve how a block drafter gets its LM head, or ``None`` to keep it dense.

        The drafter's LM head *is* the target's, so it reuses the target's initializers rather
        than quantizing a second copy (see ``adopt_target_tensors``). Only the
        symmetric/``default`` ``MatMulNBits`` convention is wired up here; any other algorithm
        or output format leaves the head dense instead of guessing at initializer names.
        """
        decoder = self.decoder
        if decoder.exclude_lm_head or not decoder.is_lm_head_quantized():
            return None
        if decoder.quant_attrs["use_qdq"]:
            print(
                "Leaving the block drafter's LM head dense: use_qdq makes the target write a "
                "DequantizeLinear/MatMul pair instead of the MatMulNBits this exporter can reuse."
            )
            return None
        head_bits, weight_name, scales_name, zero_point_name = decoder.make_tied_quantized_embedding_input_names()
        shareable = (
            weight_name == f"lm_head.MatMul.weight_Q{head_bits}"
            and scales_name == "lm_head.MatMul.weight_scales"
            and not zero_point_name
        )
        if not shareable:
            print(
                f"Leaving the block drafter's LM head dense: the target writes '{weight_name}', "
                "which this exporter does not know how to reuse."
            )
            return None
        prepack = int(decoder.matmul_attrs["weights_prepacked"]) if decoder.ep == "cuda" else 0
        adopt_target = not (prepack and decoder.io_dtype != ir.DataType.FLOAT16)
        if not adopt_target:
            print(
                "Keeping a private raw quantized block-drafter LM head because its BF16 layout "
                "cannot adopt the target's prepacked quantized weight."
            )
            prepack = 0
        return {
            "bits": head_bits,
            "block_size": int(decoder.quant_attrs["matmul_block_size"]),
            "prepack": prepack,
            "adopt_target": adopt_target,
        }

    def block_drafter_embed_quant(self):
        """Resolve the target's quantized embedding table, or ``None`` if the drafter keeps a dense one.

        The drafter embeds with the target's table, so when the target quantizes it the drafter has
        to emit the same ``GatherBlockQuantized`` over the same initializers. Emitting a dense
        ``Gather`` instead costs a second, unshareable copy of the largest tensor in either graph.
        """
        decoder = self.decoder
        if decoder.exclude_embeds or decoder.onnx_dtype not in {ir.DataType.INT4, ir.DataType.UINT4}:
            return None
        if "Gather" not in decoder.quant_attrs["op_types_to_quantize"]:
            return None
        if "/model/embed_tokens/Gather" in decoder.quant_attrs["nodes_to_exclude"]:
            return None
        if decoder.tied_quantized_embeddings:
            # Tied embeddings gather from a reshape of the quantized LM head rather than from a
            # table of their own, so there is no `model.embed_tokens.weight_Q4` to adopt.
            print(
                "Leaving the block drafter's embedding dense: shared_embeddings makes the target "
                "gather from its LM head weight, which this exporter does not know how to reuse."
            )
            return None
        # Only the symmetric/`default` convention names the table `*.weight_Q4` / `*.weight_scales`.
        if decoder.quantization_algo != "default" or not decoder.quant_attrs["is_symmetric"]:
            print(
                f"Leaving the block drafter's embedding dense: the target quantizes it with "
                f"'{decoder.quantization_algo}', whose initializer names this exporter does not know how to reuse."
            )
            return None
        return {"bits": 4, "block_size": int(decoder.quant_attrs["matmul_block_size"])}

    def make_dflash2_init(self, io_dtype, extra_options):
        """DFlash 2 block drafter, exported as an auxiliary ``dflash2.onnx``.

        ``dflash2_path`` points at the draft checkpoint. The drafter has no embedding and no
        LM head of its own, so both come from the target source; sharing on disk is conditional. SpecForge taps
        the output of each ``target_layer_ids`` entry, which is the residual stream entering the
        following layer.
        """
        self.dflash2_path = extra_options.get("dflash2_path")
        if not self.dflash2_path:
            return
        if not self.decoder.use_paged_attention:
            raise ValueError("dflash2_path requires use_paged_attention=true.")

        self.dflash2_shared_weight_policies = copy.deepcopy(
            extra_options.get("_shared_weight_policies", {"embedding": "auto", "lm_head": "auto"})
        )
        drafter_quant_config = extra_options.get("_drafter_quant_config")
        drafter_io_dtype = (
            drafter_quant_config.to_onnx_dtypes()[0]
            if drafter_quant_config is not None
            else io_dtype if self.decoder.ep == "webgpu" else ir.DataType.BFLOAT16
        )
        if self.decoder.ep == "webgpu" and drafter_io_dtype != ir.DataType.FLOAT16:
            raise ValueError("WebGPU DFlash2 requires FP16 body activations and KV caches.")

        num_draft_tokens = None
        if "dflash2_num_draft_tokens" in extra_options:
            try:
                num_draft_tokens = int(extra_options["dflash2_num_draft_tokens"])
            except (TypeError, ValueError) as e:
                raise ValueError("dflash2_num_draft_tokens must be a positive integer.") from e
            if num_draft_tokens < 1:
                raise ValueError("dflash2_num_draft_tokens must be a positive integer.")

        fuse_gate_up = str(extra_options.get("dflash2_fuse_gate_up", False)).lower()
        if fuse_gate_up not in ("true", "false"):
            raise ValueError("dflash2_fuse_gate_up must be true or false.")
        fuse_qkv = str(extra_options.get("dflash2_fuse_qkv", False)).lower()
        if fuse_qkv not in ("true", "false"):
            raise ValueError("dflash2_fuse_qkv must be true or false.")
        self.dflash2_attrs = {
            "io_dtype": io_dtype,
            "compute_dtype": drafter_io_dtype,
            "num_draft_tokens": num_draft_tokens,
            "precision": self.block_drafter_precision(extra_options, "dflash2_precision"),
            "quant_config": drafter_quant_config,
            "fuse_gate_up": fuse_gate_up == "true",
            "fuse_qkv": fuse_qkv == "true",
        }

        with open(os.path.join(self.dflash2_path, "config.json"), encoding="utf-8") as handle:
            draft_config = json.load(handle)
        dflash_config = draft_config["dflash_config"]
        checkpoint_draft_limit = int(dflash_config["block_size"]) - 1
        if num_draft_tokens is not None and num_draft_tokens > checkpoint_draft_limit:
            raise ValueError(
                f"dflash2_num_draft_tokens must not exceed the drafter checkpoint limit ({checkpoint_draft_limit})."
            )
        target_layer_ids = dflash_config["target_layer_ids"]
        self.require_specforge_aux_taps(target_layer_ids, "DFlash 2")

    def make_dflash2_model(self, input_path):
        if not self.dflash2_path:
            return
        from .dflash2 import DFlash2Builder  # noqa: PLC0415

        print("Building DFlash 2 draft model -> dflash2.onnx")
        if input_path and os.path.isdir(input_path):
            target_dir = input_path
        else:
            from huggingface_hub import snapshot_download  # noqa: PLC0415

            target_dir = snapshot_download(
                self.decoder.model_name_or_path,
                cache_dir=self.decoder.cache_dir,
                token=self.decoder.hf_token,
            )
        self.dflash2 = DFlash2Builder(
            self.dflash2_path,
            target_dir,
            self.dflash2_attrs["io_dtype"],
            self.decoder.attention_attrs["paged_block_size"],
            self.decoder.context_length,
            num_draft_tokens=self.dflash2_attrs["num_draft_tokens"],
            quant=self.block_drafter_quant(self.dflash2_attrs["precision"], self.dflash2_attrs["quant_config"]),
            lm_head_quant=self.block_drafter_lm_head_quant(),
            embed_quant=self.block_drafter_embed_quant(),
            fuse_gate_up=self.dflash2_attrs["fuse_gate_up"],
            include_attention_metadata=True,
            ep=self.decoder.ep,
            compute_dtype=self.dflash2_attrs["compute_dtype"],
            fuse_qkv=self.dflash2_attrs["fuse_qkv"],
        )
        self.dflash2.make_model()

    def save_dflash2_model(self, output_dir):
        if self.dflash2 is None:
            return
        self.dflash2_shared_initializers = self.save_block_drafter_model(
            self.dflash2,
            output_dir,
            "DFlash 2",
            getattr(self, "dflash2_shared_weight_policies", {"embedding": "auto", "lm_head": "auto"}),
        )

    def save_block_drafter_model(self, drafter, output_dir, drafter_name, shared_weight_policies=None):
        """Adopt the target's tensors, save the drafter, and fold what the two share onto one copy."""
        if hasattr(drafter, "adopt_target_tensors"):
            drafter.adopt_target_tensors(os.path.join(output_dir, self.decoder.filename))
        drafter.save_model(output_dir)

        initializer_names = set(getattr(getattr(drafter, "graph", None), "initializers", {}))
        embedding_initializers = frozenset(name for name in initializer_names if name.startswith("model.embed_tokens."))
        head_initializers = frozenset(name for name in initializer_names if name.startswith("lm_head.MatMul."))
        embedding = getattr(drafter, "embed_quant", None)
        if not embedding_initializers:
            embedding_initializers = (
                frozenset(
                    {
                        f"model.embed_tokens.weight_Q{embedding['bits']}",
                        "model.embed_tokens.weight_scales",
                    }
                )
                if embedding is not None
                else frozenset({"model.embed_tokens.weight"})
            )
        head = getattr(drafter, "lm_head_quant", None)
        if not head_initializers:
            head_initializers = (
                frozenset({f"lm_head.MatMul.weight_Q{head['bits']}", "lm_head.MatMul.weight_scales"})
                if head is not None
                else frozenset({"lm_head.MatMul.weight"})
            )
        # A head the drafter had to quantize itself must keep its private copy; an adopted one is
        # the target's own tensor and has to fold back onto it.
        adopted = set(head_initializers if head is not None and head["adopt_target"] else ())
        required = set(adopted)
        private = set(head_initializers - adopted) if head is not None else set()

        if shared_weight_policies is not None:
            for tensor_name, initializers in (
                ("embedding", embedding_initializers),
                ("lm_head", head_initializers),
            ):
                policy = shared_weight_policies[tensor_name]
                if policy == "off":
                    adopted.difference_update(initializers)
                    required.difference_update(initializers)
                    private.update(initializers)
                elif policy == "required":
                    if tensor_name == "lm_head" and head is not None and not head["adopt_target"]:
                        raise ValueError("required LM-head sharing is incompatible with the drafter's private layout")
                    adopted.update(initializers)
                    required.update(initializers)
                    private.difference_update(initializers)

        shared = self.share_initializers(
            output_dir,
            self.decoder.filename,
            drafter.filename,
            adopt_source_initializers=adopted,
            required_source_initializers=required,
            excluded_source_initializers=private,
        )
        shared_names = {entry["name"] for entry in shared}
        missing = required - shared_names
        if missing:
            raise ValueError("Required shared initializers are unavailable: " + ", ".join(sorted(missing)))
        self.warn_unshared_lm_head(
            drafter,
            shared,
            drafter_name,
            shared_weight_policies.get("lm_head", "auto") if shared_weight_policies is not None else "auto",
        )
        return shared

    def warn_unshared_lm_head(self, drafter, shared, drafter_name, sharing_policy="auto"):
        """Report a drafter head that stayed a separate copy instead of folding onto the target's.

        The drafter adopts the target's own initializers, so the two are identical by
        construction and this should never fire. If it does, the bytes on disk diverged
        somewhere after ``adopt_target_tensors``, which costs both a duplicated copy and the
        guarantee that the drafter scores with the head the target verifies with.
        """
        head = drafter.lm_head_quant
        if sharing_policy == "off" or head is None or not head["adopt_target"]:
            return
        weight_name = f"lm_head.MatMul.weight_Q{head['bits']}"
        if any(entry["name"] == weight_name for entry in shared):
            return
        print(
            f"WARNING: the {drafter_name} LM head adopted the target's '{weight_name}' but did not "
            "share it. The two copies may no longer agree."
        )

    def add_dflash2_to_genai_config(self, out_dir):
        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as config_file:
            genai_config = json.load(config_file)

        decoder = genai_config["model"]["decoder"]
        decoder.setdefault("outputs", {}).setdefault("aux_hidden_states", "aux_hidden_states")

        section = self.dflash2.genai_config_section()
        section["aux_hidden_state_layers"] = list(self.decoder.aux_hidden_state_layers)
        if self.dflash2_shared_initializers:
            existing = decoder.get("shared_initializers", [])
            known = {json.dumps(entry, sort_keys=True) for entry in existing}
            for entry in self.dflash2_shared_initializers:
                if json.dumps(entry, sort_keys=True) not in known:
                    existing.append(entry)
            decoder["shared_initializers"] = existing
            section["shared_initializers"] = self.dflash2_shared_initializers
        genai_config["model"]["dflash2"] = section
        dynamic_batching = genai_config.get("engine", {}).get("dynamic_batching")
        if dynamic_batching is not None and section.get("sliding_window", 0) <= 0:
            dynamic_batching.setdefault("prefix_caching", False)

        with open(config_path, "w") as config_file:
            config_file.write(serialize_genai_config(genai_config))
        print("Added 'dflash2' section to genai_config.json")

    def make_dspark_init(self, io_dtype, extra_options):
        """DSpark block drafter, exported as an auxiliary ``dspark.onnx``.

        ``dspark_path`` points at the draft checkpoint. SpecForge taps the *output* of each
        ``target_layer_ids`` entry (``hidden_states[layer_id + 1]``), which is the residual stream
        entering layer ``layer_id + 1`` -- the tensor aux_hidden_state_layers names. Getting that
        off by one leaves acceptance at exactly 1.0.
        """
        self.dspark_path = extra_options.get("dspark_path")
        if not self.dspark_path:
            return
        if self.dflash2_path:
            raise ValueError("dspark_path and dflash2_path are mutually exclusive.")
        if not self.decoder.use_paged_attention:
            raise ValueError("dspark_path requires use_paged_attention=true.")
        if self.decoder.ep == "webgpu":
            raise ValueError(
                "dspark_path is not supported on WebGPU: the DSpark drafter requires "
                "BF16 activations and KV caches, but WebGPU PagedAttention supports only FP16."
            )

        num_draft_tokens = None
        if "dspark_num_draft_tokens" in extra_options:
            try:
                num_draft_tokens = int(extra_options["dspark_num_draft_tokens"])
            except (TypeError, ValueError) as error:
                raise ValueError("dspark_num_draft_tokens must be between 2 and the checkpoint block size.") from error

        try:
            top_k = int(extra_options.get("dspark_top_k", 16))
        except (TypeError, ValueError) as error:
            raise ValueError("dspark_top_k must be a positive integer.") from error
        if top_k < 1:
            raise ValueError("dspark_top_k must be a positive integer.")

        with open(os.path.join(self.dspark_path, "config.json"), encoding="utf-8") as handle:
            draft_config = json.load(handle)
        checkpoint_block_size = int(draft_config["block_size"])
        if num_draft_tokens is not None and not 2 <= num_draft_tokens <= checkpoint_block_size:
            raise ValueError(
                "dspark_num_draft_tokens must be between 2 and the drafter checkpoint block size "
                f"({checkpoint_block_size})."
            )
        vocab_size = int(draft_config["vocab_size"])
        if top_k > vocab_size:
            raise ValueError(f"dspark_top_k must not exceed the drafter vocabulary size ({vocab_size}).")

        self.dspark_attrs = {
            "io_dtype": io_dtype,
            "num_draft_tokens": num_draft_tokens,
            "top_k": top_k,
        }

        target_layer_ids = draft_config["dflash_config"]["target_layer_ids"]
        self.require_specforge_aux_taps(target_layer_ids, "DSpark")

    def make_dspark_model(self, input_path):
        if not self.dspark_path:
            return
        from .dspark import DSparkBuilder  # noqa: PLC0415

        print("Building DSpark draft model -> dspark.onnx")
        target_dir = input_path if input_path and os.path.isdir(input_path) else self.decoder.model_name_or_path
        self.dspark = DSparkBuilder(
            self.dspark_path,
            target_dir,
            self.dspark_attrs["io_dtype"],
            self.decoder.attention_attrs["paged_block_size"],
            self.decoder.context_length,
            num_draft_tokens=self.dspark_attrs["num_draft_tokens"],
            top_k=self.dspark_attrs["top_k"],
            include_attention_metadata=True,
            embed_quant=self.block_drafter_embed_quant(),
            lm_head_quant=self.block_drafter_lm_head_quant(),
        )
        self.dspark.make_model()

    def save_dspark_model(self, output_dir):
        if self.dspark is None:
            return
        self.dspark_shared_initializers = self.save_block_drafter_model(self.dspark, output_dir, "DSpark")

    def add_dspark_to_genai_config(self, out_dir):
        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as config_file:
            genai_config = json.load(config_file)

        decoder = genai_config["model"]["decoder"]
        decoder.setdefault("outputs", {}).setdefault("aux_hidden_states", "aux_hidden_states")

        section = self.dspark.genai_config_section()
        section["aux_hidden_state_layers"] = list(self.decoder.aux_hidden_state_layers)
        if self.dspark_shared_initializers:
            existing = decoder.get("shared_initializers", [])
            known = {json.dumps(entry, sort_keys=True) for entry in existing}
            for entry in self.dspark_shared_initializers:
                if json.dumps(entry, sort_keys=True) not in known:
                    existing.append(entry)
            decoder["shared_initializers"] = existing
            section["shared_initializers"] = self.dspark_shared_initializers
        genai_config["model"]["dspark"] = section
        genai_config["engine"]["dynamic_batching"].setdefault("prefix_caching", False)

        with open(config_path, "w") as config_file:
            config_file.write(serialize_genai_config(genai_config))
        print("Added 'dspark' section to genai_config.json")


class Qwen35MTPModel(Qwen35MTP, Qwen35MoETextModel):
    """Qwen3.6 multi-token-prediction self-speculative head builder."""

    is_moe_mtp = True

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.is_mtp_head = True

        config = copy.deepcopy(config)
        text_config = getattr(config, "text_config", config)
        text_config.num_hidden_layers = 1
        text_config.layer_types = ["full_attention"]
        config.num_hidden_layers = 1
        config.layer_types = ["full_attention"]

        self.mtp_layer_config = copy.deepcopy(text_config)
        self.mtp_layer_config.layer_types = ["full_attention"]
        self.mtp_layer_config.num_hidden_layers = 1

        extra_options = copy.deepcopy(extra_options)
        extra_options["num_hidden_layers"] = 1
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        quant_config = extra_options.get("_quant_config")
        self.preserve_mtp_quantization = quant_config is None or quant_config.checkpoint_policy == "preserve"
        self.input_names["hidden_states"] = "hidden_states"
        self.input_types["hidden_states"] = self.io_dtype
        self.input_shapes["hidden_states"] = self.make_hidden_state_shape()

    def make_model(self, input_path):
        self.make_inputs_and_outputs()
        self.load_mtp_weights(input_path)
        self.make_preprocessing_nodes()

        projected = self.make_mtp_input_projection()
        self.layernorm_attrs["root_input"] = projected
        self.layernorm_attrs["skip_input"] = projected
        self.layernorm_attrs["first_layernorm"] = True

        self.make_layer(0, self.mtp_weights.layers[0])
        self.make_layernorm(1, self.mtp_weights.norm, skip=True, simple=True, location="final_norm")
        mtp_norm_output = self.layernorm_attrs["output_0"]
        self.make_lm_head(self.mtp_weights.lm_head)

        hidden_states_output = "hidden_states_out"
        self.make_node(
            "Identity",
            inputs=[mtp_norm_output],
            outputs=[hidden_states_output],
            name="/model/mtp/hidden_states_out/Identity",
        )
        hidden_states_value = self.make_value(
            hidden_states_output,
            self.io_dtype,
            shape=self.make_hidden_state_shape(),
        )
        self.model.graph.outputs.append(hidden_states_value)

        self.make_postprocessing_nodes()
        del self.mtp_weights

    def load_mtp_weights(self, input_path):
        model_dir = input_path if input_path and os.path.isdir(input_path) else self.model_name_or_path
        try:
            from loaders.qwen import QwenMTPModel  # noqa: PLC0415
        except ImportError:
            from onnxruntime_genai.models.loaders.qwen import QwenMTPModel  # noqa: PLC0415

        self.mtp_weights = QwenMTPModel.from_pretrained(
            self.quant_type,
            input_path,
            model_dir,
            self.mtp_layer_config,
            preserve_quantization=self.preserve_mtp_quantization,
            load_quantized_model=self.load_weights,
            is_moe=self.is_moe_mtp,
            cache_dir=self.cache_dir,
            token=self.hf_token,
        )


class Qwen35DenseMTPModel(Qwen35MTPModel):
    """Dense Qwen3.5/Qwen3.8 MTP head with one full-attention decoder layer."""

    is_moe_mtp = False

    def make_layer(self, layer_id, layer):
        return Qwen35TextModel.make_layer(self, layer_id, layer)


class Qwen35Model(Qwen35MoEModel):
    """Composite dense Qwen3.5/Qwen3.8 builder with an optional MTP graph."""

    def get_decoder_model_class(self):
        return Qwen35TextModel

    def get_mtp_model_class(self):
        return Qwen35DenseMTPModel
