import onnx_ir as ir


class Qwen:
    """Graph expansions for Qwen models predating Qwen3.5."""

    def make_qk_norm(self, layer_id, attention):
        # Before: SimplifiedLayerNorm --> Cast from FP32 to io_dtype --> Reshape --> Cast from io_dtype to FP32 --> MRotaryEmbedding
        # After:  SimplifiedLayerNorm --> Reshape --> MRotaryEmbedding
        # This allows both LayerNorm and MRoPE to be computed in FP32. Reshape is not affected by the dtype.

        self.layernorm_attrs["cast"]["output_0"] = False
        super().make_qk_norm(layer_id, attention)

        # Update dtypes for QK-norm reshapes to stay as FP32 and not cast to self.io_dtype
        self.values[self.attention_attrs["q_path"]].dtype = ir.DataType.FLOAT
        self.values[self.attention_attrs["k_path"]].dtype = ir.DataType.FLOAT

        self.layernorm_attrs["cast"]["output_0"] = True

