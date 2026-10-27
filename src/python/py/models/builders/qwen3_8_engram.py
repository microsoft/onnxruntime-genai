# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import onnx_ir as ir
import torch
from loaders.engram import EngramWeights

from .qwen3_8_graph import Qwen4ExpGraphModel


class Qwen4ExpEngramModel(Qwen4ExpGraphModel):
    CPU_EMBEDDING_ANNOTATION = EngramWeights.CPU_EMBEDDING_ANNOTATION
    prepare_engram_embedding = EngramWeights.prepare_engram_embedding
    get_engram_pad_id = EngramWeights.get_engram_pad_id

    def __init__(self, ple, config, io_dtype, scale_dtype=None):
        super().__init__(io_dtype, "engram.onnx", "qwen4_exp_engram")
        self.graph.opset_imports["com.microsoft"] = 1
        embedding = ple.ple_embedding
        eos_token_id = self.get_engram_pad_id(embedding.eos_token_id)
        ngram_size = config.ngram_size
        heads_per_ngram = config.heads_per_ngram
        ngram_heads = (ngram_size - 1) * heads_per_ngram
        ple_embed_dim = config.ple_embed_dim
        head_dim = ple_embed_dim // ngram_heads
        ple_layer_id = config.ple_layer_ids[0] - 1

        input_ids = self.make_value("input_ids", ir.DataType.INT64, ["batch_size", "sequence_length"])
        past_tokens = self.make_value(
            f"past.{ple_layer_id}.ple_tokens", ir.DataType.INT64, ["batch_size", ngram_size - 1]
        )
        engram_embeddings = self.make_value(
            "engram_embeddings", self.io_dtype, ["batch_size", "sequence_length", ple_embed_dim]
        )
        present_tokens = self.make_value(
            f"present.{ple_layer_id}.ple_tokens", ir.DataType.INT64, ["batch_size", ngram_size - 1]
        )
        self.graph.inputs.extend([input_ids, past_tokens])
        self.graph.outputs.extend([engram_embeddings, present_tokens])

        multipliers = "model.ple.layer_multipliers"
        vocab_sizes = "model.ple.head_vocab_sizes"
        offsets = "model.ple.head_offsets"
        eos = "model.ple.eos_token_id"
        self.make_initializer(embedding.layer_multipliers, multipliers)
        self.make_initializer(embedding.ngram_heads_vocab_sizes, vocab_sizes)
        self.make_initializer(embedding.ngram_heads_offsets, offsets)
        self.make_initializer(torch.tensor(eos_token_id, dtype=torch.int64), eos)

        ngram_ids = "/model/ple/NGramHashMapping/output_0"
        self.make_node(
            "NGramHashMapping",
            inputs=[input_ids.name, multipliers, vocab_sizes, past_tokens.name, offsets, eos],
            outputs=[ngram_ids, present_tokens.name],
            name="/model/ple/NGramHashMapping",
            domain="com.microsoft",
            max_ngram_size=ngram_size,
            n_head_per_ngram=heads_per_ngram,
            pad_id=eos_token_id,
            reset_on_eos=1,
        )
        self.make_value(
            ngram_ids,
            ir.DataType.INT64,
            ["batch_size", "sequence_length", ngram_heads],
        )

        table_name = "model.ple.ngram_embedding.weight"
        quantized_weight, weight_scale = self.prepare_engram_embedding(embedding.ngram_embedding)
        self.make_initializer(quantized_weight, table_name)
        self.make_initializer(
            weight_scale.reshape(1, 1),
            "model.ple.ngram_embedding.weight_scale",
            to=scale_dtype,
        )
        scale_dtype = self.graph.initializers["model.ple.ngram_embedding.weight_scale"].dtype
        gathered = "/model/ple/ngram_embedding/GatherBlockQuantized/output_0"
        self.make_node(
            "GatherBlockQuantized",
            inputs=[table_name, ngram_ids, "model.ple.ngram_embedding.weight_scale"],
            outputs=[gathered],
            name="/model/ple/ngram_embedding/GatherBlockQuantized",
            domain="com.microsoft",
            metadata_props={"layer_ann": self.CPU_EMBEDDING_ANNOTATION},
            gather_axis=0,
            quantize_axis=1,
            block_size=0,
        )
        self.make_value(
            gathered,
            scale_dtype,
            ["batch_size", "sequence_length", ngram_heads, head_dim],
        )
        if scale_dtype != self.io_dtype:
            cast_name = "/model/ple/ngram_embedding/Cast"
            self.make_cast(
                cast_name,
                gathered,
                self.io_dtype,
                ["batch_size", "sequence_length", ngram_heads, head_dim],
            )
            gathered = f"{cast_name}/output_0"
        self.make_node(
            "Reshape",
            inputs=[gathered, f"/model/constants/INT64/[0, 0, {ple_embed_dim}]"],
            outputs=[engram_embeddings.name],
            name="/model/ple/ngram_embedding/Reshape",
        )

    def save_model(self, output_dir):
        super().save_model(output_dir)
