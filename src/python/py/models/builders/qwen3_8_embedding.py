# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import onnx_ir as ir
import torch

from .qwen3_8_graph import Qwen4ExpGraphModel


class Qwen4ExpEmbeddingModel(Qwen4ExpGraphModel):
    def __init__(self, config, embedding_weight, io_dtype):
        super().__init__(io_dtype, "embedding.onnx", "qwen4_exp_embedding")
        hidden_size = embedding_weight.shape[1]
        input_ids = self.make_value("input_ids", ir.DataType.INT64, ["batch_size", "sequence_length"])
        image_features = self.make_value("image_features", self.io_dtype, ["num_image_tokens", hidden_size])
        inputs_embeds = self.make_value("inputs_embeds", self.io_dtype, ["batch_size", "sequence_length", hidden_size])
        self.graph.inputs.extend([input_ids, image_features])
        self.graph.outputs.append(inputs_embeds)

        weight_name = "model.embed_tokens.weight"
        self.make_initializer(embedding_weight, weight_name, to=self.io_dtype)
        image_token = "image_token_id"
        video_token = "video_token_id"
        self.make_initializer(torch.tensor(config.image_token_id, dtype=torch.int64), image_token)
        self.make_initializer(torch.tensor(config.video_token_id, dtype=torch.int64), video_token)
        gathered = "/model/embed_tokens/Gather/output_0"
        self.make_node("Gather", [weight_name, "input_ids"], [gathered], name="/model/embed_tokens/Gather", axis=0)
        self.make_value(gathered, self.io_dtype, ["batch_size", "sequence_length", hidden_size])
        image_mask = self.make_binary(
            "Equal",
            "/model/image_mask/Equal",
            ["input_ids", image_token],
            ir.DataType.BOOL,
            ["batch_size", "sequence_length"],
        )
        video_mask = self.make_binary(
            "Equal",
            "/model/video_mask/Equal",
            ["input_ids", video_token],
            ir.DataType.BOOL,
            ["batch_size", "sequence_length"],
        )
        multimodal_mask = self.make_binary(
            "Or",
            "/model/multimodal_mask/Or",
            [image_mask, video_mask],
            ir.DataType.BOOL,
            ["batch_size", "sequence_length"],
        )
        indices = "/model/multimodal_indices/NonZero/output_0"
        self.make_node("NonZero", [multimodal_mask], [indices], name="/model/multimodal_indices/NonZero")
        self.make_value(indices, ir.DataType.INT64, [2, "num_image_tokens"])
        scatter_indices = "/model/multimodal_indices/Transpose/output_0"
        self.make_node(
            "Transpose",
            [indices],
            [scatter_indices],
            name="/model/multimodal_indices/Transpose",
            perm=[1, 0],
        )
        self.make_value(scatter_indices, ir.DataType.INT64, ["num_image_tokens", 2])
        self.make_node(
            "ScatterND",
            [gathered, scatter_indices, "image_features"],
            ["inputs_embeds"],
            name="/model/merge_embeddings/ScatterND",
        )
