# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Export a BF16-activation DFlash2 drafter sharing the Qwen target embedding/head."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
import torch
from safetensors.torch import load_file

from ..dflash2 import DFlash2Builder


class QwenDFlash2Builder(DFlash2Builder):
    def __init__(self, checkpoint, target, precision, drafts):
        context = json.loads((checkpoint / "config.json").read_text())["max_position_embeddings"]
        self.target = onnx.load(str(target), load_external_data=False)
        rotary = next(
            (tensor for tensor in self.target.graph.initializer
             if tensor.name in ("model.rotary_emb.cos_cache", "rope.cos")), None
        )
        if rotary is None or len(rotary.dims) != 2 or rotary.dims[0] < 8:
            raise ValueError("Target must expose a nonempty two-dimensional rotary cache")
        super().__init__(str(checkpoint), "", ir.DataType.BFLOAT16, 128, max(context + 4, rotary.dims[0]),
                         num_draft_tokens=drafts)
        self.checkpoint = checkpoint
        self.target_path = target
        self.fp8 = precision == "fp8"

    def target_tensor(self, name):
        tensor = next(value for value in self.target.graph.initializer if value.name == name)
        metadata = {entry.key: entry.value for entry in tensor.external_data}
        dtype = {onnx.TensorProto.BFLOAT16: np.uint16,
                 onnx.TensorProto.FLOAT8E4M3FN: np.uint8,
                 onnx.TensorProto.FLOAT: np.float32}[tensor.data_type]
        if metadata:
            data = np.memmap(self.target_path.parent / metadata["location"], mode="r", dtype=dtype,
                             offset=int(metadata.get("offset", 0)), shape=tuple(tensor.dims))
        else:
            data = np.frombuffer(tensor.raw_data, dtype=dtype).reshape(tensor.dims)
        value = torch.from_numpy(np.array(data))
        if tensor.data_type == onnx.TensorProto.BFLOAT16:
            return value.view(torch.bfloat16)
        if tensor.data_type == onnx.TensorProto.FLOAT8E4M3FN:
            return value.view(torch.float8_e4m3fn)
        return value

    def load_weights(self):
        self.weights = load_file(str(self.checkpoint / "model.safetensors"))
        return self.weights

    def make_embedding(self, name, rows):
        self.make_initializer(self.target_tensor("model.embed_tokens.weight"), "model.embed_tokens.weight")
        output = self.out(name)
        self.make_node("Gather", ["model.embed_tokens.weight", "input_ids"], [output], name=name, axis=0)
        self.make_value(output, self.io_dtype, [rows, self.hidden_size])
        return output

    def make_lm_head(self, root, rows="num_block"):
        node = next(node for node in self.target.graph.node
                    if node.op_type == "MatMulBlockQuantizedFp8Weight" and "lm_head.weight_t" in node.input)
        for name in node.input[1:3]:
            self.make_initializer(self.target_tensor(name), name)
        output = self.out("/lm_head/MatMul")
        self.make_node(node.op_type, [root, *node.input[1:3]], [output],
                       name="/lm_head/MatMul", domain=node.domain,
                       **{attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute})
        self.make_value(output, self.io_dtype, [rows, self.vocab_size])
        return output

    def matmul(self, name, root_input, weight_tensor, in_features, out_features, rows,
               weight_name=None, quantize=True):
        if not self.fp8 or not quantize:
            return super().matmul(name, root_input, weight_tensor, in_features, out_features,
                                  rows, weight_name, quantize=False)
        weight = weight_tensor.float().reshape(out_features, in_features // 128, 128)
        scales = weight.abs().amax(dim=-1).clamp_min(1e-12) / 448.0
        quantized = (weight / scales.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        initializer = weight_name or name[1:].replace("/", ".") + ".weight"
        self.make_initializer(quantized.reshape(out_features, in_features), initializer)
        self.make_initializer(scales, initializer + ".scale")
        output = self.out(name)
        self.make_node("MatMulBlockQuantizedFp8Weight",
                       [root_input, initializer, initializer + ".scale"], [output],
                       name=name, domain="com.microsoft", block_size=128)
        self.make_value(output, self.io_dtype, [rows, out_features])
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--precision", choices=("bf16", "fp8"), default="fp8")
    parser.add_argument("--drafts", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    builder = QwenDFlash2Builder(args.checkpoint, args.target, args.precision, args.drafts)
    builder.make_model()
    args.output.mkdir(parents=True)
    builder.save_model(str(args.output))
    (args.output / "dflash2_config.json").write_text(json.dumps(builder.genai_config_section(), indent=2) + "\n")


if __name__ == "__main__":
    main()
