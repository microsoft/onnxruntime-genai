# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import torch


class EngramWeights:
    CPU_EMBEDDING_ANNOTATION = "cpu_embedding"

    def get_engram_pad_id(self, eos_token_id):
        if isinstance(eos_token_id, (list, tuple)):
            if not eos_token_id:
                raise ValueError("Engram requires a nonempty eos_token_id list.")
            eos_token_id = eos_token_id[0]
        if type(eos_token_id) is not int or eos_token_id < 0:
            raise ValueError("Engram requires a nonnegative integer eos_token_id.")
        return eos_token_id

    def prepare_engram_embedding(self, table):
        weight = table.weight.detach().cpu()
        scale = getattr(table, "weight_scale", None)
        if weight.dtype == torch.float8_e4m3fn:
            if scale is None:
                scale = torch.ones(1, dtype=torch.float32)
            else:
                scale = torch.as_tensor(scale).detach().cpu()
            if scale.numel() != 1:
                raise ValueError(f"Engram FP8 weight scale must be scalar, got shape {tuple(scale.shape)}.")
            return weight, scale

        if not weight.is_floating_point():
            raise ValueError(f"Engram embedding weight must be floating point, got {weight.dtype}.")
        max_abs = weight.abs().max().float()
        if not torch.isfinite(max_abs):
            raise ValueError("Engram embedding weight contains non-finite values.")
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = max_abs / fp8_max if max_abs > 0 else torch.ones((), dtype=torch.float32)
        quantized_weight = (weight / scale).clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)
        return quantized_weight, scale.reshape(1)


class Qwen38EngramLoader:
    get_engram_pad_id = EngramWeights.get_engram_pad_id

    def make_ple(self, prefix):
        ple = self.make_namespace(prefix)
        embedding = ple.ple_embedding
        embedding.eos_token_id = self.get_engram_pad_id(self.text_config["eos_token_id"])
        table = embedding.ngram_embedding
        if not hasattr(table, "weight"):
            shard_names = [name for name in vars(table) if name.startswith("shard_")]
            shard_names.sort(key=lambda name: int(name.removeprefix("shard_")))
            expected = self.text_config.get("split_ngram_parts", len(shard_names))
            if shard_names != [f"shard_{index}" for index in range(expected)] or not shard_names:
                raise ValueError(f"Incomplete Qwen3.8 PLE embedding shards at '{prefix}'.")
            table.weight = torch.cat([getattr(table, name).weight for name in shard_names], dim=0)
            for name in shard_names:
                delattr(table, name)
        if table.weight.dtype == torch.float8_e4m3fn:
            self.validate_positive_scalar(getattr(table, "weight_scale", None), f"{prefix}.weight_scale")
        return ple
