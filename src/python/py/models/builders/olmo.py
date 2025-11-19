# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import torch

from .base import Model


class OLMoModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_layernorm(self, layer_id, layernorm, skip, simple, location):
        layernorm.weight = torch.ones(self.hidden_size)
        layernorm.bias = torch.zeros(self.hidden_size)
        super().make_layernorm(layer_id, layernorm, skip, simple, location)
