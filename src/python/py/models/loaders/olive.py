# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .gptq import GPTQModel


class OliveModel(GPTQModel):
    """
    Olive quantization format:
    - qweight: (out_features, packed_in_features) uint8, packed along last dim
    - scales: (out_features, num_groups) float
    - qzeros: (out_features, packed_num_groups) uint8, packed along last dim
    """

    override_config_key = "overrides"

    def set_quantized_tensor_properties(self, module):
        module.out_features = module.qweight.shape[0]
        module.in_features = module.qweight.shape[1] * 8 // module.bits

    def get_layer_bits(self, layer_name):
        name = ".".join(layer_name.split(".")[:-1])
        return self.overrides.get(name, {}).get("bits", self.global_bits)

    def get_layer_group_size(self, layer_name):
        name = ".".join(layer_name.split(".")[:-1])
        return self.overrides.get(name, {}).get("group_size", self.global_group_size)

    def handle_qzeros(self, module):
        """Olive uses unsigned quantization, no offset needed."""

    def unpack(self, module):
        """Skip unpack for Olive format."""

    def repack(self, module):
        """
        Olive format:
        - qweight: (out_features, packed_in_features) uint8
        - scales: (out_features, num_groups) float
        - qzeros: (out_features, packed_num_groups) uint8

        ORT format:
        - qweight: (out_features, k_blocks, blob_size) uint8
        - scales: (out_features * num_groups,) float, flattened
        - qzeros: (out_features * packed_num_groups,) uint8, flattened
        """
        kpack = 8 // module.bits
        k_blocks = module.in_features // module.group_size
        blob_size = module.group_size // kpack

        # qweight: (out_features, packed_in_features) -> (out_features, k_blocks, blob_size)
        module.qweight = module.qweight.reshape(module.out_features, k_blocks, blob_size).contiguous()

        # scales: (out_features, num_groups) -> flatten to 1D
        module.scales = module.scales.reshape(-1).contiguous()

        # qzeros: (out_features, packed_num_groups) -> flatten to 1D
        if module.qzeros is not None and module.qzeros.numel() > 0:
            module.qzeros = module.qzeros.reshape(-1).contiguous()
