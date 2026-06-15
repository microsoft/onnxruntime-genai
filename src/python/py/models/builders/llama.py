# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from .base import Model


class LlamaModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_rope_init(self, config):
        if "low_freq_factor" in config.rope_scaling:
            # For models that rescale `inv_freq` using `low_freq_factor` and `high_freq_factor` (e.g. LLaMA-3.1)
            factor = config.rope_scaling["factor"] if "factor" in config.rope_scaling else 0
            low_freq_factor = config.rope_scaling["low_freq_factor"] if "low_freq_factor" in config.rope_scaling else 0
            high_freq_factor = (
                config.rope_scaling["high_freq_factor"] if "high_freq_factor" in config.rope_scaling else 0
            )

            self.rope_attrs["rescale_inv_freq"] = {
                "factor": factor,                      # Scale factor when calculating `new_freq` in rotary embeddings
                "low_freq_factor": low_freq_factor,    # Low freq factor when calculating `low_freq_wavelen` in rotary embeddings
                "high_freq_factor": high_freq_factor,  # High freq factor when calculating `high_freq_wavelen` in rotary embeddings
            }
