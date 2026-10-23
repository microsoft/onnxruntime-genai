# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from .standard import Standard


class WebGPU:
    make_attention_mask_reformatting_for_gqa = Standard.make_attention_mask_reformatting_for_gqa
