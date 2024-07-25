# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

# lora_A_weight: (3072, 64)
# lora_B_weight: (64, 9216)
#
lora_A_weight = np.random.rand(3072, 64)
lora_B_weight = np.random.rand(64, 9216)

# Save the parameters to the output .npz file
to_save = { "lora_A_weight" :  lora_A_weight, "lora_B_weight" :  lora_B_weight}
np.savez_compressed("two_lora_params.npz", **to_save)