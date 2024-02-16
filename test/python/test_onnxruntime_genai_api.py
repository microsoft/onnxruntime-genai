# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import os
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest


# TODO (baijumeswani) : address crash on cuda og.DeviceType.CUDA
@pytest.mark.parametrize("device", [og.DeviceType.Auto, og.DeviceType.CPU])
def test_greedy_search(device, test_data_path):
    model_path = os.fspath(
        Path(test_data_path) / "hf-internal-testing" / "tiny-random-gpt2-fp32"
    )

    model = og.Model(model_path, device)

    search_params = og.SearchParams(model)
    search_params.input_ids = np.array(
        [[0, 0, 0, 52], [0, 0, 195, 731]], dtype=np.int32
    )
    search_params.max_length = 10
    input_ids_shape = [2, 4]
    batch_size = input_ids_shape[0]

    generator = og.Generator(model, search_params)
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token_top()

    expected_sequence = np.array(
        [
            [0, 0, 0, 52, 204, 204, 204, 204, 204, 204],
            [0, 0, 195, 731, 731, 114, 114, 114, 114, 114],
        ],
        dtype=np.int32,
    )
    for i in range(batch_size):
        assert np.array_equal(
            expected_sequence[i], generator.get_sequence(i).get_array()
        )

    sequences = model.generate(search_params)
    for i in range(len(sequences)):
        assert sequences[i] == expected_sequence[i].tolist()
