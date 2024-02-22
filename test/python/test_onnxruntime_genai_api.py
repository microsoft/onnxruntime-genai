# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import os
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest


# FIXME: CUDA device does not work on the CI pipeline because the pipeline uses different cuda versions for
# onnxruntime-genai and onnxruntime. This introduces incompatibility.
# @pytest.mark.parametrize("device", [og.DeviceType.CPU, og.DeviceType.CUDA] if og.is_cuda_available() else [og.DeviceType.CPU])
@pytest.mark.parametrize("device", [og.DeviceType.CPU])
@pytest.mark.parametrize("relative_model_path", [Path("hf-internal-testing") / "tiny-random-gpt2-fp32", Path("hf-internal-testing") / "tiny-random-gpt2-fp32"])
def test_greedy_search(device, test_data_path, relative_model_path):
    model_path = os.fspath(
        Path(test_data_path) / relative_model_path
    )

    model = og.Model(model_path, device)

    search_params = og.GeneratorParams(model)
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

'''
TODO: Enable once the phi-2 model exists

@pytest.mark.parametrize("device", [og.DeviceType.CPU])
@pytest.mark.parametrize("relative_model_path", [Path("phi-2")])
def test_batching(device, test_data_path, relative_model_path):
    model_path = os.fspath(
        Path(test_data_path) / relative_model_path
    )

    model = og.Model(model_path, device)
    tokenizer = model.create_tokenizer()

    prompts = [
      "This is a test.",
      "Rats are awesome pets!",
      "The quick brown fox jumps over the lazy dog.",
    ]

    params=og.GeneratorParams(model)
    params.max_length=20 # To run faster
    params.set_input_sequences(tokenizer.encode_batch(prompts))

    output_sequences = model.generate(params)
    print(tokenizer.decode_batch(output_sequences))
'''
