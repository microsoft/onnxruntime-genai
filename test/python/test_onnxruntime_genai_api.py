# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import os
import sys
import sysconfig
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest

devices = ["cpu"]

if og.is_cuda_available():
    devices.append("cuda")

if og.is_dml_available():
    devices.append("dml")

@pytest.mark.parametrize(
    "relative_model_path",
    (
        [
            Path("hf-internal-testing") / "tiny-random-gpt2-fp32",
            Path("hf-internal-testing") / "tiny-random-gpt2-fp32-cuda",
            Path("hf-internal-testing") / "tiny-random-gpt2-fp16-cuda",
        ]
        if og.is_cuda_available()
        else [Path("hf-internal-testing") / "tiny-random-gpt2-fp32"]
    ),
)
def test_greedy_search(test_data_path, relative_model_path):
    model_path = os.fspath(Path(test_data_path) / relative_model_path)

    model = og.Model(model_path)

    search_params = og.GeneratorParams(model)
    search_params.input_ids = np.array(
        [[0, 0, 0, 52], [0, 0, 195, 731]], dtype=np.int32
    )
    search_params.set_search_options(do_sample=False, max_length=10)
    input_ids_shape = [2, 4]
    batch_size = input_ids_shape[0]

    generator = og.Generator(model, search_params)
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()

    expected_sequence = np.array(
        [
            [0, 0, 0, 52, 204, 204, 204, 204, 204, 204],
            [0, 0, 195, 731, 731, 114, 114, 114, 114, 114],
        ],
        dtype=np.int32,
    )
    for i in range(batch_size):
        assert np.array_equal(expected_sequence[i], generator.get_sequence(i))

    sequences = model.generate(search_params)
    for i in range(len(sequences)):
        assert sequences[i] == expected_sequence[i].tolist()


# TODO: CUDA pipelines use python3.6 and do not have a way to download models since downloading models
# requires pytorch and hf transformers. This test should be re-enabled once the pipeline is updated.
@pytest.mark.skipif(
    sysconfig.get_platform().endswith("arm64") or sys.version_info.minor < 8,
    reason="Python 3.8 is required for downloading models.",
)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("batch", [True, False])
def test_tokenizer_encode_decode(device, phi2_for, batch):
    model_path = phi2_for(device)

    model = og.Model(model_path)
    tokenizer = og.Tokenizer(model)

    prompts = [
        "This is a test.",
        "Rats are awesome pets!",
        "The quick brown fox jumps over the lazy dog.",
    ]
    sequences = None
    if batch:
        sequences = tokenizer.encode_batch(prompts)
        decoded_strings = tokenizer.decode_batch(sequences)
        assert prompts == decoded_strings
    else:
        for prompt in prompts:
            sequence = tokenizer.encode(prompt)
            decoded_string = tokenizer.decode(sequence)
            assert prompt == decoded_string


@pytest.mark.skipif(
    sysconfig.get_platform().endswith("arm64") or sys.version_info.minor < 8,
    reason="Python 3.8 is required for downloading models.",
)
@pytest.mark.parametrize("device", devices)
def test_tokenizer_stream(device, phi2_for):
    model = og.Model(phi2_for(device))
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()

    prompts = [
        "This is a test.",
        "Rats are awesome pets!",
        "The quick brown fox jumps over the lazy dog.",
    ]

    for prompt in prompts:
        sequence = tokenizer.encode(prompt)
        decoded_string = ""
        for token in sequence:
            decoded_string += tokenizer_stream.decode(token)

        assert decoded_string == prompt

# TODO: CUDA pipelines use python3.6 and do not have a way to download models since downloading models
# requires pytorch and hf transformers. This test should be re-enabled once the pipeline is updated.
@pytest.mark.skipif(
    sysconfig.get_platform().endswith("arm64") or sys.version_info.minor < 8,
    reason="Python 3.8 is required for downloading models.",
)
@pytest.mark.parametrize("device", devices)
def test_batching(device, phi2_for):
    model = og.Model(phi2_for(device))
    tokenizer = og.Tokenizer(model)

    prompts = [
        "This is a test.",
        "Rats are awesome pets!",
        "The quick brown fox jumps over the lazy dog.",
    ]

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=20)  # To run faster
    params.input_ids = tokenizer.encode_batch(prompts)

    if device == "dml":
        params.try_use_cuda_graph_with_max_batch_size(len(prompts))

    output_sequences = model.generate(params)
    print(tokenizer.decode_batch(output_sequences))

def test_logging():
    og.set_log_options(enabled: True, model_input_values: True, model_output_shapes: True)
    og.set_log_options(model_input_values: False, model_output_shapes: False)
    og.set_log_options(enabled: False)
