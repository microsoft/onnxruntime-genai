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

if og.is_rocm_available():
    devices.append("rocm")


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
        params.try_graph_capture_with_max_batch_size(len(prompts))

    output_sequences = model.generate(params)
    print(tokenizer.decode_batch(output_sequences))


def test_logging():
    og.set_log_options(enabled=True, generate_next_token=True)


@pytest.mark.parametrize(
    "relative_model_path",
    (
        [
            (Path("hf-internal-testing") / "tiny-random-gpt2-fp32", "CPU"),
            (Path("hf-internal-testing") / "tiny-random-gpt2-fp32-cuda", "CUDA"),
            (Path("hf-internal-testing") / "tiny-random-gpt2-fp16-cuda", "CUDA"),
        ]
        if og.is_cuda_available()
        else [(Path("hf-internal-testing") / "tiny-random-gpt2-fp32", "CPU")]
    ),
)
def test_model_device_type(test_data_path, relative_model_path):
    model_path = os.fspath(Path(test_data_path) / relative_model_path[0])

    model = og.Model(model_path)

    assert model.device_type == relative_model_path[1]


@pytest.mark.parametrize(
    "relative_model_path",
    (
        [
            Path("hf-internal-testing") / "tiny-random-gpt2-fp32",
            Path("hf-internal-testing") / "tiny-random-gpt2-fp32-cuda",
            Path("hf-internal-testing") / "tiny-random-gpt2-fp16-cuda",
        ]
        if og.is_cuda_available()
        else [
            Path("hf-internal-testing") / "tiny-random-gpt2-fp32",
        ]
    ),
)
def test_get_output(test_data_path, relative_model_path):
    model_path = os.fspath(Path(test_data_path) / relative_model_path)

    model = og.Model(model_path)

    search_params = og.GeneratorParams(model)
    search_params.input_ids = np.array(
        [[0, 0, 0, 52], [0, 0, 195, 731]], dtype=np.int32
    )
    search_params.set_search_options(do_sample=False, max_length=10)

    generator = og.Generator(model, search_params)

    # check prompt
    # full logits has shape [2, 4, 1000]. Sample 1 for every 200 tokens and the expected sampled logits has shape [2, 4, 5]
    expected_sampled_logits_prompt = np.array(
        [
            [
                [0.29694548, 0.00955007, 0.0430819, 0.10063869, 0.0437237],
                [0.27329233, 0.00841076, -0.1060291, 0.11328877, 0.13369876],
                [0.30323744, 0.0545997, 0.03894716, 0.11702324, 0.0410665],
                [-0.12675379, -0.04443946, 0.14492269, 0.03021223, -0.03212897],
            ],
            [
                [0.29694548, 0.00955007, 0.0430819, 0.10063869, 0.0437237],
                [0.27329233, 0.00841076, -0.1060291, 0.11328877, 0.13369876],
                [-0.04699047, 0.17915794, 0.20838135, 0.10888482, -0.00277808],
                [0.2938929, -0.10538938, -0.00226692, 0.12050669, -0.10622668],
            ],
        ]
    )
    generator.compute_logits()
    logits = generator.get_output("logits")
    assert np.allclose(logits[:, :, ::200], expected_sampled_logits_prompt, atol=1e-3)
    generator.generate_next_token()

    # check for the 1st token generation
    # full logits has shape [2, 1, 1000]. Sample 1 for every 200 tokens and the expected sampled logits has shape [2, 1, 5]
    expected_sampled_logits_token_gen = np.array(
        [
            [[0.03742531, -0.05752287, 0.14159015, 0.04210977, -0.1484456]],
            [[0.3041716, -0.08701379, -0.03778192, 0.07471392, -0.02049096]],
        ]
    )
    generator.compute_logits()
    logits = generator.get_output("logits")
    assert np.allclose(
        logits[:, :, ::200], expected_sampled_logits_token_gen, atol=1e-3
    )
    generator.generate_next_token()


@pytest.mark.parametrize("relative_model_path", [Path("vision-preprocessing")])
@pytest.mark.parametrize("relative_image_path", [Path("images") / "sheet.png"])
def test_vision_preprocessing(test_data_path, relative_model_path, relative_image_path):
    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    model = og.Model(model_path)

    processor = model.create_multimodal_processor()

    image_path = os.fspath(Path(test_data_path) / relative_image_path)
    images = og.Images.open(image_path)

    prompt = "<|user|>\n<|image_1|>\n Can you convert the table to markdown format?\n<|end|>\n<|assistant|>\n"
    _ = processor(prompt, images=images)


@pytest.mark.parametrize("relative_model_path", [Path("vision-preprocessing")])
@pytest.mark.parametrize("relative_image_path", [Path("images") / "sheet.png"])
def test_vision_preprocessing_load_image_from_bytes(
    test_data_path, relative_model_path, relative_image_path
):
    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    model = og.Model(model_path)

    processor = model.create_multimodal_processor()

    image_path = os.fspath(Path(test_data_path) / relative_image_path)
    images = None
    with open(image_path, "rb") as image:
        bytes = image.read()
        images = og.Images.open_bytes(bytes)

    prompt = "<|user|>\n<|image_1|>\n Can you convert the table to markdown format?\n<|end|>\n<|assistant|>\n"
    _ = processor(prompt, images=images)


@pytest.mark.parametrize("relative_model_path", [Path("vision-preprocessing")])
@pytest.mark.parametrize(
    "relative_image_paths",
    [[Path("images") / "australia.jpg", Path("images") / "sheet.png"]],
)
def test_vision_preprocessing_multiple_images(
    test_data_path, relative_model_path, relative_image_paths
):
    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    model = og.Model(model_path)

    processor = model.create_multimodal_processor()

    image_paths = [
        os.fspath(Path(test_data_path) / relative_image_path)
        for relative_image_path in relative_image_paths
    ]
    images = og.Images.open(*image_paths)

    prompt = "<|user|>\n"
    for i in range(len(relative_image_paths)):
        prompt += f"<|image_{i+1}|>\n"

    prompt += " What is shown in this two images?\n<|end|>\n<|assistant|>\n"
    _ = processor(prompt, images=images)
