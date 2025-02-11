# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path
import shutil
import onnxruntime

import numpy as np
import onnxruntime_genai as og
import pytest

if not sysconfig.get_platform().endswith("arm64"):
    # Skip importing onnx if running on ARM64
    import onnx

devices = ["cpu"]

if og.is_cuda_available():
    devices.append("cuda")

if og.is_dml_available():
    devices.append("dml")

if og.is_rocm_available():
    devices.append("rocm")

def test_config(test_data_path):
    model_path = os.fspath(Path(test_data_path) / "hf-internal-testing" / "tiny-random-gpt2-fp32")
    config = og.Config(model_path)
    config.clear_providers()
    config.append_provider("cuda")
    config.clear_providers()
    config.set_provider_option("cuda", "infinite_clock", "1")
    config.set_provider_option("quantum", "break_universe", "true")
    config.append_provider("slide rule")

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

    config = og.Config(model_path) # Test using config vs path directly
    model = og.Model(config)

    search_params = og.GeneratorParams(model)
    input_ids_shape = [2, 4]
    batch_size = input_ids_shape[0]
    search_params = og.GeneratorParams(model)
    search_params.set_search_options(do_sample=False, max_length=10, batch_size=batch_size)

    generator = og.Generator(model, search_params)
    generator.append_tokens(np.array([[0, 0, 0, 52], [0, 0, 195, 731]], dtype=np.int32))
    while not generator.is_done():
        # Test getting/setting logits
        logits = generator.get_logits()
        generator.set_logits(logits)
        generator.set_logits(logits) # twice just to be sure buffer is still valid

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
def test_rewind_cuda(test_data_path, relative_model_path):
    model_path = os.fspath(Path(test_data_path) / relative_model_path)

    model = og.Model(model_path)

    # Batch size 1 (continuous decoding) case
    input_ids_shape = [1, 4]
    batch_size = input_ids_shape[0]
    search_params = og.GeneratorParams(model)
    search_params.set_search_options(do_sample=False, max_length=10, batch_size=batch_size)

    generator = og.Generator(model, search_params)
    generator.append_tokens(np.array([[0, 0, 195, 731]], dtype=np.int32))
    while not generator.is_done():
        generator.generate_next_token()

    assert generator.get_sequence(0) is not None

    generator.rewind_to(3)

    generator.append_tokens(np.array([[731, 731]], dtype=np.int32))
    while not generator.is_done():
        generator.generate_next_token()
    
    assert generator.get_sequence(0) is not None

    # Batch size > 1 case
    input_ids_shape = [3, 4]
    batch_size = input_ids_shape[0]
    search_params = og.GeneratorParams(model)
    search_params.set_search_options(do_sample=False, max_length=10, batch_size=batch_size)

    generator = og.Generator(model, search_params)
    generator.append_tokens(np.array([[0, 0, 0, 52], [0, 0, 195, 731], [64, 65, 66, 67]], dtype=np.int32))
    while not generator.is_done():
        generator.generate_next_token()
    
    for i in range(batch_size):
        assert generator.get_sequence(i) is not None
    
    generator.rewind_to(0)

    generator.append_tokens(np.array([[52, 204, 204, 204], [731, 731, 114, 114], [67, 68, 69, 70]], dtype=np.int32))
    while not generator.is_done():
        generator.generate_next_token()

    for i in range(batch_size):
        assert generator.get_sequence(i) is not None


@pytest.mark.parametrize(
    "relative_model_path",
    (
        [Path("hf-internal-testing") / "tiny-random-gpt2-fp32"]
    ),
)
def test_rewind(test_data_path, relative_model_path):
    model_path = os.fspath(Path(test_data_path) / relative_model_path)

    model = og.Model(model_path)

    expected_sequence = np.array(
        [0, 0, 195, 731, 731, 114, 114, 114, 114, 114],
        dtype=np.int32,
    )
    
    input_ids_shape = [1, 4]
    batch_size = input_ids_shape[0]
    search_params = og.GeneratorParams(model)
    search_params.set_search_options(do_sample=False, max_length=10, batch_size=batch_size)

    generator = og.Generator(model, search_params)
    generator.append_tokens(np.array([[0, 0, 195, 731]], dtype=np.int32))
    while not generator.is_done():
        generator.generate_next_token()

    assert np.array_equal(expected_sequence, generator.get_sequence(0))

    generator.rewind_to(3)

    generator.append_tokens(np.array([[731, 731]], dtype=np.int32))
    while not generator.is_done():
        generator.generate_next_token()
    
    assert np.array_equal(expected_sequence, generator.get_sequence(0))
    

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
    params.set_search_options(max_length=20, batch_size=len(prompts))  # To run faster

    if device == "dml":
        params.try_graph_capture_with_max_batch_size(len(prompts))

    generator = og.Generator(model, params)
    generator.append_tokens(tokenizer.encode_batch(prompts))
    while not generator.is_done():
        generator.generate_next_token()
    for i in range(len(prompts)):
        print(tokenizer.decode(generator.get_sequence(0)))


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
    input_ids = np.array(
        [[0, 0, 0, 52], [0, 0, 195, 731]], dtype=np.int32
    )
    search_params.set_search_options(do_sample=False, max_length=10, batch_size=input_ids.shape[0])

    generator = og.Generator(model, search_params)
    generator.append_tokens(input_ids)

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
    logits = generator.get_output("logits")
    assert np.allclose(logits[:, :, ::200], expected_sampled_logits_prompt, atol=1e-3)
    generator.generate_next_token()
    generator.generate_next_token()

    # check for the 1st token generation
    # full logits has shape [2, 1, 1000]. Sample 1 for every 200 tokens and the expected sampled logits has shape [2, 1, 5]
    expected_sampled_logits_token_gen = np.array(
        [
            [[0.03742531, -0.05752287, 0.14159015, 0.04210977, -0.1484456]],
            [[0.3041716, -0.08701379, -0.03778192, 0.07471392, -0.02049096]],
        ]
    )
    logits = generator.get_output("logits")
    assert np.allclose(
        logits[:, :, ::200], expected_sampled_logits_token_gen, atol=1e-3
    )

@pytest.mark.skipif(
    sysconfig.get_platform().endswith("arm64") or sys.version_info.minor < 8,
    reason="Python 3.8 is required for downloading models.",
)
@pytest.mark.parametrize(
    "relative_model_path",
    (
        [
            Path("qwen/int4/cpu"),
            Path("qwen/int4/cuda"),
        ]
        if og.is_cuda_available()
        else [
            Path("qwen/int4/cpu"),
        ]
    ),
)
def test_hidden_states(test_data_path, relative_model_path):
    model_path = os.fspath(Path(test_data_path) / relative_model_path)

    model = og.Model(model_path)

    search_params = og.GeneratorParams(model)
    input_ids = np.array(
        [[0, 0, 0, 52], [0, 0, 195, 731]], dtype=np.int32
    )
    search_params.set_search_options(do_sample=False, max_length=10, batch_size=input_ids.shape[0])

    generator = og.Generator(model, search_params)
    generator.append_tokens(input_ids)
    generator.generate_next_token()
    hidden_states = generator.get_output("hidden_states")
    assert hidden_states.shape == (2, 4, 896)
    generator.generate_next_token()
    hidden_states = generator.get_output("hidden_states")
    assert hidden_states.shape == (2, 1, 896)

@pytest.mark.skipif(
    not og.is_cuda_available(), reason="Pipeline model uses a mix of CPU and CUDA EP."
)
@pytest.mark.parametrize("relative_model_path", [Path("pipeline-model")])
def test_pipeline_model(test_data_path, phi2_for, relative_model_path):
    def _extract_subgraph(
        input_path: os.PathLike,
        output_path: os.PathLike,
        input_names: list[str],
        output_names: list[str],
    ):
        """Extract a subgraph from the input model and save it to the output path"""

        model = onnx.load(input_path)

        e = onnx.utils.Extractor(model)
        extracted = e.extract_model(input_names, output_names)

        onnx.save(
            extracted,
            output_path,
            save_as_external_data=True,
            location=f"{Path(output_path).name}.data",
        )

    def _split(onnx_model_path: os.PathLike, output_dir: os.PathLike):
        """Split the model into three models: embedding model, transformer model, and lm_head model."""
        num_layers = 1
        inputs_and_outputs = [
            (["input_ids"], ["/model/embed_tokens/Gather/output_0"]),
            (
                ["/model/embed_tokens/Gather/output_0", "attention_mask"]
                + [
                    f"past_key_values.{i}.{kv}"
                    for kv in ["key", "value"]
                    for i in range(num_layers)
                ],
                ["hidden_states"]
                + [
                    f"present.{i}.{kv}"
                    for kv in ["key", "value"]
                    for i in range(num_layers)
                ],
            ),
            ([f"hidden_states"], ["logits"]),
        ]

        for i, split_name in enumerate(["embeds", "transformer", "lm_head"]):
            split_model_path = output_dir / f"{split_name}.onnx"
            _extract_subgraph(
                onnx_model_path,
                split_model_path,
                inputs_and_outputs[i][0],
                inputs_and_outputs[i][1],
            )

    _split(
        Path(phi2_for("cuda")) / "model.onnx",
        Path(test_data_path) / relative_model_path,
    )

    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    model = og.Model(model_path)
    tokenizer = og.Tokenizer(model)

    prompts = [
        "This is a test.",
        "Rats are awesome pets!",
        "The quick brown fox jumps over the lazy dog.",
    ]

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=20, batch_size=len(prompts))

    generator = og.Generator(model, params)
    generator.append_tokens(tokenizer.encode_batch(prompts))
    while not generator.is_done():
        generator.generate_next_token()

    expected_output = [
        'This is a test.\n        # TOD import * doct proofingrad',
        'Rats are awesome pets!\n    """\n\n',
        'The quick brown fox jumps over the lazy dog.\n    """\n\n',
    ]
    for i in range(len(prompts)):
        actual_output = tokenizer.decode(generator.get_sequence(i))
        equal = np.array_equal(expected_output[i], actual_output)

        if not equal:
            print("test_pipeline_model:", flush=True)
            print(f"expected = {repr(expected_output[i])}", flush=True)
            print(f"actual = {repr(actual_output)}", flush=True)
        assert equal

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


@pytest.mark.parametrize("device", devices)
@pytest.mark.skipif(
    sysconfig.get_platform().endswith("arm64"),
    reason="ONNX is not available on ARM64",
)
@pytest.mark.parametrize("multiple_adapters", [True, False])
def test_adapters(test_data_path, device, multiple_adapters, phi2_for):
    def _prepare_adapter_model(test_data_path):
        phi2_model_path = phi2_for(device)
        relative_model_path = "multiple_adapters" if multiple_adapters else "adapters"
        adapter_model_path = os.fspath(Path(test_data_path) / relative_model_path)
        if os.path.exists(adapter_model_path):
            shutil.rmtree(adapter_model_path)

        shutil.copytree(phi2_model_path, adapter_model_path)

        # Create the model with adapters
        model = onnx.load(Path(adapter_model_path) / "model.onnx")

        for node in model.graph.node:
            if node.output[0] == "logits":
                node.output[0] = "logits_0"
                break

        vocab_size = 51200
        adapter_a = onnx.helper.make_tensor_value_info(
            "adapter_a",
            onnx.TensorProto.FLOAT if device == "cpu" else onnx.TensorProto.FLOAT16,
            [vocab_size],
        )
        adapter_b = onnx.helper.make_tensor_value_info(
            "adapter_b",
            onnx.TensorProto.FLOAT if device == "cpu" else onnx.TensorProto.FLOAT16,
            [vocab_size],
        )

        model.graph.input.extend([adapter_a, adapter_b])

        for adapter_name in ["adapter_a", "adapter_b"]:
            adapter_weight = np.zeros([vocab_size], dtype=(np.float32 if device == "cpu" else np.float16))
            adapter_weight_tensor = onnx.helper.make_tensor(
                adapter_name,
                onnx.TensorProto.FLOAT if device == "cpu" else onnx.TensorProto.FLOAT16,
                [vocab_size],
                adapter_weight.flatten()
            )
            model.graph.initializer.append(adapter_weight_tensor)

        add_node = onnx.helper.make_node(
            "Add", ["adapter_a", "adapter_b"], ["adapter_output"], name="adapter_add"
        )
        add_to_logits_node = onnx.helper.make_node(
            "Add", ["adapter_output", "logits_0"], ["logits"], name="add_to_logits"
        )
        model.graph.node.extend([add_node, add_to_logits_node])

        onnx.save(
            model,
            Path(adapter_model_path) / "model.onnx",
            save_as_external_data=True,
            location="model.data",
        )

        # Create adapters for the model
        a, b = None, None
        if device == "cpu":
            a = np.random.rand(vocab_size).astype(np.float32)
            b = np.random.rand(vocab_size).astype(np.float32)
        else:
            a = np.random.rand(vocab_size).astype(np.float16)
            b = np.random.rand(vocab_size).astype(np.float16)

        onnx_dtype = 1 if device == "cpu" else 10
        adapters = {
            "adapter_a": onnxruntime.OrtValue.ortvalue_from_numpy_with_onnx_type(
                a, onnx_dtype
            ),
            "adapter_b": onnxruntime.OrtValue.ortvalue_from_numpy_with_onnx_type(
                b, onnx_dtype
            ),
        }
        if multiple_adapters:
            adapters = [{key: value} for key, value in adapters.items()]

        def _export_adapter(adapter, adapter_file_name):
            adapter_format = onnxruntime.AdapterFormat()
            adapter_format.set_adapter_version(1)
            adapter_format.set_model_version(1)
            adapter_format.set_parameters(adapter)
            adapter_format.export_adapter(adapter_file_name)

        adapter_paths = []
        if multiple_adapters:
            for i, adapter in enumerate(adapters):
                adapter_file_name = str(Path(adapter_model_path) / f"adapter_{i}.onnx_adapter")
                _export_adapter(adapter, adapter_file_name)
                adapter_paths.append(adapter_file_name)
        else:
            adapter_file_name = str(Path(adapter_model_path) / "adapters.onnx_adapter")
            _export_adapter(adapters, adapter_file_name)
            adapter_paths.append(adapter_file_name)

        return adapter_model_path, adapter_paths

    model_path, adapter_paths = _prepare_adapter_model(test_data_path)
    model = og.Model(model_path)
    adapters = og.Adapters(model)
    for i, adapter_path in enumerate(adapter_paths):
        adapters.load(adapter_path, f"adapter_{i}")

    tokenizer = og.Tokenizer(model)
    prompts = [
        "This is a test.",
        "Rats are awesome pets!",
        "The quick brown fox jumps over the lazy dog.",
    ]

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=20, batch_size=len(prompts))

    generator = og.Generator(model, params)
    for i in range(len(adapter_paths)):
        generator.set_active_adapter(adapters, f"adapter_{i}")
        
    generator.append_tokens(tokenizer.encode_batch(prompts))
    while not generator.is_done():
        generator.generate_next_token()


@pytest.mark.parametrize("device", devices)
@pytest.mark.skipif(
    sysconfig.get_platform().endswith("arm64"),
    reason="ONNX is not available on ARM64",
)
@pytest.mark.parametrize("extra_inputs", [("num_logits_to_keep", True), ("onnx::Neg_67", True), ("abcde", False)])
def test_preset_extra_inputs(test_data_path, device, phi2_for, extra_inputs):
    def _prepare_model(test_data_path):
        phi2_model_path = phi2_for(device)
        relative_model_path = "preset_extra_inputs"
        extra_inputs_model_path = os.fspath(Path(test_data_path) / relative_model_path)

        shutil.copytree(phi2_model_path, extra_inputs_model_path, dirs_exist_ok=True)

        # Create the model with the extra inputs
        model = onnx.load(Path(extra_inputs_model_path) / "model.onnx")

        for node in model.graph.node:
            if node.output[0] == "logits":
                node.output[0] = "logits_0"
                break

        extra_input_name, valid = extra_inputs
        extra_input = onnx.helper.make_tensor_value_info(
            extra_input_name,
            onnx.TensorProto.INT64,
            [],
        )

        model.graph.input.append(extra_input)

        cast_node = onnx.helper.make_node(
            "Cast", [extra_input_name], [f"{extra_input_name}_cast"], to=onnx.TensorProto.FLOAT if device == "cpu" else onnx.TensorProto.FLOAT16
        )
        add_node = onnx.helper.make_node(
            "Add", [f"{extra_input_name}_cast", "logits_0"], ["logits"], name="add_to_logits"
        )
        model.graph.node.extend([cast_node, add_node])

        onnx.save(
            model,
            Path(extra_inputs_model_path) / "model.onnx",
            save_as_external_data=True,
            location="model.data",
        )

        return extra_inputs_model_path, valid

    model_path, valid_model = _prepare_model(test_data_path)
    model = og.Model(model_path)
    tokenizer = og.Tokenizer(model)
    prompts = [
        "This is a test.",
        "Rats are awesome pets!",
        "The quick brown fox jumps over the lazy dog.",
    ]

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=20, batch_size=len(prompts))

    generator = og.Generator(model, params)
    if not valid_model:
        with pytest.raises(og.OrtException) as exc_info:
            generator.append_tokens(tokenizer.encode_batch(prompts))

        assert f"Missing Input: {extra_inputs[0]}" in str(exc_info.value)
    else:
        generator.append_tokens(tokenizer.encode_batch(prompts))

        while not generator.is_done():
            generator.generate_next_token()
