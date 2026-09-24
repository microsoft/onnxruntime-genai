import json
import os
import runpy
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import onnx
import onnxruntime_genai as og
import pytest
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, models, processors

DEFAULT_TASK = "</s><s><predict_bbox><predict_classes><output_markdown>"
CONTEXT_LENGTH = 32
DEFAULT_NORMALIZATION = {
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
}


def _save_graph(path, nodes, inputs, outputs, initializers=()):
    model = helper.make_model(
        helper.make_graph(nodes, path.stem, inputs, outputs, list(initializers)),
        opset_imports=[helper.make_opsetid("", 24)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


@pytest.fixture
def model_factory(tmp_path):
    def create(prefill_length=8, fixed_length=None, provider="cpu", fail_decoder=False,
               decoder_run_options=None, invalid_cache=False, cache_dtype=np.float32,
               default_user_prompt=DEFAULT_TASK, normalization=None,
               vision_config_filename="vision_processing.json", add_bos_token=True, add_eos_token=True,
               profile_keys=("min", "opt", "max")):
        cache_type = helper.np_dtype_to_tensor_dtype(np.dtype(cache_dtype))
        model_dir = tmp_path / f"model_{prefill_length}_{fixed_length}"
        model_dir.mkdir()
        config_path = Path(__file__).parents[2] / "configs/nemotron-parse/genai_config.json"
        config = json.loads(config_path.read_text())
        config["model"].update(context_length=CONTEXT_LENGTH, vocab_size=32)
        if default_user_prompt is None:
            config["model"].pop("default_user_prompt", None)
        else:
            config["model"]["default_user_prompt"] = default_user_prompt
        config["model"]["vision"]["num_visual_tokens"] = 1
        config["model"]["vision"]["config_filename"] = vision_config_filename
        config["model"]["decoder"].update(
            prefill_sequence_length=prefill_length,
            hidden_size=1,
            head_size=1,
            num_attention_heads=1,
            num_hidden_layers=1,
            num_key_value_heads=1,
            session_options={"intra_op_num_threads": 1},
        )
        config["search"].update(max_length=CONTEXT_LENGTH, do_sample=False)
        if provider != "cpu":
            config["model"]["decoder"]["session_options"].update({
                "provider_options": [{provider: {}}],
                "session.disable_cpu_ep_fallback": "1",
            })
        if provider == "NvTensorRtRtx":
            config["model"]["vision"]["session_options"] = {
                "provider_options": [{provider: {}}],
                "session.disable_cpu_ep_fallback": "1",
            }
            for kind in profile_keys:
                key = f"ep.nvtensorrtrtxexecutionprovider.nv_profile_{kind}_shapes"
                config["model"]["decoder"]["session_options"][key] = "decoder_input_ids:1x1"
                config["model"]["vision"]["session_options"][key] = "pixel_values:1x3x2x2"
        if decoder_run_options:
            config["model"]["decoder"]["run_options"] = decoder_run_options
        (model_dir / "genai_config.json").write_text(json.dumps(config))
        (model_dir / vision_config_filename).write_text(
            json.dumps({
                "image_height": 2,
                "image_width": 2,
                **(DEFAULT_NORMALIZATION if normalization is None else normalization),
                "processor": {
                    "name": "nemotron_parse_image_processor",
                    "transforms": [{
                        "operation": {
                            "name": "decode_image",
                            "type": "DecodeImage",
                            "attrs": {"color_space": "RGB"},
                        }
                    }],
                },
            })
        )

        special_tokens = [
            "<s>", "<pad>", "</s>", "<unk>",
            "<predict_bbox>", "<predict_classes>", "<output_markdown>",
        ]
        vocab = {token: index for index, token in enumerate(special_tokens + ["x"])}
        tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
        tokenizer.add_special_tokens(special_tokens)
        template = (["<s>"] if add_bos_token else []) + ["$A"] + (["</s>"] if add_eos_token else [])
        tokenizer.post_processor = processors.TemplateProcessing(
            single=" ".join(template), special_tokens=[(token, vocab[token]) for token in template if token != "$A"],
        )
        tokenizer.save(str(model_dir / "tokenizer.json"))
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
            })
        )

        cross_shape = [1, 1, 1, 1]
        cross_outputs = [
            helper.make_tensor_value_info(f"cross_present.0.{kind}", cache_type, cross_shape)
            for kind in ("key", "value")
        ]
        _save_graph(
            model_dir / "encoder.onnx",
            [helper.make_node("Identity", ["cross"], [output.name]) for output in cross_outputs],
            [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, 2, 2])],
            cross_outputs,
            [numpy_helper.from_array(np.zeros(cross_shape, dtype=cache_dtype), "cross")],
        )

        cache_shape = [1, 1, "invalid_cache" if invalid_cache else CONTEXT_LENGTH, 1]
        inputs = [
            helper.make_tensor_value_info(
                "decoder_input_ids", TensorProto.INT64, [1, fixed_length or "sequence_length"]
            ),
            helper.make_tensor_value_info("decoder_attention_mask", TensorProto.INT64, [1, CONTEXT_LENGTH]),
            helper.make_tensor_value_info("cache_write_indices", TensorProto.INT64, [1]),
        ]
        outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, 32])]
        nodes = [
            helper.make_node("Cast", ["decoder_input_ids"], ["tokens_float"], to=cache_type),
            helper.make_node("Unsqueeze", ["tokens_float", "cache_axes"], ["updates"]),
        ]
        if fail_decoder:
            nodes[0] = helper.make_node("Gather", ["values", "decoder_input_ids"], ["tokens_float"])
        for kind in ("key", "value"):
            inputs.extend([
                helper.make_tensor_value_info(f"past_key_values.0.{kind}", cache_type, cache_shape),
                helper.make_tensor_value_info(f"cross_past_key_values.0.{kind}", cache_type, cross_shape),
            ])
            outputs.append(helper.make_tensor_value_info(f"present.0.{kind}", cache_type, cache_shape))
            nodes.append(helper.make_node(
                "TensorScatter", [f"past_key_values.0.{kind}", "updates", "cache_write_indices"],
                [f"present.0.{kind}"], axis=2,
            ))

        nodes.extend([
            helper.make_node("ReduceSum", ["decoder_attention_mask", "axes"], ["length"], keepdims=0),
            helper.make_node("Add", ["length", "cache_write_indices"], ["next_id"]),
            helper.make_node("Mod", ["next_id", "vocab_size"], ["token_id"]),
            helper.make_node("OneHot", ["token_id", "vocab_size", "values"], ["scores"]),
            helper.make_node("Unsqueeze", ["scores", "axes"], ["logits"]),
        ])
        _save_graph(
            model_dir / "decoder.onnx", nodes, inputs, outputs,
            [
                numpy_helper.from_array(np.array([1], dtype=np.int64), "axes"),
                numpy_helper.from_array(np.array([1, 3], dtype=np.int64), "cache_axes"),
                numpy_helper.from_array(np.array(32, dtype=np.int64), "vocab_size"),
                numpy_helper.from_array(np.array([0, 1], dtype=np.float32), "values"),
            ],
        )
        return og.Model(str(model_dir))

    return create


@pytest.fixture
def images():
    image_path = Path(__file__).parents[2] / "images/10809054.jpg"
    return og.Images.open(str(image_path))


@pytest.fixture(scope="module")
def trt_rtx_provider():
    if os.environ.get("NEMOTRON_PARSE_REQUIRE_TRT_RTX") != "1":
        pytest.skip("Set NEMOTRON_PARSE_REQUIRE_TRT_RTX=1 on a TRT-RTX test host")
    library = os.environ.get("ORT_TRT_RTX_EP_LIBRARY")
    if library:
        og.register_execution_provider_library("NvTensorRTRTXExecutionProvider", library)
    return "NvTensorRtRtx"


@pytest.mark.parametrize("missing", ["min", "opt", "max"])
def test_trt_rtx_rejects_partial_explicit_profiles(model_factory, trt_rtx_provider, missing):
    with pytest.raises(RuntimeError, match="Explicit TRT-RTX profiles must specify.*together"):
        model_factory(provider=trt_rtx_provider, profile_keys=tuple(kind for kind in ("min", "opt", "max") if kind != missing))


@pytest.mark.parametrize("prefill_length", [8, 16])
def test_trt_rtx_static_and_dynamic_prefill(model_factory, images, trt_rtx_provider, prefill_length):
    model = model_factory(prefill_length=prefill_length, provider=trt_rtx_provider)
    processor = model.create_multimodal_processor()
    # Alternate paths on one resident model, including the default task when the
    # fast-path length is not eight and the maximum supported prompt length.
    for prompt in ("", "x", "x" * (prefill_length - 3), "x" * 17, "x" * 28, ""):
        inputs = processor(prompt, images=images)
        prompt_ids = inputs["input_ids"].as_numpy()[0]
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=min(len(prompt_ids) + 3, CONTEXT_LENGTH))
        generator = og.Generator(model, params)
        generator.set_inputs(inputs)
        while not generator.is_done():
            generator.generate_next_token()
            consumed = generator.get_sequence(0)[:-1]
            expected = np.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=np.float32)
            expected[0, 0, :len(consumed), 0] = consumed
            for kind in ("key", "value"):
                np.testing.assert_array_equal(generator.get_output(f"present.0.{kind}"), expected)
                np.testing.assert_array_equal(generator.get_input(f"past_key_values.0.{kind}"), expected)
            next_id = len(consumed) + (0 if len(consumed) == len(prompt_ids) else len(consumed) - 1)
            assert generator.get_sequence(0)[-1] == next_id % 32
        del generator

    # The raw API can reach length one; text processing adds special tokens.
    inputs = og.NamedTensors()
    inputs["input_ids"] = og.Tensor(np.array([[7]], dtype=np.int32))
    inputs["pixel_values"] = og.Tensor(np.zeros((1, 3, 2, 2), dtype=np.float32))
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=3)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)
    while not generator.is_done():
        generator.generate_next_token()
    np.testing.assert_array_equal(generator.get_sequence(0), [7, 1, 3])
    del generator

    for length in (0, CONTEXT_LENGTH, CONTEXT_LENGTH + 1):
        if length:
            with pytest.raises(RuntimeError, match=rf"prompt has {length} tokens.*context_length={CONTEXT_LENGTH}"):
                processor("x" * (length - 3), images=images)
        inputs["input_ids"] = og.Tensor(np.full((1, length), 7, dtype=np.int32))
        with pytest.raises(RuntimeError):
            generator = og.Generator(model, og.GeneratorParams(model))
            generator.set_inputs(inputs)
            generator.generate_next_token()


@pytest.mark.parametrize("prefill_length", [8, 16])
@pytest.mark.parametrize(
    "prompt,expected_ids",
    [
        ("", [2, 0, 2, 0, 4, 5, 6, 2]),
        (DEFAULT_TASK, [2, 0, 2, 0, 4, 5, 6, 2]),
        ("x", [2, 0, 7, 2]),
        ("x" * 17, [2, 0] + [7] * 17 + [2]),
    ],
)
def test_dynamic_prompt_prefill_and_decode(model_factory, images, prefill_length, prompt, expected_ids):
    model = model_factory(prefill_length=prefill_length)
    processor = model.create_multimodal_processor()
    inputs = processor(prompt, images=images)
    np.testing.assert_array_equal(inputs["input_ids"].as_numpy(), [expected_ids])

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(expected_ids) + 2)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)
    while not generator.is_done():
        generator.generate_next_token()
    expected_new_tokens = [len(expected_ids), (2 * len(expected_ids) + 1) % 32]
    np.testing.assert_array_equal(generator.get_sequence(0), expected_ids + expected_new_tokens)


@pytest.mark.parametrize("add_bos_token,add_eos_token", [(True, True), (False, True), (True, False), (False, False)])
@pytest.mark.parametrize("prompt", ["", DEFAULT_TASK, "x", "x" * 17])
def test_processor_uses_tokenizer_special_tokens(model_factory, tmp_path, images, prompt, add_bos_token, add_eos_token):
    model = model_factory(add_bos_token=add_bos_token, add_eos_token=add_eos_token)
    reference = Tokenizer.from_file(str(tmp_path / "model_8_None" / "tokenizer.json"))
    text = prompt or DEFAULT_TASK
    expected = [2] + reference.encode(text, add_special_tokens=True).ids
    standalone = og.Tokenizer(model)
    raw = reference.encode(text, add_special_tokens=False).ids
    np.testing.assert_array_equal(standalone.encode(text), raw)

    processor = model.create_multimodal_processor()
    for _ in range(2):
        inputs = processor(prompt, images=images)
        np.testing.assert_array_equal(inputs["input_ids"].as_numpy(), [expected])
    # Configuring the processor must not change independently created tokenizers.
    np.testing.assert_array_equal(standalone.encode(text), raw)


@pytest.mark.parametrize(
    "prefill_length,prompt,prompt_length",
    [
        (8, "", 8),
        (8, DEFAULT_TASK, 8),
        (8, "x", 4),
        (8, "x" * 17, 20),
        (16, "", 8),
        (16, "x", 4),
        (16, "x" * 13, 16),
        (16, "x" * 17, 20),
    ],
)
def test_fixed_prompt_length_checked_by_processor(model_factory, images, prefill_length, prompt, prompt_length):
    model = model_factory(prefill_length=prefill_length, fixed_length=prefill_length)
    processor = model.create_multimodal_processor()
    if prompt_length == prefill_length:
        assert processor(prompt, images=images)["input_ids"].as_numpy().shape == (1, prefill_length)
    else:
        with pytest.raises(RuntimeError, match=rf"prompt has {prompt_length} tokens.*requires exactly {prefill_length}"):
            processor(prompt, images=images)


@pytest.mark.parametrize("prompt_length", [CONTEXT_LENGTH, CONTEXT_LENGTH + 1])
def test_processor_rejects_prompt_without_generation_capacity(model_factory, images, prompt_length):
    model = model_factory()
    processor = model.create_multimodal_processor()
    with pytest.raises(RuntimeError, match=rf"prompt has {prompt_length} tokens.*context_length={CONTEXT_LENGTH}"):
        processor("x" * (prompt_length - 3), images=images)


def test_runtime_rejects_fixed_prompt_when_processor_is_bypassed(model_factory):
    model = model_factory(fixed_length=8)
    params = og.GeneratorParams(model)
    inputs = og.NamedTensors()
    inputs["input_ids"] = og.Tensor(np.array([[2, 0, 7, 2]], dtype=np.int32))
    inputs["pixel_values"] = og.Tensor(np.zeros((1, 3, 2, 2), dtype=np.float32))
    with pytest.raises(RuntimeError, match="prompt has 4 tokens.*requires exactly 8"):
        generator = og.Generator(model, params)
        generator.set_inputs(inputs)
        generator.generate_next_token()


@pytest.mark.parametrize("prompt", ["", "x", DEFAULT_TASK])
def test_single_prompt_list_matches_string(model_factory, images, prompt):
    processor = model_factory().create_multimodal_processor()
    from_list = processor([prompt], images=images)
    from_string = processor(prompt, images=images)
    for name in ("input_ids", "pixel_values"):
        np.testing.assert_array_equal(from_list[name].as_numpy(), from_string[name].as_numpy())


@pytest.mark.parametrize("prompt", ["", [""], []])
@pytest.mark.parametrize("default_prompt", [DEFAULT_TASK, "x"])
def test_empty_prompt_forms_select_default_task(model_factory, images, prompt, default_prompt):
    processor = model_factory(default_user_prompt=default_prompt).create_multimodal_processor()
    actual = processor(prompt, images=images)
    expected = processor(default_prompt, images=images)
    for name in ("input_ids", "pixel_values"):
        np.testing.assert_array_equal(actual[name].as_numpy(), expected[name].as_numpy())


@pytest.mark.parametrize("prompts", [["x", DEFAULT_TASK], ["", ""]])
def test_processor_rejects_multiple_prompts(model_factory, images, prompts):
    processor = model_factory().create_multimodal_processor()
    with pytest.raises(RuntimeError, match="does not support multiple prompts"):
        processor(prompts, images=images)


@pytest.fixture
def example_common(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "examples/python"))
    import common
    return common


@pytest.mark.parametrize("configured", [None, DEFAULT_TASK, "", "custom"])
def test_example_package_prompt_default(tmp_path, example_common, configured):
    model = {} if configured is None else {"default_user_prompt": configured}
    (tmp_path / "genai_config.json").write_text(json.dumps({"model": model}))
    fallback = "What color is the sky?"
    assert example_common.get_default_user_prompt(str(tmp_path), fallback) == (
        fallback if configured is None else configured
    )


@pytest.mark.parametrize("configured", [None, 42, [], {}])
def test_example_rejects_invalid_prompt_default(tmp_path, example_common, configured):
    (tmp_path / "genai_config.json").write_text(json.dumps({"model": {"default_user_prompt": configured}}))
    with pytest.raises(ValueError, match="must be a string"):
        example_common.get_default_user_prompt(str(tmp_path), "fallback")


@pytest.mark.parametrize("prompt", ["custom", "What color is the sky?", DEFAULT_TASK])
def test_example_preserves_explicit_prompt(example_common, prompt):
    assert example_common.get_user_prompt(prompt, non_interactive=True) == prompt


@pytest.mark.parametrize("default_prompt", [DEFAULT_TASK, "What color is the sky?"])
def test_example_interactive_empty_input_reprompts(example_common, monkeypatch, capsys, default_prompt):
    answers = iter(["", "", "custom"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert example_common.get_user_prompt(default_prompt, non_interactive=False) == "custom"
    assert capsys.readouterr().out.count("Error, input cannot be empty") == 2


@pytest.mark.parametrize("configured", [None, DEFAULT_TASK, "x", ""])
def test_processor_uses_package_prompt_default(model_factory, images, configured):
    model = model_factory(default_user_prompt=configured)
    actual = model.create_multimodal_processor()("", images=images)["input_ids"].as_numpy()
    task = DEFAULT_TASK if configured is None else configured
    expected = [2, 0, *og.Tokenizer(model).encode(task).tolist(), 2]
    np.testing.assert_array_equal(actual, [expected])


@pytest.mark.parametrize("model_type", ["nemotron_parse", "gemma3"])
@pytest.mark.parametrize("package_max, explicit_max, expected", [
    (1032, None, 1032),
    (4096, None, 4096),
    (8192, None, 7680),
    (1032, 512, 512),
    (8192, 8000, 8000),
    (1032, 2048, 2048),
])
def test_multimodal_example_max_length(
    example_common, monkeypatch, model_type, package_max, explicit_max, expected
):
    main = runpy.run_path(str(Path(__file__).parents[3] / "examples/python/model-mm.py"))["main"]
    runtime = Mock()
    runtime.Model.return_value.type = model_type
    params = runtime.GeneratorParams.return_value
    params.get_search_options.return_value = {"max_length": package_max}
    runtime.Generator.return_value.is_done.return_value = True
    monkeypatch.setitem(main.__globals__, "og", runtime)
    for name, result in {
        "register_ep": None,
        "get_config": None,
        "get_user_images": (None, 0),
        "get_user_audios": (None, 0),
        "apply_chat_template": DEFAULT_TASK,
    }.items():
        monkeypatch.setitem(main.__globals__, name, Mock(return_value=result))
    args = Namespace(
        model_path="unused", execution_provider="cpu", ep_path=None, use_winml=False,
        user_prompt=DEFAULT_TASK, system_prompt="system", debug=False, verbose=False,
        response_format="", timings=False, non_interactive=True, image_paths=[], audio_paths=[],
    )
    if explicit_max is not None:
        args.max_length = explicit_max
    main(args)
    assert params.set_search_options.call_args.kwargs["max_length"] == expected


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("prompt", ["", DEFAULT_TASK, "x", "What color is the sky?", " x\n"])
def test_exported_chat_template_preserves_task(
    model_factory, images, tmp_path, monkeypatch, example_common, structured, prompt
):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "src/python/py/models"))
    from builders.nemotron_parse import NemotronParseModel

    model = model_factory()
    package_dir = tmp_path / "model_8_None"
    (package_dir / "chat_template.jinja").write_text(NemotronParseModel.chat_template)
    content = example_common.get_user_content("nemotron_parse", 1, 0, prompt) if structured else prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "previous task"},
        {"role": "assistant", "content": "previous result"},
        {"role": "user", "content": content},
    ]
    rendered = example_common.apply_chat_template(str(package_dir), og.Tokenizer(model), json.dumps(messages), True)
    assert rendered == prompt
    processor = model.create_multimodal_processor()
    np.testing.assert_array_equal(
        processor(rendered, images=images)["input_ids"].as_numpy(),
        processor(prompt, images=images)["input_ids"].as_numpy(),
    )


def test_example_preserves_conversational_template(model_factory, tmp_path, example_common):
    model = model_factory(default_user_prompt=None)
    (tmp_path / "chat_template.jinja").write_text(
        "{{ messages[0]['content'] }}|{{ messages[-1]['content'] }}|assistant:"
    )
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "hello"}]
    rendered = example_common.apply_chat_template(str(tmp_path), og.Tokenizer(model), json.dumps(messages), True)
    assert rendered == "system|hello|assistant:"


@pytest.mark.parametrize("rewind_length", [0, 4, 9])
def test_rejected_rewind_preserves_generation(model_factory, images, rewind_length):
    model = model_factory()
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=12)
    generator = og.Generator(model, params)
    generator.set_inputs(model.create_multimodal_processor()("", images=images))
    generator.generate_next_token()
    before = generator.get_sequence(0).copy()
    cache = generator.get_output("present.0.key").copy()
    mask = generator.get_input("decoder_attention_mask").copy()
    with pytest.raises(RuntimeError, match="RewindTo is not supported for nemotron_parse"):
        generator.rewind_to(rewind_length)
    np.testing.assert_array_equal(generator.get_sequence(0), before)
    np.testing.assert_array_equal(generator.get_output("present.0.key"), cache)
    np.testing.assert_array_equal(generator.get_input("decoder_attention_mask"), mask)
    while not generator.is_done():
        generator.generate_next_token()
    np.testing.assert_array_equal(generator.get_sequence(0), [*before, 17, 19, 21])


@pytest.mark.parametrize("stopped", [False, True])
def test_failed_run_cannot_be_retried_or_reenabled(model_factory, images, stopped):
    model = model_factory(
        fail_decoder=not stopped,
        decoder_run_options={"terminate_session": "1"} if stopped else None,
    )
    generator = og.Generator(model, og.GeneratorParams(model))
    with pytest.raises(RuntimeError):
        generator.set_inputs(model.create_multimodal_processor()("", images=images))
        generator.generate_next_token()
    with pytest.raises(RuntimeError, match="unusable.*new generator"):
        generator.set_runtime_option("terminate_session", "0")
    with pytest.raises(RuntimeError, match="[Tt]erminated"):
        generator.generate_next_token()
    with pytest.raises(RuntimeError, match="unusable"):
        generator.get_output("present.0.key")


@pytest.mark.parametrize("search_options", [{"batch_size": 2}, {"num_beams": 2}])
def test_generator_validates_before_cache_construction(model_factory, search_options):
    model = model_factory(invalid_cache=True)
    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    with pytest.raises(RuntimeError, match="supports batch_size=1 and num_beams=1"):
        og.Generator(model, params)


def test_runtime_profiling_reaches_inner_sessions(model_factory, images, tmp_path):
    model = model_factory()
    generator = og.Generator(model, og.GeneratorParams(model))
    prefix = tmp_path / "nemotron_run"
    generator.set_runtime_option("enable_profiling", str(prefix))
    generator.set_inputs(model.create_multimodal_processor()("", images=images))
    generator.generate_next_token()
    generator.generate_next_token()
    generator.set_runtime_option("enable_profiling", "0")
    del generator
    del model
    profiles = list(tmp_path.glob("nemotron_run*.json"))
    assert profiles, "Run-level profiling did not reach either inner session"
    events = [event for path in profiles for event in json.loads(path.read_text())]
    assert any("TensorScatter" in event.get("name", "") for event in events)


@pytest.mark.parametrize("provider", ["cpu", "cuda"])
@pytest.mark.parametrize("cache_dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
def test_in_place_cache_across_multiple_steps(model_factory, images, tmp_path, provider, cache_dtype):
    if provider == "cuda" and not og.is_cuda_available():
        if os.environ.get("NEMOTRON_PARSE_REQUIRE_CUDA") == "1":
            pytest.fail("CUDA coverage was required but CUDA is unavailable")
        pytest.skip("CUDA is unavailable; set NEMOTRON_PARSE_REQUIRE_CUDA=1 to require it")
    model = model_factory(provider=provider, cache_dtype=cache_dtype)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=14)
    generator = og.Generator(model, params)
    generator.set_runtime_option("enable_profiling", str(tmp_path / "cache_run_0"))
    inputs = model.create_multimodal_processor()("", images=images)
    generator.set_inputs(inputs)
    for step in range(6):
        if step:
            # Run-level profiles use timestamps; fast decode steps can otherwise overwrite one another.
            generator.set_runtime_option("enable_profiling", str(tmp_path / f"cache_run_{step}"))
        generator.generate_next_token()
        generator.set_runtime_option("enable_profiling", "0")
        consumed = generator.get_sequence(0)[:-1]
        expected_mask = np.zeros((1, CONTEXT_LENGTH), dtype=np.int64)
        expected_mask[:, :len(consumed)] = 1
        np.testing.assert_array_equal(generator.get_input("decoder_attention_mask"), expected_mask)
        expected = np.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=cache_dtype)
        expected[0, 0, :len(consumed), 0] = consumed
        for kind in ("key", "value"):
            np.testing.assert_array_equal(generator.get_output(f"present.0.{kind}"), expected)
            np.testing.assert_array_equal(generator.get_input(f"past_key_values.0.{kind}"), expected)
        assert generator.get_input("cache_write_indices")[0] == (0 if step == 0 else len(consumed) - 1)
    profiles = list(tmp_path.glob("cache_run*.json"))
    assert profiles, "No profile was written for cache validation"
    scatter_events = [
        event for path in profiles for event in json.loads(path.read_text())
        if event.get("args", {}).get("op_name") == "TensorScatter"
        and event.get("args", {}).get("provider")
    ]
    assert len(scatter_events) == 12, "Expected key/value TensorScatter kernels for all six steps"
    expected_provider = "CUDAExecutionProvider" if provider == "cuda" else "CPUExecutionProvider"
    assert {event["args"]["provider"] for event in scatter_events} == {expected_provider}


@pytest.mark.parametrize("height,width", [(1, 1), (2, 2), (7, 3), (3, 7)])
@pytest.mark.parametrize("normalization", [
    DEFAULT_NORMALIZATION,
    {"image_mean": [0.1, 0.2, 0.3], "image_std": [0.5, 0.75, 1.0]},
], ids=["checkpoint", "configured"])
def test_pixels_match_checkpoint_resize_pad_normalize(model_factory, tmp_path, height, width, normalization):
    cv2 = pytest.importorskip("cv2")
    image_module = pytest.importorskip("PIL.Image")
    pixels = np.random.default_rng(1).integers(0, 256, (height, width, 3), dtype=np.uint8)
    path = tmp_path / "pixels.png"
    image_module.fromarray(pixels).save(path)
    processor = model_factory(normalization=normalization).create_multimodal_processor()
    actual = processor("", images=og.Images.open(str(path)))["pixel_values"].as_numpy()

    resized_h, resized_w = height, width
    if height > 2:
        resized_h, resized_w = 2, int(2 * width / height)
    if resized_w > 2:
        resized_w, resized_h = 2, int(2 * height / width)
    resized_h, resized_w = max(1, resized_h), max(1, resized_w)
    resized = cv2.resize(pixels, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((2, 2, 3), 255, dtype=np.uint8)
    top, left = (2 - resized_h) // 2, (2 - resized_w) // 2
    canvas[top:top + resized_h, left:left + resized_w] = resized
    mean = np.array(normalization["image_mean"], dtype=np.float32)
    std = np.array(normalization["image_std"], dtype=np.float32)
    expected = ((canvas.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None]
    # OpenCV rounds resized uint8 pixels; the native bilinear path retains fractions.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1.1 / (255 * std.min()))


@pytest.mark.parametrize("field,value", [
    ("image_mean", None), ("image_std", None),
    ("image_mean", []), ("image_mean", [0.1, 0.2]),
    ("image_std", [1, 1, 1, 1]), ("image_std", [1, 0, 1]),
    ("image_std", [1, -1, 1]), ("image_std", [1, 1e-50, 1]),
    ("image_mean", [1e100, 0, 0]), ("image_std", [1, "invalid", 1]),
    ("image_mean", 0.5), ("image_std", True),
])
def test_rejects_invalid_normalization_config(model_factory, field, value):
    normalization = dict(DEFAULT_NORMALIZATION)
    if value is None:
        del normalization[field]
    else:
        normalization[field] = value
    model = model_factory(normalization=normalization)
    with pytest.raises(RuntimeError, match=field):
        model.create_multimodal_processor()


def test_uses_configured_vision_processing_filename(model_factory, images):
    model = model_factory(vision_config_filename="custom_vision.json")
    processor = model.create_multimodal_processor()
    assert processor("", images=images)["pixel_values"].as_numpy().shape == (1, 3, 2, 2)


@pytest.mark.parametrize("field", ["image_height", "image_width"])
@pytest.mark.parametrize("value", [0, -1, 3, 2.5, "2", None])
def test_rejects_processing_dimensions_inconsistent_with_graph(model_factory, field, value):
    model = model_factory(normalization={**DEFAULT_NORMALIZATION, field: value})
    with pytest.raises(RuntimeError, match="Invalid Nemotron Parse vision processing config"):
        model.create_multimodal_processor()


def test_legacy_processing_dimensions_come_from_graph(model_factory, tmp_path, images):
    model = model_factory()
    path = tmp_path / "model_8_None" / "vision_processing.json"
    config = json.loads(path.read_text())
    del config["image_height"], config["image_width"]
    path.write_text(json.dumps(config))
    processor = model.create_multimodal_processor()
    assert processor("", images=images)["pixel_values"].as_numpy().shape == (1, 3, 2, 2)


def test_missing_vision_processing_file_has_clear_error(model_factory, tmp_path):
    model = model_factory()
    (tmp_path / "model_8_None" / "vision_processing.json").unlink()
    with pytest.raises(RuntimeError, match="Cannot open.*vision_processing.json"):
        model.create_multimodal_processor()
