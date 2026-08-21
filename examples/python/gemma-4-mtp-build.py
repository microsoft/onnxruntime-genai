#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Build a Gemma 4 target + assistant pair for speculative decoding with ``og.MtpGenerator``.

``builder.py`` has no Gemma 4 support, so the graphs come from an external exporter --
mobius (https://github.com/onnxruntime/mobius) -- and this script applies the post-processing
that ``og.MtpGenerator`` needs on top of them. See ``gemma-4-mtp.md`` for the resulting contract
and ``gemma-4-mtp.py`` to run the pair.

    # Everything, including the two mobius exports (needs `uv` on PATH):
    python gemma-4-mtp-build.py all --out-dir models

    # Or drive the stages yourself:
    python gemma-4-mtp-build.py target   <hf_config.json> <exported> <prepared> --assistant <head>
    python gemma-4-mtp-build.py assistant <hf_config.json> <exported> <prepared> --target <prepared_target>
    python gemma-4-mtp-build.py quantize <prepared> <quantized>

``target``/``assistant`` need ``onnx``; ``quantize`` also needs ``onnxruntime`` and ``onnx_ir``.
"""

import argparse
import errno
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import onnx
from onnx import helper

NAMESPACE = "ortgenai/mtp"
DEFAULT_VERIFY_WINDOW = 6
DEFAULT_TARGET_MODEL = "google/gemma-4-E4B-it"
DEFAULT_ASSISTANT_MODEL = "google/gemma-4-E4B-it-assistant"
# Graph and node names below depend on the exporter, so both pins are load-bearing.
DEFAULT_MOBIUS_REF = "66e9c5edea92e9553d98be15f64126ae48be8a53"
DEFAULT_TRANSFORMERS = "5.14.0"

GENAI_CONFIG = "genai_config.json"
MODEL_FILE = "model.onnx"
# The target keeps its decoder in a subdirectory; the assistant is a single flat graph.
TARGET_MODEL_RELPATH = f"decoder/{MODEL_FILE}"


# ---------------------------------------------------------------------------- helpers


def link_or_copy(source: str, destination: str) -> str:
    """Hard-link multi-gigabyte weights when the destination is on the same filesystem."""
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        return shutil.copy2(source, destination)
    return destination


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | str | None]:
    return [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param or None
        for dim in value.type.tensor_type.shape.dim
    ]


def load_text_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    return config.get("text_config", config)


def save_graph(model: onnx.ModelProto, model_path: Path) -> None:
    temporary = model_path.with_suffix(".onnx.tmp")
    onnx.save(model, temporary)
    temporary.replace(model_path)
    # Catches malformed surgery (wrong node arity, unsorted nodes) here rather than at session load.
    onnx.checker.check_model(str(model_path), full_check=False)


def stage_package(source: Path, output: Path, model_relpath: str) -> Path:
    """Copy a package, then unlink and re-copy the graph so edits do not touch the source."""
    if not (source / model_relpath).is_file():
        raise SystemExit(f"model not found: {source / model_relpath}")
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    shutil.copytree(source, output, copy_function=link_or_copy)
    staged = output / model_relpath
    staged.unlink()
    shutil.copy2(source / model_relpath, staged)
    for extra in (source / model_relpath).parent.glob(f"{MODEL_FILE}.data*"):
        target = staged.parent / extra.name
        target.unlink(missing_ok=True)
        shutil.copy2(extra, target)
    return staged


# ---------------------------------------------------------------------------- target


def derive_shared_kv_layers(text_config: dict[str, Any]) -> list[int]:
    """Cache indices the assistant reads: the last sliding and last full attention layer.

    Gemma 4's trailing layers share the KV of earlier ones, so the cache is shorter than the layer
    stack and the assistant binds the final sliding/full pair.
    """
    layer_types = text_config["layer_types"]
    layer_count = text_config["num_hidden_layers"]
    if len(layer_types) != layer_count:
        raise ValueError(f"layer_types has {len(layer_types)} entries, expected {layer_count}")

    cache_count = layer_count - text_config["num_kv_shared_layers"]
    owned = layer_types[:cache_count]
    sliding = max(i for i, kind in enumerate(owned) if kind == "sliding_attention")
    full = max(i for i, kind in enumerate(owned) if kind == "full_attention")
    if (sliding, full) != (cache_count - 2, cache_count - 1):
        raise ValueError(
            "final cache sources are not the adjacent trailing sliding/full pair: "
            f"sliding={sliding}, full={full}, caches={cache_count}"
        )
    return [sliding, full]


def add_int64_initializers(model: onnx.ModelProto, vectors: dict[str, list[int]], scalars: dict[str, int]) -> None:
    model.graph.initializer.extend(
        helper.make_tensor(name, onnx.TensorProto.INT64, [1], values) for name, values in vectors.items()
    )
    model.graph.initializer.extend(
        helper.make_tensor(name, onnx.TensorProto.INT64, [], [value]) for name, value in scalars.items()
    )


def add_hidden_state_output(model, hidden_name: str, element_type: int, width: int, batch) -> None:
    """Slice the LM head's input at the last position and expose it as `final_hidden_state`."""
    vectors = {
        f"{NAMESPACE}/hidden_starts": [-1],
        f"{NAMESPACE}/hidden_ends": [2**63 - 1],
        f"{NAMESPACE}/hidden_axes": [1],
        f"{NAMESPACE}/hidden_steps": [1],
    }
    add_int64_initializers(model, vectors, {})
    model.graph.node.append(
        helper.make_node(
            "Slice",
            [
                hidden_name,
                f"{NAMESPACE}/hidden_starts",
                f"{NAMESPACE}/hidden_ends",
                f"{NAMESPACE}/hidden_axes",
                f"{NAMESPACE}/hidden_steps",
            ],
            ["final_hidden_state"],
            name=f"{NAMESPACE}/final_hidden_state",
        )
    )
    model.graph.output.append(helper.make_tensor_value_info("final_hidden_state", element_type, [batch, 1, width]))


def bound_verify_logits(model, head_index: int, logits, hidden_name: str, window: int) -> None:
    """Gate the LM head so inputs longer than `window` produce last-token logits only.

    A prefill over a long prompt would otherwise materialize [1, prompt_len, vocab] logits, while a
    verify forward still needs one row per draft token.
    """
    vectors = {
        f"{NAMESPACE}/logits_ends": [2**63 - 1],
        f"{NAMESPACE}/logits_axes": [1],
        f"{NAMESPACE}/logits_steps": [1],
        f"{NAMESPACE}/logits_unsqueeze_axes": [0],
    }
    scalars = {
        f"{NAMESPACE}/logits_zero": 0,
        f"{NAMESPACE}/logits_one": 1,
        f"{NAMESPACE}/logits_window": window,
        f"{NAMESPACE}/logits_gather_index": 1,
    }
    add_int64_initializers(model, vectors, scalars)
    chain = [
        helper.make_node("Shape", [hidden_name], [f"{NAMESPACE}/hidden_shape"], name=f"{NAMESPACE}/hidden_shape"),
        helper.make_node(
            "Gather",
            [f"{NAMESPACE}/hidden_shape", f"{NAMESPACE}/logits_gather_index"],
            [f"{NAMESPACE}/sequence_length"],
            name=f"{NAMESPACE}/sequence_length",
            axis=0,
        ),
        helper.make_node(
            "Greater",
            [f"{NAMESPACE}/sequence_length", f"{NAMESPACE}/logits_window"],
            [f"{NAMESPACE}/is_prefill"],
            name=f"{NAMESPACE}/is_prefill",
        ),
        helper.make_node(
            "Sub",
            [f"{NAMESPACE}/sequence_length", f"{NAMESPACE}/logits_one"],
            [f"{NAMESPACE}/last_index"],
            name=f"{NAMESPACE}/last_index",
        ),
        helper.make_node(
            "Where",
            [f"{NAMESPACE}/is_prefill", f"{NAMESPACE}/last_index", f"{NAMESPACE}/logits_zero"],
            [f"{NAMESPACE}/start_scalar"],
            name=f"{NAMESPACE}/start_scalar",
        ),
        helper.make_node(
            "Unsqueeze",
            [f"{NAMESPACE}/start_scalar", f"{NAMESPACE}/logits_unsqueeze_axes"],
            [f"{NAMESPACE}/logits_starts"],
            name=f"{NAMESPACE}/logits_starts",
        ),
        helper.make_node(
            "Slice",
            [
                hidden_name,
                f"{NAMESPACE}/logits_starts",
                f"{NAMESPACE}/logits_ends",
                f"{NAMESPACE}/logits_axes",
                f"{NAMESPACE}/logits_steps",
            ],
            [f"{NAMESPACE}/bounded_hidden_state"],
            name=f"{NAMESPACE}/bounded_hidden_state",
        ),
    ]
    # The chain feeds the LM head, so it has to precede it in the node list.
    for offset, node in enumerate(chain):
        model.graph.node.insert(head_index + offset, node)
    model.graph.node[head_index + len(chain)].input[0] = f"{NAMESPACE}/bounded_hidden_state"
    logits.type.tensor_type.shape.dim[1].dim_param = f"mtp_verify_sequence_len_{window}"


def transform_target(model_path: Path, width: int, window: int) -> None:
    model = onnx.load(model_path, load_external_data=False)
    outputs = {output.name: output for output in model.graph.output}
    if "logits" not in outputs:
        raise ValueError("decoder does not expose logits")
    if "final_hidden_state" in outputs:
        raise ValueError("decoder already exposes final_hidden_state; package is already prepared")
    logits = outputs["logits"]
    logits_shape = tensor_shape(logits)
    if len(logits_shape) != 3 or logits_shape[1] == 1:
        raise ValueError(f"MTP target requires unpruned [batch, sequence, vocab] logits; found {logits_shape}")

    heads = [
        (index, node)
        for index, node in enumerate(model.graph.node)
        if node.op_type in {"MatMul", "MatMulNBits"} and "/lm_head/" in node.name
    ]
    if len(heads) != 1:
        raise ValueError(f"expected exactly one LM-head projection, found {len(heads)}")
    head_index, head = heads[0]
    hidden_name = head.input[0]

    add_hidden_state_output(model, hidden_name, logits.type.tensor_type.elem_type, width, logits_shape[0])
    if window > 0:
        bound_verify_logits(model, head_index, logits, hidden_name, window)
    save_graph(model, model_path)


def write_target_config(package: Path, shared_kv_layers: list[int], window: int) -> dict[str, Any]:
    config_path = package / GENAI_CONFIG
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = config["model"]
    decoder = model_config["decoder"]
    decoder.setdefault("outputs", {})["hidden_states"] = "final_hidden_state"
    if window > 0:
        decoder["max_logits_sequence_length"] = window

    # The assistant reads the target's token embeddings from the embedding stage, the only output
    # carrying them that the runtime binds.
    embeddings = model_config.get("embedding", {}).get("outputs", {}).get("embeddings", "inputs_embeds")
    model_config["mtp"] = {
        "main_hidden_states": "final_hidden_state",
        "main_inputs_embeds": embeddings,
        "shared_kv_layers": shared_kv_layers,
        "inputs": {
            "hidden_states": "inputs_embeds",
            "attention_mask": "attention_mask",
            "shared_key_names": ["shared_kv.sliding_attention.key", "shared_kv.full_attention.key"],
            "shared_value_names": ["shared_kv.sliding_attention.value", "shared_kv.full_attention.value"],
        },
        "outputs": {"logits": "logits", "hidden_states": "projected_state"},
    }
    config_path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
    return model_config["mtp"]


# ---------------------------------------------------------------------------- assistant


def reachable_nodes(model: onnx.ModelProto) -> set[int]:
    """Node indices still feeding a graph output, walked backwards to a fixed point."""
    required = {output.name for output in model.graph.output}
    retained: set[int] = set()
    changed = True
    while changed:
        changed = False
        for index, node in enumerate(model.graph.node):
            if index in retained or not required.intersection(node.output):
                continue
            retained.add(index)
            required.update(name for name in node.input if name)
            changed = True
    return retained


def densify_assistant(model_path: Path) -> None:
    """Replace the exporter's ordered (sampled-vocab) head with a dense LM projection.

    Speculative verification compares full-vocabulary argmaxes, so the head must score every token.
    """
    model = onnx.load(model_path, load_external_data=False)
    outputs = {output.name for output in model.graph.output}
    missing = {"logits", "projected_state"} - outputs
    if missing:
        raise ValueError(f"assistant is missing outputs: {', '.join(sorted(missing))}")

    post_projections = [
        node for node in model.graph.node if node.op_type == "MatMul" and "post_projection" in node.name
    ]
    if len(post_projections) != 1:
        raise ValueError(f"expected one post_projection MatMul, found {len(post_projections)}")
    hidden_states = post_projections[0].input[0]

    producers = [node for node in model.graph.node if "logits" in node.output]
    if len(producers) != 1:
        raise ValueError(f"expected one logits producer, found {len(producers)}")
    producers[0].output[list(producers[0].output).index("logits")] = f"{NAMESPACE}/ordered_logits_unused"

    kept = [node for node in model.graph.node if not node.name.startswith(f"{NAMESPACE}/lm_head/")]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    model.graph.node.extend(
        [
            helper.make_node(
                "Transpose",
                ["lm_head.weight"],
                [f"{NAMESPACE}/lm_head.weight_t"],
                name=f"{NAMESPACE}/lm_head/Transpose",
                perm=[1, 0],
            ),
            helper.make_node(
                "MatMul",
                [hidden_states, f"{NAMESPACE}/lm_head.weight_t"],
                ["logits"],
                name=f"{NAMESPACE}/lm_head/MatMul",
            ),
        ]
    )

    retained = reachable_nodes(model)
    nodes = [node for index, node in enumerate(model.graph.node) if index in retained]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {name for node in nodes for name in node.input if name}
    initializers = [initializer for initializer in model.graph.initializer if initializer.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    save_graph(model, model_path)


def write_assistant_config(package: Path, assistant_config: dict[str, Any], target_package: Path) -> None:
    """Derive the head's genai_config from its HF config, taking vocab/EOS from the target."""
    target = json.loads((target_package / GENAI_CONFIG).read_text(encoding="utf-8"))["model"]
    config = {
        "model": {
            "type": "gemma4_assistant",
            "vocab_size": target["vocab_size"],
            "context_length": target["context_length"],
            "decoder": {
                "filename": MODEL_FILE,
                "hidden_size": assistant_config["hidden_size"],
                "head_size": assistant_config["head_dim"],
                "num_attention_heads": assistant_config["num_attention_heads"],
                "num_hidden_layers": assistant_config["num_hidden_layers"],
                "num_key_value_heads": assistant_config["num_key_value_heads"],
                "inputs": {"inputs_embeds": "inputs_embeds", "attention_mask": "attention_mask"},
                "outputs": {"logits": "logits", "hidden_states": "projected_state"},
            },
            "eos_token_id": target["eos_token_id"],
            "pad_token_id": target["pad_token_id"],
        }
    }
    (package / GENAI_CONFIG).write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------- validation


def validate(target_package: Path, assistant_package: Path | None) -> None:
    """Check every name in the target's mtp block against the graph that must provide it."""
    mtp = json.loads((target_package / GENAI_CONFIG).read_text(encoding="utf-8"))["model"]["mtp"]
    decoder = onnx.load(target_package / TARGET_MODEL_RELPATH, load_external_data=False)
    decoder_outputs = {output.name for output in decoder.graph.output}
    required = [mtp["main_hidden_states"]]
    required += [f"present.{layer}.{kind}" for layer in mtp["shared_kv_layers"] for kind in ("key", "value")]
    missing = [name for name in required if name not in decoder_outputs]
    if missing:
        raise ValueError(f"target is missing outputs: {', '.join(missing)}")

    if assistant_package is None:
        return
    head = onnx.load(assistant_package / MODEL_FILE, load_external_data=False)
    head_inputs = {value.name for value in head.graph.input}
    head_outputs = {output.name for output in head.graph.output}
    expected = [mtp["inputs"]["hidden_states"], mtp["inputs"]["attention_mask"]]
    expected += mtp["inputs"]["shared_key_names"] + mtp["inputs"]["shared_value_names"]
    missing = [name for name in expected if name not in head_inputs]
    missing += [name for name in mtp["outputs"].values() if name not in head_outputs]
    if missing:
        raise ValueError(f"assistant head is missing: {', '.join(missing)}")


# ---------------------------------------------------------------------------- stages


def stage_target(args) -> None:
    text_config = load_text_config(args.hf_config)
    shared_kv_layers = derive_shared_kv_layers(text_config)
    staged = stage_package(args.source, args.output, TARGET_MODEL_RELPATH)
    try:
        transform_target(staged, int(text_config["hidden_size"]), args.verify_window)
        write_target_config(args.output, shared_kv_layers, args.verify_window)
        validate(args.output, args.assistant)
    except Exception:
        shutil.rmtree(args.output, ignore_errors=True)
        raise
    print(f"wrote {args.output} (shared_kv_layers={shared_kv_layers}, verify_window={args.verify_window})")


def stage_assistant(args) -> None:
    assistant_config = load_text_config(args.hf_config)
    staged = stage_package(args.source, args.output, MODEL_FILE)
    try:
        densify_assistant(staged)
        write_assistant_config(args.output, assistant_config, args.target)
        validate(args.target, args.output)
    except Exception:
        shutil.rmtree(args.output, ignore_errors=True)
        raise
    print(f"wrote {args.output}")


def stage_quantize(args) -> None:
    # Deferred so the graph-editing stages only need onnx.
    try:
        from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils  # noqa: PLC0415
    except ImportError as error:
        raise SystemExit(f"the quantize stage needs onnxruntime and its quantization extras: {error}") from error

    source_model = args.source / TARGET_MODEL_RELPATH
    if not source_model.is_file():
        raise SystemExit(f"decoder not found: {source_model}")

    def skip_weights(path: str, names: list[str]) -> set[str]:
        if Path(path) == args.source / "decoder":
            return {name for name in names if name.startswith(MODEL_FILE)}
        return set()

    shutil.rmtree(args.output, ignore_errors=True)
    shutil.copytree(args.source, args.output, ignore=skip_weights)
    config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=32,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul", "Gather"),
        quant_axes=(("MatMul", 0), ("Gather", 1)),
    )
    quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(str(source_model), algo_config=config)
    quantizer.process()
    quantizer.model.save_model_to_file(str(args.output / TARGET_MODEL_RELPATH), True)
    print(f"wrote {args.output}")


def mobius_export(args, model_id: str, output: Path, extra: list[str]) -> None:
    spec = f"mobius-onnx[ort-genai] @ git+https://github.com/onnxruntime/mobius.git@{args.mobius_ref}"
    command = [
        "uvx",
        "--from",
        spec,
        "--with",
        f"transformers=={args.transformers}",
        "mobius",
        "build",
        "--model",
        model_id,
        str(output),
        "--ep",
        args.ep,
        "--dtype",
        "f16",
        *extra,
    ]
    print(">>", " ".join(command))
    subprocess.run(command, check=True)


def stage_all(args) -> None:
    if shutil.which("uvx") is None:
        raise SystemExit("uvx is required to run the pinned mobius exporter; install uv first")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    hf_target = out / "target-config.json"
    hf_assistant = out / "assistant-config.json"
    for path in (hf_target, hf_assistant):
        if not path.is_file():
            raise SystemExit(
                f"missing {path}; download config.json for {args.target_model} and "
                f"{args.assistant_model} and save them as target-config.json / assistant-config.json"
            )

    exported_target, exported_assistant = out / "target-fp16", out / "assistant-ordered"
    if not exported_target.exists():
        mobius_export(args, args.target_model, exported_target, ["--runtime", "ort-genai"])
    if not exported_assistant.exists():
        mobius_export(args, args.assistant_model, exported_assistant, ["--task", "gemma4-assistant"])

    prepared_target = out / "target-mtp-fp16"
    stage_target(
        argparse.Namespace(
            hf_config=hf_target,
            source=exported_target,
            output=prepared_target,
            verify_window=args.verify_window,
            assistant=None,
        )
    )
    stage_assistant(
        argparse.Namespace(
            hf_config=hf_assistant,
            source=exported_assistant,
            output=out / "assistant-fp16",
            target=prepared_target,
        )
    )
    if not args.no_quantize:
        stage_quantize(argparse.Namespace(source=prepared_target, output=out / "target-mtp-int4"))


# ---------------------------------------------------------------------------- cli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    target = sub.add_parser("target", help="Add the MTP outputs and mtp config block to a target package")
    target.add_argument("hf_config", type=Path, help="config.json of the Hugging Face target model")
    target.add_argument("source", type=Path, help="Exported target package")
    target.add_argument("output", type=Path, help="Destination package (must not exist)")
    target.add_argument("--assistant", type=Path, default=None, help="Assistant package to cross-check names against")
    target.set_defaults(func=stage_target)

    assistant = sub.add_parser("assistant", help="Densify the assistant head and write its genai_config")
    assistant.add_argument("hf_config", type=Path, help="config.json of the Hugging Face assistant model")
    assistant.add_argument("source", type=Path, help="Exported assistant package")
    assistant.add_argument("output", type=Path, help="Destination package (must not exist)")
    assistant.add_argument("--target", type=Path, required=True, help="Prepared target package")
    assistant.set_defaults(func=stage_assistant)

    quantize = sub.add_parser("quantize", help="Quantize a prepared target package to INT4 weights")
    quantize.add_argument("source", type=Path)
    quantize.add_argument("output", type=Path)
    quantize.set_defaults(func=stage_quantize)

    every = sub.add_parser("all", help="Export both models with mobius, then run every stage")
    every.add_argument("--out-dir", type=Path, default=Path("models"))
    every.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    every.add_argument("--assistant-model", default=DEFAULT_ASSISTANT_MODEL)
    every.add_argument("--mobius-ref", default=DEFAULT_MOBIUS_REF)
    every.add_argument("--transformers", default=DEFAULT_TRANSFORMERS)
    every.add_argument("--ep", default="webgpu")
    every.add_argument("--no-quantize", action="store_true")
    every.set_defaults(func=stage_all)

    for command in (target, every):
        command.add_argument(
            "--verify-window",
            type=int,
            default=DEFAULT_VERIFY_WINDOW,
            help="Bound the logits sequence dimension to this many rows; 0 leaves the logits unbounded",
        )

    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
