import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

MODEL_BUILDER_ROOT = Path(__file__).resolve().parents[1] / "src/python/py"
sys.path.insert(0, str(MODEL_BUILDER_ROOT))
sys.path.insert(0, str(MODEL_BUILDER_ROOT / "models"))

MTPModel = importlib.import_module("models.builders.mtp").MTPModel
serialize_genai_config = importlib.import_module("models.builder_config").serialize_genai_config


def export_package(source, output, max_draft_tokens=7):
    source = Path(source).resolve()
    output = Path(output).absolute()
    if output.exists():
        raise ValueError("The output directory must not exist; existing models are never overwritten.")
    if max_draft_tokens < 1:
        raise ValueError("IndexShare export requires at least one draft token.")
    with (source / "genai_config.json").open() as config_file:
        config = json.load(config_file)
    mtp = config["model"]["mtp"]
    if Path(mtp["filename"]).name != mtp["filename"]:
        raise ValueError("Side-by-side conversion requires a top-level MTP graph filename.")
    output.mkdir(parents=True)
    try:
        for path in source.iterdir():
            destination = output / path.name
            if path.name == "genai_config.json":
                continue
            if path.suffix == ".onnx" and path.is_file():
                shutil.copy2(path, destination)
            elif path.is_file():
                destination.hardlink_to(path.resolve())
            else:
                destination.symlink_to(path, target_is_directory=path.is_dir())
        metadata = MTPModel().export_indexshare_graphs(str(output), mtp["filename"], max_draft_tokens)
        mtp.pop("index_share", None)
        mtp.pop("max_draft_tokens", None)
        mtp["base_capacity"] = metadata["base_capacity"]
        inputs = mtp.setdefault("inputs", {})
        inputs["past_indices"] = "past_indices"
        inputs["past_counts"] = "past_counts"
        outputs = mtp.setdefault("outputs", {})
        outputs.pop("indices", None)
        outputs.pop("counts", None)
        outputs["present_indices"] = metadata["indices_output"]
        outputs["present_counts"] = metadata["counts_output"]
        outputs["indexer_status"] = metadata["status_output"]
        config.setdefault("speculative", {}).setdefault("max_draft_tokens", metadata["max_draft_tokens"])
        with (output / "genai_config.json").open("w") as config_file:
            config_file.write(serialize_genai_config(config))
    except Exception:
        shutil.rmtree(output)
        raise
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Create a lossless IndexShare MTP model package.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-draft-tokens", type=int, default=7)
    args = parser.parse_args()
    metadata = export_package(args.input, args.output, args.max_draft_tokens)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
