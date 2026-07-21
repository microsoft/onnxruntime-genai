# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib
import importlib.util
import logging
import os
import subprocess
import sys

try:
    from huggingface_hub import model_info
except ImportError:
    model_info = None

# Execution providers shipped as separate plug-in libraries that must be registered with
# ONNX Runtime before use. Maps the GenAI provider name to the Python package that exposes
# the plug-in library path via get_library_path().
PLUGIN_EP_PACKAGES = {
    "cuda": "onnxruntime_ep_cuda",
    "webgpu": "onnxruntime_ep_webgpu",
}

# Tracks plug-in EPs already registered in this process. Registration is process-wide on the
# shared ORT environment, so it must happen at most once even when multiple test modules import
# this helper and register at import time.
_registered_plugin_eps: set[str] = set()


def is_ep_plugin_available(provider_name: str) -> bool:
    """Returns True if the plug-in package backing the given GenAI provider name is installed."""
    package_name = PLUGIN_EP_PACKAGES.get(provider_name)
    if package_name is None:
        return False
    return importlib.util.find_spec(package_name) is not None


def is_webgpu_ep_available() -> bool:
    """WebGPU ships as a separate plug-in package (onnxruntime-ep-webgpu) rather than in the base
    onnxruntime build, so its availability is determined by whether that package is installed."""
    return is_ep_plugin_available("webgpu")


def register_plugin_ep(provider_name: str, log: logging.Logger | None = None) -> bool:
    """Registers a single plug-in execution provider library with ONNX Runtime GenAI.

    Registration is process-wide and idempotent: repeated calls (including from different test
    modules) register the library at most once. Returns True if the EP is registered (or was
    already), False if its plug-in package is not installed.
    """
    if provider_name in _registered_plugin_eps:
        return True

    package_name = PLUGIN_EP_PACKAGES.get(provider_name)
    if package_name is None:
        raise ValueError(f"Unknown plug-in execution provider: {provider_name!r}")

    try:
        ep_module = importlib.import_module(package_name)
    except ImportError:
        if log:
            log.info("Skipping plug-in EP '%s': package '%s' is not installed.", provider_name, package_name)
        return False

    import onnxruntime_genai as og  # noqa: PLC0415 - imported lazily so tests that don't need an EP don't pay for it

    try:
        og.register_execution_provider_library(ep_module.get_ep_name(), ep_module.get_library_path())
        _registered_plugin_eps.add(provider_name)
        if log:
            log.info("Registered plug-in EP '%s' from package '%s'.", provider_name, package_name)
        return True
    except Exception as exc:
        if log:
            log.warning("Failed to register plug-in EP '%s': %s", provider_name, exc)
        return False


def register_webgpu_plugin(log: logging.Logger | None = None) -> bool:
    """Registers the onnxruntime-ep-webgpu plug-in with ONNX Runtime GenAI (idempotent).

    Returns True if the WebGPU EP is registered (or was already), False if the plug-in package is
    not installed.
    """
    return register_plugin_ep("webgpu", log)


def register_plugin_providers(log: logging.Logger | None = None) -> None:
    """Registers every available plug-in execution provider library with ONNX Runtime GenAI.

    A provider is skipped if its package is not installed; other failures are logged but not raised
    so the absence of an optional EP never blocks the test session.
    """
    for provider_name in PLUGIN_EP_PACKAGES:
        register_plugin_ep(provider_name, log)


def is_windows():
    return sys.platform.startswith("win")


def run_subprocess(
    args: list[str],
    cwd: str | bytes | os.PathLike | None = None,
    capture: bool = False,
    dll_path: str | bytes | os.PathLike | None = None,
    shell: bool = False,
    env: dict[str, str] | None = None,
    log: logging.Logger | None = None,
):
    if log:
        log.info(f"Running subprocess in '{cwd or os.getcwd()}'\n{args}")
    user_env = os.environ.copy()
    if env is None:
        env = {}
    user_env.update(env)
    if dll_path:
        if is_windows():
            user_env["PATH"] = dll_path + os.pathsep + user_env["PATH"]
        else:
            if "LD_LIBRARY_PATH" in user_env:
                user_env["LD_LIBRARY_PATH"] += os.pathsep + dll_path
            else:
                user_env["LD_LIBRARY_PATH"] = dll_path

    stdout, stderr = (subprocess.PIPE, subprocess.STDOUT) if capture else (None, None)
    completed_process = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        stdout=stdout,
        stderr=stderr,
        env=user_env,
        shell=shell,
    )

    if log:
        log.debug("Subprocess completed. Return code=%s", completed_process.returncode)
    return completed_process


def get_ci_data_path():
    if is_windows():
        ci_data_path = os.path.join(R"C:\\", "data", "models", "ortgenai")
    else:
        ci_data_path = os.path.join(os.path.abspath(os.sep), "data", "ortgenai")
    return ci_data_path


def get_model_paths():
    # TODO: Uncomment the following models as needed in the CI pipeline.

    # Format: model alias: (HF repo name, create only 1 layer, enable graph capture)
    hf_paths = {
        # "olmo": "amd/AMD-OLMo-1B-SFT-DPO",
        # "phi-3.5": "microsoft/Phi-3.5-mini-instruct",
        # "llama-3.2": "meta-llama/Llama-3.2-1B-instruct",
        # "granite-3.0": "ibm-granite/granite-3.0-2b-instruct",
        "phi-4-mini": ("microsoft/Phi-4-mini-instruct", True, False),
        "qwen-2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", False, False),
        "qwen-2.5-0.5b-graph": ("Qwen/Qwen2.5-0.5B-Instruct", False, True),
        "lfm2.5-350m": ("LiquidAI/LFM2.5-350M", False, False),
        "lfm2.5-1.2b": ("LiquidAI/LFM2.5-1.2B-Instruct", False, False),
    }

    ci_data_path = os.path.join(get_ci_data_path(), "pytorch")
    if not os.path.exists(ci_data_path):
        return {}, hf_paths

    # Note: If a model has over 4B parameters, please add a quantized version
    # to `ci_paths` instead of `hf_paths` to reduce file size and testing time.
    # Format: model alias: (OS path, create only 1 layer, enable graph capture)
    ci_paths = {
        # "llama-2": os.path.join(ci_data_path, "Llama-2-7B-Chat-GPTQ"),
        # "llama-3": os.path.join(ci_data_path, "Meta-Llama-3-8B-AWQ"),
        # "mistral-v0.2": os.path.join(ci_data_path, "Mistral-7B-Instruct-v0.2-GPTQ"),
        "phi-2": (os.path.join(ci_data_path, "phi2"), True, False),
        # "gemma-2b": os.path.join(ci_data_path, "gemma-1.1-2b-it"),
        # "gemma-7b": os.path.join(ci_data_path, "gemma-7b-it-awq"),
        # "phi-3-mini": os.path.join(ci_data_path, "phi3-mini-128k-instruct"),
        # "gemma-2-2b": os.path.join(ci_data_path, "gemma-2-2b-it"),
        # "llama-3.2": os.path.join(ci_data_path, "llama-3.2b-1b-instruct"),
        # "qwen-2.5-0.5b": os.path.join(ci_data_path, "qwen2.5-0.5b-instruct"),
        # "nemotron-mini": os.path.join(ci_data_path, "nemotron-mini-4b"),
    }

    return ci_paths, hf_paths


def download_model(model_name, input_path, output_path, precision, device, one_layer, enable_graph_capture):
    command = [
        sys.executable,
        "-m",
        "onnxruntime_genai.models.builder",
    ]

    if model_name is not None:
        # If model_name is provided:
        # python -m onnxruntime_genai.models.builder -m <model_name> -o <output_path> -p <precision> -e <device>
        command += ["-m", model_name]
    elif input_path != "":
        # If input_path is provided:
        # python -m onnxruntime_genai.models.builder -i <input_path> -o <output_path> -p <precision> -e <device>
        command += ["-i", input_path]
    else:
        raise Exception("Either `model_name` or `input_path` can be provided for PyTorch models, not both.")

    command += [
        "-o",
        output_path,
        "-p",
        precision,
        "-e",
        device,
    ]

    extra_options = ["--extra_options", "include_hidden_states=1", "hf_token=0", "hf_remote=0"]
    if device == "cpu" and precision == "int4":
        extra_options += ["accuracy_level=4"]
    if one_layer:
        extra_options += ["num_hidden_layers=1"]
    # Graph capture is a generic model option and maps to EP-specific builder flags.
    if enable_graph_capture and device == "webgpu":
        extra_options += ["enable_webgpu_graph=true"]
    if enable_graph_capture and device == "cuda":
        extra_options += ["enable_cuda_graph=true"]
    if len(extra_options) > 1:
        command += extra_options

    run_subprocess(command).check_returncode()


# Devices that support graph capture. Models with enable_graph_capture=True
# are only built for these devices.
#
# CUDA is intentionally excluded: the Windows CUDA CI consistently fails to
# download this new model from Hugging Face. Will re-add "cuda" once the
# CI download issue is resolved.
#
# Note: nvtensorrtrtx is included here for model generation but does not have
# dedicated CI coverage yet — tests using NvTensorRtRtx models are guarded with
# GTEST_SKIP when the model artifacts are not present.
_GRAPH_CAPTURE_DEVICES = {"webgpu", "dml", "nvtensorrtrtx"}


def download_models(download_path, precision, device, log):
    log.debug(f"Downloading models to {download_path} with precision {precision} and device {device}")

    ci_paths, hf_paths = get_model_paths()
    output_paths = []

    log.debug(f"Downloading {len(ci_paths)} PyTorch models and {len(hf_paths)} Hugging Face models")

    # python -m onnxruntime_genai.models.builder -i <input_path> -o <output_path> -p <precision> -e <device>
    for model_name, (input_path, one_layer, graph_capture) in ci_paths.items():
        if graph_capture and device.lower() not in _GRAPH_CAPTURE_DEVICES:
            continue
        try:
            output_path = os.path.join(download_path, model_name, precision, device)
            log.debug(f"Downloading {model_name} from {input_path} to {output_path}")
            if not os.path.exists(output_path):
                download_model(None, input_path, output_path, precision, device, one_layer, graph_capture)
                output_paths.append(output_path)
        except Exception as e:
            log.warning(f"Error: {e}. Skipping CI model.")
            continue

    # python -m onnxruntime_genai.models.builder -m <model_name> -o <output_path> -p <precision> -e <device>
    for model_name, (hf_name, one_layer, graph_capture) in hf_paths.items():
        if graph_capture and device.lower() not in _GRAPH_CAPTURE_DEVICES:
            continue
        try:
            if model_info is None:
                raise ImportError
            model_info(hf_name)
        except ImportError:
            log.warning("huggingface_hub is not installed. Skipping downloading hugging face models.")
            continue
        except Exception as e:
            log.warning(f"Error: {e}. Skipping downloading hugging face models")
            continue
        output_path = os.path.join(download_path, model_name, precision, device)

        log.debug(f"Downloading {model_name} from {hf_name} to {output_path}")

        if not os.path.exists(output_path):
            download_model(hf_name, "", output_path, precision, device, one_layer, graph_capture)
            output_paths.append(output_path)

    log.info(f"Successfully downloaded {len(output_paths)} models")

    return output_paths
