# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""End-to-end benchmark for SPECULATIVE decoding in onnxruntime-genai.

Compares speculative decoding (big ``target`` decoder + small ``draft``) against
standard decoding of the target alone, on identical prompts, and reports
wall-clock decode throughput, speedup, and acceptance statistics.

This benchmark MIRRORS chat.py: every prompt is wrapped in the model's chat
template (Qwen3 no-think mode) and decoded as a real assistant answer, so the
acceptance/speedup numbers reflect genuine generation -- NOT the degenerate
repetition loops the old length-padded prompts produced. (The previous version
selected prompts by token length and *repeated* any prompt shorter than the
requested length; that primed a repetition loop in which the small draft
trivially matched the target's argmax, inflating greedy acceptance to a
misleading ~100%.)

Only TWO things are swept:
  * mode -- greedy or sampling
  * K    -- max_draft_tokens

Everything else is fixed for reproducibility: a built-in set of real prompts, a
fixed max-new-tokens budget, and a fixed sampling seed (so sampling runs
reproduce -- unlike chat.py, which uses a random seed).

Why a dedicated script (vs benchmark_e2e.py)? benchmark_e2e.py attributes one
token to each ``generate_next_token()`` call and times calls individually. The
speculative round structure makes per-call latency lumpy (one heavy propose+
verify call followed by cheap drains), so the fair measure is total decode time
over total NEW tokens. That is what this script measures.

The speculative config is composed on the fly from each standalone model's own
``genai_config.json`` (architecture), so ANY target/draft pair works (e.g.
1.7b/0.6b now, 8b/1.7b later) without hand-maintaining a folder per pair.

Prompts: a small built-in smoke-test set by default, or -- recommended -- the
Spec-Bench dataset via ``--dataset question.jsonl`` (6 tasks x 80 prompts:
mt_bench, translation, summarization, qa, math_reasoning, rag). We use each
prompt's first turn only. Results are reported PER TASK (acceptance is
domain-dependent) plus an overall geometric-mean speedup. ``--limit-per-task N``
subsamples each task so a full run stays tractable on CPU.

Examples:
    # Quick built-in smoke test (local flat layout: <models-root>/qwen3-8b/, qwen3-1.7b/)
    python benchmark_speculative.py --target 8b --draft 1.7b --k 1,2,4,8

    # Spec-Bench, 10 prompts/task, all 6 tasks, greedy + sampling
    python benchmark_speculative.py --target 8b --draft 1.7b \
        --dataset question.jsonl --limit-per-task 10 \
        --k 1,2,4,8 --modes greedy,sampling --reps 2 -o results/spec_8b_1.7b

    # Foundry-Local layout + installed wheel (ep-cert style). Models resolve from
    # <models-root>/<id>/onnx/<device-dir>/v<N>/, and the pip-installed
    # onnxruntime-genai wheel is used (not a local build tree):
    #   CPU:    -e cpu    --device-dir cpu_and_mobile
    #   CUDA:   -e cuda   --device-dir cuda
    #   WebGPU: -e webgpu --device-dir webgpu   (needs `pip install onnxruntime-ep-webgpu`)
    #   NPU:    -e OpenVINOExecutionProvider --device npu --device-dir <fl-npu-dir> --use-winml
    python benchmark_speculative.py --use-installed \
        --models-root %ORTGENAI_MODEL_ROOT% --model-prefix "" \
        --target qwen3-8b --draft qwen3-0.6b \
        -e cuda --device-dir cuda --k 1,2,4,8 --modes greedy,sampling \
        -o results/cuda/spec_qwen3-8b_qwen3-0.6b

Device/EP coverage note (Foundry-Local): the target+draft pair must both be present under the SAME
onnx/<device-dir> for the chosen EP, and must share a vocab/tokenizer. Some EPs (e.g. NPU/DML)
may not have both models published in the catalog; in that case model resolution fails fast (a
FileNotFoundError naming the missing path) rather than silently running on CPU -- which is the
intended "let unsupported devices fail" behavior for a cross-EP sweep.
"""
from __future__ import annotations

import argparse
import atexit
import copy
import csv
import gc
import json
import os
import re
import statistics
import sys
import tempfile
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# onnxruntime_genai import shim: prefer the freshly built extension in the build
# tree (the installed wheel may be an older release without the speculative API).
# ---------------------------------------------------------------------------

def _import_og(build_root: str, use_installed: bool = False):
    """Import onnxruntime_genai with the speculative API.

    Default: prefer a freshly built extension in the local build tree (handy for
    local dev, since an installed wheel may predate the speculative API), falling
    back to the installed wheel when no build tree is present.

    use_installed=True skips the build tree entirely and imports the pip-installed
    wheel -- this is what ep-cert / CI should use so the package under test is the
    one pulled from the ORT-Nightly feed, never a stale local build.
    """
    if not use_installed:
        pyd = os.path.join(build_root, "build", "Windows", "Release", "src", "python", "Release")
        dll = os.path.join(build_root, "build", "Windows", "Release", "Release")
        if os.path.isdir(dll) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(dll)
        if os.path.isdir(pyd):
            sys.path.insert(0, pyd)
    else:
        pyd = "(installed wheel)"
    import onnxruntime_genai as og  # noqa: E402
    if not hasattr(og.GeneratorParams, "set_speculative_options"):
        raise RuntimeError(
            "Imported onnxruntime_genai lacks the speculative API. Install a wheel "
            "built from the speculative-decoding branch (or build locally). "
            f"Loaded from: {og.__file__}")
    return og


class _Tee:
    """Duplicate a text stream to several sinks (console + per-run .log file).

    Only Python-level writes are captured; native EP logs (e.g. OpenVINO) go to
    the OS stderr and are not mirrored here, keeping the .log to the run's own
    progress + summary (and any Python traceback).
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def _setup_run_log(out_prefix: str) -> str:
    """Tee all console output to ``out_prefix + '.log'`` for the duration of the run."""
    log_path = out_prefix + ".log"
    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    atexit.register(log_fh.close)
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)
    return log_path


# ---------------------------------------------------------------------------
# Model resolution: Foundry-Local layout (<root>/<id>/onnx/<device>/v<N>/) with
# a flat-layout fallback (<root>/<id>/) for local dev. Mirrors the integration
# suite's resolver (test/python/integration/resolver.py): newest vN wins.
# ---------------------------------------------------------------------------

_VERSION_DIR = re.compile(r"^v(\d+)$")

# Default Foundry-Local onnx/<device> subdir per benchmark EP. NPU/GPU IHV dir
# names vary in the catalog, so those require an explicit --device-dir.
_FL_DEVICE_DIRS = {
    None: "cpu_and_mobile",   # follow_config / cpu
    "cpu": "cpu_and_mobile",
    "cuda": "cuda",
    "webgpu": "webgpu",
    "dml": "dml",
}


def _newest_version_dir(base: str) -> str | None:
    if not os.path.isdir(base):
        return None
    best, best_n = None, -1
    for name in os.listdir(base):
        m = _VERSION_DIR.match(name)
        full = os.path.join(base, name)
        if m and os.path.isdir(full):
            n = int(m.group(1))
            if n > best_n:
                best_n, best = n, full
    return best


def default_fl_device_dir(provider: str | None) -> str | None:
    """FL onnx/<device> subdir implied by the chosen EP (None if it must be given)."""
    return _FL_DEVICE_DIRS.get(provider)


def resolve_model_dir(models_root: str, logical_id: str, device_dir: str | None) -> str:
    """Resolve a logical model id to an on-disk genai model directory.

    Tries the Foundry-Local layout first when a device_dir is known
    (``<root>/<id>/onnx/<device_dir>/v<N>/``, newest N, or that dir directly),
    then falls back to the flat layout ``<root>/<id>/``. Raises with both
    attempted paths if neither has a genai_config.json -- a loud, intended
    failure for a device whose model isn't published (e.g. an unsupported EP).
    """
    attempted = []
    if device_dir:
        base = os.path.join(models_root, logical_id, "onnx", device_dir)
        for cand in (_newest_version_dir(base), base):
            if cand:
                attempted.append(cand)
                if os.path.exists(os.path.join(cand, "genai_config.json")):
                    return cand
    flat = os.path.join(models_root, logical_id)
    attempted.append(flat)
    if os.path.exists(os.path.join(flat, "genai_config.json")):
        return flat
    raise FileNotFoundError(
        f"No genai_config.json for model '{logical_id}' (device_dir={device_dir!r}). "
        f"Tried: {', '.join(attempted)}")


def resolve_model_arg(models_root: str, value: str, model_prefix: str,
                      device_dir: str | None) -> str:
    """Resolve a --target/--draft value to an on-disk genai model directory.

    Accepts either:
      * a short KEY (e.g. ``8b``), resolved as ``model_prefix + key`` under
        ``models_root`` via resolve_model_dir (Foundry-Local ``onnx/<device>/v<N>``
        layout, then flat), or
      * an explicit model DIRECTORY (absolute or relative). If it already holds a
        genai_config.json it is used as-is; otherwise the newest ``v<N>`` subdir or
        the first immediate subdir containing a genai_config.json is used. This is
        the path ep-cert/CI passes: the Foundry cache dir
        ``<cache>/Microsoft/<name>-<version>/<subdir>/`` resolved on the agent.
    """
    if os.path.isdir(value):
        if os.path.exists(os.path.join(value, "genai_config.json")):
            return os.path.abspath(value)
        cand = _newest_version_dir(value)
        if cand and os.path.exists(os.path.join(cand, "genai_config.json")):
            return os.path.abspath(cand)
        for name in sorted(os.listdir(value)):
            sub = os.path.join(value, name)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "genai_config.json")):
                return os.path.abspath(sub)
        raise FileNotFoundError(
            f"model directory {value!r} has no genai_config.json "
            f"(checked it, newest v<N>, and immediate subdirs)")
    return resolve_model_dir(models_root, f"{model_prefix}{value}", device_dir)


# ---------------------------------------------------------------------------
# Speculative config composition (target + draft -> one speculative config dir).
# ---------------------------------------------------------------------------

def build_spec_config(target_path: str, draft_path: str, out_dir: str,
                      provider: str | None = None, device: str | None = None) -> str:
    """Splice already-resolved target + draft model dirs into one speculative config.

    ``target_path`` / ``draft_path`` are absolute genai model directories (each
    holding a genai_config.json + its onnx weights), as returned by
    resolve_model_dir. If ``provider`` is set, the same EP and optional ``device``
    filter (cpu/gpu/npu) is written into BOTH the target (decoder) and draft
    blocks -- speculative decoding requires identical EP config on both, and the
    device filter pins e.g. OpenVINO to the GPU/NPU instead of its AUTO default.
    """
    with open(os.path.join(target_path, "genai_config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(draft_path, "genai_config.json")) as f:
        draft_cfg = json.load(f)

    cfg["model"]["type"] = "speculative"
    # onnxruntime-genai resolves `filename` by concatenating it onto the config
    # dir (it does NOT honor absolute paths), so express each model path as a
    # path RELATIVE to the generated config dir. Read the actual onnx filename
    # from each config (Foundry-Local models are not always named model.onnx).
    # External weights (e.g. model.onnx.data) then load from alongside the
    # resolved onnx file.
    out_abs = os.path.abspath(out_dir)
    tgt_onnx = os.path.abspath(os.path.join(target_path, cfg["model"]["decoder"]["filename"]))
    dft_onnx = os.path.abspath(os.path.join(draft_path, draft_cfg["model"]["decoder"]["filename"]))
    cfg["model"]["decoder"]["filename"] = os.path.relpath(tgt_onnx, out_abs)
    draft_block = copy.deepcopy(draft_cfg["model"]["decoder"])
    draft_block["filename"] = os.path.relpath(dft_onnx, out_abs)
    cfg["model"]["draft"] = draft_block
    # `speculative` is a sibling of `model`/`search` (placement-sensitive parser).
    # K is overridden per-run via set_speculative_options, so this default is moot.
    cfg["speculative"] = {"max_draft_tokens": 4}
    # Speculative decoding rewinds the target KV cache on each rejection. On the
    # fully-dynamic CPU / WebGPU path we disable the shared past/present buffer so
    # DefaultKeyValueCache::RewindTo resizes the cache. But hardware backends chosen
    # by an explicit --device (OpenVINO / VitisAI / QNN NPU, TensorRT) COMPILE to
    # static shapes and REQUIRE the shared buffer: forcing it off leaves the KV
    # seq_len dim symbolic and the NPU compiler aborts ("Got non broadcastable
    # dimensions ... to_shape was called on a dynamic shape"). For those we inherit
    # the stock model's value (the same setting the standard baseline loads, which
    # compiles + runs on the NPU). This stays correct for speculative decoding:
    # under a shared buffer RewindTo is a no-op and the target is re-anchored via the
    # total_length passed to the next Run (the same mechanism CUDA uses), so rejected
    # KV entries are simply overwritten on the following step.
    if not device:
        cfg["search"]["past_present_share_buffer"] = False

    # Pin the execution provider on BOTH blocks (target decoder + draft). The
    # engine requires target and draft to use the same EP, and the device filter
    # (hardware_device_type) selects GPU/NPU instead of OpenVINO's AUTO default.
    if provider:
        opts: dict = {}
        if device:
            opts["device_filtering_options"] = {"hardware_device_type": device.upper()}
        prov_opts = [{provider: opts}]
        for block in ("decoder", "draft"):
            cfg["model"][block].setdefault("session_options", {})["provider_options"] = \
                copy.deepcopy(prov_opts)

    # Recreate the output dir from scratch so support files (tokenizer, chat
    # template, etc.) always match the current target. Reusing a stale dir from a
    # previous target/draft pair would silently load the wrong tokenizer.
    import shutil
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    # Copy tokenizer / chat-template support files from the target (skip weights).
    for name in os.listdir(target_path):
        if name.endswith((".onnx", ".onnx.data")) or name in ("genai_config.json", "model_config.json"):
            continue
        src = os.path.join(target_path, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, name))
    with open(os.path.join(out_dir, "genai_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return out_dir


# ---------------------------------------------------------------------------
# Execution-provider registration + model loading (mirrors benchmark_e2e.py /
# examples/python/common.py + winml.py).
# ---------------------------------------------------------------------------

# Plug-in EPs registered through the Windows ML catalog. All ship as MSIX EP
# packages that the catalog enumerates + registers; requesting any of them sets
# use_winml=True so register_execution_providers() takes the EpCatalog path.
_WINML_PLUGIN_EPS = {
    "OpenVINOExecutionProvider",         # Intel (LNL): gpu=Arc, npu=AI Boost
    "VitisAIExecutionProvider",          # AMD (STX)
    "QNNExecutionProvider",              # Qualcomm (QNN)
    "NvTensorRTRTXExecutionProvider",    # NVIDIA (RTX): TensorRT-RTX
}


def _maybe_register_webgpu(og) -> bool:
    """Register the onnxruntime-ep-webgpu plugin so append_provider("webgpu") works.

    The base onnxruntime package doesn't ship a WebGPU EP; the plugin package
    provides it as a separate shared library that must be registered with ORT
    GenAI first. Mirrors test/python/integration/test_integration_text.py.
    Returns True on success (or if already registered), False if the plugin
    package is not installed.
    """
    if getattr(_maybe_register_webgpu, "_done", False):
        return True
    try:
        import onnxruntime_ep_webgpu as webgpu_ep  # noqa: PLC0415
    except ImportError:
        return False
    og.register_execution_provider_library("webgpu", webgpu_ep.get_library_path())
    _maybe_register_webgpu._done = True
    return True


def register_execution_providers(og, use_winml: bool, ep_library_path: str | None,
                                 ep_name: str | None) -> None:
    """Register plug-in execution providers with onnxruntime-genai.

    With ``use_winml`` the Windows ML catalog is queried and every ready provider
    (e.g. OpenVINO for Intel GPU/NPU) is registered; otherwise an explicit
    ``ep_library_path`` is loaded. CPU / follow_config needs neither.
    """
    if use_winml:
        try:
            from windowsml import EpCatalog
        except ImportError as e:
            raise RuntimeError(
                "windowsml is required to register the OpenVINO EP via Windows ML. "
                "Install it with `pip install windowsml`.") from e
        with EpCatalog() as catalog:
            for prov in catalog.find_all_providers():
                prov.ensure_ready()
                if prov.library_path:
                    og.register_execution_provider_library(prov.name, prov.library_path)
                    print(f"  registered WinML EP: {prov.name}")
    elif ep_library_path and ep_name:
        og.register_execution_provider_library(ep_name, ep_library_path)
        print(f"  registered {ep_name} from {ep_library_path}")


def load_model(og, model_dir: str, provider: str | None, device: str | None):
    """Load a single (non-speculative) model, optionally forcing an EP + device.

    Mirrors benchmark_e2e.py: clear the config's providers and append the chosen
    EP, then pin the hardware device (OpenVINO: gpu=Arc, npu=AI Boost). With no
    provider the model loads exactly as its genai_config specifies (CPU).
    """
    if not provider:
        return og.Model(model_dir)
    cfg = og.Config(model_dir)
    cfg.clear_providers()
    cfg.append_provider(provider)
    if device:
        cfg.set_decoder_provider_options_hardware_device_type(provider, device.upper())
    return og.Model(cfg)


# ---------------------------------------------------------------------------
# Prompts (mimic chat.py: real instructions, wrapped in the chat template).
# ---------------------------------------------------------------------------

# A fixed, reproducible set of real-world prompts. They are decoded as genuine
# assistant answers (via the chat template), so they exercise normal-entropy
# generation -- the regime where draft/target agreement is actually meaningful.
# Edit this list to benchmark your own workload; results stay reproducible.
PROMPTS = [
    "How are astronauts launched into space quickly on those rockets?",
    "Today, we will learn how to bake a chocolate cake. First, you need to "
    "have all of the ingredients to bake. Otherwise, the chocolate cake won't "
    "be tasty. You will also need a large baking pan to hold the batter.",
    "Explain how a transformer neural network works in simple terms.",
    "Write a short story about a robot that learns to paint.",
    "What are the main differences between Python and C++?",
    "Give me three practical tips for improving my sleep quality.",
]

# Spec-Bench tasks whose outputs heavily overlap the prompt (the model echoes the
# input), which structurally inflates acceptance. We still report them, but flag
# them so they're never mistaken for general open-ended generation performance.
INPUT_GUIDED_TASKS = {"translation", "summarization", "rag"}

# The 8 MT-Bench subcategories collapse into a single "mt_bench" task.
_MT_BENCH_SUBCATS = {"writing", "roleplay", "reasoning", "math", "coding",
                     "extraction", "stem", "humanities"}


def _task_of(category):
    return "mt_bench" if category in _MT_BENCH_SUBCATS else category


def builtin_prompts(max_prompts=0):
    """Wrap the built-in smoke-test PROMPTS as prompt items."""
    lst = PROMPTS if max_prompts <= 0 else PROMPTS[:max_prompts]
    return [{"task": "builtin", "question_id": i, "subcategory": "builtin",
             "text": t} for i, t in enumerate(lst)]


def load_dataset_prompts(path, tasks=None, limit_per_task=0,
                         mt_bench_by_subcategory=False):
    """Load a Spec-Bench-style question.jsonl into prompt items.

    Each line is {"question_id", "category", "turns": [...]}. We use turns[0]
    ONLY (single-turn: turn 2 of MT-Bench says "your previous response", which is
    meaningless without first generating turn 1 -- out of scope here). The 8
    MT-Bench subcategories normally collapse into one "mt_bench" task. --tasks
    filters to a subset; --limit-per-task takes the first N of each task so a full
    480-prompt run stays tractable on CPU. Returns dicts: {task, question_id,
    subcategory, text}.

    mt_bench_by_subcategory=True keeps ONLY the 8 MT-Bench subcategories and
    treats each one as its own task (writing, roleplay, reasoning, math, coding,
    extraction, stem, humanities), so --limit-per-task N yields N prompts per
    subcategory and the summary reports each subcategory separately.
    """
    import collections
    buckets = collections.OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            turns = r.get("turns") or []
            if not turns:
                continue
            category = r.get("category", "unknown")
            if mt_bench_by_subcategory:
                # MT-Bench only; each subcategory is its own task.
                if category not in _MT_BENCH_SUBCATS:
                    continue
                task = category
            else:
                task = _task_of(category)
            if tasks and task not in tasks:
                continue
            buckets.setdefault(task, []).append({
                "task": task,
                "question_id": r.get("question_id"),
                "subcategory": category,
                "text": turns[0],
            })
    items = []
    for task, lst in buckets.items():
        items.extend(lst if limit_per_task <= 0 else lst[:limit_per_task])
    return items


def encode_prompt(tokenizer, prompt, chat=True, think=False):
    """Wrap a prompt exactly like chat.py and return its token ids.

    chat=True applies the model's chat/instruct template (so the model answers
    instead of doing raw text completion); think=False seeds an empty
    ``<think></think>`` block to put Qwen3 in concise no-think mode. chat=False
    falls back to raw text completion.
    """
    if chat:
        msgs = json.dumps([{"role": "user", "content": prompt}])
        try:
            prompt = tokenizer.apply_chat_template(
                messages=msgs, add_generation_prompt=True)
            if not think:
                prompt += "<think>\n\n</think>\n\n"
        except Exception as e:  # noqa: BLE001 - fall back to raw completion
            print(f"[chat template failed, using raw: {e}]")
    return list(int(t) for t in tokenizer.encode(prompt))


# ---------------------------------------------------------------------------
# One measured generation.
# ---------------------------------------------------------------------------

def run_once(og, model, ids, max_new, mode, speculative, K, seed):
    import numpy as np
    p = og.GeneratorParams(model)
    # Speculative decoding rewinds the KV cache, so it needs a non-shared
    # past/present buffer. We use it for BOTH decoders so the only thing the
    # comparison measures is speculative vs standard (matches chat.py). min_length
    # is left unset -- speculative rejects min_length > 0, and chat.py never pins
    # it -- so both decoders may stop early on EOS, the fair/realistic behavior.
    opts = dict(max_length=len(ids) + max_new, past_present_share_buffer=False)
    if mode == "sampling":
        # Same knobs as chat.py, but a FIXED seed so sampling runs reproduce.
        opts.update(do_sample=True, temperature=0.7, top_k=20, top_p=0.95,
                    random_seed=seed)
    else:
        opts.update(do_sample=False)
    p.set_search_options(**opts)
    if speculative:
        p.set_speculative_options(max_draft_tokens=K)

    g = og.Generator(model, p)

    t0 = time.perf_counter()
    g.append_tokens(np.array([ids], dtype=np.int32))
    prefill_s = time.perf_counter() - t0

    start_len = g.token_count()
    target = start_len + max_new
    t1 = time.perf_counter()
    while not g.is_done() and g.token_count() < target:
        g.generate_next_token()
    decode_s = time.perf_counter() - t1

    new_tokens = g.token_count() - start_len
    seq = list(int(t) for t in g.get_sequence(0))
    stats = g.get_speculative_stats() if speculative else None
    del g
    gc.collect()
    return dict(prefill_s=prefill_s, decode_s=decode_s, new_tokens=new_tokens,
                tail=seq[start_len:], stats=stats)


CSV_COLUMNS = [
    "mode", "K", "task", "subcategory", "question_id", "prompt_id", "rep", "decoder",
    "new_tokens", "prefill_s", "decode_s", "decode_tok_s", "e2e_tok_s",
    "speedup_decode", "speedup_e2e",
    "acceptance_rate", "rounds", "draft_proposed", "draft_accepted",
    "corrections", "bonuses", "mean_accepted_tokens", "greedy_match",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    ap.add_argument("--build-root", default=repo_root, help="repo root holding build/")
    ap.add_argument("--models-root",
                    default=os.path.join(repo_root, "test", "test_models", "qwen3-speculative"),
                    help="dir holding qwen3-<size> model folders")
    ap.add_argument("--target", default="8b", help="target model key, e.g. 8b")
    ap.add_argument("--draft", default="1.7b", help="draft model key, e.g. 1.7b")
    ap.add_argument("--model-prefix", default="qwen3-", help="folder = prefix + key")
    # ---- the only two swept variables ----
    ap.add_argument("--modes", default="greedy,sampling", help="sweep var 1: greedy and/or sampling")
    ap.add_argument("--k", default="1,2,4,8", help="sweep var 2: comma list of max_draft_tokens")
    # ---- fixed run configuration (NOT swept) ----
    ap.add_argument("--max-new-tokens", type=int, default=128,
                    help="new tokens to generate per prompt (fixed)")
    # ---- prompt source: Spec-Bench dataset (preferred) or built-in smoke-test ----
    ap.add_argument("--dataset", default=None,
                    help="path to Spec-Bench question.jsonl; overrides the built-in "
                         "PROMPTS. Uses turns[0] only (single-turn).")
    ap.add_argument("--tasks", default=None,
                    help="comma list of tasks to include when --dataset is set: "
                         "mt_bench,translation,summarization,qa,math_reasoning,rag "
                         "(default: all present)")
    ap.add_argument("--limit-per-task", type=int, default=0,
                    help="cap prompts per task (0 = all, e.g. 80); subsample for CPU runs")
    ap.add_argument("--mt-bench-by-subcategory", action="store_true",
                    help="with --dataset: keep ONLY the 8 MT-Bench subcategories "
                         "(writing/roleplay/reasoning/math/coding/extraction/stem/"
                         "humanities), each as its own task, so --limit-per-task N "
                         "gives N prompts per subcategory (e.g. 5 -> 40 prompts).")
    ap.add_argument("--max-prompts", type=int, default=0,
                    help="cap built-in prompts (0 = all); only used without --dataset")
    ap.add_argument("--reps", type=int, default=2, help="measured repetitions per config")
    ap.add_argument("--warmup", type=int, default=1, help="warmup generations per model")
    ap.add_argument("--seed", type=int, default=0, help="FIXED sampling seed (reproducibility)")
    ap.add_argument("--raw", action="store_true",
                    help="raw text completion (skip the chat template)")
    ap.add_argument("--think", action="store_true",
                    help="enable Qwen3 reasoning (default: concise no-think)")
    ap.add_argument("-o", "--output", default=None, help="output path prefix (no extension)")
    # ---- execution provider (mirrors benchmark_e2e.py) ----
    ap.add_argument("-e", "--execution-provider", default="follow_config",
                    choices=["follow_config", "cpu", "OpenVINOExecutionProvider",
                             "VitisAIExecutionProvider", "QNNExecutionProvider",
                             "NvTensorRTRTXExecutionProvider", "cuda", "dml", "webgpu"],
                    help="EP to run on. follow_config/cpu = genai_config as-is (CPU). "
                         "OpenVINOExecutionProvider = Intel GPU/NPU (use with --device).")
    ap.add_argument("--device", default=None, choices=["cpu", "gpu", "npu"],
                    help="hardware device filter for the EP (OpenVINO: gpu=Arc, npu=AI Boost).")
    ap.add_argument("--device-dir", default=None,
                    help="Foundry-Local onnx/<device-dir> subfolder to resolve models from "
                         "(e.g. cpu_and_mobile, cuda, webgpu). Defaults from -e for "
                         "cpu/cuda/webgpu/dml; REQUIRED for NPU/IHV EPs. Ignored for flat layout.")
    ap.add_argument("--use-installed", action="store_true",
                    help="import the pip-installed onnxruntime-genai wheel instead of a local "
                         "build tree (use in ep-cert/CI so the feed package is what runs).")
    ap.add_argument("--use-winml", action="store_true",
                    help="register EPs via the Windows ML catalog (auto-on for OpenVINO/QNN/VitisAI).")
    ap.add_argument("--ep-library-path", default=None,
                    help="explicit EP plug-in library to register (instead of --use-winml).")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in ("greedy", "sampling"):
            ap.error(f"unknown mode {m!r} (greedy|sampling)")
    ks = [int(x) for x in args.k.split(",") if x.strip()]
    for k in ks:
        if not (1 <= k <= 16):
            ap.error(f"K={k} out of range [1,16]")

    if args.mt_bench_by_subcategory and not args.dataset:
        ap.error("--mt-bench-by-subcategory requires --dataset (the question.jsonl "
                 "holding the MT-Bench subcategories)")

    if args.dataset:
        tasks_filter = (set(t.strip() for t in args.tasks.split(",") if t.strip())
                        if args.tasks else None)
        prompt_items = load_dataset_prompts(args.dataset, tasks_filter,
                                            args.limit_per_task,
                                            mt_bench_by_subcategory=args.mt_bench_by_subcategory)
        if not prompt_items:
            ap.error("no prompts loaded from --dataset (check the path / --tasks / "
                     "--limit-per-task / --mt-bench-by-subcategory)")
    else:
        prompt_items = builtin_prompts(args.max_prompts)
    chat = not args.raw
    think = args.think
    max_new = args.max_new_tokens

    import collections as _c
    task_counts = _c.Counter(it["task"] for it in prompt_items)

    # --target/--draft may be a short key (composed with --model-prefix) or an
    # explicit model directory (ep-cert passes the resolved Foundry cache dir);
    # use a short leaf label for filenames/prints either way.
    tgt_label = (os.path.basename(os.path.normpath(args.target))
                 if os.path.isdir(args.target) else args.target)
    dft_label = (os.path.basename(os.path.normpath(args.draft))
                 if os.path.isdir(args.draft) else args.draft)
    # Resolve the execution provider + device filter from the EP args.
    provider = None if args.execution_provider in ("follow_config", "cpu") else args.execution_provider
    device = args.device
    use_winml = args.use_winml or (provider in _WINML_PLUGIN_EPS and not args.ep_library_path)
    # Foundry-Local onnx/<device-dir> to resolve models from: explicit, else derived from the EP.
    device_dir = args.device_dir or default_fl_device_dir(provider)
    # Results land in results/<ep-tag>/spec_<target>_<draft>_<timestamp>.{csv,json}
    ep_folder = (device or ("cpu" if provider is None else provider)).lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_prefix = args.output or os.path.join(
        here, "results", ep_folder, f"spec_{tgt_label}_{dft_label}_{timestamp}")
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    log_path = _setup_run_log(out_prefix)

    og = _import_og(args.build_root, use_installed=args.use_installed)
    print(f"onnxruntime_genai: {og.__file__}")
    print(f"prompts={len(prompt_items)}  tasks={dict(task_counts)}  modes={modes}  "
          f"K={ks}  max_new={max_new}  chat={chat}  think={think}  seed={args.seed}")
    print(f"execution_provider={provider or 'cpu'}  device={device or '-'}  "
          f"device_dir={device_dir or '-'}  use_winml={use_winml}  results={out_prefix}")

    # Register plug-in EPs before any model load. CPU / follow_config needs none.
    # WebGPU ships as a separate plugin package (onnxruntime-ep-webgpu); the IHV
    # EPs (OpenVINO/QNN/VitisAI) come via the Windows ML catalog or an explicit lib.
    if provider == "webgpu":
        if not _maybe_register_webgpu(og):
            raise RuntimeError(
                "webgpu EP requested but 'onnxruntime-ep-webgpu' is not installed "
                "(pip install onnxruntime-ep-webgpu).")
    elif provider:
        register_execution_providers(og, use_winml, args.ep_library_path, provider)

    # Resolve target + draft to on-disk dirs (Foundry-Local layout, flat fallback).
    # A missing model here fails fast (naming the tried paths) -- the intended
    # behavior when an EP's model isn't published for this device.
    target_path = resolve_model_arg(args.models_root, args.target, args.model_prefix, device_dir)
    draft_path = resolve_model_arg(args.models_root, args.draft, args.model_prefix, device_dir)
    print(f"target={target_path}")
    print(f"draft ={draft_path}")

    spec_dir = build_spec_config(target_path, draft_path,
                                 os.path.join(tempfile.gettempdir(),
                                              f"ogspec_{tgt_label}_{dft_label}"),
                                 provider=provider, device=device)
    std_dir = target_path

    rows = []
    csv_path = out_prefix + ".csv"
    json_path = out_prefix + ".json"

    def flush():
        # Rewrites the full CSV+JSON after every completed config. Called
        # incrementally in both phases so a mid-sweep failure (e.g. an
        # unsupported EP that throws on model load) still leaves all completed
        # metrics on disk for the CI artifact step, while the exception
        # propagates and the process exits non-zero ("let unsupported fail").
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)

    # baseline (per mode, prompt) + speculative (per mode, prompt, K)
    total_cfgs = len(modes) * len(prompt_items) * (1 + len(ks))
    done = 0
    bench_t0 = time.time()

    # Two-phase design (memory-safe): run all STANDARD baselines with only the
    # target loaded, free it, then load the speculative pair (target+draft) and
    # run all speculative configs. Peak RAM = target+draft, not target*2+draft,
    # which matters for the larger pairs (e.g. 8b target).
    baselines = {}  # (mode, prompt_id) -> dict(dec, e2e, tail)

    # ---- Phase 1: standard baseline (target only) ----
    print(f"[phase 1/2] Loading standard baseline (target={tgt_label}) ...", flush=True)
    t0 = time.time()
    std_model = load_model(og, std_dir, provider, device)
    print(f"  loaded in {time.time()-t0:.1f}s")
    tokenizer = og.Tokenizer(std_model)

    # Pre-encode every prompt once (chat template applied here); the token-id
    # lists are reused verbatim by the speculative phase (same Qwen3 tokenizer).
    encoded = [encode_prompt(tokenizer, it["text"], chat=chat, think=think)
               for it in prompt_items]
    for idx, (it, ids) in enumerate(zip(prompt_items, encoded)):
        print(f"  [{idx}] {it['task']}/{it['question_id']}: {len(ids)} tokens")

    if args.warmup:
        print("Warming up (standard) ...", flush=True)
        for _ in range(args.warmup):
            run_once(og, std_model, encoded[0], 16, "greedy", False, 0, args.seed)

    for mode in modes:
        for idx, (it, ids) in enumerate(zip(prompt_items, encoded)):
            base_dec, base_e2e, base_tail = [], [], None
            for rep in range(args.reps):
                r = run_once(og, std_model, ids, max_new, mode, False, 0, args.seed)
                dec_tps = r["new_tokens"] / r["decode_s"] if r["decode_s"] else 0.0
                e2e_tps = r["new_tokens"] / (r["prefill_s"] + r["decode_s"]) \
                    if (r["prefill_s"] + r["decode_s"]) else 0.0
                base_dec.append(dec_tps)
                base_e2e.append(e2e_tps)
                base_tail = r["tail"]
                rows.append(dict(
                    mode=mode, K="", task=it["task"], subcategory=it["subcategory"],
                    question_id=it["question_id"], prompt_id=idx, rep=rep, decoder="standard",
                    new_tokens=r["new_tokens"], prefill_s=round(r["prefill_s"], 4),
                    decode_s=round(r["decode_s"], 4), decode_tok_s=round(dec_tps, 3),
                    e2e_tok_s=round(e2e_tps, 3), speedup_decode="", speedup_e2e="",
                    acceptance_rate="", rounds="", draft_proposed="", draft_accepted="",
                    corrections="", bonuses="", mean_accepted_tokens="", greedy_match=""))
            baselines[(mode, idx)] = dict(
                dec=statistics.median(base_dec), e2e=statistics.median(base_e2e),
                tail=base_tail)
            done += 1
            print(f"[{done}/{total_cfgs}] {mode} {it['task']}/{it['question_id']} standard: "
                  f"{baselines[(mode, idx)]['dec']:.1f} tok/s decode", flush=True)
            flush()

    # Free the standalone target before loading the (larger) speculative pair.
    del std_model, tokenizer
    gc.collect()

    # ---- Phase 2: speculative (target + draft) ----
    print(f"[phase 2/2] Loading speculative model "
          f"(target={tgt_label}, draft={dft_label}) ...", flush=True)
    t0 = time.time()
    spec_model = og.Model(spec_dir)
    print(f"  loaded in {time.time()-t0:.1f}s")

    if args.warmup:
        print("Warming up (speculative) ...", flush=True)
        for _ in range(args.warmup):
            run_once(og, spec_model, encoded[0], 16, "greedy", True, ks[0], args.seed)

    for mode in modes:
        for idx, (it, ids) in enumerate(zip(prompt_items, encoded)):
            b = baselines[(mode, idx)]
            base_dec_med, base_e2e_med, base_tail = b["dec"], b["e2e"], b["tail"]
            for K in ks:
                s_dec = []
                last = None
                for rep in range(args.reps):
                    r = run_once(og, spec_model, ids, max_new, mode, True, K, args.seed)
                    st = r["stats"] or {}
                    dec_tps = r["new_tokens"] / r["decode_s"] if r["decode_s"] else 0.0
                    e2e_tps = r["new_tokens"] / (r["prefill_s"] + r["decode_s"]) \
                        if (r["prefill_s"] + r["decode_s"]) else 0.0
                    s_dec.append(dec_tps)
                    last = st
                    match = ""
                    if mode == "greedy" and base_tail is not None:
                        n = min(len(base_tail), len(r["tail"]))
                        match = round(sum(1 for a, c in zip(base_tail[:n], r["tail"][:n])
                                          if a == c) / n, 4) if n else ""
                    rows.append(dict(
                        mode=mode, K=K, task=it["task"], subcategory=it["subcategory"],
                        question_id=it["question_id"], prompt_id=idx, rep=rep,
                        decoder="speculative",
                        new_tokens=r["new_tokens"], prefill_s=round(r["prefill_s"], 4),
                        decode_s=round(r["decode_s"], 4), decode_tok_s=round(dec_tps, 3),
                        e2e_tok_s=round(e2e_tps, 3),
                        speedup_decode=round(dec_tps / base_dec_med, 3) if base_dec_med else "",
                        speedup_e2e=round(e2e_tps / base_e2e_med, 3) if base_e2e_med else "",
                        acceptance_rate=round(st.get("acceptance_rate", 0.0), 4),
                        rounds=st.get("rounds", ""),
                        draft_proposed=st.get("draft_tokens_proposed", ""),
                        draft_accepted=st.get("draft_tokens_accepted", ""),
                        corrections=st.get("correction_tokens", ""),
                        bonuses=st.get("bonus_tokens", ""),
                        mean_accepted_tokens=round(
                            st.get("mean_accepted_tokens")
                            or st.get("effective_speedup", 0.0), 3),
                        greedy_match=match))
                s_dec_med = statistics.median(s_dec)
                acc = (last or {}).get("acceptance_rate", 0.0)
                done += 1
                sp = s_dec_med / base_dec_med if base_dec_med else 0.0
                print(f"[{done}/{total_cfgs}] {mode} {it['task']}/{it['question_id']} "
                      f"spec K={K}: {s_dec_med:.1f} tok/s  speedup x{sp:.2f}  "
                      f"accept={acc:.0%}", flush=True)
                flush()

    flush()
    print(f"\nDone in {time.time()-bench_t0:.0f}s. Wrote {csv_path}, {json_path} and {log_path}")
    print_summary(rows, modes, ks)


def print_summary(rows, modes, ks):
    """Per-task median (across prompts x reps) of speedup / acceptance /
    mean-accepted-tokens, plus an overall geometric-mean speedup per (mode, K).

    Per-task reporting is mandatory for speculative decoding: acceptance is
    entropy/domain-dependent, and input-guided tasks (translation/summarization/
    rag) inflate it via prompt->output overlap. The headline number the field
    reports is the GEOMETRIC MEAN of per-task speedups (Spec-Bench convention).
    """
    import math

    # Preserve task order as first seen among speculative rows.
    tasks = []
    for r in rows:
        if r["decoder"] == "speculative" and r["task"] not in tasks:
            tasks.append(r["task"])

    def med_spec(mode, task, K, key):
        vals = [r[key] for r in rows
                if r["decoder"] == "speculative" and r["mode"] == mode
                and r["task"] == task and r["K"] == K and r[key] != ""]
        return statistics.median(vals) if vals else None

    def med_std(mode, task):
        vals = [r["decode_tok_s"] for r in rows
                if r["decoder"] == "standard" and r["mode"] == mode
                and r["task"] == task and r["decode_tok_s"] != ""]
        return statistics.median(vals) if vals else None

    print("\n===== SUMMARY: per-task median across prompts x reps "
          "(mean_acc_tok = #Mean Accepted Tokens) =====")
    for mode in modes:
        print(f"\n#### mode = {mode} ####")
        for K in ks:
            print(f"\n  K={K}")
            print(f"    {'task':16} {'std tok/s':>9} {'spec tok/s':>10} {'speedup':>8} "
                  f"{'accept':>7} {'mean_acc_tok':>12} {'match':>6}")
            speedups = []
            for task in tasks:
                std = med_std(mode, task)
                spec = med_spec(mode, task, K, "decode_tok_s")
                if std is None or spec is None:
                    continue
                sp = med_spec(mode, task, K, "speedup_decode")
                if sp is None:
                    sp = spec / std if std else 0.0
                acc = med_spec(mode, task, K, "acceptance_rate") or 0.0
                mat = med_spec(mode, task, K, "mean_accepted_tokens") or 0.0
                gm = med_spec(mode, task, K, "greedy_match")
                matchs = f"{gm:.0%}" if gm not in (None, "") else "-"
                flag = "*" if task in INPUT_GUIDED_TASKS else ""
                speedups.append(sp)
                print(f"    {task + flag:16} {std:>9.1f} {spec:>10.1f} "
                      f"{'x' + format(sp, '.2f'):>8} {acc:>6.0%} {mat:>12.2f} {matchs:>6}")
            if speedups:
                geo = math.exp(sum(math.log(max(s, 1e-9)) for s in speedups) / len(speedups))
                print(f"    {'OVERALL geomean':16} {'':>9} {'':>10} "
                      f"{'x' + format(geo, '.2f'):>8}")
    print("\n  * input-guided task (prompt->output overlap inflates acceptance; "
          "report separately)")
    print("=" * 84)


if __name__ == "__main__":
    main()
