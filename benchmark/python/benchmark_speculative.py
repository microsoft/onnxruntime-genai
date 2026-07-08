# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""End-to-end benchmark for speculative decoding in onnxruntime-genai.

Compares speculative decoding (large ``target`` decoder + small ``draft``) against
standard target-only decoding on identical prompts, reporting decode throughput,
speedup, acceptance rate, and peak memory. Prompts are wrapped in the model's chat
template so results reflect real generation, and total decode time over total new
tokens is measured (not per-token latency, which is uneven across speculative rounds).

Two things are swept: ``--modes`` (greedy/sampling) and ``--k`` (max_draft_tokens).
The speculative config is composed on the fly from each model's own
``genai_config.json``, so any target/draft pair works. Prompts default to the bundled
Spec-Bench dataset (``question.jsonl``); ``--limit-per-task`` subsamples it and
``--builtin`` uses a small smoke-test set. Results are reported per task plus an
overall geometric-mean speedup.

Examples:
    # Default local flat layout (<models-root>/qwen3-8b/, qwen3-1.7b/)
    python benchmark_speculative.py --target 8b --draft 1.7b --k 1,2,4,8 --limit-per-task 10

    # Quick built-in smoke test (no dataset)
    python benchmark_speculative.py --target 8b --draft 1.7b --k 1,2,4,8 --builtin

    # Foundry-Local layout + installed wheel on a specific EP
    python benchmark_speculative.py --use-installed --models-root %ORTGENAI_MODEL_ROOT% \
        --model-prefix "" --target qwen3-8b --draft qwen3-0.6b \
        -e cuda --device-dir cuda --k 1,2,4,8 -o results/cuda/spec_qwen3-8b_qwen3-0.6b

Run ``benchmark_speculative.py -h`` for the full list of options.
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
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime


# ---------------------------------------------------------------------------
# Peak-memory monitoring (mirrors benchmark_e2e.py / benchmark_multimodal.py):
# nvidia-smi for GPU memory on NVIDIA systems, else psutil for host RAM. Other
# accelerators (DML/WebGPU/OpenVINO/QNN/VitisAI) report host RAM only.
# ---------------------------------------------------------------------------
try:
    subprocess.run(["nvidia-smi"], check=True, capture_output=True)
    IS_NVIDIA_SYSTEM = True
except Exception:
    IS_NVIDIA_SYSTEM = False


class PeakMemoryMonitor:
    """Background sampler of peak GPU (nvidia-smi) or host RAM (psutil), in GiB.

    Used once per load+run phase so the target-only baseline and target+draft
    speculative footprints are reported separately.
    """

    _warned_no_psutil = False

    def __init__(self):
        self.peak_gpu_gib = 0.0
        self.peak_cpu_gib = 0.0
        self._stop = False
        self._thread = None
        try:
            import psutil  # noqa: PLC0415
            self._psutil = psutil
        except ImportError:
            self._psutil = None
            if not IS_NVIDIA_SYSTEM and not PeakMemoryMonitor._warned_no_psutil:
                print("[warn] psutil not installed and no NVIDIA GPU: peak-memory columns "
                      "will be 0. Install it with `pip install psutil`.")
                PeakMemoryMonitor._warned_no_psutil = True

    def _run(self):
        while not self._stop:
            if IS_NVIDIA_SYSTEM:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    check=False, capture_output=True, text=True)
                lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
                if lines:
                    self.peak_gpu_gib = max(
                        self.peak_gpu_gib, round(max(float(x) for x in lines) / 1024, 2))
            if self._psutil is not None:
                self.peak_cpu_gib = max(
                    self.peak_cpu_gib,
                    round(self._psutil.virtual_memory().used / 1024**3, 2))
            time.sleep(0.1)

    def start(self):
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


def _backfill_memory(rows, decoder, monitor):
    """Stamp a phase's peak memory onto its rows (memory is per-phase, not per-row)."""
    for row in rows:
        if row["decoder"] == decoder:
            row["peak_gpu_mem_gib"] = monitor.peak_gpu_gib if IS_NVIDIA_SYSTEM else ""
            row["peak_cpu_mem_gib"] = monitor.peak_cpu_gib

# ---------------------------------------------------------------------------
# onnxruntime_genai import shim: prefer the freshly built extension in the build
# tree (the installed wheel may be an older release without the speculative API).
# ---------------------------------------------------------------------------

def _import_og(build_root: str, use_installed: bool = False):
    """Import onnxruntime_genai with the speculative API.

    Default prefers a freshly built extension in the local build tree (an
    installed wheel may predate the speculative API), falling back to the
    installed wheel. use_installed=True skips the build tree and imports the
    pip-installed wheel (for ep-cert/CI, so the feed package is what runs).
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
    OS stderr and are not mirrored here.
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

    Tries the Foundry-Local layout when a device_dir is known
    (``<root>/<id>/onnx/<device_dir>/v<N>/``, newest N, or that dir), then the
    flat layout ``<root>/<id>/``. Raises with both attempted paths if neither
    has a genai_config.json.
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

    Accepts either a short KEY (e.g. ``8b``), resolved as ``model_prefix + key``
    under ``models_root`` via resolve_model_dir, or an explicit model DIRECTORY:
    used as-is if it holds a genai_config.json, else the newest ``v<N>`` subdir
    or first immediate subdir that does (the path ep-cert/CI passes).
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

    ``target_path`` / ``draft_path`` are absolute genai model directories from
    resolve_model_dir. If ``provider`` is set, the same EP and optional
    ``device`` filter (cpu/gpu/npu) is written into BOTH the decoder and draft
    blocks (speculative decoding requires identical EP config on both).
    """
    with open(os.path.join(target_path, "genai_config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(draft_path, "genai_config.json")) as f:
        draft_cfg = json.load(f)

    cfg["model"]["type"] = "speculative"
    # onnxruntime-genai resolves `filename` relative to the config dir (absolute
    # paths are not honored), so express each model path relative to out_dir. The
    # onnx filename is read from each config (FL models aren't always model.onnx);
    # external weights (e.g. model.onnx.data) load alongside it.
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
    # past_present_share_buffer is inherited from the target's stock config, so the
    # speculative decoder uses the same KV-buffer policy as the standalone baseline
    # (keeping the comparison apples-to-apples). The shared buffer is the qwen3
    # default and is required by static-shape NPU/compiled EPs.

    # Pin the EP on BOTH blocks (decoder + draft); the engine requires them to match.
    # Preserve each block's stock provider_options and only MERGE the device filter in --
    # do NOT replace them: compiled backends (e.g. OpenVINO NPU) carry stock options
    # (device_type/cache_dir/load_config) pointing at a precompiled static-shape blob, and
    # dropping them forces a fresh compile of the dynamic-shape ONNX, which the NPU compiler
    # rejects ("to_shape was called on a dynamic shape").
    if provider:
        def _merge_provider_options(block_name: str) -> None:
            session_options = cfg["model"][block_name].setdefault("session_options", {})
            po_list = session_options.setdefault("provider_options", [])
            norm = lambda n: n.replace("ExecutionProvider", "")
            entry = None
            for item in po_list:
                if item and norm(next(iter(item))) == norm(provider):
                    entry = item[next(iter(item))]
                    break
            if entry is None:
                entry = {}
                po_list.append({provider: entry})
            if device:
                entry["device_filtering_options"] = {"hardware_device_type": device.upper()}
        _merge_provider_options("decoder")
        _merge_provider_options("draft")
        # Diagnostic: log the composed EP config for both blocks so a mismatch
        # (which the engine's identical-provider-options check rejects) is visible.
        print("decoder provider_options:",
              json.dumps(cfg["model"]["decoder"]["session_options"]["provider_options"]))
        print("draft provider_options:  ",
              json.dumps(cfg["model"]["draft"]["session_options"]["provider_options"]))

    # Recreate the output dir from scratch so support files (tokenizer, chat
    # template) always match the current target, not a stale previous pair.
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

# Plug-in EPs registered through the Windows ML catalog. Requesting any of them
# sets use_winml=True so register_execution_providers() takes the EpCatalog path.
_WINML_PLUGIN_EPS = {
    "OpenVINOExecutionProvider",         # Intel (LNL): gpu=Arc, npu=AI Boost
    "VitisAIExecutionProvider",          # AMD (STX)
    "QNNExecutionProvider",              # Qualcomm (QNN)
    "NvTensorRTRTXExecutionProvider",    # NVIDIA (RTX): TensorRT-RTX
}


def _maybe_register_webgpu(og) -> bool:
    """Register the onnxruntime-ep-webgpu plugin so append_provider("webgpu") works.

    The base onnxruntime package doesn't ship a WebGPU EP; the plugin provides it
    as a separate shared library that must be registered first. Returns True on
    success (or if already registered), False if the plugin isn't installed.
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

# A fixed, reproducible set of real-world prompts, decoded as genuine assistant
# answers via the chat template. Edit this list to benchmark your own workload.
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

# Spec-Bench tasks whose outputs heavily overlap the prompt, structurally inflating
# acceptance. Still reported, but flagged so they're not mistaken for open-ended gen.
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


def _sample_across_subcategories(items, limit_per_task):
    """Take up to ``limit_per_task`` items, round-robin across subcategories.

    ``mt_bench``'s 8 subcategories are stored as contiguous blocks, so a naive
    ``items[:N]`` would sample only the FIRST. Round-robin picks one per
    subcategory in turn (deterministic, no RNG). Single-subcategory tasks reduce
    to ``items[:N]``.
    """
    if limit_per_task <= 0 or limit_per_task >= len(items):
        return items
    import collections
    groups = collections.OrderedDict()
    for it in items:
        groups.setdefault(it["subcategory"], []).append(it)
    out, idx = [], 0
    while len(out) < limit_per_task:
        advanced = False
        for g in groups.values():
            if idx < len(g):
                out.append(g[idx])
                advanced = True
                if len(out) >= limit_per_task:
                    break
        if not advanced:
            break
        idx += 1
    return out


def load_dataset_prompts(path, tasks=None, limit_per_task=0,
                         mt_bench_by_subcategory=False):
    """Load a Spec-Bench-style question.jsonl into prompt items.

    Each line is {"question_id", "category", "turns": [...]}; we use turns[0]
    only (single-turn). The 8 MT-Bench subcategories normally collapse into one
    "mt_bench" task; --tasks filters to a subset and --limit-per-task caps each.
    Returns dicts: {task, question_id, subcategory, text}.

    mt_bench_by_subcategory=True keeps every category as its own task instead
    (8 MT-Bench subcategories + 5 Spec-Bench tasks = 13, nothing collapsed).
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
                task = category  # every category is its own task (no collapsing)
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
        items.extend(_sample_across_subcategories(lst, limit_per_task))
    return items


def encode_prompt(tokenizer, prompt, chat=True, think=False):
    """Wrap a prompt like chat.py and return its token ids.

    chat=True applies the model's chat template; think=False seeds an empty
    ``<think></think>`` block for Qwen3 concise no-think mode. chat=False does
    raw text completion.
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
    # past_present_share_buffer and min_length are inherited from each model's genai_config
    # (not overridden), so baseline and speculative decoders share the same policy and both
    # may stop early on EOS -- keeping the comparison apples-to-apples and faithful to chat.py.
    opts = dict(max_length=len(ids) + max_new)
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
    # token_count() forces prefill to finish, so prefill_s is accurate even on async
    # accelerators (append_tokens can otherwise return before the compute completes).
    start_len = g.token_count()
    prefill_s = time.perf_counter() - t0

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
    "provider", "device",
    "new_tokens", "prefill_s", "decode_s", "decode_tok_s", "e2e_tok_s",
    "speedup_decode", "speedup_e2e",
    "acceptance_rate", "rounds", "draft_proposed", "draft_accepted",
    "corrections", "bonuses", "mean_accepted_tokens",
    "avg_draft_ms_per_token", "avg_target_ms_per_token", "effective_speedup",
    "greedy_match",
    "peak_gpu_mem_gib", "peak_cpu_mem_gib",
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
    # ---- prompt source: Spec-Bench dataset (default) or built-in smoke-test ----
    _default_dataset = os.path.join(here, "question.jsonl")
    ap.add_argument("--dataset",
                    default=(_default_dataset if os.path.exists(_default_dataset) else None),
                    help="path to a Spec-Bench question.jsonl. Defaults to the bundled "
                         "question.jsonl next to this script (the WHOLE dataset -- use "
                         "--limit-per-task to subsample for CPU runs). Uses turns[0] only "
                         "(single-turn). Pass --builtin to force the small smoke-test set.")
    ap.add_argument("--builtin", action="store_true",
                    help="use the built-in smoke-test PROMPTS instead of the dataset "
                         "(overrides --dataset).")
    ap.add_argument("--tasks", default=None,
                    help="comma list of tasks to include when --dataset is set: "
                         "mt_bench,translation,summarization,qa,math_reasoning,rag "
                         "(default: all present)")
    ap.add_argument("--limit-per-task", type=int, default=0,
                    help="cap prompts per task (0 = all/whole dataset, e.g. 80). Samples "
                         "round-robin across subcategories so the subset is representative "
                         "(mt_bench spreads across its 8 subcategories). Subsample for CPU runs.")
    ap.add_argument("--by-category", "--mt-bench-by-subcategory",
                    dest="mt_bench_by_subcategory", action="store_true",
                    help="with --dataset: report EVERY category as its own task -- the 8 "
                         "MT-Bench subcategories (writing/roleplay/reasoning/math/coding/"
                         "extraction/stem/humanities) are split out AND the 5 Spec-Bench "
                         "tasks (math_reasoning/qa/rag/summarization/translation) are kept, "
                         "13 tasks total (nothing collapsed or dropped). --limit-per-task N "
                         "then gives N prompts per category. "
                         "(--mt-bench-by-subcategory is a backward-compatible alias.)")
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

    if args.mt_bench_by_subcategory and (args.builtin or not args.dataset):
        ap.error("--by-category/--mt-bench-by-subcategory requires a dataset "
                 "(the question.jsonl to split per category); it is incompatible with --builtin")

    if args.dataset and not args.builtin:
        tasks_filter = (set(t.strip() for t in args.tasks.split(",") if t.strip())
                        if args.tasks else None)
        prompt_items = load_dataset_prompts(args.dataset, tasks_filter,
                                            args.limit_per_task,
                                            mt_bench_by_subcategory=args.mt_bench_by_subcategory)
        if not prompt_items:
            ap.error("no prompts loaded from --dataset (check the path / --tasks / "
                     "--limit-per-task / --by-category)")
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
    # Self-describing labels recorded in every output row so cross-EP artifacts are unambiguous.
    provider_label = args.execution_provider
    device_label = device or ""
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
    # A missing model fails fast, naming the tried paths.
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
        # Rewrite the full CSV+JSON after every completed config, so a mid-sweep
        # failure still leaves all completed metrics on disk for the CI artifact.
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
    # target loaded, free it, then load the speculative pair and run all spec
    # configs. Peak RAM = target+draft, not target*2+draft (matters for 8b).
    baselines = {}  # (mode, prompt_id) -> dict(dec, e2e, tail)

    # ---- Phase 1: standard baseline (target only) ----
    # Monitor peak memory for the target-only footprint (baseline of the two-model cost).
    mem_baseline = PeakMemoryMonitor().start()
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
                    provider=provider_label, device=device_label,
                    new_tokens=r["new_tokens"], prefill_s=round(r["prefill_s"], 4),
                    decode_s=round(r["decode_s"], 4), decode_tok_s=round(dec_tps, 3),
                    e2e_tok_s=round(e2e_tps, 3), speedup_decode="", speedup_e2e="",
                    acceptance_rate="", rounds="", draft_proposed="", draft_accepted="",
                    corrections="", bonuses="", mean_accepted_tokens="",
                    avg_draft_ms_per_token="", avg_target_ms_per_token="", effective_speedup="",
                    greedy_match="", peak_gpu_mem_gib="", peak_cpu_mem_gib=""))
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
    mem_baseline.stop()
    _backfill_memory(rows, "standard", mem_baseline)
    if IS_NVIDIA_SYSTEM:
        print(f"  [phase 1] peak GPU memory (target only): {mem_baseline.peak_gpu_gib:.2f} GiB",
              flush=True)
    print(f"  [phase 1] peak host RAM (target only): {mem_baseline.peak_cpu_gib:.2f} GiB",
          flush=True)

    # ---- Phase 2: speculative (target + draft) ----
    # Monitor peak memory for the target+draft footprint (the two-model cost of spec decode).
    mem_spec = PeakMemoryMonitor().start()
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
                        provider=provider_label, device=device_label,
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
                        mean_accepted_tokens=round(st.get("mean_accepted_tokens", 0.0), 3),
                        avg_draft_ms_per_token=round(st.get("avg_draft_ms_per_token", 0.0), 4),
                        avg_target_ms_per_token=round(st.get("avg_target_ms_per_token", 0.0), 4),
                        effective_speedup=round(st.get("effective_speedup", 0.0), 3),
                        greedy_match=match,
                        peak_gpu_mem_gib="", peak_cpu_mem_gib=""))
                s_dec_med = statistics.median(s_dec)
                acc = (last or {}).get("acceptance_rate", 0.0)
                done += 1
                sp = s_dec_med / base_dec_med if base_dec_med else 0.0
                print(f"[{done}/{total_cfgs}] {mode} {it['task']}/{it['question_id']} "
                      f"spec K={K}: {s_dec_med:.1f} tok/s  speedup x{sp:.2f}  "
                      f"accept={acc:.0%}", flush=True)
                flush()

    flush()
    mem_spec.stop()
    _backfill_memory(rows, "speculative", mem_spec)
    flush()
    print(f"\nDone in {time.time()-bench_t0:.0f}s. Wrote {csv_path}, {json_path} and {log_path}")
    print_summary(
        rows, modes, ks,
        run_ctx=dict(target=tgt_label, draft=dft_label, provider=provider_label,
                     device=device_label, n_prompts=len(prompt_items),
                     reps=args.reps, max_new=max_new),
        mem_baseline=mem_baseline, mem_spec=mem_spec)


def print_summary(rows, modes, ks, *, run_ctx=None, mem_baseline=None, mem_spec=None):
    """Interpretable summary: a headline BEST config, a speedup-by-K pivot per
    mode (regressions flagged with '!'), and a best-K detail table.

    Speedup is the measured decode-throughput ratio vs the target-only baseline
    (>1.0 = faster); the headline is the geometric mean of per-task speedups
    (Spec-Bench convention). Full per-token metrics live in the CSV/JSON.
    """
    import math

    # Preserve task order as first seen among speculative rows.
    tasks = []
    for r in rows:
        if r["decoder"] == "speculative" and r["task"] not in tasks:
            tasks.append(r["task"])
    if not tasks:
        print("\n(no speculative rows to summarize)")
        return

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

    def task_speedup(mode, task, K):
        sp = med_spec(mode, task, K, "speedup_decode")
        if sp is None:
            std, spec = med_std(mode, task), med_spec(mode, task, K, "decode_tok_s")
            sp = (spec / std) if (std and spec) else None
        return sp

    def geomean(vals):
        vals = [v for v in vals if v is not None]
        return math.exp(sum(math.log(max(v, 1e-9)) for v in vals) / len(vals)) if vals else None

    def median_over_tasks(mode, K, key):
        vals = [v for v in (med_spec(mode, t, K, key) for t in tasks) if v is not None]
        return statistics.median(vals) if vals else 0.0

    # Geomean speedup for every (mode, K), plus the winning K per mode.
    geo = {(mode, K): geomean([task_speedup(mode, t, K) for t in tasks])
           for mode in modes for K in ks}
    best_k = {}
    for mode in modes:
        cand = [(K, geo[(mode, K)]) for K in ks if geo[(mode, K)] is not None]
        best_k[mode] = max(cand, key=lambda kv: kv[1])[0] if cand else None

    width = 78

    # ---- run-context header ----
    print("\n" + "=" * width)
    print("SPECULATIVE DECODING SUMMARY".center(width))
    print("=" * width)
    if run_ctx:
        ep = run_ctx["provider"] + (f"/{run_ctx['device']}" if run_ctx.get("device") else "")
        print(f"target={run_ctx['target']}  draft={run_ctx['draft']}   EP={ep}   "
              f"prompts={run_ctx['n_prompts']}  reps={run_ctx['reps']}  "
              f"max_new={run_ctx['max_new']}")

    # ---- headline BEST config (highest geomean speedup across all mode x K) ----
    best = max((kv for kv in geo.items() if kv[1] is not None),
               key=lambda kv: kv[1], default=None)
    if best is not None:
        (bmode, bK), bgeo = best
        bacc = median_over_tasks(bmode, bK, "acceptance_rate")
        bmat = median_over_tasks(bmode, bK, "mean_accepted_tokens")
        if bgeo >= 1.0:
            print(f"\n>> BEST: {bmode} K={bK}  ->  {bgeo:.2f}x faster decode   "
                  f"(accept {bacc:.0%}, {bmat:.1f} tokens/round)")
        else:
            print(f"\n>> BEST: {bmode} K={bK}  ->  {bgeo:.2f}x  (speculative did NOT beat "
                  f"baseline on average; accept {bacc:.0%})")

    # ---- memory (baseline -> speculative, and the draft's added cost) ----
    if mem_baseline is not None and mem_spec is not None:
        if IS_NVIDIA_SYSTEM:
            base_gib, spec_gib, unit = mem_baseline.peak_gpu_gib, mem_spec.peak_gpu_gib, "GiB GPU"
        else:
            base_gib, spec_gib, unit = mem_baseline.peak_cpu_gib, mem_spec.peak_cpu_gib, "GiB host RAM"
        print(f">> Memory: {base_gib:.1f} -> {spec_gib:.1f} {unit}  "
              f"(+{max(spec_gib - base_gib, 0.0):.1f} for the draft model)")

    # ---- legend ----
    print("\nLegend: speedup > 1.0 = speculative faster (higher is better); '!' = slower than baseline")
    print("        accept   = % of drafted tokens the target accepted")
    print("        mean_acc = avg tokens accepted per round (higher = fewer target calls)")
    print("        match    = greedy output matches baseline (sanity, expect ~100%; '-' in sampling)")
    print("        *        = input-guided task (prompt->output overlap inflates acceptance)")

    def sp_cell(sp):
        if sp is None:
            return f"{'-':>7} "
        return f"{sp:>7.2f}" + ("!" if sp < 1.0 else " ")

    for mode in modes:
        # ---- speedup-by-K pivot (one row per task, geomean at the bottom) ----
        print(f"\n---- mode = {mode} : decode speedup (x) by K ----")
        print(f"  {'task':16}" + "".join(f"{'K=' + str(K):>7} " for K in ks) + f"  {'best':>5}")
        for task in tasks:
            flag = "*" if task in INPUT_GUIDED_TASKS else ""
            cells, bk, bsp = "", None, None
            for K in ks:
                sp = task_speedup(mode, task, K)
                cells += sp_cell(sp)
                if sp is not None and (bsp is None or sp > bsp):
                    bsp, bk = sp, K
            print(f"  {task + flag:16}{cells}  {('K=' + str(bk)) if bk else '-':>5}")
        print(f"  {'-' * (16 + 8 * len(ks))}")
        gcells = "".join(sp_cell(geo[(mode, K)]) for K in ks)
        bestlbl = f"K={best_k[mode]}" if best_k[mode] else "-"
        print(f"  {'GEOMEAN':16}{gcells}  {bestlbl:>5}")

        # ---- detail at the winning K (acceptance / mean_acc / greedy match) ----
        bK = best_k[mode]
        if bK is not None:
            print(f"\n  detail at best K={bK}:")
            print(f"  {'task':16}{'std t/s':>9}{'spec t/s':>9}{'accept':>8}"
                  f"{'mean_acc':>9}{'match':>7}")
            for task in tasks:
                std, spec = med_std(mode, task), med_spec(mode, task, bK, "decode_tok_s")
                if std is None or spec is None:
                    continue
                acc = med_spec(mode, task, bK, "acceptance_rate") or 0.0
                mat = med_spec(mode, task, bK, "mean_accepted_tokens") or 0.0
                gm = med_spec(mode, task, bK, "greedy_match")
                matchs = f"{gm:.0%}" if gm not in (None, "") else "-"
                flag = "*" if task in INPUT_GUIDED_TASKS else ""
                print(f"  {task + flag:16}{std:>9.1f}{spec:>9.1f}{acc:>7.0%} "
                      f"{mat:>9.2f}{matchs:>7}")

    print("\nFull per-prompt metrics (per-token ms, effective_speedup, memory) are in the CSV/JSON.")
    print("=" * width)


if __name__ == "__main__":
    main()
