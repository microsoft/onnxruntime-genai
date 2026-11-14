# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Shared helpers of the qwen38_flash_graph Engine qualification scripts (smoke, gate, behaviours, benchmarks).

The scripts drive a package (see README.md) through the public Engine API (og.Engine / og.Request). Prompts are built from
two natural-text pools that the caller supplies (`--text-pool`, `--prose-pool`, or $QWEN38_TEXT_POOL / $QWEN38_PROSE_POOL):
either a text file (tokenized with the package's own tokenizer) or a `.npy` file of token ids. Random-token and chat prompts
need no pool. Nothing is read from any other location.
"""

import copy
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

_WIN_DLL_HANDLES = []
if os.name == "nt":
    _extra_dirs = [d for d in os.environ.get("ORT_GENAI_DLL_DIRS", "").split(os.pathsep) if d]
    _cudnn_home = os.environ.get("CUDNN_HOME")
    if not _cudnn_home:
        for _cand in (
            r"D:\cudnn\9.24.1.1_cuda13",
            r"C:\cuDNN\9.27.0.42_cuda13",
            r"C:\cudnn-windows-arm64-9.27.0.42_cuda13",
        ):
            if os.path.isdir(_cand):
                _cudnn_home = _cand
                break
    if _cudnn_home:
        for _sub in (os.path.join("bin", "arm64"), os.path.join("bin", "x64"), "bin"):
            _extra_dirs.append(os.path.join(_cudnn_home, _sub))
    _cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4")
    if _cuda_path:
        for _sub in (os.path.join("bin", "arm64"), os.path.join("bin", "x64"), "bin"):
            _extra_dirs.append(os.path.join(_cuda_path, _sub))
    for _d in _extra_dirs:
        if os.path.isdir(_d):
            _WIN_DLL_HANDLES.append(os.add_dll_directory(str(Path(_d).resolve())))
            if _d not in os.environ.get("PATH", ""):
                os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")

import numpy as np  # noqa: E402 - after the Windows DLL directory setup above
import onnxruntime_genai as og  # noqa: E402

EOS_IDS = (248046, 248044)

# Test hook of the qwen38_flash_graph engine process; it is not a genai_config key.
FORCE_REJECT_ENV = "ORT_QWEN38_FORCE_REJECT"
HOST_EXPERT_BYTES_ENV = "QWEN38_HOST_EXPERT_BYTES"
HOST_EXPERT_BYTES_ENTRY = "session.moe.offload.host_expert_bytes"

# Models and tokenizers can open package assets after load_model returns.
_BUDGET_PACKAGE_DIRS = []

# Chat template switches that turn the default "thinking" generation prompt into the non-thinking one used by the gates.
_THINKING_ON = "enable_thinking is undefined or enable_thinking is true"
_THINKING_OFF = "enable_thinking is defined and enable_thinking is false"


def add_prompt_arguments(parser):
    parser.add_argument(
        "--text-pool",
        default=os.environ.get("QWEN38_TEXT_POOL"),
        help="natural text (a text file, or a .npy of token ids; at least ~40k tokens) for the 'text' prompts",
    )
    parser.add_argument(
        "--prose-pool",
        default=os.environ.get("QWEN38_PROSE_POOL"),
        help="second natural-text pool for the 'prose' prompts (same formats)",
    )


class Prompts:
    """Prompt builders of the vLLM benchmark protocol: a window of a natural-text pool, random tokens, or a chat prompt.

    A window starts at (seed * 7919) % (len(pool) - n), so a seed names the same prompt for the same pool.
    """

    def __init__(self, model, package, text_pool=None, prose_pool=None):
        self._package = Path(package)
        self._tokenizer = og.Tokenizer(model)
        self._paths = {"text": text_pool, "prose": prose_pool}
        self._pools = {}

    @classmethod
    def from_args(cls, model, args):
        return cls(model, args.model, args.text_pool, args.prose_pool)

    def _pool(self, kind):
        if kind not in self._pools:
            path = self._paths[kind]
            if not path:
                raise SystemExit(
                    f"the '{kind}' prompts need a natural-text pool: pass --{kind}-pool <file> "
                    f"(or set QWEN38_{kind.upper()}_POOL); a text file or a .npy of token ids"
                )
            if str(path).endswith(".npy"):
                tokens = np.load(path).tolist()
            else:
                tokens = [int(t) for t in self._tokenizer.encode(Path(path).read_text(encoding="utf-8", errors="ignore"))]
            self._pools[kind] = tokens
        return self._pools[kind]

    def make(self, kind, n, seed):
        if kind == "random":
            rng = random.Random(seed)
            return [rng.randrange(1000, 200000) for _ in range(n)]
        if kind not in ("text", "prose"):
            raise ValueError(kind)
        pool = self._pool(kind)
        if len(pool) <= n:
            if not pool:
                raise SystemExit(f"the {kind} pool is empty; {n + 1} tokens are needed")
            reps = (n + 1 + len(pool) - 1) // len(pool)
            pool = pool * reps
        start = (seed * 7919) % (len(pool) - n)
        return pool[start : start + n]

    def chat_prompt(self, question):
        """Non-thinking chat prompt of the package's own chat template (GenAI tokenizer)."""
        template = (self._package / "chat_template.jinja").read_text(encoding="utf-8")
        template = template.replace(_THINKING_ON, "false").replace(_THINKING_OFF, "true")
        messages = json.dumps([{"role": "user", "content": question}])
        text = self._tokenizer.apply_chat_template(messages, template_str=template, add_generation_prompt=True)
        return [int(x) for x in self._tokenizer.encode(text)]


def token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest()


def set_force_reject(enabled):
    """Selects the draft-rejection test hook (every draft is replaced, so the stream is the no-MTP greedy stream).

    The engine reads $ORT_QWEN38_FORCE_REJECT once, when it is constructed, so this has to run before `load_model` and
    `og.Engine`. A value inherited from the caller's environment is dropped when `enabled` is false: the mode that a run records
    must be the mode the engine ran in.
    """
    inherited = os.environ.get(FORCE_REJECT_ENV)
    if enabled:
        os.environ[FORCE_REJECT_ENV] = "1"
    elif inherited is not None:
        del os.environ[FORCE_REJECT_ENV]
        print(
            f"note: ignoring {FORCE_REJECT_ENV}={inherited}; pass --force-reject to replace every draft",
            file=sys.stderr,
        )


def environment_overlay():
    raw = os.environ.get("QWEN38_OVERLAY_JSON")
    overlay = json.loads(raw) if raw else {}
    if not isinstance(overlay, dict):
        raise ValueError("QWEN38_OVERLAY_JSON must be a JSON object")
    return overlay


def host_expert_budget():
    raw = os.environ.get(HOST_EXPERT_BYTES_ENV)
    if raw is None:
        return None
    try:
        budget = int(raw)
    except ValueError as exc:
        raise ValueError(f"{HOST_EXPERT_BYTES_ENV} must be a positive integer byte budget") from exc
    if not 0 < budget < 1 << 64:
        raise ValueError(f"{HOST_EXPERT_BYTES_ENV} must be a positive integer byte budget below 2**64")
    return budget


def execution_protocol():
    return {"overlay": environment_overlay(), "host_expert_bytes": host_expert_budget()}


def with_host_expert_budget(document, budget):
    document = copy.deepcopy(document)
    options = document.setdefault("model", {}).setdefault("decoder", {}).setdefault("session_options", {})
    options[HOST_EXPERT_BYTES_ENTRY] = str(budget)
    for profile in document.get("runtime_profiles", []):
        options = (
            profile.setdefault("overlay", {})
            .setdefault("model", {})
            .setdefault("decoder", {})
            .setdefault("session_options", {})
        )
        options[HOST_EXPERT_BYTES_ENTRY] = str(budget)
    return document


def _link_package_asset(link_path: Path, target: Path) -> None:
    try:
        link_path.symlink_to(target, target_is_directory=target.is_dir())
    except OSError:
        if os.name != "nt":
            raise
        if target.is_dir():
            import _winapi  # noqa: PLC0415

            _winapi.CreateJunction(str(target), str(link_path))
        else:
            os.link(target, link_path)


def _needs_windows_builtin_plane_overlay(package: Path, package_config: dict, *overlays: dict | None) -> bool:
    """Return True on Windows when `plane_library` names an existing `.so` with no `.dll` alias and no overlay sets it.

    The engine loads such a `.so` as configured, which fails on Windows (see ResolveWindowsSharedLibraryPath in
    src/models/qwen38_graph_plane.cpp, whose `.dll` aliases this mirrors); an empty `plane_library` selects the
    built-in plane instead.
    """
    if os.name != "nt":
        return False
    for doc in overlays:
        if isinstance(doc, dict) and "plane_library" in doc.get("model", {}).get("qwen38_graph", {}):
            return False
    configured = package_config.get("model", {}).get("qwen38_graph", {}).get("plane_library", "")
    if not configured or not configured.endswith(".so"):
        return False
    so_path = package / configured
    if not so_path.exists():
        return False
    if so_path.with_suffix(".dll").exists():
        return False
    stem = so_path.stem
    if stem.startswith("lib") and len(stem) > 3 and (so_path.parent / f"{stem[3:]}.dll").exists():
        return False
    return True


def load_model(package, provider=None, overlay=None):
    """Registers the plugin CUDA EP (`provider` path, default $QWEN_PROVIDER or <package>/provider/libonnxruntime_providers_cuda.so) and loads the package."""
    package = Path(package).resolve()
    package_config = json.loads((package / "genai_config.json").read_text(encoding="utf-8"))
    env_overlay = environment_overlay()
    validate_mtp_mode(package_config, env_overlay, overlay or {})
    budget = host_expert_budget()
    if os.name == "nt":
        pkg_libs = package / "libs"
        if pkg_libs.is_dir():
            _WIN_DLL_HANDLES.append(os.add_dll_directory(str(pkg_libs)))
    provider = provider or os.environ.get("QWEN_PROVIDER")
    if not provider:
        pkg_provider = (
            Path(package)
            / "provider"
            / ("onnxruntime_providers_cuda.dll" if os.name == "nt" else "libonnxruntime_providers_cuda.so")
        )
        if pkg_provider.exists():
            provider = str(pkg_provider)
    if provider:
        provider = str(Path(provider).resolve())
        if os.name == "nt":
            prov_dir = Path(provider).parent
            if prov_dir.is_dir():
                _WIN_DLL_HANDLES.append(os.add_dll_directory(str(prov_dir)))
        try:
            og.register_execution_provider_library("CUDAExecutionProvider", provider)
        except RuntimeError as exc:
            if "already registered" not in str(exc):
                raise
    win_plane_overlay = (
        {"model": {"qwen38_graph": {"plane_library": ""}}}
        if _needs_windows_builtin_plane_overlay(package, package_config, env_overlay, overlay)
        else None
    )
    directory = None
    try:
        config_package = package
        if budget is not None:
            # Config.overlay appends runtime profiles; patch the base profiles before loading, without changing the package.
            directory = tempfile.TemporaryDirectory(prefix="qwen38_ram_budget_")
            config_package = Path(directory.name)
            for asset in package.iterdir():
                if asset.name != "genai_config.json":
                    _link_package_asset(config_package / asset.name, asset)
            (config_package / "genai_config.json").write_text(
                json.dumps(with_host_expert_budget(package_config, budget)), encoding="utf-8"
            )
        config = og.Config(str(config_package))
        for document in (win_plane_overlay, env_overlay, overlay):
            if document:
                effective = with_host_expert_budget(document, budget) if budget is not None else document
                config.overlay(json.dumps(effective))
        model = og.Model(config)
        if directory is not None:
            _BUDGET_PACKAGE_DIRS.append(directory)
            directory = None
        return model
    finally:
        if directory is not None:
            directory.cleanup()


def validate_mtp_mode(*configs):
    """Rejects no-MTP on folded graphs after applying the package and overlays in order."""
    mtp, fold = True, False
    for config in configs:
        graph = config.get("model", {}).get("qwen38_graph", {})
        mtp = graph.get("mtp", {}).get("enabled", mtp)
        fold = graph.get("fold", {}).get("enabled", fold)
    if fold and not mtp:
        raise ValueError(
            "--no-mtp requires a non-folded package with S=1 graphs; this package has fold.enabled=true. "
            "Use --force-reject with engine_gate.py for an identity reference; forced rejection does not measure S=1 performance."
        )


@dataclass(frozen=True)
class TurnUsage:
    prompt_tokens: int
    generated_tokens: int
    cached_prompt_tokens: int


class TurnResult:
    def __init__(self):
        self.tokens = []
        self.times = []  # perf_counter() after the Engine.run call that delivered each token
        self.finish_reason = None
        self.usage = None
        self.matched_stop_string_index = None
        self.runs = 0
        self.events = 0
        self.start = None
        self.end = None


def run_turn(engine, buffer, request, prompt, max_new, *, stop_strings=None, turn_setup=None, until=None):
    """Begins a greedy turn and runs the Engine until the Engine has no work (or `until(result)` returns True).

    Returns the TurnResult and the turn id. Raises on a failed or capacity-blocked event.
    """
    turn = og.TurnOptions(request)
    turn.set_do_sample(False)
    if max_new is not None:
        turn.set_max_generated_tokens(max_new)
    if stop_strings:
        turn.set_stop_strings(stop_strings)
    if turn_setup:
        turn_setup(turn)
    result = TurnResult()
    result.start = time.perf_counter()
    turn_id = request.begin_turn(np.asarray(prompt, dtype=np.int32), turn)
    pump(engine, buffer, result, until)
    return result, turn_id


def pump(engine, buffer, result, until=None):
    while engine.has_pending_requests():
        events = engine.run(buffer)
        now = time.perf_counter()
        result.runs += 1
        for event in events:
            result.events += 1
            if event.flags & (og.EngineEventFlags.FAILED | og.EngineEventFlags.CAPACITY_BLOCKED):
                raise RuntimeError(f"Engine failure: flags={event.flags} error={event.error_code}")
            if event.flags & og.EngineEventFlags.TOKEN:
                result.tokens.append(int(event.token))
                result.times.append(now)
            if event.flags & og.EngineEventFlags.TURN_FINISHED:
                result.end = now
                result.finish_reason = event.finish_reason
                usage = event.usage
                result.usage = TurnUsage(
                    int(usage.prompt_tokens), int(usage.generated_tokens), int(usage.cached_prompt_tokens)
                )
                result.matched_stop_string_index = event.matched_stop_string_index
        if until is not None and until(result):
            break
    return result
