# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Self-speculative decoding integration test.

Wraps each model under test as a *self-speculative* model (draft == target) and
checks that speculative decoding is healthy on the requested execution provider.

Why acceptance rate, not exact-match against standard greedy:
speculative decoding commits, at every position, the argmax of the target's
*batched* verify forward (K proposed tokens scored in one pass). Standard greedy
commits the argmax of a *sequential* one-token-at-a-time forward. On quantized
(int4) models -- which is what CI ships in ``cpu_and_mobile`` -- those two
forwards are not bit-identical (the int4 matmul kernels differ across batch
shapes), so at a near-tie the batched argmax can pick a different, equally-valid
token. That makes token-for-token equality with sequential greedy an invalid
invariant for these models: it flips on harmless numerical near-ties, not bugs.

The invariant that actually reflects correctness is the draft *acceptance rate*.
For self-speculative (draft == target) the draft proposes its own argmax and the
target verifies with its batched argmax; because they are the same weights these
agree at the vast majority of positions, so acceptance is high (~0.9-1.0). A real
defect -- KV-cache misalignment, a device/dtype mishandle in the verify read, or
an EP-selection bug -- makes the target's verify uncorrelated with the draft and
collapses acceptance toward zero. So we assert the model runs, produces output,
and clears a conservative acceptance floor.

To exercise a range of inputs (and surface how acceptance / speedup vary by
workload) the test runs one prompt from each category of the benchmark question
set and logs the generated text and speculative stats for every prompt.

Following ``test_integration_text.py``'s style, the on-disk model directory is
treated as immutable: the real ``genai_config.json`` is loaded with ``og.Config``
and only the speculative wiring is layered on with ``Config.overlay`` --
``model.type`` becomes ``speculative`` and a ``draft`` block that is a faithful
deep copy of the real decoder (draft == target) is added. Nothing is written to
disk and no config field is hand-reconstructed, so every model-specific setting
(input names, RoPE, quantization, ``past_present_share_buffer``, and the decoder's
own ``session_options`` / ``provider_options``) is preserved verbatim.

The execution provider comes from the model directory itself: the resolver hands
back a device-specific build (``onnx/cuda/...``, ``onnx/cpu_and_mobile/...``, ...),
whose shipped config already selects the right EP. Copying the decoder verbatim
into the draft keeps target and draft byte-identical, which is what speculative
decoding requires -- it compares full provider *options*, not just the provider
name -- so this works across CPU / CUDA / WebGPU without per-EP special-casing.

It reuses the suite's ``model`` / ``device`` / ``model_path`` fixtures, so the
integration pipeline runs it once per (model, ep) job with no pipeline change,
inheriting the CPU / CUDA / WebGPU coverage the suite already provides.
"""

from __future__ import annotations

import copy
import gc
import json
import time
from pathlib import Path

import onnxruntime_genai as og

from . import ep_support

_MAX_NEW_TOKENS = 24
_MAX_DRAFT_TOKENS = 4
_MAX_PROMPT_TOKENS = 512  # cap long benchmark prompts so prefill stays bounded / within context

# Self-speculative (draft == target) acceptance is ~0.9-1.0 on a healthy model;
# a broken verify read / KV / EP path drives it toward 0. This conservative floor
# separates "working" from "broken" without tripping on the occasional near-tie.
_MIN_ACCEPTANCE_RATE = 0.5

# One prompt per category, sampled from the benchmark question set
# (benchmark/python/question.jsonl -- MT-bench-style). Embedded rather than read
# from that file so the test is self-contained: the file is large and not part of
# the checked-out tree in CI. This spread of categories surfaces how acceptance
# and throughput vary by workload. Keep them concise and ASCII.
_CATEGORY_PROMPTS: tuple[tuple[str, str], ...] = (
    ("writing", "Describe a vivid and unique character, using strong imagery and creative language. Please answer in fewer than two paragraphs."),
    ("roleplay", "Pretend you are Elon Musk. Speak like Elon Musk as much as possible. Why do we need to go to Mars?"),
    ("reasoning", "Which word does not belong with the others: tyre, steering wheel, car, engine?"),
    ("math", "Given x + y = 4z and x * y = 4z^2, express x - y in terms of z."),
    ("coding", "Write a C++ program to find the nth Fibonacci number using recursion."),
    ("extraction", "Extract all unique variable names from this equation and return them as a JSON list: y = (3/4)x^3 - e^(2x) + sin(pi*x) - sqrt(7)."),
    ("stem", "What is the central dogma of molecular biology? What processes are involved? Who named this?"),
    ("humanities", "What are some business etiquette norms when doing business in Japan?"),
    ("translation", "Translate the following German to English: Es gibt nicht viel auszusetzen."),
    ("summarization", "Summarize in one sentence: renewable energy adoption is accelerating worldwide as costs fall and government policy support grows."),
    ("qa", "What is the meaning of cc and bcc in email?"),
    ("math_reasoning", "Geb is 10 less than half the age of Haley. If Haley is 26 years old, how old is Geb?"),
    ("rag", "Context: George Harrison released a cover of 'Got My Mind Set on You' in 1987 on his album Cloud Nine. Question: who released the 1987 version of the song?"),
)


def _self_speculative_config(model_dir: Path) -> og.Config:
    """Load ``model_dir`` as a self-speculative ``og.Config`` (draft == target).

    The model directory is device-specific (the resolver picks ``onnx/cuda/...``,
    ``onnx/cpu_and_mobile/...``, etc.), so its shipped config already carries the
    correct ``session_options`` -- including ``provider_options`` -- for that
    execution provider. We add a ``draft`` block that is a *faithful* deep copy of
    the decoder and change ``model.type`` to ``speculative``; the decoder itself is
    left untouched.

    Keeping the draft byte-identical to the decoder is essential: speculative
    decoding requires target and draft to use the same execution provider, and it
    compares the full provider *options* (not just the provider name). Overriding
    or clearing/appending providers on only one side makes the options differ and
    trips "Target and draft must use the same execution provider" on CUDA/WebGPU
    (where the shipped config carries non-empty provider options). A plain deep
    copy sidesteps that entirely -- whatever the options are, both sides match.
    ORT GenAI derives the draft's providers list from its provider_options when the
    overlay is applied, mirroring what the constructor does for the decoder.
    """
    real = json.loads((model_dir / "genai_config.json").read_text())
    draft = copy.deepcopy(real["model"]["decoder"])

    overlay = {
        "model": {"type": "speculative", "draft": draft},
        "speculative": {"max_draft_tokens": _MAX_DRAFT_TOKENS},
    }

    config = og.Config(str(model_dir))
    config.overlay(json.dumps(overlay))
    return config



def _plain_config(model_dir: Path) -> og.Config:
    """Load ``model_dir`` as an ordinary (non-speculative) ``og.Config`` -- the
    target model on its own, for the sequential-decode speedup baseline.

    Loaded natively (no clear/append) so it engages the exact same device and
    provider options as the target inside the speculative config, making the
    speedup a like-for-like comparison. The directory is device-specific, so its
    shipped provider config already selects the right EP."""
    return og.Config(str(model_dir))


def _encode(tokenizer: og.Tokenizer, prompt: str) -> list[int]:
    ids = [int(t) for t in tokenizer.encode(prompt)]
    return ids[:_MAX_PROMPT_TOKENS]


def _timed_generate(model: og.Model, prompt_ids: list[int]):
    """Greedy-decode ``prompt_ids`` and time the whole generation (prefill +
    decode loop) as the user would experience it. Returns (new_tokens, seconds,
    speculative_stats). The stats are all-zero for a non-speculative model.

    Search options are pinned to a clean greedy configuration rather than the
    model's shipped defaults: many configs default ``do_sample`` and carry a
    ``repetition_penalty``/``min_length`` tuned for sampling quality. Speculative
    decoding intentionally does not implement those (they need cross-position
    bookkeeping), so leaving the config defaults in place would make a *supported*
    model fail on a search-option value, not a real incompatibility. Pinning them
    also makes the standard baseline a true greedy reference for the speedup /
    acceptance comparison. ``max_length`` is the only per-call value.
    """
    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=len(prompt_ids) + _MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
        min_length=0,
    )
    generator = og.Generator(model, params)

    start = time.perf_counter()
    generator.append_tokens(prompt_ids)
    while not generator.is_done():
        generator.generate_next_token()
    elapsed = time.perf_counter() - start

    new_tokens = list(generator.get_sequence(0))[len(prompt_ids):]
    return new_tokens, elapsed, generator.get_speculative_stats()


def _tokens_per_second(n_tokens: int, seconds: float) -> float:
    return (n_tokens / seconds) if seconds > 0 else 0.0


def _snippet(text: str, width: int = 60) -> str:
    text = " ".join(text.split())
    if len(text) > width:
        text = text[: width - 1] + "..."
    # ASCII-only so the ADO log never hits a console-encoding (cp1252) error when
    # a prompt/output contains non-ASCII (e.g. umlauts, smart quotes) on Windows.
    return text.encode("ascii", "replace").decode("ascii")


def test_self_speculative_decoding(device, model, model_path):
    """Self-speculative decoding must run and keep a healthy draft-acceptance rate
    on the requested execution provider, across a spread of prompt categories.

    Also measures the wall-clock speedup of speculative decoding vs. plain
    sequential decoding of the same target (speedup = spec tokens/s / standard
    tokens/s). For self-speculative (draft == target) this is expected to be < 1
    -- the draft costs as much as the target, so there is no size advantage to
    win back; speedup > 1 only when the draft is smaller than the target. It is
    logged for insight, not asserted; correctness is judged by acceptance rate.

    This test deliberately never skips from our side: an unsupported architecture
    or an EP that can't run the model raises and fails the test loudly instead of
    hiding it. (The shared ``model_path`` fixture still skips a model that has no
    build for the requested device -- that is a catalog fact, not a restriction
    imposed here.)
    """
    # WebGPU's EP ships as a separate plug-in shared library; register it so the
    # provider is resolvable. This is infrastructure, not a skip gate -- if the EP
    # still can't load below, we let it fail.
    if device == "webgpu":
        ep_support.register_webgpu_plugin_once()

    model_dir = Path(model_path)

    # Build the speculative model (target + draft). An unsupported architecture or
    # an unavailable EP raises here and fails the test -- intentionally not caught.
    # This is also the peak memory point (two model copies: target + draft).
    spec_model = og.Model(_self_speculative_config(model_dir))

    tokenizer = og.Tokenizer(spec_model)
    encoded = [(category, prompt, _encode(tokenizer, prompt)) for category, prompt in _CATEGORY_PROMPTS]

    # Speculative pass (target + draft). Capture output, timing and stats.
    spec_rows = []
    for category, prompt, prompt_ids in encoded:
        new_tokens, seconds, stats = _timed_generate(spec_model, prompt_ids)
        spec_rows.append({
            "category": category,
            "prompt": prompt,
            "output": tokenizer.decode(new_tokens),
            "n_tokens": len(new_tokens),
            "seconds": seconds,
            "stats": stats,
        })
    del spec_model
    gc.collect()

    # Standard pass: the target alone, sequential greedy -- the speedup baseline.
    std_model = og.Model(_plain_config(model_dir))
    std_seconds_by_category = {}
    std_tokens_by_category = {}
    for category, _prompt, prompt_ids in encoded:
        new_tokens, seconds, _ = _timed_generate(std_model, prompt_ids)
        std_seconds_by_category[category] = seconds
        std_tokens_by_category[category] = len(new_tokens)
    del std_model
    gc.collect()

    header = f"Self-speculative decoding -- {model} on {device}  (K={_MAX_DRAFT_TOKENS}, max_new={_MAX_NEW_TOKENS})"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    print(
        "columns: accept=draft acceptance rate (accepted/proposed); "
        "mean_tok=tokens committed per verify round (max K+1); "
        "speedup=spec vs sequential wall-clock tokens/s (<1 expected for self-spec: draft==target)."
    )
    print("-" * len(header))

    total_proposed = 0
    total_accepted = 0
    total_spec_seconds = 0.0
    total_std_seconds = 0.0
    total_spec_tokens = 0
    total_std_tokens = 0
    empty_outputs = 0
    for row in spec_rows:
        category = row["category"]
        stats = row["stats"]
        total_proposed += stats["draft_tokens_proposed"]
        total_accepted += stats["draft_tokens_accepted"]
        if not row["output"].strip():
            empty_outputs += 1

        std_tps = _tokens_per_second(std_tokens_by_category[category], std_seconds_by_category[category])
        spec_tps = _tokens_per_second(row["n_tokens"], row["seconds"])
        speedup = (spec_tps / std_tps) if std_tps > 0 else 0.0

        total_spec_seconds += row["seconds"]
        total_std_seconds += std_seconds_by_category[category]
        total_spec_tokens += row["n_tokens"]
        total_std_tokens += std_tokens_by_category[category]

        print(
            f"[{category:<14}] accept={stats['acceptance_rate']:.2f} "
            f"mean_tok={stats['mean_accepted_tokens']:.2f} "
            f"std={std_seconds_by_category[category]:.2f}s spec={row['seconds']:.2f}s speedup={speedup:.2f}x"
            f"  | {_snippet(row['prompt'], 40)!r} -> {_snippet(row['output'])!r}"
        )

    aggregate_acceptance = (total_accepted / total_proposed) if total_proposed else 0.0
    agg_std_tps = _tokens_per_second(total_std_tokens, total_std_seconds)
    agg_spec_tps = _tokens_per_second(total_spec_tokens, total_spec_seconds)
    aggregate_speedup = (agg_spec_tps / agg_std_tps) if agg_std_tps > 0 else 0.0
    print("-" * len(header))
    print(
        f"AGGREGATE: prompts={len(spec_rows)}  empty_outputs={empty_outputs}  "
        f"accepted={total_accepted}/{total_proposed}  "
        f"acceptance={aggregate_acceptance:.3f}  (floor={_MIN_ACCEPTANCE_RATE})  "
        f"speedup={aggregate_speedup:.2f}x  (std {agg_std_tps:.1f} tok/s vs spec {agg_spec_tps:.1f} tok/s)"
    )
    print("=" * len(header))

    # Health gates are aggregate, not per-prompt: a single prompt can legitimately
    # produce an empty greedy completion (an early EOS on a raw, non-chat-templated
    # prompt), which is model behavior, not a speculative fault. What must hold is
    # that the model generated tokens, the speculative path actually engaged, and
    # the draft acceptance rate is healthy.

    # 1. The model generated tokens across the prompt set (it isn't dead / mis-loaded).
    assert total_spec_tokens > 0, (
        f"[{model}/{device}] speculative decoding produced no tokens on any prompt"
    )

    # 2. Draft tokens must have been proposed (speculative path really engaged).
    assert total_proposed > 0, (
        f"[{model}/{device}] no draft tokens were proposed; speculative path did not run"
    )

    # 3. Acceptance must clear the health floor. High on a working model; a real
    #    verify/KV/EP bug collapses it toward zero. (Wall-clock speedup is logged
    #    above for insight but not asserted -- it is <1 by design for self-spec.)
    assert aggregate_acceptance >= _MIN_ACCEPTANCE_RATE, (
        f"[{model}/{device}] self-speculative acceptance {aggregate_acceptance:.3f} "
        f"below floor {_MIN_ACCEPTANCE_RATE}; the draft's proposals are being rejected far more "
        f"than near-tie numerics can explain, which points to a verify/KV/EP bug."
    )
