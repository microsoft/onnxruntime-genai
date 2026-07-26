# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Calibrate symmetric KV-cache scales for the ONNX Runtime GenAI model builder.

Quantizing the KV cache (``extra_options["kv_cache_quant_type"]``) requires per-layer
scales, supplied to the builder through ``extra_options["kv_cache_scale_file"]``. This
module produces that file.

It runs an **FP16-KV baseline** ONNX model (the same model built without
``kv_cache_quant_type``) over a set of calibration sequences, captures the
``present.<layer>.key`` / ``present.<layer>.value`` outputs -- post-RoPE K and raw V,
exactly the tensors the quantized ``GroupQueryAttention`` node quantizes -- and computes
per-channel (or per-tensor) symmetric scales::

    scale = threshold / qmax     (qmax = 128 for INT8, 8 for INT4, 448 for FP8 E4M3)

where ``threshold`` per channel is the abs-max (``method="minmax"``), a high percentile of
``|x|`` (``method="percentile"``, tames outliers), or the clip point that minimizes the
quantization mean squared error (``method="mse"``). The output JSON is::

    {"scales": {"k_scales": [<per-layer>...], "v_scales": [<per-layer>...]}}

Typical two-step flow::

    # 1. build the FP16-KV baseline (no kv_cache_quant_type)
    python -m onnxruntime_genai.models.builder -m <hf_model> -o <baseline_dir> -p int4 -e cuda

    # 2. calibrate, then rebuild with the quantized KV cache
    python -m onnxruntime_genai.models.kv_cache_calibration \
        --model <baseline_dir> --tokenizer <hf_model> --out kv_scales.json
    python -m onnxruntime_genai.models.builder -m <hf_model> -o <final_dir> -p int4 -e cuda \
        --extra_options kv_cache_quant_type=int8_per_channel kv_cache_scale_file=kv_scales.json

Layer count, KV head count and head size are auto-detected from the baseline model's
``past_key_values.*`` inputs, so the tool is model-agnostic.

**Rotary models:** keep ``k_rotary_envelope=True`` (the default). K is quantized *after*
RoPE, and the low-frequency RoPE dimensions barely rotate inside a short calibration
window, so raw post-RoPE thresholds are only valid near the calibration length. See
:func:`_pair_envelope`. On gpt-oss-20b, disabling it inflates the K quantization RMS error
at position 32k by ~11x (0.86% -> 9.32%) and costs several points of accuracy on long
chain-of-thought workloads.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Diverse, long-form corpus spanning MMLU-ish domains (STEM, humanities, social science,
# reasoning). Passages are concatenated and sliced to TARGET_SEQ tokens so calibration covers
# the same post-RoPE position range that the eval context length exercises.
CORPUS = [
    "The theory of general relativity, formulated by Albert Einstein in 1915, describes gravity "
    "not as a force but as the curvature of spacetime caused by mass and energy. Massive objects "
    "such as stars and planets warp the geometry of spacetime, and this curvature dictates how "
    "other objects move. The theory predicted phenomena such as the bending of light by gravity, "
    "the precession of Mercury's orbit, gravitational time dilation, and the existence of black "
    "holes and gravitational waves, all of which have since been confirmed experimentally.",
    "In computer science, a hash table is a data structure that implements an associative array, "
    "mapping keys to values using a hash function to compute an index into an array of buckets. "
    "Collisions, where two keys hash to the same bucket, are resolved by chaining or open "
    "addressing. Under reasonable assumptions the average cost of lookup, insertion, and deletion "
    "is constant time, which makes hash tables one of the most widely used structures in practice, "
    "underpinning database indexes, caches, symbol tables in compilers, and set membership tests.",
    "Photosynthesis is the process by which green plants, algae, and some bacteria convert "
    "sunlight, water, and carbon dioxide into glucose and oxygen. It occurs in two stages: the "
    "light-dependent reactions in the thylakoid membranes, which capture energy and produce ATP "
    "and NADPH, and the Calvin cycle in the stroma, which fixes carbon dioxide into sugars. "
    "Chlorophyll absorbs light most strongly in the blue and red parts of the spectrum, reflecting "
    "green light, which is why leaves appear green to the human eye under normal daylight.",
    "The French Revolution began in 1789 and led to the end of the absolute monarchy, the rise of "
    "radical political factions such as the Jacobins, the Reign of Terror, and ultimately the "
    "ascent of Napoleon Bonaparte. It was driven by financial crisis, Enlightenment ideas about "
    "popular sovereignty and natural rights, and deep resentment of aristocratic privilege. The "
    "Declaration of the Rights of Man proclaimed liberty, equality, and fraternity, principles that "
    "reshaped European politics and inspired revolutionary and nationalist movements worldwide.",
    "Quantum mechanics is the branch of physics that studies matter and energy at the scale of "
    "atoms and subatomic particles, where classical intuitions break down. Particles exhibit both "
    "wave and particle behavior, quantities such as energy are quantized, and the act of "
    "measurement affects the system. The Heisenberg uncertainty principle sets a fundamental limit "
    "on the simultaneous knowledge of position and momentum, while the Schrodinger equation "
    "governs how the wavefunction, encoding the probabilities of outcomes, evolves over time.",
    "In economics, supply and demand describe how prices are determined in a competitive market. "
    "The demand curve slopes downward because consumers buy more of a good at lower prices, while "
    "the supply curve slopes upward because producers supply more at higher prices. The equilibrium "
    "price occurs where the two curves intersect, clearing the market. Shifts in demand or supply, "
    "caused by changes in income, preferences, technology, or input costs, move the equilibrium and "
    "explain phenomena such as shortages, surpluses, and the effects of taxes and price controls.",
    "The human circulatory system transports oxygen, nutrients, hormones, and waste products "
    "throughout the body. The heart, a muscular pump, drives blood through arteries, capillaries, "
    "and veins. Oxygenated blood leaves the left ventricle through the aorta to the tissues, while "
    "deoxygenated blood returns to the right side of the heart and is pumped to the lungs for gas "
    "exchange. Red blood cells carry oxygen bound to hemoglobin, and the coordinated contraction of "
    "cardiac muscle is regulated by electrical signals originating in the sinoatrial node.",
    "Machine learning models learn patterns from data by minimizing a loss function through "
    "iterative optimization such as gradient descent, generalizing from training examples to unseen "
    "inputs. Overfitting occurs when a model memorizes noise in the training set and fails to "
    "generalize; it is mitigated by regularization, early stopping, dropout, and larger or more "
    "diverse datasets. The bias-variance tradeoff captures the tension between models that are too "
    "simple to fit the data and models so flexible that they are sensitive to sampling noise.",
    "In organic chemistry, the carbon atom's ability to form four covalent bonds allows it to build "
    "long chains, branched structures, and rings, giving rise to the vast diversity of organic "
    "molecules. Functional groups such as hydroxyl, carbonyl, carboxyl, and amino groups determine "
    "the chemical reactivity and physical properties of compounds. Isomers share a molecular formula "
    "but differ in connectivity or spatial arrangement, and stereochemistry, including chirality, "
    "profoundly affects how molecules interact with biological systems such as enzymes and receptors.",
    "The United States Constitution, ratified in 1788, establishes a federal system that divides "
    "power between the national government and the states, and separates the national government "
    "into legislative, executive, and judicial branches. A system of checks and balances lets each "
    "branch limit the others: Congress writes laws and controls spending, the President enforces "
    "laws and commands the military, and the courts interpret laws and can strike down those that "
    "violate the Constitution. The Bill of Rights guarantees fundamental liberties such as speech.",
    "Plate tectonics is the theory that Earth's rigid outer shell, the lithosphere, is divided into "
    "plates that move slowly over the more fluid asthenosphere beneath. At divergent boundaries "
    "plates pull apart and new crust forms; at convergent boundaries plates collide, producing "
    "mountains, volcanoes, and deep ocean trenches through subduction; and at transform boundaries "
    "plates slide past one another, generating earthquakes. This unifying theory explains the "
    "distribution of continents, the pattern of seismic activity, and the geologic history of Earth.",
    "In probability theory, the central limit theorem states that the sum or average of a large "
    "number of independent, identically distributed random variables tends toward a normal "
    "distribution, regardless of the underlying distribution, provided the variance is finite. This "
    "result explains why the bell curve appears so often in nature and underpins much of statistical "
    "inference, including confidence intervals and hypothesis tests. The standard deviation of the "
    "sample mean shrinks in proportion to the inverse square root of the sample size as data grows.",
    "The Industrial Revolution, beginning in Britain in the late eighteenth century, transformed "
    "economies from agrarian and handcraft production to machine-based manufacturing. Innovations "
    "such as the steam engine, the power loom, and improvements in iron and steel production raised "
    "productivity dramatically. Urbanization accelerated as workers moved to factory towns, living "
    "standards eventually rose, and new social classes emerged. The period also brought harsh labor "
    "conditions, child labor, and pollution, prompting reform movements, labor unions, and new laws.",
    "In cell biology, mitochondria are membrane-bound organelles that generate most of the cell's "
    "supply of adenosine triphosphate, used as chemical energy. Through oxidative phosphorylation, "
    "electrons are passed along a chain of protein complexes in the inner membrane, pumping protons "
    "to create a gradient that drives ATP synthase. Mitochondria possess their own circular DNA and "
    "are thought to have originated from an ancient endosymbiotic bacterium, a hypothesis supported "
    "by their double membrane, independent replication, and similarities to modern prokaryotes.",
    "Linguistics is the scientific study of language and its structure, encompassing phonetics, "
    "phonology, morphology, syntax, semantics, and pragmatics. Phonology studies sound systems, "
    "morphology the structure of words, syntax the rules that combine words into sentences, and "
    "semantics the meaning conveyed. Languages change over time through sound shifts, borrowing, and "
    "grammaticalization, and comparative methods reconstruct ancestral languages. Chomsky's theory "
    "of universal grammar proposed that humans share an innate capacity underlying all languages.",
    "The greenhouse effect is the process by which certain gases in a planet's atmosphere, such as "
    "carbon dioxide, methane, and water vapor, trap heat by absorbing and re-emitting infrared "
    "radiation. This natural effect keeps Earth warm enough to support life, but human activities "
    "since the Industrial Revolution have increased greenhouse gas concentrations, enhancing the "
    "effect and driving global warming. Consequences include rising sea levels, more frequent "
    "extreme weather, ocean acidification, and shifts in ecosystems and agricultural patterns.",
]

def _get_quant_type_max(quant_type: str) -> float:
    """Return the qmax divisor for the given quant_type (int8/int4/fp8 prefix like int8_per_channel)."""
    qmax = {"int8": 128.0, "int4": 8.0, "fp8": 448.0}

    for key in qmax:
        if quant_type.startswith(key):
            return qmax[key]

    raise ValueError(f"Unsupported kv_cache quant_type '{quant_type}' (expect int8/int4/fp8 prefix).")


def _pair_envelope(x: np.ndarray, num_kv_heads: int, head_size: int) -> np.ndarray:
    """Map post-RoPE K to its rotation-invariant per-pair envelope.

    RoPE rotates the channel pair ``(d, d + head_size/2)`` of each head as a 2-vector:

        k'_d(p)        =  k_d cos(theta_p) - k_{d+h} sin(theta_p)
        k'_{d+h}(p)    =  k_d sin(theta_p) + k_{d+h} cos(theta_p)

    A rotation preserves the pair norm, so ``||(k'_d, k'_{d+h})||`` does NOT depend on the
    position ``p`` and it upper-bounds ``|k'_d(p)|`` at *every* position. Calibrating the
    threshold on that norm therefore covers positions far beyond the calibration window.

    This matters because the low-frequency RoPE dims barely rotate inside a short calibration
    window (at 512 tokens the dims with periods >100k tokens sit at theta ~ 0, so only ``k_d``
    itself is observed), yet reach theta = O(1) rad during a long reasoning generation, at which
    point they pick up the partner component ``k_{d+h}`` that calibration never saw.

    Returns an array shaped like ``x`` where both channels of every pair hold the pair norm.

    Assumes the half-rotated (non-interleaved) RoPE layout that the builder emits
    (``rotary_interleaved=0``), where the partner of channel ``d`` is ``d + head_size/2``. For
    an interleaved model the pairing is wrong, but since ``||(a, b)|| >= |a|`` the result is
    still an upper bound on ``|x|``, so scales stay conservative (coarser, never clipping).
    """
    if head_size % 2 != 0:
        raise ValueError(f"RoPE pair envelope requires an even head_size, got {head_size}.")
    if x.ndim != 2 or x.shape[1] != num_kv_heads * head_size:
        raise ValueError(
            f"Expected x with shape [tokens, {num_kv_heads * head_size}], got {x.shape}."
        )
    half = head_size // 2
    pairs = x.reshape(x.shape[0], num_kv_heads, head_size)
    norm = np.sqrt(pairs[:, :, :half].astype(np.float32) ** 2 + pairs[:, :, half:].astype(np.float32) ** 2)
    return np.concatenate([norm, norm], axis=-1).reshape(x.shape[0], -1)


def _tokenize_corpus(
    tokenizer, num_seqs: int, target_seq: int, corpus: list[str] | None = None
) -> list[np.ndarray]:
    """Build ``num_seqs`` calibration sequences of length ``target_seq`` tokens.

    Concatenates the corpus into one long token stream and slices non-overlapping windows so
    every window covers positions 0..target_seq-1 (matching the eval context length), exercising
    the full post-RoPE position range rather than only the first few dozen positions.

    ``corpus`` overrides the built-in :data:`CORPUS` with a user-supplied list of passages, e.g.
    text drawn from the target domain. Falls back to :data:`CORPUS` when ``None`` or empty.

    When the corpus is shorter than ``num_seqs * target_seq`` tokens the passage order is rotated
    on every repeat instead of replaying the identical token stream: byte-identical windows would
    otherwise contribute duplicate samples that skew the percentile/MSE thresholds (and trigger
    the model's repetition-induction behaviour, which is not representative of real traffic).
    """
    if num_seqs <= 0:
        raise ValueError(f"num_seqs must be positive, got {num_seqs}.")
    if target_seq <= 0:
        raise ValueError(f"target_seq must be positive, got {target_seq}.")

    passages = list(corpus) if corpus else list(CORPUS)
    needed = num_seqs * target_seq
    ids: list[int] = []
    repeat = 0
    while len(ids) < needed:
        offset = repeat % len(passages)
        encoded = tokenizer.encode("\n\n".join(passages[offset:] + passages[:offset]))
        if not encoded:
            raise ValueError("Tokenizer produced no token IDs for the calibration corpus.")
        if repeat == 0 and len(encoded) < needed:
            logger.warning(
                "Calibration corpus holds %d tokens but %d are needed; passages will be reused in "
                "rotated order. Pass a larger corpus (--corpus-file) for a better estimate.",
                len(encoded),
                needed,
            )
        ids.extend(encoded)
        repeat += 1
    seqs = []
    for s in range(num_seqs):
        window = ids[s * target_seq : (s + 1) * target_seq]
        if len(window) < target_seq:
            break
        seqs.append(np.asarray(window, dtype=np.int64).reshape(1, target_seq))
    return seqs


def _detect_kv_shape(sess, model_path: str) -> tuple[int, int]:
    """Infer ``(num_kv_heads, head_size)`` for the baseline model.

    The builder emits ``past_key_values.*`` as ``[batch, num_kv_heads, past_seq_len, head_size]``,
    but the last dim is symbolic (``kv_cache_dim``) on builds that support KV quantization, so
    fall back to ``genai_config.json`` next to the model for whatever is not concrete.
    """
    num_kv_heads = head_size = None
    for meta in sess.get_inputs():
        if not meta.name.startswith("past_key_values."):
            continue
        shape = meta.shape
        if len(shape) == 4:
            if isinstance(shape[1], int):
                num_kv_heads = shape[1]
            if isinstance(shape[3], int):
                head_size = shape[3]
        break

    if num_kv_heads is None or head_size is None:
        config_path = Path(model_path).parent / "genai_config.json"
        if config_path.is_file():
            with open(config_path, encoding="utf-8") as file:
                decoder = json.load(file).get("model", {}).get("decoder", {})
            num_kv_heads = num_kv_heads if num_kv_heads is not None else decoder.get("num_key_value_heads")
            head_size = head_size if head_size is not None else decoder.get("head_size")

    if num_kv_heads is None or head_size is None:
        raise ValueError(
            "Could not infer num_kv_heads/head_size from the model's past_key_values.* inputs or "
            f"from a genai_config.json next to {model_path}; pass them explicitly."
        )
    logger.info("KV geometry: num_kv_heads=%d head_size=%d", num_kv_heads, head_size)
    return int(num_kv_heads), int(head_size)


def _subsample(x: np.ndarray, budget: int) -> np.ndarray:
    """Return at most ``budget`` rows of ``x``, evenly spread over the full row range.

    Rows are token positions, so taking a prefix would bias the threshold towards early
    positions -- exactly where RoPE has barely rotated. Striding keeps the sample
    representative of the whole calibration window while bounding memory.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}.")
    if budget >= x.shape[0]:
        return x
    return x[np.linspace(0, x.shape[0] - 1, budget).astype(np.int64)]


def _mse_threshold(
    samples: np.ndarray,
    amax: np.ndarray,
    qmax: float,
    qneg: float,
    qpos: float,
    num_candidates: int = 24,
) -> np.ndarray:
    """Per-channel clip threshold that minimizes the symmetric quantization MSE.

    Sweeps candidate thresholds in ``[0.2 * amax, amax]`` and keeps, per channel, the one whose
    round-trip ``dequant(quant(x))`` has the lowest mean squared error. This is the objective
    behind the ``mse`` calibrators in ONNX Runtime's quantization tools and in TensorRT Model
    Optimizer. Being a squared-error objective it is much less aggressive than ``percentile`` at
    discarding rare extreme values, so it is the safer choice when clipping a tail sample is
    costlier than losing resolution.
    """
    amax = amax.astype(np.float32)
    best_thr = amax.copy()
    best_err = np.full(amax.shape, np.inf, dtype=np.float32)
    for ratio in np.linspace(0.2, 1.0, num_candidates, dtype=np.float32):
        thr = np.maximum(amax * ratio, 1e-6)
        scale = thr / qmax
        dequantized = np.clip(np.rint(samples / scale), qneg, qpos) * scale
        err = np.mean(np.square(dequantized - samples), axis=0, dtype=np.float64).astype(np.float32)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_thr = np.where(better, thr, best_thr)
    return best_thr


def _numpy_dtype(ort_type: str) -> np.dtype:
    """Map the ONNX Runtime input type strings used by builder models to NumPy dtypes."""
    dtypes = {
        "tensor(float)": np.dtype(np.float32),
        "tensor(float16)": np.dtype(np.float16),
        "tensor(int32)": np.dtype(np.int32),
        "tensor(int64)": np.dtype(np.int64),
    }
    try:
        return dtypes[ort_type]
    except KeyError as error:
        raise ValueError(f"Unsupported calibration model input type '{ort_type}'.") from error


def calibrate_kv_scales(
    model_path: str,
    tokenizer_path: str,
    out_json: str,
    quant_type: str = "int8_per_channel",
    method: str = "percentile",
    percentile: float = 99.99,
    target_seq: int = 512,
    num_seqs: int = 24,
    sample_budget: int = 12288,
    num_layers: int | None = None,
    num_kv_heads: int | None = None,
    head_size: int | None = None,
    k_rotary_envelope: bool = True,
    corpus: list[str] | None = None,
) -> str:
    """Run the FP16-KV baseline and write calibrated symmetric KV scales to ``out_json``.

    ``num_layers``, ``num_kv_heads`` and ``head_size`` are auto-detected from the model when
    left as ``None``.

    ``corpus`` supplies a user-provided list of calibration passages instead of the built-in
    :data:`CORPUS`; pass domain-specific text to better match your target workload. Falls back
    to :data:`CORPUS` when ``None`` or empty.

    ``method`` selects the threshold objective: ``minmax`` (abs-max), ``percentile`` (a high
    quantile of ``|x|``) or ``mse`` (the clip point minimizing the round-trip quantization
    error, see :func:`_mse_threshold`; not available for FP8). ``percentile`` and ``mse`` buffer
    up to ``sample_budget`` token rows per layer, strided evenly across every calibration
    sequence so the estimate is not biased towards early positions.

    ``k_rotary_envelope`` calibrates the K threshold on the rotation-invariant pair norm
    (see :func:`_pair_envelope`) instead of the raw post-RoPE values. Without it, the K scales
    are only valid for positions inside the calibration window: measured on gpt-oss-20b, the
    worst channel overshoots its 512-token threshold by 25x at position 32k and the K
    quantization RMS error grows 11x (0.86% -> 9.32%), which shows up as a large accuracy loss
    on long chain-of-thought workloads. V is never rotated, so it always uses raw values.
    Set it to ``False`` only for models without rotary position embeddings.

    Returns the path to the written scale file.
    """
    import onnxruntime as ort  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    quant_type = quant_type.lower()
    valid_quant_types = {
        "int8_per_tensor",
        "int8_per_channel",
        "int4_per_tensor",
        "int4_per_channel",
        "fp8_per_tensor",
        "fp8_per_channel",
    }
    if quant_type not in valid_quant_types:
        raise ValueError(f"quant_type must be one of {sorted(valid_quant_types)}, got '{quant_type}'.")
    if method not in {"minmax", "percentile", "mse"}:
        raise ValueError(f"Unknown method={method!r} (minmax|percentile|mse)")
    if method == "percentile" and not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be between 0 and 100, got {percentile}.")
    if sample_budget <= 0:
        raise ValueError(f"sample_budget must be positive, got {sample_budget}.")
    qmax = _get_quant_type_max(quant_type)
    per_channel = quant_type.endswith("per_channel")
    is_fp8 = quant_type.startswith("fp8")
    if method == "mse" and is_fp8:
        raise ValueError("method='mse' models integer rounding and cannot be used with FP8 quant types.")
    # Full-range signed int: the quantizer clamps to [-qmax, qmax-1] (kInt8 [-128,127], kInt4 [-8,7]).
    qneg, qpos = (-qmax, qmax) if is_fp8 else (-qmax, qmax - 1.0)

    # Prepacked MatMulNBits baselines require the fpA_intB path to load; harmless otherwise.
    os.environ.setdefault("ORT_FPA_INTB_GEMM", "1")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    available_providers = ort.get_available_providers()
    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available_providers
    ]
    if not providers:
        raise RuntimeError(
            "Neither CUDAExecutionProvider nor CPUExecutionProvider is available in this "
            f"onnxruntime install (available: {available_providers})."
        )
    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)

    if num_kv_heads is None or head_size is None:
        detected_heads, detected_head_size = _detect_kv_shape(sess, model_path)
        num_kv_heads = detected_heads if num_kv_heads is None else num_kv_heads
        head_size = detected_head_size if head_size is None else head_size
    if num_kv_heads <= 0 or head_size <= 0:
        raise ValueError(f"num_kv_heads and head_size must be positive, got {num_kv_heads} and {head_size}.")
    if k_rotary_envelope and head_size % 2 != 0:
        raise ValueError(f"k_rotary_envelope requires an even head_size, got {head_size}.")

    output_names = [o.name for o in sess.get_outputs()]
    present_keys = {
        int(name.split(".")[1]): name
        for name in output_names
        if name.startswith("present.") and name.endswith(".key") and name.split(".")[1].isdigit()
    }
    present_values = {
        int(name.split(".")[1]): name
        for name in output_names
        if name.startswith("present.") and name.endswith(".value") and name.split(".")[1].isdigit()
    }
    if num_layers is None:
        num_layers = len(present_keys)
    if num_layers <= 0:
        raise ValueError("Model has no present.*.key outputs; cannot calibrate KV scales.")
    expected_layers = set(range(num_layers))
    missing_keys = sorted(expected_layers - present_keys.keys())
    missing_values = sorted(expected_layers - present_values.keys())
    if missing_keys or missing_values:
        raise ValueError(
            "Calibration requires contiguous present.<layer>.key/value outputs starting at layer 0; "
            f"missing keys={missing_keys}, values={missing_values}."
        )
    read_names = [present_keys[i] for i in range(num_layers)] + [present_values[i] for i in range(num_layers)]
    channels = num_kv_heads * head_size

    input_metas = {meta.name: meta for meta in sess.get_inputs()}
    for required_name in ("input_ids", "attention_mask"):
        if required_name not in input_metas:
            raise ValueError(f"Calibration model is missing required input '{required_name}'.")
    past_metas = [
        meta
        for meta in input_metas.values()
        if meta.name.startswith("past_key_values.") and meta.name.endswith((".key", ".value"))
    ]
    supported_inputs = {"input_ids", "attention_mask", "position_ids", *(meta.name for meta in past_metas)}
    unsupported_inputs = sorted(input_metas.keys() - supported_inputs)
    if unsupported_inputs:
        raise ValueError(f"Calibration model has unsupported required inputs: {unsupported_inputs}.")

    seqs = _tokenize_corpus(tokenizer, num_seqs, target_seq, corpus)
    logger.info(
        "KV calibration: %d sequences x %d tokens (quant=%s method=%s qmax=%g)",
        len(seqs),
        target_seq,
        quant_type,
        method,
        qmax,
    )

    k_amax = np.zeros((num_layers, channels), dtype=np.float32)
    v_amax = np.zeros((num_layers, channels), dtype=np.float32)
    need_samples = method in {"percentile", "mse"}
    per_seq_budget = max(1, sample_budget // max(1, len(seqs)))
    k_buf = [[] for _ in range(num_layers)] if need_samples else None
    v_buf = [[] for _ in range(num_layers)] if need_samples else None

    for pi, input_ids in enumerate(seqs):
        seq_len = input_ids.shape[1]
        feeds = {
            "input_ids": input_ids.astype(_numpy_dtype(input_metas["input_ids"].type), copy=False),
            "attention_mask": np.ones(
                (1, seq_len), dtype=_numpy_dtype(input_metas["attention_mask"].type)
            ),
        }
        if "position_ids" in input_metas:
            position_meta = input_metas["position_ids"]
            position_ids = np.arange(seq_len, dtype=_numpy_dtype(position_meta.type)).reshape(1, seq_len)
            if len(position_meta.shape) == 3:
                position_axes = position_meta.shape[0]
                if not isinstance(position_axes, int):
                    raise ValueError("Rank-3 position_ids must have a concrete leading dimension.")
                position_ids = np.broadcast_to(position_ids, (position_axes, 1, seq_len)).copy()
            elif len(position_meta.shape) != 2:
                raise ValueError(f"Unsupported position_ids shape {position_meta.shape}.")
            feeds["position_ids"] = position_ids
        for meta in past_metas:
            feeds[meta.name] = np.zeros(
                (1, num_kv_heads, 0, head_size), dtype=_numpy_dtype(meta.type)
            )
        outputs = sess.run(read_names, feeds)

        for i in range(num_layers):
            k = outputs[i].astype(np.float32).reshape(num_kv_heads, seq_len, head_size)
            v = outputs[num_layers + i].astype(np.float32).reshape(num_kv_heads, seq_len, head_size)
            k = np.transpose(k, (1, 0, 2)).reshape(seq_len, channels)
            v = np.transpose(v, (1, 0, 2)).reshape(seq_len, channels)
            if k_rotary_envelope:
                k = _pair_envelope(k, num_kv_heads, head_size)
            k_amax[i] = np.maximum(k_amax[i], np.abs(k).max(axis=0))
            v_amax[i] = np.maximum(v_amax[i], np.abs(v).max(axis=0))
            if need_samples:
                k_buf[i].append(_subsample(k, per_seq_budget).astype(np.float16))
                v_buf[i].append(_subsample(v, per_seq_budget).astype(np.float16))
        logger.info("[%d/%d] seq_len=%d k_amax<L0>=%.4f v_amax<L0>=%.4f", pi + 1, len(seqs), seq_len, k_amax[0].max(), v_amax[0].max())

    # A NaN/Inf anywhere in the captured KV poisons the whole channel's threshold, so fail loudly
    # instead of writing unusable scales (the abs-max reduction propagates non-finite values).
    if not np.isfinite(k_amax).all() or not np.isfinite(v_amax).all():
        raise ValueError(
            "Baseline model produced non-finite present.*.key/value values; cannot calibrate KV scales."
        )

    k_thr = k_amax.copy()
    v_thr = v_amax.copy()
    if need_samples:
        for i in range(num_layers):
            ks = np.concatenate(k_buf[i], axis=0).astype(np.float32)
            vs = np.concatenate(v_buf[i], axis=0).astype(np.float32)
            if method == "percentile":
                k_thr[i] = np.percentile(np.abs(ks), percentile, axis=0)
                v_thr[i] = np.percentile(np.abs(vs), percentile, axis=0)
            else:
                k_thr[i] = _mse_threshold(ks, k_amax[i], qmax, qneg, qpos)
                v_thr[i] = _mse_threshold(vs, v_amax[i], qmax, qneg, qpos)

    # Fraction of channels whose observed abs-max exceeds the calibrated threshold, i.e. the
    # channels where `percentile`/`mse` chose to clip the tail. Not the fraction of clipped values.
    k_clipped_channels = float(np.mean(k_amax > k_thr * 1.0001))
    v_clipped_channels = float(np.mean(v_amax > v_thr * 1.0001))

    if not per_channel:
        k_thr = k_thr.max(axis=1, keepdims=True)
        v_thr = v_thr.max(axis=1, keepdims=True)

    k_scales = np.maximum(k_thr, 1e-6) / qmax
    v_scales = np.maximum(v_thr, 1e-6) / qmax

    out_json = os.path.abspath(out_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as file:
        json.dump(
            {"scales": {"k_scales": k_scales.tolist(), "v_scales": v_scales.tolist()}},
            file,
            allow_nan=False,
        )

    logger.info(
        "Wrote %s (quant=%s, per_channel=%s, k_rotary_envelope=%s). "
        "k clipped channels=%.3f v clipped channels=%.3f k_scale[%.6f,%.6f] v_scale[%.6f,%.6f]",
        out_json,
        quant_type,
        per_channel,
        k_rotary_envelope,
        k_clipped_channels,
        v_clipped_channels,
        float(k_scales.min()),
        float(k_scales.max()),
        float(v_scales.min()),
        float(v_scales.max()),
    )
    return out_json


def _load_corpus_file(path: str) -> list[str]:
    """Load user calibration passages from ``path``.

    A ``.json`` file is parsed as a JSON array of strings; any other extension is treated as
    plain text whose passages are separated by blank lines. Returns the non-empty passages.
    """
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".json":
        loaded = json.loads(text)
        if not isinstance(loaded, list) or not all(isinstance(passage, str) for passage in loaded):
            raise ValueError(f"Corpus file '{path}' must contain a JSON array of strings.")
        passages = [passage.strip() for passage in loaded]
    else:
        passages = [block.strip() for block in text.split("\n\n")]
    passages = [p for p in passages if p]
    if not passages:
        raise ValueError(f"Corpus file '{path}' contains no passages.")
    return passages


def _main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Calibrate symmetric KV-cache scales from an FP16-KV ONNX model.")
    p.add_argument("--model", required=True, help="Path to the FP16-KV baseline model.onnx (or its directory).")
    p.add_argument("--tokenizer", required=True, help="HF model id/path for the tokenizer.")
    p.add_argument("--out", required=True, help="Output scale JSON path.")
    p.add_argument("--quant-type", default="int8_per_channel")
    p.add_argument("--method", default="percentile", choices=["minmax", "percentile", "mse"])
    p.add_argument("--percentile", type=float, default=99.99)
    p.add_argument("--target-seq", type=int, default=512)
    p.add_argument("--num-seqs", type=int, default=24)
    p.add_argument(
        "--sample-budget",
        type=int,
        default=12288,
        help="Token rows buffered per layer for method=percentile|mse (strided across all sequences).",
    )
    p.add_argument(
        "--corpus-file",
        default=None,
        help="Path to a user calibration corpus: a .json file with a JSON array of strings, or "
        "any other file treated as plain text with passages separated by blank lines. Defaults "
        "to the built-in corpus.",
    )
    p.add_argument("--num-layers", type=int, default=None, help="Auto-detected from the model if omitted.")
    p.add_argument("--num-kv-heads", type=int, default=None, help="Auto-detected from the model if omitted.")
    p.add_argument("--head-size", type=int, default=None, help="Auto-detected from the model if omitted.")
    p.add_argument(
        "--no-k-rotary-envelope",
        dest="k_rotary_envelope",
        action="store_false",
        help="Calibrate K on raw post-RoPE values (legacy; only valid near the calibration length).",
    )
    args = p.parse_args()

    model_path = args.model
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "model.onnx")
    corpus = _load_corpus_file(args.corpus_file) if args.corpus_file else None
    calibrate_kv_scales(
        model_path=model_path,
        tokenizer_path=args.tokenizer,
        out_json=args.out,
        quant_type=args.quant_type,
        method=args.method,
        percentile=args.percentile,
        target_seq=args.target_seq,
        num_seqs=args.num_seqs,
        sample_budget=args.sample_budget,
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        k_rotary_envelope=args.k_rotary_envelope,
        corpus=corpus,
    )


if __name__ == "__main__":
    _main()
