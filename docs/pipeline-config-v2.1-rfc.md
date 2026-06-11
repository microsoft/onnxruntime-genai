# RFC: Pipeline-as-Config v2.1 — Speculative Decoding & Modern Inference Optimizations

- **Author:** Rusty (Lead / Architect)
- **Date:** 2026-06-11
- **Status:** RFC — for team-review discussion. A working prototype (PR-A…F) already exists on branch `squad/2114-pipeline-as-config`.
- **Issue / PR:** microsoft/onnxruntime-genai#2114; draft PR #2210 (prototype stack).
- **Deep design appendix:** [`docs/pipeline-config-v2.1-design.md`](./pipeline-config-v2.1-design.md) — the full 37 KB technical spec. This RFC summarizes and frames decisions; it does **not** duplicate the spec.
- **Read time:** ~10 min.

---

## 1. TL;DR

v2.1 evolves Pipeline-as-Config from a **static DAG run once per token** into a config that can also express **speculative decoding** and a small, composable set of modern inference optimizations (multi-session roles, KV-cache rollback, ordered logit-processor chain, runtime-feature namespace, controller escape hatch). It adds exactly **one** new control-flow primitive — a `speculative` flow strategy — plus orthogonal building blocks; everything irregular stays behind a plugin. All v1/v2.0 configs parse and run byte-for-byte unchanged (both stay `version: 2`; the discriminator is *block presence*, not a version bump). **A working prototype already exists on this branch (PR-A…F, all reviewed/approved)**, so this RFC is grounded in code, not speculation — the goal of the meeting is to decide which design choices to keep, harden, or revisit before splitting the stack into mergeable PRs.

---

## 2. Motivation / problem

v2.0 expresses a **static, acyclic DAG of ONNX sessions executed once per token**, with a prompt-vs-decode phase split (`run_on_prompt` / `run_on_token_gen`) and static dataflow wiring (the base schema itself — version, structural routing, and the Wire/dataflow model — is presented for discussion in §4). That already covers nearly all "bucket A" optimizations — including disaggregated prefill/decode, the highest-value scheduling case.

What it **cannot** express, concretely:

- **An inner draft loop** — propose K tokens before a verify. The executor runs one DAG per token (`DecoderOnlyPipelineState::Run`).
- **A data-dependent accept/reject branch** — there is no runtime branch anywhere in the system; sampling/verification live in the *outer* `Generator`, not in config-declared stages.
- **Variable tokens-per-step** — the loop advances exactly one token per call.
- **KV-cache rollback** — the caches *implement* `RewindTo`, but the pipeline executor never wired it up, and `Generator::RewindToLength` historically **threw** for `decoder-pipeline`.
- **Hidden-state dataflow edges** — EAGLE/EAGLE-3/MTP draft modules consume the target's *intermediate activations*, not just logits/KV.

This reaffirms the prior architecture verdict in `.squad/decisions.md` ("Can Pipeline-as-Config v2 express speculative decoding?"): **pure v2 config cannot express speculative decoding today**, and the right home is a v2.1 native `speculative` strategy with explicit draft/verify roles, acceptance rule, variable advancement, and KV-rollback hooks.

---

## 3. Goals / Non-goals

**Goals**
- Add a first-class, parameterized `speculative` flow strategy (bucket B) covering the 2024–2026 spec-decode family (draft-model, self-spec, ngram/PLD, extra-heads) via one schema block.
- Provide the reusable executor primitives it needs: inner draft loop, single-pass multi-candidate verify, accept/reject + variable advance, KV rollback, hidden-state edges.
- Generalize the single grammar hook into an ordered **logit-processor / sampler chain** (also unlocks contrastive/CFG).
- Formalize a **runtime vs build-time** feature namespace under `session_options` ("declared, never synthesized").
- Keep backward compatibility absolute: v1/v2.0 unchanged, additive-only, refactor-not-rewrite.

**Non-goals**
- **Not** synthesizing build-time graph artifacts (Medusa/EAGLE heads, MTP heads, early-exit heads, GPTQ/AWQ weights, MQA/GQA, SWA-in-weights). v2.1 may *declare* and *validate* these, never *produce* them.
- **Not** changing any v2.0 behavior or the default sampling path (byte-for-byte identical).
- **Not** delivering tree-attention in the prototype — it requires a model-side graph change (see §6c); the shipped path is an honest linear-K fallback.
- **Numeric KV-quant / paging effects out of scope** for now — PR-F lands schema + validation only ("declared, never synthesized"), per design §7.
- **Not** targeting engine-level (continuous-batching) speculative scheduling in the first cut — the prototype targets the single-request `Generator` path.

---

## 4. The v2.0 base schema (Pipeline-as-Config) — for discussion

> **Discussion item, presented for the same team review as v2.1.** v2.0 — the Pipeline-as-Config *base* from issue microsoft/onnxruntime-genai#2114 (PRs 1–5) — already landed on this branch but, like v2.1, **was never put to the team**. v2.1 is a strict extension of v2.0, so this section comes first: it frames what v2.0 introduced and the base-schema decisions to ratify before we layer the speculative/optimization work (§6) on top. Everything below is grounded in the committed code, not a proposal to write.

### What v2.0 introduced

- **A `version: 2` schema with a top-level `pipeline` block.** `version` defaults to `1` (the legacy `model.*` layout); `version: 2` adds the `pipeline` section (`src/config.h:466`). A `Config::Pipeline` struct (`src/config.h:473`) carries `sessions[]`, a `flow[]` of `FlowStep`s, a `dataflow[]` of `Wire`s, normalized `state` (KV-cache format, optional cross-cache, position-id strategy), and the optional `plugin`/`controller` escape hatches. The block sets `pipeline.present = true` only when it is actually parsed or derived (`src/config.h:588`).

- **Structural / block-presence routing replaces `model_type` dispatch.** The production loader no longer branches on `model.type` strings. `CreateModel()` checks `config->pipeline.present` and, when set (always, post-PR1), routes through `CreatePipeline()` → `ClassifyStructuralRoute()` (`src/models/model.cpp:987`, `:905`). Routing decisions are made on **config structure** — which ONNX sessions exist, whether there is an encoder / frozen cross-attention cache, a vision/speech/embedding session, a multi-stage `decoder.pipeline`, or a combined KV tensor — yielding the same concrete model classes as before: **DecoderOnly**, encoder-decoder (**Whisper** / Marian tie-break), **MultiModal**, explicit **DecoderOnlyPipeline**, plus `gpt2` (combined-KV) and the residual type-keyed cases (RNNT/TDT transducers, `lfm2`, the Qwen-VL pipeline variant) that carry no structural tell (`src/models/model.cpp:905-958`). The legacy `model.type` chain survives only as `ClassifyLegacyRoute()` — kept as the **ground-truth oracle**, no longer the decision-maker (`src/models/model.cpp:867`).

- **The pipeline / dataflow Wire concept.** Dataflow between sessions is expressed declaratively as `Wire{ from, to }`, where each endpoint is a `"session.tensor"` string (`src/config.h:488-491`, `dataflow` at `:537`). `FlowStep` carries the lifecycle phase (`when`: `init` | `step` | `final`) and loop mode (`loop`: `batched` | `per_image`) (`src/config.h:480-487`). This is the static-DAG substrate v2.1's `dataflow` hidden-state edges and `roles` extend — the speculative work adds blocks alongside these, it does not change them.

### v1 → v2 migration story

- **v1 configs are untouched and auto-translated to a pipeline *view*.** For a v1 input (`version < 2 && !pipeline.present`), `TranslateV1ToPipeline()` (`src/config.cpp:2578`) *derives* a normalized `Config::Pipeline` from the legacy `model.*` fields — choosing `extends` (`autoregressive-decoder` / `encoder-decoder` / `vision-language` / `speech-language`) and building the `flow`/`dataflow` from which sessions and cross-attention exist — while leaving `config.model.*` **exactly as parsed**. That is the safe, zero-behavior-change path: the derived view feeds structural dispatch, the original fields still feed every existing consumer.
- **v2 configs lower back into the legacy fields.** For a v2 input (`version >= 2 && pipeline.present`), `ResolvePipelineExtends()` resolves the preset and `LowerPipelineToModel()` lowers session files + top-level `tokens`/`generation`/`metadata` back into `config.model.*` / `config.search.*` so existing consumers keep working even for a pure-v2 config with no legacy `model` block (`src/config.cpp:2522`, `:2549`, `:2733`). `metadata.model_type` is human-facing only and merely seeds `model.type` (`src/config.cpp:2425-2434`).
- **Worked example.** `examples/pipeline-config/05-v1-to-v2/` shows the same `gpt2` model as a v1 `genai_config.json` and its v2 equivalent (`version: 2`, `extends: "autoregressive-decoder"`, a single `decoder` session, combined KV) — the canonical before/after for the translator.

### Key decisions to discuss (v2.0)

Framed like the v2.1 list (§6): **Options → Tradeoffs → Rusty's recommendation.** *The team decides.* These are base-schema calls we inherited from the prototype but never ratified.

#### (a) Structural routing vs explicit `model_type` dispatch

- **Options:** (1) Route on config **structure** (which sessions/caches exist), keeping `model.type` only for residuals with no structural tell. (2) Keep the explicit `model_type` if/else chain as the production decision-maker.
- **Tradeoffs:** Structural routing means new architectures that reuse an existing session shape need *no* dispatch edit, and the legacy chain is retained verbatim as a testable oracle. But it moves the decision into a less obvious place and leaves a handful of honest type residuals (RNNT/TDT, `lfm2`, marian-vs-whisper, the Qwen-VL pipeline variant).
- **Rusty's recommendation:** **Keep structural routing** (as prototyped), with the residual type predicates documented in the PR5 decision note, **and keep `ClassifyLegacyRoute()` as the permanent equivalence oracle.**

#### (b) Schema version-bump policy (v1 default, v2 opt-in, no v3 yet)

- **Options:** (1) `version` defaults to `1`; `2` is opt-in and is the *last* bump — future features ride block-presence within `version: 2`. (2) Bump the integer for each schema generation (v2, v3, …).
- **Tradeoffs:** A stable `version: 2` with additive blocks avoids a parser-version matrix and matches how v2.1 is gated (block-presence, no bump — §6a). A per-generation bump is a louder human signal but multiplies route tables.
- **Rusty's recommendation:** **Default `1`, opt-in `2`, and treat `2` as the stable base** that v2.1 extends without a bump.

#### (c) Backward-compat guarantee for existing `genai_config.json`

- **Options:** (1) Absolute guarantee — every existing v1 `genai_config.json` parses and runs **byte-for-byte unchanged**, with v2 derived as a parallel view. (2) Require a migration pass to v2 to benefit from structural routing.
- **Tradeoffs:** The absolute guarantee is what the translator delivers today (v1 fields untouched; pipeline view derived) and is the entire reason structural dispatch could ship without a regression. A forced migration would be cleaner internally but breaks every shipped model directory.
- **Rusty's recommendation:** **Hold the absolute guarantee.** v1 stays first-class indefinitely; v2 is opt-in. The `StructuralMatchesLegacyForEveryFixture` gate (below) is the contract.

#### (d) The discriminator: `version >= 2 && pipeline.present`

- **Options:** (1) Decide v1-vs-v2 handling by the pair `(version, pipeline.present)` — resolve+lower when both hold, translate when neither does. (2) Branch on `version` alone, or on `pipeline` presence alone.
- **Tradeoffs:** The paired predicate (`src/config.cpp:2733`) is unambiguous and symmetric, and rejects mixed/garbage states early; either single signal alone admits a contradictory config (e.g. `version: 2` with no `pipeline`, or a stray `pipeline` under `version: 1`).
- **Rusty's recommendation:** **Keep the paired discriminator**, and decide explicitly what a contradictory `(version, pipeline.present)` combination should do (throw vs best-effort) — currently it simply falls through to neither branch.

#### (e) Preset inheritance (`extends`) vs fully explicit configs

- **Options:** (1) Offer built-in presets (`autoregressive-decoder`, `encoder-decoder`, `vision-language`, `speech-language`) resolved by `ResolvePipelineExtends()`, with explicit blocks overriding. (2) Require every config to spell out sessions/flow/state fully.
- **Tradeoffs:** Presets keep the common case to a few lines (see example 01/05) and are how the translator emits v2; fully-explicit configs are more transparent but verbose and error-prone.
- **Rusty's recommendation:** **Keep `extends` presets**, document the resolution/override order, and treat the preset names as a small, reviewed, append-only set.

#### (f) The `Wire` dataflow model (`"session.tensor"` string edges)

- **Options:** (1) Express inter-session dataflow as `Wire{from,to}` string endpoints, validated structurally. (2) A richer typed edge object (dtypes, shapes, optionality) up front.
- **Tradeoffs:** String `"session.tensor"` edges are minimal and already carry encoder→decoder and (in v2.1) hidden-state wiring; a fully-typed edge would catch more mistakes at parse time but is heavier and not yet needed. The current gap is that edge endpoints aren't validated against actual session I/O names.
- **Rusty's recommendation:** **Keep the string `Wire` shape for v2.0**, and add endpoint-existence validation (`from`/`to` name against the named session's tensors) as a follow-up — the same "validate, don't redesign" stance as §6f.

### What v2.0 proved (evidence)

The zero-regression contract is a real, code-grounded test: **`PipelineDispatchTests.StructuralMatchesLegacyForEveryFixture`** asserts `ClassifyStructuralRoute(config) == ClassifyLegacyRoute(config)` for every fixture (`test/pipeline_dispatch_tests.cpp:79`), with companion tests pinning specific structural signals (`Gpt2RoutesViaCombinedKvFormat`, `QwenVlCarriesMropeStrategy`, `Phi3vUsesBatchedVisionLoop` — `:104`, `:112`, `:119`). This equivalence gate is the invariant every later PR (including v2.1's PR-A…F) must keep green.

**Honestly deferred at the v2.0 level:** dataflow endpoint validation against real session tensor names (string edges are unchecked today); a typed edge schema; and surfacing the remaining type residuals (RNNT/TDT, `lfm2`) as structural signals so `ClassifyLegacyRoute` could eventually be retired rather than merely kept as an oracle.

---

## 5. Design overview

### The capabilities at a glance

| Capability | What it adds | Status | Where (code) |
|---|---|---|---|
| KV-cache rollback for pipelines | `RewindTo` override + un-gate `RewindToLength` for `decoder-pipeline` | **Prototyped** (PR-A) | `src/models/decoder_only_pipeline.{h,cpp}`, `src/generators.cpp` (`RewindToLength`) |
| `speculative` strategy + roles schema | New `strategy`/`roles` blocks; vanilla draft-target executor | **Prototyped** (PR-B) | `src/config.{h,cpp}`, `src/models/speculative_decoder.{h,cpp}` |
| Inner draft loop + single-pass verify | Draft proposes K; target verifies K in one pass (`GetRawLogits`) | **Prototyped** (PR-B) | `src/models/speculative_decoder.cpp`, `src/generators.cpp` (`GetRawLogits`) |
| Accept/reject + variable advance | Longest greedy-matching prefix accepted (1…K+1 tokens), rollback both roles | **Prototyped** (PR-B, greedy) | `src/models/speculative_decoder.cpp` |
| Hidden-state dataflow edges | `GetHiddenStates`; expose intermediate activation, device-resident | **Prototyped** (PR-C) | `src/generators.cpp` (`GetHiddenStates`), `decoder_only_pipeline.cpp` |
| Token-tree verify (tree attention) | `medusa_choices` parsed; **degrades to linear-K** | **Partial / deferred** (PR-C) | `src/models/speculative_decoder.cpp` (`tree_linear_k_fallback_`) |
| Ordered logit-processor / sampler chain | Declarative `generation.logits[]`; default == legacy | **Prototyped** (PR-D) | `src/logits_processor_chain.{h,cpp}`, `src/generators.cpp` (`logits_chain_`) |
| Controller-plugin escape hatch (bucket C) | Stable C-ABI step hook; host vtable of primitives | **Prototyped, dlopen gated** (PR-E) | `src/models/plugin_api.h`, `src/models/controller_host.{h,cpp}` |
| Runtime vs build-time namespace | `session_options.runtime` / `build_requires`; parse + validate only | **Partial (declared-only)** (PR-F) | `src/config.{h,cpp}` (`RuntimeFeatures`, `BuildRequires`) |

### The speculative loop (linear-K, as prototyped)

```
                 ┌─────────────────────── outer decode step ───────────────────────┐
                 │                                                                  │
  accepted ──►   │  DRAFT role (small)         VERIFY role (target)                 │
  prefix         │  ┌───────────┐  K tokens   ┌────────────┐  K+1 rows of logits    │
                 │  │ draft.Run │ ──────────► │ target.Run │ ───────┐               │
                 │  │  × K      │  (proposals)│  (1 pass)  │        ▼               │
                 │  └───────────┘             └────────────┘   accept/reject        │
                 │       ▲                                     (longest prefix       │
                 │       │                                      i where draft==      │
                 │       │   RewindTo(accepted_len)             target_greedy)       │
                 │       └──────────────◄───── KV rollback ◄────────┤               │
                 │            (both role caches)                    │               │
                 │                                          commit n accepted        │
                 │                                          + 1 bonus token          │
                 └──────────────────────────────────────────────────────────────────┘

Roles & dataflow model:
  roles: { target -> target_session, draft -> draft_session }   each role = its own KV cache
  dataflow (EAGLE, PR-C): { from: "target.hidden_states", to: "eagle_draft.prev_hidden" }
  Generator owns: accept/reject branch, variable advance, sampling chain.
  State owns:     K+1-row logits, hidden-state exposure, per-role RewindTo.
```

---

## 6. Key decisions to discuss (v2.1)

Each is framed as **Options → Tradeoffs → Rusty's recommendation**. *The team decides.*

### (a) Discriminator: block-presence vs version-bump

- **Options:** (1) Both v2.0 and v2.1 stay `version: 2`; recognize features by *presence* of the new optional block (`strategy`/`roles`/`logits`/`runtime`/`controller`). (2) Bump to `version: 3` for v2.1.
- **Tradeoffs:** Block-presence is fully additive, needs no new dispatch tier, and unknown keys still throw (so old parsers reject new configs loudly). A version bump is a clearer signal to humans but forces a parser-version matrix and risks splitting the route table.
- **Rusty's recommendation:** **Block-presence, stay `version: 2`** (as prototyped, guarded by `version >= 2 && pipeline.present`). It matches the existing zero-regression dispatch and is what PR-B…F already validate.

### (b) Speculative composition: two Generators vs single-State role→cache map

- **Options:** (1) **Prototype (PR-B):** compose two independent `Generator`s (target + draft), each owning its own session + KV cache. (2) **Design's preferred shape:** one `DecoderOnlyPipelineState` with a role→cache map.
- **Tradeoffs:** Two-Generator composition reuses battle-tested per-model machinery, keeps the hot single-`Run` path untouched, and shipped working + output-equivalent. The single-State map is more elegant and avoids duplicate outer-loop bookkeeping, but is a heavier refactor with real regression risk on the hot path.
- **Rusty's recommendation:** **Ship the two-Generator composition now; treat the role→cache fold-in as a fast-follow refactor** behind the same schema. The schema is identical either way, so we are not locking users in. Revisit if/when engine-level batching needs a single shared state.

### (c) Tree attention: model-side mask input vs shipped linear-K fallback

- **Options:** (1) Add a model-side `[batch,1,q,kv]` additive tree-attention mask input and drop the in-graph causal Trilu (a **build-time graph-export change**). (2) Keep the shipped **linear-K fallback** (degrade any `medusa_choices` tree to its best linear chain, flagged `tree_linear_k_fallback()==true`).
- **Tradeoffs:** Real tree attention unlocks Medusa/EAGLE-2/SpecInfer acceptance gains, but requires regenerating every target graph and new ORT kernel plumbing — the single biggest feasibility unknown. Linear-K is honest, already covers vanilla/PLD/self-spec, and never fakes a tree.
- **Rusty's recommendation:** **Keep linear-K as the default**; gate true tree attention behind an opt-in `build_requires` declaration and a separate graph-export workstream. Only invest if benchmark gains justify the export-pipeline change (an open question, §9).

### (d) Logit-processor chain ordering semantics

- **Options:** (1) Honor declared array order literally for every op, including scalar sampler ops. (2) **Prototype (PR-D):** apply in-place transforms (penalty/bias/grammar/combine) in declared order, but **collect scalar sampler ops (`temperature`/`top_k`/`top_p`) into the terminal fused sampler regardless of their position**.
- **Tradeoffs:** Literal ordering is conceptually pure but would fork the fused sampler kernels (numeric divergence risk + perf cost). The prototype's "scalar sampler ops realized by the terminal fused sampler" keeps sampler numerics identical to the legacy path; in-place ops (e.g. `logit_bias`) remain genuinely order-sensitive and tested as such.
- **Rusty's recommendation:** **Keep the prototype semantics**, but **document the rule prominently in the schema** so users aren't surprised that sampler-op position is cosmetic. Reject configs that interleave a sampler op *between* two transforms only if it changes meaning — otherwise accept and normalize.

### (e) Controller escape hatch: C-ABI surface & stability commitments

- **Options:** (1) Minimal vtable limited to the core primitives (GetSequenceLength, GetEosTokenId, IsDone, GetTokens, GetLogits, GetHiddenStates, AppendTokens, RewindTo). (2) Richer surface (custom masks, per-role rewind, distribution access) to support more research up front.
- **Tradeoffs:** A minimal opaque-handle ABI is easy to keep stable and hard to misuse; richer surfaces invite unbounded scope creep and ABI churn. There is a **known limitation**: `RewindTo` + `GetLogits` mid-step does not currently replicate the legacy path's `Action::rewound` re-append, so a real rewinding controller could see off-by-one KV-length behavior.
- **Rusty's recommendation:** **Keep the minimal vtable surface and explicitly mark the controller ABI "unstable / preview" until a real `.so` controller exercises it.** Fix the `rewound` re-append semantics (add a test) **before** committing to ABI stability or advertising rewinding controllers.

### (f) Runtime vs `build_requires` namespace boundary

- **Options:** (1) "Declared, never synthesized" — parse + validate + fail-fast on namespace misuse, but never flip a session knob or synthesize an artifact (prototype/PR-F). (2) Wire runtime effects (KV dtype/quant, paging, prefix cache, chunked prefill) through to the engine/cache now.
- **Tradeoffs:** The declarative-only boundary keeps the honest split visible in the config and is low-risk; users get clear errors (e.g. an AWQ token in `runtime.kv_cache.dtype` points them to `build_requires.quantization`). Wiring effects now pulls in engine/cache numeric behavior with real parity risk and a much larger test burden.
- **Rusty's recommendation:** **Hold the "declared, never synthesized" boundary for v2.1**; land per-feature runtime effects as separate, individually-benchmarked PRs. Also add **graph-match validation** for `build_requires` (currently enum-allowlist only) as a follow-up so declarations fail against the actual exported graph.

---

## 7. What the prototype proved (evidence)

Per-PR, honestly. Commit SHAs are real (this branch). Test names are real (verified in tree).

- **PR-A — KV rollback (`99018ba`).** `DecoderOnlyPipelineState::RewindTo` override (drains async partial KV updates, then rolls back position/KV/recurrent state); `Generator::RewindToLength` un-gated for `decoder-pipeline`. **Verified:** `CAPITests.RewindDecoderPipelineFp32CAPI` (rollback no longer throws, KV rollback correct). **Deferred:** `TruncateToAccepted` delta fast-path, multi-session per-role rollback, share-buffer e2e parity.
- **PR-B — speculative strategy + vanilla draft-target (`6ccc580`).** `Config::Pipeline::{Roles,Speculative}` schema; `SpeculativeDecoder` composing two Generators; `Logits::GetAll()` + `GetRawLogits` to read K+1-row logits from one verify pass. **Verified:** `SpeculativeDecodingTests.{SchemaParsesStrategyAndRoles, GreedyMatchesBaselineDistinctDraft, AllAcceptedWhenDraftEqualsTarget}` — greedy speculative output is **token-for-token identical** to plain greedy (the correctness invariant; see the design appendix in §11). **Deferred:** non-greedy acceptance (rejection/typical), ngram/extra-heads producers — all **parse-but-throw**, never silently wrong.
- **PR-C — hidden-state edges + tree linear-K (`b0b1f30`).** Real `GetHiddenStates` edge (device-resident, fp16/bf16→fp32); `medusa_choices` parsed but degraded to linear-K. **Verified:** `HiddenStateEdgeGreedyMatchesBaseline`, `HiddenStateEdgeExposedWithShape`, `TreeDegradesToLinearKGreedyMatchesBaseline` (asserts `tree_linear_k_fallback()==true`). **Deferred & WHY (code-grounded):** true tree attention — the in-tree decoder-pipeline `PositionInputs` only builds a 1D `{batch,seq}` padding mask and hardcodes causal masking via an in-graph Trilu; a `[batch,1,q,kv]` tree mask needs a model-side graph change.
- **PR-D — ordered logit-processor chain (`17f7d47`).** `generation.logits[]` → `LogitsProcessorChain`; scalar sampler ops folded into the terminal fused sampler. **Verified:** `LogitsChainTests.{SchemaParsesChainInOrder, BiasMaskShiftsDeterministicArgmax, BackCompatDefaultMatchesLegacy}` — default (no chain) is byte-for-byte legacy; in-place ops are order-sensitive. **Deferred:** `combine` (contrastive/CFG) throws a clear "deferred" error; `grammar` op needs **`USE_GUIDANCE=ON`** (throws a clear build error otherwise, never a silent skip).
- **PR-E — controller-plugin hook (`a0d6005`).** Additive C-ABI (`OgaDecodeController`, `OgaDecodeContext`, `OgaDecodeStepContext` vtable); `controller_host.{h,cpp}` binds `Generator` to the vtable. **Verified:** `ControllerHookTests.StepContextPrimitivesAreConsistent` + an in-tree stub that reproduces greedy **through the vtable only**. **Deferred & WHY:** real external `.so` load needs **`USE_GENAI_PLUGINS=ON`** (dlopen body compiled out otherwise; `#else` throws a clear rebuild message). **Known limitation:** `RewindTo`+`GetLogits` mid-step doesn't yet honor `Action::rewound` re-append — test before relying on it.
- **PR-F — runtime/build namespace (`9ad8ff5`).** `session_options.runtime` + `build_requires` schema, SAX parse, and cross-namespace validation. **Verified:** `RuntimeFeatureNamespaceTests.{SchemaParsesRuntimeAndBuildNamespace, BackCompatNoNamespaceUnchanged, MisNamespacedBuildFeatureInRuntimeThrows, UnknownBuildEnumValueThrows, …}`. **Deferred & WHY:** actual runtime *effects* (KV-quant/paging numeric behavior) — "declared, never synthesized" per design §7; only a guarded log-warning is emitted, no knob is flipped. Graph-match validation for `build_requires` is enum-allowlist only (follow-up).

> All PR-A…F reviews are **APPROVE** or **APPROVE-WITH-NITS** (no blocking items); see `.squad/decisions.md`.

---

## 8. Forward-compatibility: audio-to-audio (speech-to-speech)

> **Discussion item, not a v2.1 commitment.** The question here is narrow: *does the v2.1 design leave room to GROW to audio-OUTPUT additively — no v2.0 break, no schema redesign — and what would it take?* This is for the team to weigh; nothing below is proposed for the initial v2.1 scope.

**Framing.** Audio **input** is already supported today (the speech-encoder config section, the `MultiModal` route with `has_speech_model`, and the Whisper encoder-decoder class — `src/config.h:32` speech-encoder names, `src/models/model.cpp:842` / `885`, `src/models/whisper.{h,cpp}`). The design question is audio **output**. Up-front answer: **yes for the structural/wiring layer — the extension points already exist and are additive — but audio-out needs new *runtime primitives* that v2.1 can DECLARE while the executor must later grow to honor them**, exactly consistent with the "declared, never synthesized" principle (§3, §6f).

### (a) Extension points that already fit — the compatibility argument

These are **additive** uses of structure v2.1 already has; no schema redesign required.

| Existing v2.1 mechanism | How audio-out reuses it (additively) | Where (code) |
|---|---|---|
| Two-model encoder-decoder wiring (Whisper), run **in reverse** | LM → codec/vocoder **decoder** as a trailing `pipeline[]` stage at `when:"final"` — structurally Whisper backwards | `src/models/whisper.{h,cpp}`; `src/models/model.cpp:842,885` |
| `dataflow[]` edges | Wire LM outputs into the trailing codec stage as named edges | design §dataflow; `src/models/decoder_only_pipeline.cpp` |
| **PR-C hidden-state edges** | A Talker / vocoder conditioned on LM **hidden states** (Qwen2.5-Omni Thinker-Talker) is the same edge shape as an EAGLE draft | `src/generators.cpp` (`GetHiddenStates`), PR-C |
| **PR-D logit-processor chain** | Natural home for **per-codebook** sampling ops (an ordered, declarative sampler chain) | `src/logits_processor_chain.{h,cpp}`, `src/generators.cpp` (`logits_chain_`) |
| **PR-E controller hook** | Streaming / full-duplex turn-taking lives behind the controller plugin (bucket C escape hatch) | `src/models/plugin_api.h`, `src/models/controller_host.{h,cpp}` |

### (b) New runtime primitives needed — the honest gap

These do **not** exist anywhere in `src/`. The schema can *declare* them; the executor/graph must *grow* to honor them.

| Gap | What it is | v2.1 schema-declarable? | Needs executor/graph growth? |
|---|---|---|---|
| (a) Audio vocabulary + audio-token output head | A second vocab + a head emitting audio tokens; today there is a **single** `vocab_size` | Declarable (e.g. `model.audio_vocab_size`, `decoder.outputs.audio_logits`) | Yes — `src/config.h:189`, `src/generators.cpp:533-541` assume one vocab |
| (b) Typed codec/vocoder **final stage** + new dataflow **source** = the generated token *sequence* | Today dataflow wires per-step `session.tensor`, not the whole accumulated sequence | Stage type declarable (`audio`/`codec` stage); new source declarable | Yes — a sequence-as-source edge + a typed decode stage are new |
| (c) Multi-codebook **per-step** sampler | Moshi/Mimi RQ-/depth-transformer emits several codebook tokens per frame | Partly (sampler ops via PR-D chain) | Yes — the single-token-per-step path (`src/search.h:12`, single int32 stream) must become multi-codebook |
| (d) Waveform output buffer + C-API getter | A non-text sink, peer of `GetSequence` | Declarable as an output kind | Yes — output sink is text-only int32 today (`src/generators.cpp:808-810`) |
| (e) Streaming / full-duplex audio I/O step primitives | Audio-frame emit / audio-chunk ingest on the controller vtable | Declarable as controller capability | Yes — `OgaDecodeStepContext` exposes only token primitives (§6e) |

### (c) A/B/C bucketing (mirrors the speculative-decode analysis)

- **A/B boundary — smallest first step.** Single-codebook speech-out (audio tokens through the existing single token stream) + a trailing codec/vocoder decoder stage at `when:"final"`. Structurally Whisper-in-reverse; the only genuinely new pieces are gap (b) — a dataflow source for the accumulated generated sequence — and gap (d) — a waveform sink.
- **B — parameterized loop/branch.** A second/audio vocabulary, **multi-codebook per-step** generation, and interleaved text+audio output. Reuses the *shape* of the v2.1 speculative inner loop (§5) and variable-tokens-per-step, plus PR-C hidden-state edges (a Talker on LM hidden states), but adds the multi-codebook sampler the single-vocab path lacks.
- **C — custom controller.** Full-duplex streaming dialog (Moshi-style listen-while-speak, VAD turn-taking). Lives behind the PR-E controller plugin — and even that must be extended with new audio I/O step primitives (gap (e)).

### (d) Compatibility verdict + recommendation — *Rusty's*

v2.1's schema (pipeline stages, `dataflow`, roles, outputs incl. `hidden_states`, the logit/sampler chain, the controller hook, and the runtime/build namespace) is a **superset** that can express audio-out *wiring* additively. The only hard constraints are **executor-level**: the single-token / single-`vocab_size` assumption (`src/config.h:189`, `src/generators.cpp:533-541`, `src/search.h:12`) and the text-only output sink (`src/generators.cpp:808-810`). Both would be grown **behind new schema blocks** — **no v2.0 break, no redesign.**

**Recommendation (team decides):** keep audio-out **OUT** of the initial v2.1 scope, but (i) **reserve the schema namespace now** — e.g. `decoder.outputs.audio_logits`, an `audio`/`codec` stage type, and `model.audio_vocab_size` — so later support is purely additive; and (ii) carry it as an explicit open question. Reserving namespace is cheap insurance against a future schema break; the team should weigh whether the reservation is worth committing to before any audio-out demand is concrete.

---

## 9. Open questions for the team

- **Is tree attention worth a graph-export change?** True Medusa/EAGLE-2 trees need a model-side `[batch,1,q,kv]` mask input + dropped in-graph Trilu. Do the acceptance-rate gains justify regenerating graphs and new ORT kernel plumbing, or is linear-K "good enough" for v2.1?
- **Controller ABI stability.** When do we commit to a stable controller C-ABI? Should it stay "preview" until a real Lookahead/cascade controller ships? What's the minimal surface we're willing to freeze?
- **Config ergonomics.** Is the `strategy` + `roles` + `dataflow` triple too verbose for the common vanilla draft-target case? Should we offer a preset/sugar (e.g. `strategy.kind: "draft_target"` with implicit roles)?
- **Speculative × multi-modal.** How do speculative roles interact with VLM/ALM pipelines (per-image stages, prefill/decode split)? Is a draft model meaningful for multi-modal prefill, or is speculative decode-only?
- **Acceptance numerics.** Rejection/typical acceptance need the *draft's* token distribution (not just argmax), which `ngram` cannot provide. Confirm the restriction "`ngram` ⇒ greedy acceptance only" and the golden-test strategy for stochastic acceptance.
- **Two-Generator vs single-State** (§6b): do we ever *need* the single-State fold-in, or is composition the permanent shape?
- **Audio-out: reserve schema namespace now?** (§8) Is it worth reserving `model.audio_vocab_size` / `decoder.outputs.audio_logits` / an `audio`/`codec` stage type in v2.1 to keep later speech-to-speech support additive, or premature until demand is concrete?
- **Audio-out target: single- vs multi-codebook first?** (§8) Single-codebook speech-out + trailing codec stage (A/B boundary) vs a Moshi/Mimi multi-codebook per-step path (bucket B)?
- **Is full-duplex in scope for this architecture at all?** (§8) Does streaming listen-while-speak belong behind the controller hook here, or is it a separate effort entirely?

---

## 10. Phased rollout / PR plan

Dependency-ordered (from design §9). The prototype proved the ordering is right: **PR-A is the smallest and highest value-per-complexity** because the caches already implement `RewindTo` — the pipeline simply never wired it up.

```
PR-A (KV rollback)  ──►  PR-B (speculative strategy)  ──►  ┌─ PR-C (tree / hidden-state edges)
   foundation              first data-dependent branch     └─ PR-D (logit-processor chain)
PR-E (controller hook)  — independent (uses PR-B role/Generator surface)
PR-F (runtime namespace) — independent (standalone)
```

| PR | Scope | Landed? | Commit |
|---|---|---|---|
| PR-A | KV-cache rollback for pipelines | ✅ landed | `99018ba` |
| PR-B | `speculative` strategy + vanilla draft-target | ✅ landed | `6ccc580` |
| PR-C | hidden-state edges + tree linear-K fallback | ✅ landed | `b0b1f30` |
| PR-D | ordered logit-processor / sampler chain | ✅ landed | `17f7d47` |
| PR-E | controller-plugin escape hatch | ✅ landed | `a0d6005` |
| PR-F | runtime vs build-time namespace | ✅ landed | `9ad8ff5` |

**Next (post-discussion):** `TruncateToAccepted` delta-rewind + multi-session per-role rollback; non-greedy acceptance (rejection/typical); true tree attention (pending §6c decision); `combine`/contrastive/CFG ops; per-feature runtime effects + `build_requires` graph-match validation; fix controller `Action::rewound` semantics.

**Suggested merge/review strategy:** the prototype is **currently a stack on draft PR #2210** on this branch. Recommend **splitting into the six independently-reviewable PRs above** in dependency order — PR-A and PR-F can merge early (foundational / standalone, low risk); PR-B is the high-risk gate (first data-dependent branch + multi-session KV) and deserves its own focused review; PR-C/D/E follow. Keep the zero-regression dispatch gate (`ClassifyStructuralRoute == ClassifyLegacyRoute`) green on every PR.

---

## 11. Appendix link

- **Full technical design (deep appendix):** [`docs/pipeline-config-v2.1-design.md`](./pipeline-config-v2.1-design.md) — schema, struct/parser mapping, executor primitives, role model, logit chain, runtime namespace, controller ABI, testing matrix, open risks.
- **Example configs:** [`examples/pipeline-config/`](../examples/pipeline-config/) — incl. `01-preset-decoder`, `02-explicit-encoder-decoder`, `03-vlm-per-image`, `04-plugin-escape-hatch`, `05-v1-to-v2`, `06-multimodal-single-pass`, `07-prefill-decode`.
- **Decision ledger:** `.squad/decisions.md` — speculative-decoding feasibility verdict (bucket A/B/C), design-doc review, and PR-A…F implementation + review records.
