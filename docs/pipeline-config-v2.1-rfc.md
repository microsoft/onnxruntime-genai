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

v2.0 expresses a **static, acyclic DAG of ONNX sessions executed once per token**, with a prompt-vs-decode phase split (`run_on_prompt` / `run_on_token_gen`) and static dataflow wiring. That already covers nearly all "bucket A" optimizations — including disaggregated prefill/decode, the highest-value scheduling case.

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
- **Not** delivering tree-attention in the prototype — it requires a model-side graph change (see §5c); the shipped path is an honest linear-K fallback.
- **Numeric KV-quant / paging effects out of scope** for now — PR-F lands schema + validation only ("declared, never synthesized"), per design §7.
- **Not** targeting engine-level (continuous-batching) speculative scheduling in the first cut — the prototype targets the single-request `Generator` path.

---

## 4. Design overview

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

## 5. Key decisions to discuss

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
- **Rusty's recommendation:** **Keep linear-K as the default**; gate true tree attention behind an opt-in `build_requires` declaration and a separate graph-export workstream. Only invest if benchmark gains justify the export-pipeline change (an open question, §7).

### (d) Logit-processor chain ordering semantics

- **Options:** (1) Honor declared array order literally for every op, including scalar sampler ops. (2) **Prototype (PR-D):** apply in-place transforms (penalty/bias/grammar/combine) in declared order, but **collect scalar sampler ops (`temperature`/`top_k`/`top_p`) into the terminal fused sampler regardless of their position**.
- **Tradeoffs:** Literal ordering is conceptually pure but would fork the fused sampler kernels (numeric divergence risk + perf cost). The prototype's "scalar sampler ops realized by the terminal fused sampler" keeps sampler numerics identical to the legacy path; in-place ops (e.g. `logit_bias`) remain genuinely order-sensitive and tested as such.
- **Rusty's recommendation:** **Keep the prototype semantics**, but **document the rule prominently in the schema** so users aren't surprised that sampler-op position is cosmetic. Reject configs that interleave a sampler op *between* two transforms only if it changes meaning — otherwise accept and normalize.

### (e) Controller escape hatch: C-ABI surface & stability commitments

- **Options:** (1) Minimal vtable limited to the §4 primitives (GetSequenceLength, GetEosTokenId, IsDone, GetTokens, GetLogits, GetHiddenStates, AppendTokens, RewindTo). (2) Richer surface (custom masks, per-role rewind, distribution access) to support more research up front.
- **Tradeoffs:** A minimal opaque-handle ABI is easy to keep stable and hard to misuse; richer surfaces invite unbounded scope creep and ABI churn. There is a **known limitation**: `RewindTo` + `GetLogits` mid-step does not currently replicate the legacy path's `Action::rewound` re-append, so a real rewinding controller could see off-by-one KV-length behavior.
- **Rusty's recommendation:** **Keep the minimal §4 surface and explicitly mark the controller ABI "unstable / preview" until a real `.so` controller exercises it.** Fix the `rewound` re-append semantics (add a test) **before** committing to ABI stability or advertising rewinding controllers.

### (f) Runtime vs `build_requires` namespace boundary

- **Options:** (1) "Declared, never synthesized" — parse + validate + fail-fast on namespace misuse, but never flip a session knob or synthesize an artifact (prototype/PR-F). (2) Wire runtime effects (KV dtype/quant, paging, prefix cache, chunked prefill) through to the engine/cache now.
- **Tradeoffs:** The declarative-only boundary keeps the honest split visible in the config and is low-risk; users get clear errors (e.g. an AWQ token in `runtime.kv_cache.dtype` points them to `build_requires.quantization`). Wiring effects now pulls in engine/cache numeric behavior with real parity risk and a much larger test burden.
- **Rusty's recommendation:** **Hold the "declared, never synthesized" boundary for v2.1**; land per-feature runtime effects as separate, individually-benchmarked PRs. Also add **graph-match validation** for `build_requires` (currently enum-allowlist only) as a follow-up so declarations fail against the actual exported graph.

---

## 6. What the prototype proved (evidence)

Per-PR, honestly. Commit SHAs are real (this branch). Test names are real (verified in tree).

- **PR-A — KV rollback (`99018ba`).** `DecoderOnlyPipelineState::RewindTo` override (drains async partial KV updates, then rolls back position/KV/recurrent state); `Generator::RewindToLength` un-gated for `decoder-pipeline`. **Verified:** `CAPITests.RewindDecoderPipelineFp32CAPI` (rollback no longer throws, KV rollback correct). **Deferred:** `TruncateToAccepted` delta fast-path, multi-session per-role rollback, share-buffer e2e parity.
- **PR-B — speculative strategy + vanilla draft-target (`6ccc580`).** `Config::Pipeline::{Roles,Speculative}` schema; `SpeculativeDecoder` composing two Generators; `Logits::GetAll()` + `GetRawLogits` to read K+1-row logits from one verify pass. **Verified:** `SpeculativeDecodingTests.{SchemaParsesStrategyAndRoles, GreedyMatchesBaselineDistinctDraft, AllAcceptedWhenDraftEqualsTarget}` — greedy speculative output is **token-for-token identical** to plain greedy (the §10 correctness invariant). **Deferred:** non-greedy acceptance (rejection/typical), ngram/extra-heads producers — all **parse-but-throw**, never silently wrong.
- **PR-C — hidden-state edges + tree linear-K (`b0b1f30`).** Real `GetHiddenStates` edge (device-resident, fp16/bf16→fp32); `medusa_choices` parsed but degraded to linear-K. **Verified:** `HiddenStateEdgeGreedyMatchesBaseline`, `HiddenStateEdgeExposedWithShape`, `TreeDegradesToLinearKGreedyMatchesBaseline` (asserts `tree_linear_k_fallback()==true`). **Deferred & WHY (code-grounded):** true tree attention — the in-tree decoder-pipeline `PositionInputs` only builds a 1D `{batch,seq}` padding mask and hardcodes causal masking via an in-graph Trilu; a `[batch,1,q,kv]` tree mask needs a model-side graph change.
- **PR-D — ordered logit-processor chain (`17f7d47`).** `generation.logits[]` → `LogitsProcessorChain`; scalar sampler ops folded into the terminal fused sampler. **Verified:** `LogitsChainTests.{SchemaParsesChainInOrder, BiasMaskShiftsDeterministicArgmax, BackCompatDefaultMatchesLegacy}` — default (no chain) is byte-for-byte legacy; in-place ops are order-sensitive. **Deferred:** `combine` (contrastive/CFG) throws a clear "deferred" error; `grammar` op needs **`USE_GUIDANCE=ON`** (throws a clear build error otherwise, never a silent skip).
- **PR-E — controller-plugin hook (`a0d6005`).** Additive C-ABI (`OgaDecodeController`, `OgaDecodeContext`, `OgaDecodeStepContext` vtable); `controller_host.{h,cpp}` binds `Generator` to the vtable. **Verified:** `ControllerHookTests.StepContextPrimitivesAreConsistent` + an in-tree stub that reproduces greedy **through the vtable only**. **Deferred & WHY:** real external `.so` load needs **`USE_GENAI_PLUGINS=ON`** (dlopen body compiled out otherwise; `#else` throws a clear rebuild message). **Known limitation:** `RewindTo`+`GetLogits` mid-step doesn't yet honor `Action::rewound` re-append — test before relying on it.
- **PR-F — runtime/build namespace (`9ad8ff5`).** `session_options.runtime` + `build_requires` schema, SAX parse, and cross-namespace validation. **Verified:** `RuntimeFeatureNamespaceTests.{SchemaParsesRuntimeAndBuildNamespace, BackCompatNoNamespaceUnchanged, MisNamespacedBuildFeatureInRuntimeThrows, UnknownBuildEnumValueThrows, …}`. **Deferred & WHY:** actual runtime *effects* (KV-quant/paging numeric behavior) — "declared, never synthesized" per design §7; only a guarded log-warning is emitted, no knob is flipped. Graph-match validation for `build_requires` is enum-allowlist only (follow-up).

> All PR-A…F reviews are **APPROVE** or **APPROVE-WITH-NITS** (no blocking items); see `.squad/decisions.md`.

---

## 7. Open questions for the team

- **Is tree attention worth a graph-export change?** True Medusa/EAGLE-2 trees need a model-side `[batch,1,q,kv]` mask input + dropped in-graph Trilu. Do the acceptance-rate gains justify regenerating graphs and new ORT kernel plumbing, or is linear-K "good enough" for v2.1?
- **Controller ABI stability.** When do we commit to a stable controller C-ABI? Should it stay "preview" until a real Lookahead/cascade controller ships? What's the minimal surface we're willing to freeze?
- **Config ergonomics.** Is the `strategy` + `roles` + `dataflow` triple too verbose for the common vanilla draft-target case? Should we offer a preset/sugar (e.g. `strategy.kind: "draft_target"` with implicit roles)?
- **Speculative × multi-modal.** How do speculative roles interact with VLM/ALM pipelines (per-image stages, prefill/decode split)? Is a draft model meaningful for multi-modal prefill, or is speculative decode-only?
- **Acceptance numerics.** Rejection/typical acceptance need the *draft's* token distribution (not just argmax), which `ngram` cannot provide. Confirm the restriction "`ngram` ⇒ greedy acceptance only" and the golden-test strategy for stochastic acceptance.
- **Two-Generator vs single-State** (§5b): do we ever *need* the single-State fold-in, or is composition the permanent shape?

---

## 8. Phased rollout / PR plan

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

**Next (post-discussion):** `TruncateToAccepted` delta-rewind + multi-session per-role rollback; non-greedy acceptance (rejection/typical); true tree attention (pending §5c decision); `combine`/contrastive/CFG ops; per-feature runtime effects + `build_requires` graph-match validation; fix controller `Action::rewound` semantics.

**Suggested merge/review strategy:** the prototype is **currently a stack on draft PR #2210** on this branch. Recommend **splitting into the six independently-reviewable PRs above** in dependency order — PR-A and PR-F can merge early (foundational / standalone, low risk); PR-B is the high-risk gate (first data-dependent branch + multi-session KV) and deserves its own focused review; PR-C/D/E follow. Keep the zero-regression dispatch gate (`ClassifyStructuralRoute == ClassifyLegacyRoute`) green on every PR.

---

## 9. Appendix link

- **Full technical design (deep appendix):** [`docs/pipeline-config-v2.1-design.md`](./pipeline-config-v2.1-design.md) — schema, struct/parser mapping, executor primitives, role model, logit chain, runtime namespace, controller ABI, testing matrix, open risks.
- **Example configs:** [`examples/pipeline-config/`](../examples/pipeline-config/) — incl. `01-preset-decoder`, `02-explicit-encoder-decoder`, `03-vlm-per-image`, `04-plugin-escape-hatch`, `05-v1-to-v2`, `06-multimodal-single-pass`, `07-prefill-decode`.
- **Decision ledger:** `.squad/decisions.md` — speculative-decoding feasibility verdict (bucket A/B/C), design-doc review, and PR-A…F implementation + review records.
