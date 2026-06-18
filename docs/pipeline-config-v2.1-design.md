# Pipeline-as-Config v2.1 — Design: Speculative Decoding & a Framework for Modern Inference Optimizations

- **Author:** Rusty (Lead / Architect)
- **Date:** 2026-06-10
- **Status:** Design proposal (no implementation). Targets `onnxruntime-genai` issue #2114, branch `squad/2114-pipeline-as-config`, draft PR microsoft/onnxruntime-genai#2210.
- **Inputs:** `.squad-task/research-inference-optimizations.md` (the 2024–2026 optimization landscape, bucketed A/B/C); prior architecture verdict in `.squad/decisions.md` ("Can Pipeline-as-Config v2 express speculative decoding?").
- **Scope of this doc:** schema + executor design only. It cites the *current* code (verified file:line) and proposes additive changes. It does not modify any source or tests.

Throughout, the research memo's **buckets** are used:
- **(A) Static-DAG-expressible** — declarative graph of sessions, no data-dependent branching.
- **(B) Parameterized loop/branch strategy in the executor** — a known, bounded control-flow pattern (inner loop, accept/reject, rollback) the executor implements generically and config parameterizes.
- **(C) Custom controller (plugin)** — irregular, stateful control flow needing imperative code.

---

## 1. Motivation & scope

### 1.1 What v2.0 already covers

The v2.0 pipeline schema (`src/config.h:417-470`) plus the legacy decoder pipeline array (`src/config.h:386-404`) expresses a **static, acyclic DAG of ONNX sessions executed once per token**, with a prompt-vs-decode phase split:

- Per-stage lifecycle gating via `run_on_prompt` / `run_on_token_gen` (`src/config.h:395-396`), resolved into `PipelineFlow::Phase {Init, Step, Final}` (`src/models/decoder_only_pipeline.h:39-63`) and applied in `RunPipeline` (`src/models/decoder_only_pipeline.cpp:328-350`).
- Static dataflow wiring via `inputs` / `outputs` / `output_names_forwarder` (`src/config.h:392-394`) and the v2 `dataflow[]` `{from,to}` wires (`src/config.h:432-435`, parsed at `src/config.cpp:1625-1655`, applied in `RunStage` at `src/models/decoder_only_pipeline.cpp:412-465`).
- Per-session `session_options` (`src/config.h:388`, `src/config.h:421`), `is_lm_head` (`src/config.h:397`), `reset_session_idx` (`src/config.h:398-401`).
- Structural dispatch on config shape via `ClassifyStructuralRoute` (`src/models/model.cpp:891-945`), gated against the legacy oracle `ClassifyLegacyRoute` (`src/models/model.cpp:853-878`) by the zero-regression test (`PipelineDispatchTests`).
- An existing **plugin escape hatch** that hands over *model construction only* (`src/models/plugin_api.h:61-64`; loaded in `CreatePipeline` at `src/models/model.cpp:951-960`).
- Existing **constrained decoding** via llguidance (`src/constrained_logits_processor.{h,cpp}`, `docs/ConstrainedDecoding.md`), invoked from the outer loop in `Generator::GenerateNextToken` (`src/generators.cpp:659-662`).

Per the memo (§2, §3), this **already covers nearly all of bucket (A)** — including the highest-value scheduling case, disaggregated prefill/decode, which is literally the existing `run_on_prompt`/`run_on_token_gen` split plus a KV handoff.

### 1.2 The gap

The executor is a single DAG run once per token (`DecoderOnlyPipelineState::Run`, `src/models/decoder_only_pipeline.cpp:511-559`). Sampling and verification live in the **outer** `Generator`, not in config-declared stages (`Generator::GenerateNextToken`, `src/generators.cpp:628-695`; sampling switch at `src/generators.cpp:679-694`). It has **no** inner draft loop, **no** runtime accept/reject branch, **no** variable tokens-per-step, and **no** KV-cache rollback wired in for pipelines — `Generator::RewindToLength` explicitly **throws for `decoder-pipeline`** (`src/generators.cpp:697-699`), and `DecoderOnlyPipelineState` does not override `State::RewindTo` (default no-op at `src/models/model.h:34`), even though the underlying caches *do* implement rollback (e.g. `DefaultKeyValueCache::RewindTo`, `src/models/kv_cache.cpp:508`).

This reaffirms the prior verdict in `.squad/decisions.md`: **pure v2 config cannot express speculative decoding today**, and the correct home is a v2.1 native `speculative` flow strategy with explicit draft/verify roles, acceptance rule, variable token advancement, and KV rollback hooks.

### 1.3 What v2.1 targets

| Bucket | v2.1 stance |
|---|---|
| **(A)** static-DAG / runtime knobs | Mostly already expressible. v2.1 *formalizes* a runtime-feature namespace (§7) and adds the static two-session **contrastive / CFG combine** (a special case of §5 + §6). |
| **(B)** parameterized loop/branch | **Primary target.** A first-class `speculative` flow strategy (§3) + the executor primitives it needs (§4): inner draft loop, batched/tree verify, accept/reject, KV rollback, variable tokens/step, hidden-state edges. Also the **logit-processor chain** (§6, the FSM-grammar part of bucket B/E). |
| **(C)** custom controller | **Stays plugin-only.** Lookahead's stateful Jacobi pool, deeply nested cascades, and novel research go through an *extended* controller-plugin hook (§8). The declarative core stays small. |

**Honesty boundary (memo §3 item 7):** v2.1 cleanly separates *schema/executor changes* (B — doable here) from *model-build artifacts* (Medusa heads, EAGLE draft module, MTP heads, GPTQ/AWQ weights, MQA/GQA, SWA-in-weights), which v2.1 only *references* and never *produces*, and from *needs-plugin* (C).

---

## 2. Design principles

1. **Keep the declarative core small.** v2.1 adds exactly one new control-flow primitive (the `speculative` strategy, §3) plus a small set of orthogonal, composable building blocks (multi-session roles §5, logit-processor chain §6, runtime-feature namespace §7). Everything irregular is pushed behind the controller plugin (§8).

2. **Backward-compatible & additive, gated by block-presence.** All v1 and v2.0 configs parse and run byte-for-byte unchanged. v2.0 and v2.1 are **both** `version: 2`, so the discriminator is the *presence of the new optional block* (`strategy`/`roles`/`logits`/`runtime`), not a version bump; v2.1 features are *only* recognized when such a block is present (dispatch `version >= 2 && pipeline.present` at `src/config.cpp:2194-2198`, version field at `src/config.cpp:1905-1906`). No existing key changes meaning. New SAX elements are added alongside the existing ones (`PipelineConfig_Element` at `src/config.cpp:1754-1795`); an absent `strategy`/`roles`/`logits`/`runtime` block leaves the v2.0 path intact.

3. **Refactor, not rewrite.** Extend the existing `DecoderOnlyPipelineState` executor (`src/models/decoder_only_pipeline.{h,cpp}`) and `Generator` (`src/generators.cpp`); do **not** fork a parallel "speculative model" class. The draft loop becomes an inner loop *inside* the existing `Run`/`RunPipeline` structure (`src/models/decoder_only_pipeline.cpp:511-559`, `:328-350`); accept/reject and variable-token advancement reuse the existing `search_`/`Sequences` plumbing in `Generator` (`src/generators.cpp:628-695`). KV rollback reuses the caches' existing `RewindTo` (`src/models/kv_cache.cpp:508`, `:72`, `:708`, `:856`).

4. **Honesty about runtime vs build-time.** Every v2.1 field is tagged in the schema as `runtime` (schedulable at load) or `build` (a property of the exported ONNX graphs/weights). Speculative *heads* and *draft modules* are build artifacts; the *loop* that drives them is runtime. The schema may *declare* a build requirement (so loading fails fast with a clear message) but never claims to synthesize one.

5. **Zero-regression gate stays green.** `ClassifyStructuralRoute == ClassifyLegacyRoute` for every existing fixture (`src/models/model.cpp:891-945` vs `:853-878`). A `speculative` strategy is a *new* structural signal that routes to the (extended) `DecoderOnlyPipeline`; it cannot change the route of any config that lacks the block (§10).

---

## 3. The core new primitive: a `speculative` flow strategy

### 3.1 Schema

A new optional `strategy` object lives under `pipeline` (peer of `sessions`/`flow`/`dataflow`/`state`/`plugin`). When `strategy.kind == "speculative"`, the executor runs the parameterized draft→verify→accept loop instead of the single per-token DAG.

> **Note:** the `//` comments in the JSON snippets below (and throughout §3) are illustrative only — `genai_config.json` is parsed as strict JSON, which does **not** permit comments. The fences are marked `jsonc` purely for readability; real configs must omit the `//` annotations.

```jsonc
"pipeline": {
  "sessions": { "...": {} },
  "strategy": {
    "kind": "speculative",
    "draft": {
      "producer": "draft_model",   // "draft_model" | "self_speculative" | "ngram" | "extra_heads"
      "session": "draft",          // session role name (see §5); omitted for ngram
      "depth": null,               // self_speculative: exit-layer / shallow-depth (build property)
      "heads": null,               // extra_heads / medusa / mtp: head group name (build property)
      "ngram": null                // ngram: {"min_match": 2, "max_draft": 8, "window": 256}
    },
    "verify": { "session": "target" },
    "num_speculative_tokens": 5,    // K
    "acceptance": "rejection_sampling", // "greedy" | "rejection_sampling" | "typical"
    "typical_threshold": 0.09,      // only for acceptance=="typical"
    "tree": null                    // null = linear K; or { "topology": "medusa_choices"|"dynamic", ... }
  }
}
```

Field semantics:

- `draft.producer` selects the **drafting mechanism** (covers the whole memo §1A family with one block):
  - `draft_model` — a separate small ONNX session (vanilla Leviathan/Chen draft-target; TRT-LLM's two-`Executor` Draft-Target-Model). Bucket B.
  - `self_speculative` — same target weights run to a shallow `depth` (Self-Spec 2309.08168 / LayerSkip 2404.16710). `depth` is a **build** property: the exported graph must expose an early-exit head; the schema only *names* it.
  - `ngram` — draft-model-free string match over prompt/recent output (PLD; vLLM `prompt_lookup`). No NN; `draft.session` omitted.
  - `extra_heads` — Medusa/EAGLE/MTP heads attached to the target (**build** artifact). `heads` names the head group; verification is typically a tree (`tree`).
- `verify.session` — the role (§5) whose single forward pass verifies all K candidates.
- `num_speculative_tokens` (K) — draft length per outer step. Variable tokens-per-step on output is 1…K+1.
- `acceptance` — the data-dependent rule: `greedy` (argmax match), `rejection_sampling` (modified rejection sampling, Leviathan 2211.17192), or `typical` (Medusa typical acceptance, 2401.10774).
- `tree` — optional token-tree topology (Medusa `medusa_choices` static tree, or `dynamic` for EAGLE-2/SpecInfer), enabling the tree-verify primitive (§4b).

### 3.2 Concrete examples

**(a) Vanilla draft–target (separate draft model), rejection sampling, K=5**

```jsonc
{
  "version": 2,
  "pipeline": {
    "sessions": {
      "target": { "file": "target.onnx" },
      "draft":  { "file": "draft.onnx" }
    },
    "roles": { "target": "target", "draft": "draft" },
    "strategy": {
      "kind": "speculative",
      "draft":  { "producer": "draft_model", "session": "draft" },
      "verify": { "session": "target" },
      "num_speculative_tokens": 5,
      "acceptance": "rejection_sampling"
    }
  },
  "tokens": { "eos": [2], "pad": 0 },
  "generation": { "max_length": 4096, "sampling": { "temperature": 0.7 } },
  "metadata": { "model_type": "decoder-pipeline" }
}
```

**(b) EAGLE-style, hidden-state-coupled draft + dynamic tree verify**

The draft module consumes the target's **last hidden state** (not logits), so the dataflow wires an intermediate-tensor edge (§4e). The EAGLE draft head is a build artifact.

```jsonc
{
  "version": 2,
  "pipeline": {
    "sessions": {
      "target":      { "file": "target.onnx" },
      "eagle_draft": { "file": "eagle_draft.onnx" }
    },
    "roles": { "target": "target", "draft": "eagle_draft" },
    "dataflow": [
      { "from": "target.hidden_states", "to": "eagle_draft.prev_hidden" }
    ],
    "strategy": {
      "kind": "speculative",
      "draft":  { "producer": "draft_model", "session": "eagle_draft" },
      "verify": { "session": "target" },
      "num_speculative_tokens": 8,
      "acceptance": "typical",
      "typical_threshold": 0.09,
      "tree": { "topology": "dynamic", "max_nodes": 60, "max_depth": 6 }
    }
  },
  "tokens": { "eos": [2] },
  "generation": { "max_length": 8192, "sampling": { "temperature": 0.6 } },
  "metadata": { "model_type": "decoder-pipeline" }
}
```

**(c) PLD / n-gram (draft-model-free)**

```jsonc
{
  "version": 2,
  "pipeline": {
    "sessions": { "target": { "file": "target.onnx" } },
    "roles": { "target": "target" },
    "strategy": {
      "kind": "speculative",
      "draft":  { "producer": "ngram",
                  "ngram": { "min_match": 2, "max_draft": 8, "window": 256 } },
      "verify": { "session": "target" },
      "num_speculative_tokens": 8,
      "acceptance": "greedy"
    }
  },
  "tokens": { "eos": [2] },
  "generation": { "max_length": 4096 },
  "metadata": { "model_type": "decoder-pipeline" }
}
```

### 3.3 Struct + parser mapping

New `Config::Pipeline` members (additive; alongside `src/config.h:462-469`):

```cpp
struct Pipeline {                       // src/config.h:417
  // ... existing sessions/flow/dataflow/state/plugin ...
  struct Roles {                        // §5: logical role -> session name
    std::optional<std::string> target, draft, amateur, expert, unconditional;
  };
  std::optional<Roles> roles;

  struct Speculative {
    std::string kind{"speculative"};
    struct Draft {
      std::string producer{"draft_model"};            // draft_model|self_speculative|ngram|extra_heads
      std::optional<std::string> session;             // role/session name
      std::optional<int> depth;                       // self_speculative (build)
      std::optional<std::string> heads;               // extra_heads/medusa/mtp (build)
      struct Ngram { int min_match{2}, max_draft{8}, window{256}; };
      std::optional<Ngram> ngram;
    } draft;
    struct Verify { std::string session; } verify;
    int num_speculative_tokens{5};                    // K
    std::string acceptance{"greedy"};                 // greedy|rejection_sampling|typical
    std::optional<float> typical_threshold;
    struct Tree { std::string topology; int max_nodes{0}, max_depth{0};
                  std::vector<std::vector<int>> medusa_choices; };
    std::optional<Tree> tree;
  };
  std::optional<Speculative> strategy;                 // present() only when strategy.kind set
};
```

Parser hooks (additive SAX elements, mirroring the existing style):
- Add `Strategy_Element` (+ nested `Draft_Element`, `Ngram_Element`, `Tree_Element`) and a `Roles_Element`, and route them from `PipelineConfig_Element::OnObject` (`src/config.cpp:1765-1775`): `if (name == "strategy") return strategy_; if (name == "roles") return roles_;`.
- Unknown keys still `throw JSON::unknown_value_error{}` — so a v2.0 parser (no strategy element) would reject these keys, which is why recognition is gated on block-presence (the `strategy`/`roles` block) and additive — both v2.0 and v2.1 share `version: 2`.
- Lowering: `LowerPipelineToModel` (`src/config.cpp:2010`) gains a branch that, when `strategy` is present, materializes the draft/target session files into `model.decoder.pipeline[]` entries tagged with their role so the existing `DecoderOnlyPipelineModel` session-loading path loads both graphs. Sessions are loaded eagerly in the model constructor — `sessions_.emplace_back(CreateSession(...))` at `src/models/decoder_only_pipeline.cpp:139` — and lazily (re)created on first use at `src/models/decoder_only_pipeline.cpp:194-198`.

---

## 4. Executor primitives required

These are the generic, reusable mechanisms the executor/Generator must gain. Each is tied to where it hooks into current code, and which object it touches.

### (a) Inner draft loop (K steps)

- **What:** produce K candidate tokens before a single verify.
- **Where:** a new `RunSpeculativeStep()` on `DecoderOnlyPipelineState` that wraps the existing per-stage machinery. The draft producer runs K times (separate-draft: K calls into the draft session via the existing `RunStage` path, `src/models/decoder_only_pipeline.cpp:352-509`; ngram: a pure CPU string match, no session). This sits *inside* the executor, between `UpdateInputsOutputs` (`:597-609`) and the verify call, replacing the single `RunPipeline` invocation in `Run` (`:511-559`).
- **Touches:** **DecoderOnlyPipelineState** (new inner loop) and, for token bookkeeping, **Generator** (the K provisional tokens must be visible to verify; see (c)).

### (b) Batched multi-candidate / tree verify + custom attention mask

- **What:** verify all K candidates (linear) or a candidate *tree* (Medusa/EAGLE-2/SpecInfer) in **one** target forward pass, which needs a non-causal **tree attention mask**.
- **Where:** the verify session is the existing target stage, but `input_ids` carries K+ candidate tokens and the attention mask is supplied by a new mask builder. The mask currently comes from `position_inputs_`/`attention_mask` plumbing (`PositionInputs`, used in `UpdateInputsOutputs`, `src/models/decoder_only_pipeline.cpp:601`); v2.1 adds a `SpeculativeMask` provider that emits the tree mask when `strategy.tree` is set.
- **Touches:** **DecoderOnlyPipelineState** (mask construction, multi-token `input_ids`, multi-row `logits` from `Logits` — the `logits_` member `src/models/decoder_only_pipeline.h:159`, updated via `logits_.Update` at `src/models/decoder_only_pipeline.cpp:608`). The ORT graph/attention kernel must accept an explicit additive mask — a known risk (§11).

### (c) Accept/reject comparison + variable tokens-per-step

- **What:** the **first data-dependent branch** in the whole system: compare draft proposals against target distribution under the chosen `acceptance` rule, accept the longest valid prefix (1…K+1 tokens), feed *all accepted tokens* back into the search state.
- **Where:** this belongs in the **outer** `Generator`, next to sampling, because that is where `search_`/`Sequences` and the sampling methods live (`Generator::GenerateNextToken`, `src/generators.cpp:628-695`; sampling switch `:679-694`; `search_->AppendTokens` already exists, `src/generators.cpp:494`, `:552`, `:656`). A new `Generator::AcceptSpeculative(draft_tokens, target_logits)` computes the accepted count, calls `search_->AppendTokens` for each accepted token (variable advancement), and emits one bonus token from the target distribution. The state's `Run` contract still returns target logits (`src/generators.cpp:538`, `src/models/model.h:31`), so the logits contract is unchanged — only *how many* tokens advance per outer call changes.
- **Touches:** **both.** Generator owns the branch + advancement; State owns producing the K+1-row logits the branch consumes.

### (d) KV-cache rollback / checkpoint — the biggest gap

- **What:** after accepting N of K, truncate the KV cache of **every** session to the accepted prefix; on full rejection, roll the draft session back too.
- **Current state:** the caches already implement `RewindTo` (`DefaultKeyValueCache::RewindTo` `src/models/kv_cache.cpp:508`; `CombinedKeyValueCache::RewindTo` `:72`; `ModelManagedKeyValueCache::RewindTo` `:708`; `LFM2Cache::RewindTo` `:856`). But **`DecoderOnlyPipelineState` does not override `State::RewindTo`** (default no-op, `src/models/model.h:34`), and **`Generator::RewindToLength` throws for `decoder-pipeline`** (`src/generators.cpp:697-699`). This is the precise spot to implement.
- **Design:** 
  1. Override `DecoderOnlyPipelineState::RewindTo(size_t length)` to call `key_value_cache_->RewindTo(length)` for all per-session caches it owns (`src/models/decoder_only_pipeline.h:161`), plus `position_inputs_`/`recurrent_state_` reset.
  2. Remove `decoder-pipeline` from the throw list in `Generator::RewindToLength` (`src/generators.cpp:698`) **once** the override exists, and add a lighter `Generator::TruncateToAccepted(n)` fast-path that rewinds by a small delta each step (rollback is per-step and cheap-ish, vs. the full reset `RewindToLength` implies).
  3. Multi-session: rollback must address each role's cache independently (draft and target advance by different amounts).
- **Touches:** **both** (State override + Generator un-gating and fast-path).

### (e) Intermediate hidden-state edges in dataflow

- **What:** EAGLE/EAGLE-3/MTP draft modules consume the target's **hidden states** (EAGLE-3: multiple layers), not just `logits`/KV.
- **Where:** the dataflow wiring already moves *named* non-managed tensors between stages via the `ortvalue_store_` and explicit `dataflow[]` wires (`src/models/decoder_only_pipeline.cpp:412-465`, `:490-504`; `Config::Pipeline::Wire` `src/config.h:432-435`). The extension is to allow a wire `from` an **intermediate activation** (e.g. `target.hidden_states`) — which requires the target ONNX graph to *expose* that output (a **build** property) and the schema to declare it as a `from` endpoint. No new wiring mechanism is needed; we only relax the assumption that non-managed outputs are `logits`/KV and document that intermediate tensors may be large/device-resident (today non-managed outputs are assumed CPU, `src/models/decoder_only_pipeline.cpp:490-491` — a real constraint to fix for hidden-state edges).
- **Touches:** **DecoderOnlyPipelineState** (relax the CPU-only assumption for forwarded tensors; keep them on device).

| Primitive | DecoderOnlyPipelineState | Generator |
|---|:--:|:--:|
| (a) inner draft loop | ✓ | (token visibility) |
| (b) tree verify + mask | ✓ | |
| (c) accept/reject + variable advance | (K+1 logits) | ✓ |
| (d) KV rollback | ✓ (override RewindTo) | ✓ (un-gate + fast-path) |
| (e) hidden-state edges | ✓ | |

---

## 5. Multi-session roles

v2.1 lets ≥2 sessions be referenced under named **roles**, each with its own KV state:

```jsonc
"roles": {
  "target": "target_session",
  "draft": "draft_session",
  "amateur": "small_session",       // contrastive
  "expert": "large_session",        // contrastive
  "unconditional": "target_session" // CFG: same weights, different context
}
```

- Backed by `Config::Pipeline::Roles` (§3.3). Each role resolves to a `sessions[]` entry (`src/config.h:418-422`) and gets an independent KV cache instance in the executor (the executor already owns one `key_value_cache_`, `src/models/decoder_only_pipeline.h:161`; v2.1 generalizes this to a small role→cache map).
- **Unlocks contrastive decoding & CFG cheaply (bucket A).** These need *no* draft loop, accept/reject, or rollback — they are a **static two-session combine** run every step (memo §1E, §3 item 2). With roles + the logit-processor chain (§6), `logit = expert − α·amateur` (contrastive, 2210.15097) or `logit = uncond + g·(cond − uncond)` (CFG, 2306.17806) is just two stage runs feeding a `combine` processor. No new control flow — they reuse the v2.0 static DAG. This is why roles are introduced as a *general* primitive, not a speculative-only one.

---

## 6. Logit-processor / sampler chain

Generalize the existing single llguidance hook (`src/generators.cpp:659-662`; `ConstrainedLogitsProcessor` interface at `src/constrained_logits_processor.h:24-34`) into a **config-listed, ordered chain** applied to logits before sampling:

```jsonc
"generation": {
  "logits": [
    { "op": "repetition_penalty", "value": 1.1 },
    { "op": "logit_bias", "map": { "50256": -100 } },
    { "op": "grammar", "backend": "llguidance", "grammar": "json_schema:...", "stateful": true },
    { "op": "combine", "mode": "contrastive", "alpha": 0.5,
      "expert": "expert", "amateur": "amateur" },
    { "op": "temperature", "value": 0.7 },
    { "op": "top_k", "value": 50 },
    { "op": "top_p", "value": 0.9 },
    { "op": "sample" }
  ]
}
```

- **Order matters:** penalties → bias → grammar-FSM mask → combine (contrastive/CFG) → temp/top-k/p → sample. This mirrors the order already hard-coded in `GenerateNextToken`: guidance (`:659-662`) → `ApplyMinLength`/`ApplyRepetitionPenalty` (`:665-666`) → sampling switch (`:679-694`). v2.1 makes that sequence *data-driven* and *extensible* instead of fixed C++.
- **Grammar/FSM** reuses `ConstrainedLogitsProcessor` (`src/constrained_logits_processor.cpp`) verbatim as the `grammar` op — including its stateful `CommitTokens`/`Reset`/`GetFFTokens` lifecycle (`src/constrained_logits_processor.h:24-34`, fast-forward handling at `src/generators.cpp:546-562`). This is the memo's bucket-B "generic processor slot".
- **`combine`** is where contrastive/CFG (§5) plug in — a stateless two-input op over role logits.
- **Backward-compat:** absent `logits` chain ⇒ the existing fixed order runs unchanged. The chain is opt-in and additive.
- **Touches:** **Generator** (replace the fixed sequence in `GenerateNextToken` with a chain walk; keep the default chain identical to today's behavior). `sampling_method_`/`SamplingMethod` (`src/generators.h:139-144`) becomes the terminal `sample` op.

---

## 7. Runtime-feature namespace

Formalize per-session runtime attributes under the existing per-session `session_options` (`src/config.h:388`, `:421`), explicitly tagged **runtime** vs **build** (memo §3 item 7). Most of bucket (A) lives here.

```jsonc
"sessions": {
  "target": {
    "file": "target.onnx",
    "runtime": {                       // NEW: schedulable at load, no graph change
      "kv_cache": { "dtype": "fp8", "quant": "per_token" },   // KVQuant/KIVI/FP8 KV
      "paging":   { "enabled": true, "block_size": 256 },     // PagedAttention
      "prefix_cache": { "enabled": true },                    // APC / RadixAttention
      "sliding_window": { "size": 4096, "sink_tokens": 4 },   // StreamingLLM (runtime part)
      "chunked_prefill": { "max_batched_tokens": 2048 },      // Sarathi
      "precision": "fp16"
    },
    "build_requires": {                // NEW: declared, validated, never synthesized
      "attention": "gqa",             // MQA/GQA — in weights
      "quantization": "awq",          // GPTQ/AWQ — in weights
      "extra_heads": "medusa"         // Medusa/MTP heads — in graph
    }
  }
}
```

- `runtime.*` maps onto existing engine/cache machinery: paging/prefix-cache to `src/engine/paged_key_value_cache.{h,cpp}` and `src/engine/cache_manager.*`; chunked prefill onto the existing `search.chunk_size` (`src/config.h:489`) and the sliding-window chunk loop already in `Run` (`src/models/decoder_only_pipeline.cpp:521-537`); KV dtype/quant onto the cache constructors (`src/models/kv_cache.cpp`).
- `build_requires.*` is **declarative only**: load-time validation fails fast with a clear message if the graph doesn't match (e.g. EAGLE draft requested but no `hidden_states` output). The schema never builds these.
- This keeps the honest runtime/build split visible *in the config itself*.

---

## 8. Controller plugin hook (bucket C)

Lookahead (stateful Jacobi n-gram pool + bespoke 2-D mask), deeply nested cascades, and novel research stay **plugin-only**. The current ABI hands over *model construction only* (`OgaCreatePipelineFn` returns an opaque model handle, `src/models/plugin_api.h:61-64`); the generation loop stays in `Generator`. To let a controller own the decode loop, the ABI must additionally expose a **decode-step callback** surface, still as a stable C ABI (opaque handles, plain pointers, int status — per the header's existing contract, `src/models/plugin_api.h:6-20`):

What `plugin_api.h` would need to **additionally** expose (all opaque-handle C, no C++ types crossing the boundary):

```c
// NEW, additive: a controller the runtime calls once per outer decode step.
typedef struct OgaDecodeController OgaDecodeController;

// Borrowed handles the controller may call back into (all opaque; adapted inside the runtime):
//   ctx -> run a named session, read/append tokens, rewind a role's KV, read logits.
typedef int (*OgaControllerStepFn)(OgaDecodeController* self,
                                    void* step_ctx /* OgaDecodeStepContext* */,
                                    int* tokens_emitted_out);

typedef void (*OgaDecodeControllerDestroyFn)(OgaDecodeController* self);

// Optional second entry point a controller plugin MAY export (alongside OgaCreatePipelineFn):
typedef int (*OgaCreateDecodeControllerFn)(void* config,
                                           OgaDecodeController** controller_out,
                                           OgaControllerStepFn* step_out,
                                           OgaDecodeControllerDestroyFn* destroy_out);
```

Plus a runtime-side `OgaDecodeStepContext` vtable (opaque) exposing the §4 primitives as C calls: `RunSession(name)`, `AppendAccepted(tokens,n)`, `RewindRole(role,len)`, `GetLogits()`, `SetMask(...)`. The controller owns custom state/masks/acceptance; the runtime owns sessions, KV, and search. When `pipeline.plugin` declares a controller entry point, `Generator::GenerateNextToken` (`src/generators.cpp:628`) delegates the step to `OgaControllerStepFn` instead of running the built-in path. This is the minimal extension that keeps Lookahead/cascades expressible without bloating the declarative core.

---

## 9. Phased rollout plan

Ordered by dependency and value-per-complexity. The memo (§3) is explicit that **KV-rollback + the speculative loop unlock the most coverage**, so they come first.

| PR | Scope | Files touched | Risk |
|---|---|---|---|
| **PR-A — KV rollback API for pipelines** *(foundation; highest value-per-complexity)* | Override `DecoderOnlyPipelineState::RewindTo` to rewind all owned caches + position/recurrent state; un-gate `decoder-pipeline` in `Generator::RewindToLength`; add `TruncateToAccepted` fast-path. No schema change. | `src/models/decoder_only_pipeline.{h,cpp}` (add `RewindTo` override), `src/generators.cpp:697-714` (un-gate), reuse `src/models/kv_cache.cpp:508` etc. | Med — must preserve `past_present_share_buffer` semantics; verify no regression to non-speculative pipelines. |
| **PR-B — `speculative` strategy schema + vanilla draft-target** | `Config::Pipeline::Speculative`/`Roles` structs + SAX elements; `LowerPipelineToModel` branch; inner draft loop (4a); linear verify (4b minus tree); accept/reject + variable advance (4c) using PR-A rollback. Greedy + rejection_sampling. | `src/config.h:417-470`, `src/config.cpp:1754-1795` (+ new elements), `src/models/decoder_only_pipeline.{h,cpp}`, `src/generators.cpp:628-695`, `src/models/model.cpp:891-945` (route the new structural signal). | High — first data-dependent branch + multi-session KV; depends on PR-A. |
| **PR-C — tree / EAGLE hidden-state edges** | Tree-attention mask builder (4b tree); `strategy.tree`; intermediate hidden-state edges (4e) — relax CPU-only forwarded-output assumption. Enables Medusa/EAGLE/MTP (heads are build artifacts). | `src/models/decoder_only_pipeline.cpp:412-504` (mask + device-resident forwarding), `src/config.h` (`Tree`), parser. | High — tree-attention mask plumbing through ORT kernels is the main unknown (§11). |
| **PR-D — logit-processor chain** | Config-listed ordered chain; wrap existing penalties/guidance/sampling into ops; `combine` op unlocks contrastive/CFG (§5/§6). Default chain == today's behavior. | `src/generators.cpp:659-694`, `src/generators.h:139-144`, reuse `src/constrained_logits_processor.*`, `src/config.cpp` (`Generation_Element` at `:1859-1884`). | Med — mostly a refactor of an existing fixed order; guard exact-equivalence default. |
| **PR-E — controller plugin loop hook** | Additive C ABI (`OgaCreateDecodeControllerFn` + step context); delegate the decode step when declared. Enables Lookahead/cascades (bucket C). | `src/models/plugin_api.h` (additive), `src/models/plugin_loader.*`, `src/generators.cpp:628`. | Med — ABI surface design; must stay opaque-handle-only. |
| **PR-F — runtime-feature namespace** | `session.runtime` / `build_requires` schema + load-time validation; map onto engine caches. Formalizes bucket A. | `src/config.h:386-404`/`:418-422`, `src/config.cpp` (`PipelineSession_Element` at `:1548`), `src/engine/*`, `src/models/kv_cache.cpp`. | Low-Med — mostly declarative + validation; runtime knobs already exist. |

Dependency order: **PR-A → PR-B → {PR-C, PR-D}**; PR-E and PR-F are independent and can land any time after the schema groundwork (PR-B for PR-E's role context; standalone for PR-F).

---

## 10. Backward-compat & testing

### Zero-regression gate

- `ClassifyStructuralRoute == ClassifyLegacyRoute` (`src/models/model.cpp:891-945` vs `:853-878`) stays green: a `speculative` strategy is a **new** structural signal. Configs without `strategy`/`roles`/`logits`/`runtime` classify exactly as today (the new predicates are only evaluated when those optional fields are present). `PipelineDispatchTests` continues to assert equivalence across all existing fixtures.
- Block-presence gating: v1 and v2.0 configs (no `strategy`/`roles`/`logits`/`runtime` block) never reach the new parser branches (`src/config.cpp:1905-1906`, `:2194-2198` — `version >= 2 && pipeline.present`); v2.0 and v2.1 are both `version: 2`, so a version bump is **not** the gate. Unknown keys still throw, so the new keys are inert in old configs.
- The default logit-processor chain (PR-D) must be byte-identical to the current fixed order (`src/generators.cpp:659-694`) — a golden-logits test guards this.

### New tests/fixtures per PR

| PR | Unit-testable (no real models) | Needs real models |
|---|---|---|
| PR-A | `RewindTo` on a stub pipeline cache; `RewindToLength` no longer throws for `decoder-pipeline`; idempotence at length==current. | Round-trip correctness after rewind on a small real decoder. |
| PR-B | Schema parse/lower of `strategy`/`roles`; route resolves to `DecoderOnlyPipeline`; acceptance-rule math (greedy/rejection) on synthetic logits; variable-advance bookkeeping in `search_`. | End-to-end with a real **draft + target** pair (accept rate, output equals non-speculative greedy under greedy acceptance — the correctness invariant). |
| PR-C | Tree-mask construction from `medusa_choices`; hidden-state edge wiring keeps tensor on device. | EAGLE/Medusa acceptance on real heads. |
| PR-D | Chain ordering; default chain == legacy (golden logits); `combine` math (contrastive/CFG). | Grammar decoding parity with existing llguidance tests. |
| PR-E | ABI shim round-trip with a stub controller `.so`; status-code propagation. | A real Lookahead controller. |
| PR-F | `runtime`/`build_requires` parse + validation errors. | KV-quant/paging numeric parity. |

**Key invariant for PR-B:** under `acceptance: greedy` with temperature 0, speculative output **must** equal plain greedy decoding token-for-token — a deterministic, model-pair test that catches rollback/advancement bugs without statistical flakiness.

---

## 11. Open questions / risks

1. **KV-rollback cost.** Per-step truncation across multiple role caches may dominate the speedup if implemented as a full re-copy. PR-A's `TruncateToAccepted` delta-rewind must be O(rejected tokens), not O(sequence). Interaction with `past_present_share_buffer` (`src/config.h:487`) and the partial-KV-update worker thread (`src/models/decoder_only_pipeline.cpp:368-388`, `:479-488`) needs care — speculative rollback and async partial updates must not race.
2. **Tree-attention mask through ORT.** The verify pass needs an explicit non-causal additive mask. Whether the target ONNX graph + attention kernel accept a caller-supplied tree mask (vs. an internal causal mask) is the single biggest feasibility unknown (PR-C). If unsupported, fall back to linear-K verify (no tree), which still covers vanilla/PLD/self-spec.
3. **Self-speculative / early-exit needs partial-graph execution.** Running the target to a shallow `depth` requires the exported graph to expose an early-exit head; ORT does not easily run a sub-graph "to layer N". This is a **build** requirement (LayerSkip recipe), declared via `build_requires`, not something the schema can synthesize.
4. **Hidden-state edges are large & device-resident.** Today non-managed forwarded outputs are assumed CPU (`src/models/decoder_only_pipeline.cpp:490-491`). EAGLE hidden states must stay on device; PR-C must relax this without breaking existing CPU-forwarded vision/embedding wiring.
5. **Scheduler interactions (engine).** Continuous batching, paging, and prefix cache live in `src/engine/*` (`scheduler.{h,cpp}`, `paged_key_value_cache.*`, `cache_manager.*`). Variable tokens-per-step and per-role KV rollback complicate batch accounting; v2.1's first cut should target the single-request `Generator` path and treat engine-level speculative batching as a follow-on (it is where vLLM/TRT-LLM spend most of their complexity).
6. **Acceptance-rule numerics.** Rejection sampling (Leviathan 2211.17192) requires the *draft's* token probabilities, not just argmax — the draft producer must surface a distribution, which `ngram` cannot (hence `ngram` is restricted to `greedy` acceptance).
7. **Controller ABI scope creep.** The PR-E step-context vtable could grow unboundedly. Keep it to the §4 primitives (run session, append accepted, rewind role, get logits, set mask); richer needs stay inside the controller's own state.

---

## 12. Summary

v2.1 = today's static session-DAG (covers all of bucket A) **+** one parameterized **`speculative` flow strategy** with {multi-session roles, acceptance rule, KV rollback, variable/tree tokens, hidden-state edges} (covers ~all of bucket B) **+** a **logit-processor chain** (grammar/guidance/contrastive/CFG) **+** an **extended controller-plugin hook** (bucket C). The entire 2024–2026 speculative-decoding family collapses into a single reusable executor strategy, and the highest value-per-complexity work — **PR-A (KV rollback for pipelines)** — is also the smallest and most self-contained, because the caches already implement `RewindTo`; the pipeline executor and `Generator` simply never wire it up today.
