# `qwen38_flash_graph`: CUDA-graph decode with a built-in MTP(1) loop, served by the Engine

`model.type = "qwen38_flash_graph"` runs a Qwen3.8-Flash-Next style decoder that has been exported for CUDA-graph capture
(fixed in-place KV store, a ring of recurrent-state banks, device-side greedy token selection) together with a one-layer
multi-token-prediction (MTP) head. The draft -> verify -> commit loop runs on the GPU; the host only waits for one small
result record per cycle. The model is served by the **Engine API** (`og.Engine` / `OgaEngine` / `Engine::Run`) on its
static path; it has no Generator API support (`og.Generator` reports that the Engine API is required).

The Engine side of the design (cache manager, executor, `Request::CommitExternalTokens`) is described in
[paged_attention_engine.md](paged_attention_engine.md#static-single-slot-backend-that-selects-its-own-tokens). This page
describes the model type itself.

## Scope and limits

* Linux + CUDA only. The CUDA data plane is `dlopen`-ed and the PLE table is `mmap`-ed; on other platforms the sources
  compile, but creating the model fails with a clear message.
* Greedy decoding only. `BeginTurn` rejects sampling, `repetition_penalty`, `no_repeat_ngram_size`, `min_generated_tokens`,
  per-turn seeds and guidance before it changes the request.
* One resident request. Further requests queue behind it (static-batching semantics). One Engine per model at a time.
* No prefix cache, no continuous batching, no caller-supplied draft tokens (`MaxDraftTokensPerStep()` is 0).
* A continued turn and the replay after `RewindToStartOfTurn` re-ingest the whole retained history: the look-ahead cycle and
  the recurrent-state bank ring cannot be cropped to an arbitrary token. Prompt ingestion uses the S=2 target graph (two
  tokens per step), so a long history is slow to re-ingest.
* A session holds at most `kv_capacity - 4` tokens (a verify step writes two rows past the committed length).
* The Engine never delivers the stop token itself (finish reason `EOS`), as for every other model.

## What a package must provide

GenAI does not export this model. The package is produced by an exporter and consists of:

```text
<package>/
  genai_config.json
  tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json
  target/model.onnx (+ external data)        # the S=1 / S=2 target graphs (see "Graph contract")
  mtp_head/model_s1.onnx, model_s2.onnx       # the MTP head for one and two rows (only if MTP is enabled)
  libs/<custom op libraries>.so              # every custom-op library the graphs use (see below)
  libs/libqwen38_plane.so                    # built from src/qwen38_plane (-DENABLE_QWEN38_PLANE=ON)
  ple_table.txt                              # description of the host-side PLE table (see "PLE table")
  ple/<shards>                               # the FP8 PLE shard files named by ple_table.txt
```

All paths in `genai_config.json` are relative to the package folder and must stay inside it (absolute paths and `..` are
rejected, as are plane libraries that resolve outside the folder through symbolic links).

Runtime requirements (none of them is needed to build GenAI or to run its unit tests):

* **Custom-op libraries supplied by the package.** The exported graphs use custom ops that are not part of ONNX Runtime or
  GenAI. They are listed in `session_options.custom_ops_libraries` (new, see below) and loaded in order through
  `RegisterCustomOpsLibrary`.
* **The plugin CUDA execution provider**, registered before the model is loaded
  (`og.register_execution_provider_library("CUDAExecutionProvider", <path>)`), built from an ONNX Runtime that supports the
  graphs' operators, the S=2 target graphs and the `arena_extend_strategy=kSameAsRequested` allocator option.
  The performance numbers of the pull request that introduced the model type depend on provider kernels that are not part
  of this repository.
* **The PLE table in the page cache.** The per-layer embedding (PLE) table is large (about 51 GB for the
  Qwen3.8-Flash-Next package) and is read zero-copy over PCIe by the gather kernel; it is mapped read-only and registered with the CUDA
  driver (`cudaHostRegister`) when the model is created. Creating the model takes as long as it takes to fault in the files.
* CUDA graph capture of the decoder: `enable_cuda_graph` and `min_num_runs_before_cuda_graph_capture` in the provider options
  (see the example), and a model-creation warm-up (`model.qwen38_graph.warmup`, default on) that runs every graph often
  enough to capture it.

## `genai_config.json`

```jsonc
{
  "model": {
    "type": "qwen38_flash_graph",
    "eos_token_id": [248046, 248044],        // stop tokens: the device loop stops after one of them
    "vocab_size": 248320,
    "context_length": 33024,                 // kv_capacity - 4
    "decoder": {                             // the target session
      "filename": "target/model.onnx",
      "hidden_size": 2560, "head_size": 256, "num_attention_heads": 24, "num_hidden_layers": 48, "num_key_value_heads": 2,
      "session_options": {
        "graph_optimization_level": "ORT_DISABLE_ALL",
        "provider_options": [{"cuda": {"enable_cuda_graph": "1", "min_num_runs_before_cuda_graph_capture": "4",
                                       "arena_extend_strategy": "kSameAsRequested"}}],
        "custom_ops_libraries": ["libs/<op library 1>.so", "libs/<op library 2>.so"]
      }
    },
    "qwen38_graph": {
      "plane_library": "libs/libqwen38_plane.so",
      "ple_table": "ple_table.txt",
      "kv_capacity": 33028,
      "mtp": {"enabled": true, "prefill_head": true},
      "head": {
        "filename_s1": "mtp_head/model_s1.onnx",
        "filename_s2": "mtp_head/model_s2.onnx",
        "session_options": {"custom_ops_libraries": ["libs/<op library 1>.so", "libs/<op library 2>.so", "libs/<head library>.so"]}
      }
    }
  },
  "search": {"do_sample": false, "max_length": 33024}
}
```

### `model.qwen38_graph`

| key | default | meaning |
| --- | --- | --- |
| `plane_library` | required | CUDA data-plane library (`libqwen38_plane.so`), `dlopen`-ed at model creation. Must resolve inside the package. |
| `ple_table` | required | PLE table description file (see below). |
| `head.filename_s1`, `head.filename_s2` | required (`head.filename_s1` selects the section; `filename_s2` is required when MTP is enabled) | MTP head graphs for one and for two rows. |
| `head.session_options` | inherits the decoder's | Session options of the head sessions. Unset fields (providers, provider options, config entries, `custom_ops_library`, `custom_ops_libraries`, graph optimization level) are inherited from `model.decoder.session_options`. |
| `kv_capacity` | 33028 | Positions of the fixed KV store. Must equal the sequence dimension of the graph's `past_key_values.*.key/value` inputs. A session holds `kv_capacity - 4` tokens. |
| `target_s1_graph_id`, `target_s2_graph_id` | 41, 51 | First `gpu_graph_id` run option of the S=1 / S=2 target graphs; three consecutive ids are used (one per bank phase). |
| `head_s1_graph_id`, `head_s2_graph_id` | 61, 63 | First `gpu_graph_id` of the head graphs; two consecutive ids are used (double-buffered head state). |
| `prefill_chunk_tokens` | 2048 | Prompt tokens ingested per Engine step (rounded down to an even number, at least 2). A step that only advances the prompt delivers no event, so `CancelRequest` stays responsive. |
| `warmup` | true | Run every graph often enough to capture it when the model is created. |
| `mtp.enabled` | true | `false` runs the plain S=1 graph decode (no draft/verify); the head sessions are not created. |
| `mtp.prefill_head` | true | Feed the target's hidden rows of the prompt to the head. `false` starts the head without prompt context (lower acceptance). |

Unknown keys are rejected. Config overlays (`og.Config.overlay`) can change any of these, for example `{"model":{"qwen38_graph":{"mtp":{"enabled":false}}}}`
for a no-MTP run of the same package.

### `session_options.custom_ops_libraries`

A list of custom-op library paths, available in every `session_options` object. The libraries are registered after
`custom_ops_library` (if any), in list order. Every entry goes through the same path checks and allow-list as
`custom_ops_library`: no `..` in the path, a relative path is searched in the model folder, the directories of the
registered execution-provider libraries and the working directory (in that order), and the resolved library must lie in one
of those directories. All entries are validated before any of them is registered. `custom_ops_libraries` is inherited like the other session options.

## Graph contract

The loop binds fixed device buffers to the graphs by name, so the exported graphs must follow this contract. Dynamic
dimensions are bound as 1 (batch size 1).

Target (`model.decoder.filename`), S rows per call, S = 1 or 2:

| inputs | |
| --- | --- |
| `input_ids` `[1,S]` int64, `position_ids` `[1,S]` int64, `attention_mask` `[1,S]` int64 (ones) | token rows of this step |
| `committed_length` `[1]` int32 | committed length before the step |
| `sequence_length_minus_one` `[1]` int32, `cumulative_sequence_lengths` `[2]` int32 | `S-1`, `{0,S}` |
| `logits_indices` `[S]` int64 (optional) | `0..S-1` |
| `ple_embeddings.<layer>` `[1,S,ple_width]` bf16 | exactly one per graph; produced by the data plane from the PLE table (`ple_width` must equal `n_heads * head_dim` of the table) |
| `past_position_ids`, `past_key_values.<i>.key`, `.value` | in-place KV store (one buffer, bound as input and output) |
| `past_key_values.<i>.index_key` | in-place index store (same treatment) |
| every other `past_key_values.*` | recurrent state; the engine keeps three identical banks plus a scratch bank |

| outputs | |
| --- | --- |
| `present_position_ids`, `present.<i>.key/.value/.index_key` | the in-place stores |
| `present.<other>` | committed recurrent state, written to the next bank of the ring |
| `present_mid.<other>` | recurrent state after the first of two rows (S=2); written to the scratch bank when S=1 |
| `logits` `[1,S,vocab]`, `next_tokens` `[1,S]` int64 | `next_tokens` are the greedy tokens of both rows |
| `hidden_states_out` (or `hidden_states`) `[1,S,hidden]` bf16 | fed to the head |
| `validation.aggregate` `[N]` int32 | non-zero entries mark a device-side validation failure; the model's metadata key `experiment.validation_status_names` (a JSON-like list of quoted names) is optional and only names the entries in the error message |

MTP head (`head.filename_s1` for one row, `head.filename_s2` for two rows): inputs `input_ids`, `position_ids`,
`hidden_states`, `committed_length` (S=1) or the same names with a `.0` / `.1` suffix per row (S=2);
`past_key_values.0.key/.value/.index_key` (in-place) and `past_key_values.0.index_buffer/.index_lengths` (double-buffered).
Outputs: `logits`, `hidden_states_out`, `next_token`, `validation.aggregate`, `present_mid.*`, `present.*`.

Any other input or output name is rejected when the model is created ("unsupported target input ...").

## PLE table (`ple_table.txt`)

A text file read by `ParseQwen38PleDescription`:

```text
qwen38_ple_table 1
n_heads 2
head_dim 4
heads_per_ngram 1
eos 7
scale 0.5
mult 1 2 3 0
vocab 11 13
offset 0 11
shard ple/a.safetensors 128 11
shard ple/b.safetensors 0 13
```

`n_heads` (at most 32) heads of `head_dim` FP8 (E4M3) values are gathered per token. The row index of head `h` is a hash
of the current token and the previous one (heads below `heads_per_ngram`) or of the current token and the two previous ones
(the other heads): the tokens are multiplied by `mult`, XOR-ed, reduced modulo `vocab[h]` and shifted by `offset[h]`. The
older of the two previous tokens is replaced by `eos` when the previous token is `eos`. `scale` multiplies the dequantised values. A `shard <file> <byte offset> <rows>` line names a file inside the
package (relative, no `..`), the byte offset of its first row, and its row count; shards are concatenated in order (at most
256). The mapping and the hash are implemented in `src/models/qwen38_graph_plane.cpp` and `src/qwen38_plane/qwen38_plane.cu`.

## Data-plane library

`libqwen38_plane.so` (`-DENABLE_QWEN38_PLANE=ON`, Linux + CUDA, default off) contains the per-row input preparation (including
the PLE gather), a deterministic argmax, the verify/step epilogues that publish a result record to pinned host memory, the
head launch preparation and thin CUDA runtime helpers. The core GenAI library never links the CUDA runtime: the C ABI is
declared in `src/qwen38_plane/qwen38_plane_abi.h` (version 1, checked when the library is loaded), shared by the library and by the loader.

## Use

```python
og.register_execution_provider_library("CUDAExecutionProvider", PROVIDER_SO)
model = og.Model(og.Config(PACKAGE))
engine = og.Engine(model)
buffer = engine.create_event_buffer(16)
request = engine.create_request()
turn = og.TurnOptions(request)
turn.set_do_sample(False)                  # greedy only
turn.set_max_generated_tokens(256)
request.begin_turn(prompt_tokens, turn)
while engine.has_pending_requests():
    for event in engine.run(buffer):       # one Run = one prompt chunk or one cycle (1-2 tokens)
        ...
```

`engine.get_speculative_stats()` reports one round per cycle, accepted drafts, bonus/correction tokens and the
acceptance-length histogram. `engine.get_capabilities()` reports one row and `max_request_length = kv_capacity - 4`.

Qualification scripts that need a real package are in `tools/qwen38_graph/`; the logic that does not need a GPU is covered by
`test/cpp/engine/qwen38_graph_*_tests.cpp` and `custom_ops_libraries_tests.cpp` (a scripted device drives the real Engine).
