# LiLiCorr (DFlash) drafter

LiLiCorr is the DFlash 2 block drafter that NVIDIA ships for the Nemotron-3.5 Bolt models
(`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr`). Its backbone is the DFlash 2 one
(a non-causal block of one anchor plus `block_size - 1` mask tokens, grouped causal convolutions, one
sliding-window attention sink per head). The candidate selector is replaced by the LiLiCorr lattice
head: the top-`k` target tokens of every mask row are re-scored by a small transformer over the
`(slot, rank)` lattice, and the anchor of the lattice is `hidden_norm(fc(aux)[last committed row])`.

The exported graph keeps the DFlash 2 runtime contract. It still produces `draft_candidate_ids
[B, n, k]` and `draft_scores [B, n, k, k]` (row 0 of the scores carries the start scores), so the host
walk and the paged drafter cache are shared with DFlash 2. The only extra input is the optional
`anchor_row_index` (int32, one entry per query block): the packed `aux_hidden_states` row of the
request's last committed context row. Configure it with `model.dflash2.inputs.anchor_row_index`.
Like DFlash 2, the CUDA graph uses packed QKV projections and `qkv_row_map`; it does not
declare the unused `q_row_map` input, avoiding a host copy during CUDA graph capture.

## Export

The drafter embeds the target's embedding table and LM head (LiLiCorr scores candidates with the
target's log-softmax), so the exporter reads both checkpoints:

```bash
cd src/python/py/models
python -m builders.lilicorr \
  --draft_dir <NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr> \
  --target_dir <NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB> \
  --output_dir <out>            # dflash2.onnx + dflash2_genai_config_section.json
```

* `--precision bf16|int4|int8` selects the body precision (large NVFP4 projections are dequantized to bf16
  for `bf16`).
* `--lm_head_precision int8|bf16`: the target's ggml Q8_0 head runs as an exact int8 `MatMulNBits`
  (default) or as a dense bf16 matmul.
* Merge `dflash2_genai_config_section.json` into the target's `genai_config.json` under `model.dflash2`
  (it lists `anchor_row_index` and `aux_hidden_state_layers`).

The target must expose the residual stream entering the layers in `aux_hidden_state_layers`
(`1, 18, 35, 51, 68, 85` for Bolt-74B) as `aux_hidden_states`.

## Notes

* Avoid `ReduceLogSumExp` / `ReduceMax` over the `[rows, vocab]` logits: on the CUDA execution
  providers they run a cuDNN reduction that is ~100x slower than TopK + Exp + ReduceSum. The exporter
  derives the log-partition from the sorted top-k instead.
* `GatherSliceToSplitFusion` can merge the selector's axis-1 slices of `[batch, 7, 8, 1024]` into
  an unequal `Split` with sizes `1 + 6`. Its general CUDA path copies host-side pointer and split
  arrays; these host buffers must remain valid for CUDA graph replay. This was verified with fusion
  enabled on ORT stack #33967 at `dc85408` (API 33, CUDA 12.8, H100), which retains captured host
  buffers. Candidate IDs, scores, and drafts matched the unfused path across capture/replay and
  fresh context lengths 1-8. The fixture's legacy Q8 head was migrated to the current `MatMulPacked`
  Q8_0 contract without changing effective scales or quantized bytes.
* The Bolt adapter retains this narrow exclusion because the supported external runtimes do not
  yet guarantee the captured-host-buffer fix. An API version alone does not establish that
  guarantee. Existing caller exclusions are retained when adding the workaround; the adapter does
  not lower the overall graph optimization level. The successful qualification above is evidence
  for that exact build, not a claim that every older runtime fails.
* With capture enabled, Bolt warms disposable target state with four verification calls and,
  when present, the drafter with four proposal calls. These conservatively cover the legacy
  two-eager-run capture delay, capture, and a subsequent replay. Newer ORT can perform warm-up
  and capture internally within one `Run`, so the four calls do not always map to those four
  phases. Caches, cursors, and statistics are reset before the real prompt.
* With capture the drafter step takes ~2.5 ms against ~3.3 ms eager (H200, 8 context rows).
* `lilicorr_drafter_tests` (CUDA kernel tests) replays recorded drafter steps through the real
  `Dflash2Drafter`; it is skipped unless `LILICORR_TEST_DIR` is set.
