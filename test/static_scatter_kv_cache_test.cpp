// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "models/static_scatter_indices.h"

#include <gtest/gtest.h>

namespace Generators::test {

// The StaticScatterIndexTracker drives the write_indices / nonpad_kv_seqlen
// inputs of a mobius static-scatter (TensorScatter) KV cache. These tests pin
// the off-by-one / init contract: the first (prefill) step writes at row 0 and
// reports nonpad == prefill length, and each later step appends contiguously.

TEST(StaticScatterIndexTracker, StartsEmpty) {
  StaticScatterIndexTracker tracker;
  EXPECT_EQ(tracker.valid_tokens(), 0);
}

TEST(StaticScatterIndexTracker, FirstPrefillWritesAtRowZero) {
  StaticScatterIndexTracker tracker;
  // Prompt of 5 tokens: write at row 0, 5 valid tokens after.
  const StaticScatterIndices prefill = tracker.Advance(5);
  EXPECT_EQ(prefill.write_index, 0);  // NOT -1: the crux off-by-one
  EXPECT_EQ(prefill.nonpad_seqlen, 5);
  EXPECT_EQ(tracker.valid_tokens(), 5);
}

TEST(StaticScatterIndexTracker, DecodeStepsAppendContiguously) {
  StaticScatterIndexTracker tracker;
  tracker.Advance(5);  // prefill

  const StaticScatterIndices decode1 = tracker.Advance(1);
  EXPECT_EQ(decode1.write_index, 5);     // this step's offset == previous nonpad
  EXPECT_EQ(decode1.nonpad_seqlen, 6);

  const StaticScatterIndices decode2 = tracker.Advance(1);
  EXPECT_EQ(decode2.write_index, 6);
  EXPECT_EQ(decode2.nonpad_seqlen, 7);
}

TEST(StaticScatterIndexTracker, SingleTokenPrefillThenDecode) {
  // Degenerate prompt length 1 must still write at row 0, then decode at row 1.
  StaticScatterIndexTracker tracker;
  const StaticScatterIndices prefill = tracker.Advance(1);
  EXPECT_EQ(prefill.write_index, 0);
  EXPECT_EQ(prefill.nonpad_seqlen, 1);

  const StaticScatterIndices decode = tracker.Advance(1);
  EXPECT_EQ(decode.write_index, 1);
  EXPECT_EQ(decode.nonpad_seqlen, 2);
}

TEST(StaticScatterIndexTracker, ResetReturnsToEmptyCache) {
  StaticScatterIndexTracker tracker;
  tracker.Advance(4);
  tracker.Advance(1);
  ASSERT_EQ(tracker.valid_tokens(), 5);

  tracker.Reset();
  EXPECT_EQ(tracker.valid_tokens(), 0);

  const StaticScatterIndices reprefill = tracker.Advance(3);
  EXPECT_EQ(reprefill.write_index, 0);
  EXPECT_EQ(reprefill.nonpad_seqlen, 3);
}

TEST(StaticScatterIndexTracker, MatchesSliceAFixtureGolden) {
  // Exact values from the #366 slice-A fixture golden_io.npz: a 4-token prefill
  // then a 1-token decode. Keeps this producer locked to the frozen contract.
  StaticScatterIndexTracker tracker;
  const StaticScatterIndices prefill = tracker.Advance(4);
  EXPECT_EQ(prefill.write_index, 0);
  EXPECT_EQ(prefill.nonpad_seqlen, 4);

  const StaticScatterIndices decode = tracker.Advance(1);
  EXPECT_EQ(decode.write_index, 4);
  EXPECT_EQ(decode.nonpad_seqlen, 5);
}

// Scaffolding for the end-to-end cache test owned by the build-test-genai (qa)
// task. It drives the mobius slice-A fixture (#366, bias-aware external-KV
// static-cache decoder: static_cache_bias_decoder.onnx + golden_io.npz, 2
// layers, key_cache.{i}/value_cache.{i} [batch,16,32], write_indices/
// nonpad_kv_seqlen [batch] int64) through CreateKeyValueCache and asserts:
//   * StaticScatterKeyValueCache::IsStaticScatterCache(model) selects this
//     variant (and does NOT fire for a non-scatter model).
//   * Per-layer 3D buffers [batch, max_seq_len, kv_hidden] are allocated at the
//     declared max_seq_len/kv_hidden (batch from BatchBeamSize, symbolic in the
//     graph), incl. per-layer kv_hidden variation (Gemma-4 Phase 2).
//   * Add() binds key_cache.{i} and updated_key_cache.{i} to the SAME OrtValue
//     (in-place share-buffer aliasing) and Update()/RewindTo() leave it intact.
//   * Prefill (seq=4) then a decode step (write@4, nonpad=5) match the golden
//     logits/updated caches (parity).
// Needs a GPU/CPU build with opset-24 TensorScatter/Attention kernels, so it
// cannot run here yet.
TEST(StaticScatterKeyValueCache, DISABLED_EndToEndFixtureParity) {
  GTEST_SKIP() << "Requires the #366 slice-A fixture wired into a genai_config.json "
                  "and a build with opset-24 TensorScatter/Attention kernels "
                  "(build-test-genai task).";
}

}  // namespace Generators::test
