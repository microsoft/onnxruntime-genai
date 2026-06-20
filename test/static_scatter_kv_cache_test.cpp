// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <gtest/gtest.h>

#include "span.h"
#define OGA_USE_SPAN 1
#include <ort_genai.h>

#include "models/static_scatter_indices.h"
#include "test_utils.h"

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
  EXPECT_EQ(prefill.nonpad_kv_seqlen, 5);
  EXPECT_EQ(tracker.valid_tokens(), 5);
}

TEST(StaticScatterIndexTracker, DecodeStepsAppendContiguously) {
  StaticScatterIndexTracker tracker;
  tracker.Advance(5);  // prefill

  const StaticScatterIndices decode1 = tracker.Advance(1);
  EXPECT_EQ(decode1.write_index, 5);     // this step's offset == previous nonpad
  EXPECT_EQ(decode1.nonpad_kv_seqlen, 6);

  const StaticScatterIndices decode2 = tracker.Advance(1);
  EXPECT_EQ(decode2.write_index, 6);
  EXPECT_EQ(decode2.nonpad_kv_seqlen, 7);
}

TEST(StaticScatterIndexTracker, SingleTokenPrefillThenDecode) {
  // Degenerate prompt length 1 must still write at row 0, then decode at row 1.
  StaticScatterIndexTracker tracker;
  const StaticScatterIndices prefill = tracker.Advance(1);
  EXPECT_EQ(prefill.write_index, 0);
  EXPECT_EQ(prefill.nonpad_kv_seqlen, 1);

  const StaticScatterIndices decode = tracker.Advance(1);
  EXPECT_EQ(decode.write_index, 1);
  EXPECT_EQ(decode.nonpad_kv_seqlen, 2);
}

TEST(StaticScatterIndexTracker, MatchesSliceAFixtureGolden) {
  // Exact values from the #366 slice-A fixture golden_io.npz: a 4-token prefill
  // then a 1-token decode. Keeps this producer locked to the frozen contract.
  StaticScatterIndexTracker tracker;
  const StaticScatterIndices prefill = tracker.Advance(4);
  EXPECT_EQ(prefill.write_index, 0);
  EXPECT_EQ(prefill.nonpad_kv_seqlen, 4);

  const StaticScatterIndices decode = tracker.Advance(1);
  EXPECT_EQ(decode.write_index, 4);
  EXPECT_EQ(decode.nonpad_kv_seqlen, 5);
}

// M1 fail-loud contract: StaticScatterKeyValueCache::RewindTo MUST throw rather
// than silently no-op. RewindTo cannot reset the write_indices/nonpad_kv_seqlen
// stream (it lives in InputIDs with no rewind hook), so a no-op would leave the
// index tracker stale -> wrong scatter slots + over-reported nonpad => silently
// wrong logits. DecoderOnly_State::RewindTo reaches kv_cache_->RewindTo and
// OgaGenerator::RewindTo is a public API not blocked for these models, so the
// throw is the only thing standing between a rewind call and silent corruption.
//
// Constructing the cache requires a full Model/State, so this asserts the throw
// through the public generator API against the #366 slice-A fixture wired as a
// genai model dir. It is DISABLED until that fixture lands under
// test/models/static-scatter-fixture/ (owned by the build-test-genai task); the
// e2e parity test below shares the same dir. Enable both together.
TEST(StaticScatterKeyValueCache, DISABLED_RewindToThrows) {
  auto model = OgaModel::Create(MODEL_PATH "static-scatter-fixture");
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 16);
  params->SetSearchOption("batch_size", 1);

  auto generator = OgaGenerator::Create(*model, *params);
  const std::vector<int32_t> prompt{0, 1, 2, 3};
  generator->AppendTokens(prompt);
  generator->GenerateNextToken();  // advance the index stream past prefill

  // Rewinding the static-scatter cache is unsupported and must fail loud.
  EXPECT_THROW(generator->RewindTo(2), std::exception);
}

// Scaffolding for the end-to-end cache parity test owned by the build-test-genai
// (qa) task. It drives the mobius slice-A fixture (#366, bias-aware external-KV
// static-cache decoder: static_cache_bias_decoder.onnx + golden_io.npz, 2
// layers, key_cache.{i}/value_cache.{i} [batch,16,32], write_indices/
// nonpad_kv_seqlen [batch] int64) through CreateKeyValueCache and asserts:
//   * StaticScatterKeyValueCache::IsStaticScatterCache(model) selects this
//     variant (and does NOT fire for a non-scatter model).
//   * Per-layer 3D buffers [batch, max_seq_len, kv_hidden] are allocated at the
//     declared max_seq_len/kv_hidden (batch from BatchBeamSize, symbolic in the
//     graph), incl. per-layer kv_hidden variation (Gemma-4 Phase 2).
//   * Add() binds key_cache.{i} and updated_key_cache.{i} to the SAME OrtValue
//     (in-place share-buffer aliasing) and Update() leaves it intact.
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
