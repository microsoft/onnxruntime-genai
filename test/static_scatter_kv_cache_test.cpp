// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "span.h"
#define OGA_USE_SPAN 1
#include <ort_genai.h>
#include <gtest/gtest.h>

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
  EXPECT_EQ(decode1.write_index, 5);  // this step's offset == previous nonpad
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

// Shared helpers + frozen golden for the fixture-backed cache tests below. The
// #366 slice-A fixture (test/models/static-scatter-bias-decoder) is a bias-aware
// external-KV static-cache decoder: mistral backbone, 2 layers,
// key_cache.{i}/value_cache.{i} [batch,16,32] FLOAT, write_indices /
// nonpad_kv_seqlen [batch] int64. It runs on the CPU EP (empty provider_options)
// so it matches the CPU MEA (opset 24) golden captured in golden_io.npz.
//
// These fixture-backed tests (and their helpers/golden) drive the real C++ cache
// path end-to-end through the Oga generator, which executes the fixture's
// TensorScatter(24) node. TensorScatter has CPU and CUDA kernels but no DirectML
// implementation, so under a USE_DML build (which routes these models onto the
// DML EP) session.run throws "Could not find an implementation for
// TensorScatter(24)". The static-cache Flash feature is CPU/CUDA-targeted, so
// these tests are excluded only on DML builds (the StaticScatterIndexTracker
// unit tests above are pure C++ and run on every EP).
#if !USE_DML
namespace {

// Last-token logits summary from golden_io.npz (CPU MEA, opset 24).
// prefill: input_ids [1,2,3,4] @ write_index 0, nonpad 4 -> argmax(last)=11.
// decode:  input_ids [5]       @ write_index 4, nonpad 5 -> argmax=36.
constexpr int kPrefillArgmax = 11;
constexpr float kPrefillLastTokLogitsSum = -9.80959f;
constexpr int kDecodeArgmax = 36;
constexpr float kDecodeLogitsSum = 78.96353f;  // matches manifest logits_sum.

// updated_key_cache.0 element-sums from golden (proves the in-place scatter
// actually wrote the cache, not just that logits are right).
constexpr float kPrefillUpdatedKeyCache0Sum = 4.31156f;
constexpr float kDecodeUpdatedKeyCache0Sum = -37.31835f;

std::vector<float> ToFloatVector(OgaTensor& tensor) {
  auto shape = tensor.Shape();
  int64_t count = std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
  const float* data = static_cast<const float*>(tensor.Data());
  return std::vector<float>(data, data + count);
}

int ArgMax(const std::vector<float>& values) {
  return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

float Sum(const std::vector<float>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0f);
}

// The slice-A fixture updates its KV cache with an opset-24 TensorScatter node.
// Some ORT packages used in CI (e.g. the DirectML and certain CUDA lanes) don't
// ship a TensorScatter(24) kernel yet, so the static-scatter graph cannot execute
// there and model run throws "Could not find an implementation for TensorScatter".
// Probe the runtime once so the cache tests skip cleanly where the op is absent
// (this path is intentionally not gated on a specific ORT release); lanes whose
// ORT has the kernel still run and assert full parity.
bool StaticScatterRuntimeAvailable() {
  try {
    auto model = OgaModel::Create(MODEL_PATH "static-scatter-bias-decoder");
    auto params = OgaGeneratorParams::Create(*model);
    auto generator = OgaGenerator::Create(*model, *params);
    generator->AppendTokens(std::vector<int32_t>{1});
  } catch (const std::exception& e) {
    if (std::string(e.what()).find("TensorScatter") != std::string::npos) {
      return false;
    }
    throw;  // Unrelated failure: let it surface as a real test error.
  }
  return true;
}

}  // namespace

// M1 fail-loud contract: StaticScatterKeyValueCache::RewindTo MUST throw rather
// than silently no-op. RewindTo cannot reset the write_indices/nonpad_kv_seqlen
// stream (it lives in InputIDs with no rewind hook), so a no-op would leave the
// index tracker stale -> wrong scatter slots + over-reported nonpad => silently
// wrong logits with no error. DecoderOnly_State::RewindTo reaches
// kv_cache_->RewindTo and OgaGenerator::RewindTo is a public API not blocked for
// these models, so the throw is the only guard against silent corruption. The
// happy-path e2e parity test never exercises rewind, so this pins it explicitly.
TEST(StaticScatterKeyValueCache, RewindToThrows) {
  if (!StaticScatterRuntimeAvailable()) {
    GTEST_SKIP() << "ORT runtime lacks the opset-24 TensorScatter kernel; "
                    "the static-scatter KV-cache path is unavailable on this build.";
  }
  auto model = OgaModel::Create(MODEL_PATH "static-scatter-bias-decoder");
  auto params = OgaGeneratorParams::Create(*model);
  auto generator = OgaGenerator::Create(*model, *params);

  const std::vector<int32_t> prompt{1, 2, 3, 4};
  generator->AppendTokens(prompt);

  // Rewinding the static-scatter cache is unsupported and must fail loud.
  EXPECT_THROW(generator->RewindTo(2), std::exception);
}

// End-to-end parity test for the mobius slice-A fixture (#366, bias-aware
// external-KV static-cache decoder). It drives the real
// StaticScatterKeyValueCache C++ path through the public Oga generator API:
// AppendTokens runs the model, DefaultInputIDs feeds write_indices /
// nonpad_kv_seqlen via StaticScatterIndexTracker, and StaticScatterKeyValueCache
// binds the 3D in-place share-buffer KV cache. We force the exact prompt + decode
// token from golden_io.npz and compare the produced logits / updated caches to
// the frozen golden (ORT CPU MEA reference, opset 24).
TEST(StaticScatterKeyValueCache, EndToEndFixtureParity) {
  if (!StaticScatterRuntimeAvailable()) {
    GTEST_SKIP() << "ORT runtime lacks the opset-24 TensorScatter kernel; "
                    "the static-scatter KV-cache path is unavailable on this build.";
  }
  auto model = OgaModel::Create(MODEL_PATH "static-scatter-bias-decoder");
  auto params = OgaGeneratorParams::Create(*model);
  auto generator = OgaGenerator::Create(*model, *params);

  // --- Prefill: force the 4-token golden prompt. StaticScatterIndexTracker
  // must bind write_index=0, nonpad=4; TensorScatter writes rows 0..3. ---
  const std::vector<int32_t> prompt{1, 2, 3, 4};
  generator->AppendTokens(prompt);

  auto prefill_logits = generator->GetLogits();
  auto prefill_logits_vec = ToFloatVector(*prefill_logits);  // last token only: [1,1,256]
  EXPECT_EQ(ArgMax(prefill_logits_vec), kPrefillArgmax);
  EXPECT_NEAR(Sum(prefill_logits_vec), kPrefillLastTokLogitsSum, 1e-2f);

  auto prefill_cache = generator->GetOutput("updated_key_cache.0");
  EXPECT_NEAR(Sum(ToFloatVector(*prefill_cache)), kPrefillUpdatedKeyCache0Sum, 1e-2f);

  // --- Decode: force golden token 5. The tracker advances to write_index=4,
  // nonpad=5; the decode step must read the slot prefill just wrote. ---
  const std::vector<int32_t> decode_token{5};
  generator->AppendTokens(decode_token);

  auto decode_logits = generator->GetLogits();
  auto decode_logits_vec = ToFloatVector(*decode_logits);  // [1,1,256]
  EXPECT_EQ(ArgMax(decode_logits_vec), kDecodeArgmax);
  EXPECT_NEAR(Sum(decode_logits_vec), kDecodeLogitsSum, 1e-2f);

  auto decode_cache = generator->GetOutput("updated_key_cache.0");
  EXPECT_NEAR(Sum(ToFloatVector(*decode_cache)), kDecodeUpdatedKeyCache0Sum, 1e-2f);
}
#endif  // !USE_DML

}  // namespace Generators::test
