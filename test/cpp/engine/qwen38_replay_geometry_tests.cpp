// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// The validation of the geometry of the compact state-update replay (src/models/qwen38_replay_geometry.h). It needs neither a
// package nor a device: the engine calls it with the shapes of the tensors while it builds the replay descriptors.

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "models/qwen38_replay_geometry.h"
#include "qwen38_plane/qwen38_plane_abi.h"

namespace Generators::test {

namespace {

// The message of the error ValidateQwen38ReplayDescriptor throws for `request`; empty when it accepts the request.
std::string RejectionOf(const Qwen38ReplayRequest& request) {
  try {
    ValidateQwen38ReplayDescriptor(request);
  } catch (const std::runtime_error& error) {
    return error.what();
  }
  return {};
}

void ExpectRejected(const Qwen38ReplayRequest& request, const std::vector<std::string>& fragments) {
  const std::string message = RejectionOf(request);
  ASSERT_FALSE(message.empty()) << "the request was accepted";
  EXPECT_EQ(message.rfind("qwen38_graph: the compact replay of " + request.update.name + " into " + request.state.name, 0),
            0u)
      << message;
  for (const std::string& fragment : fragments) {
    EXPECT_NE(message.find(fragment), std::string::npos) << "\"" << fragment << "\" is not in: " << message;
  }
}

// A GatedDeltaNet state of `heads_v` value heads of [width_v, width_k] whose capsule holds, for each of the `capacity`
// transitions, the decays [heads_v], the keys [heads_k, width_k] and the deltas [heads_v, width_v].
Qwen38ReplayRequest GatedDeltaNet(int64_t heads_v = 4, int64_t width_v = 64, int64_t width_k = 128, int64_t heads_k = 2,
                                  int capacity = 7) {
  Qwen38ReplayRequest request;
  request.kind = Qwen38ReplayKind::kGatedDeltaNet;
  request.capacity = capacity;
  request.state = {"past_key_values.3.recurrent_state", {1, heads_v, width_v, width_k}, sizeof(float)};
  request.update = {"state_update.3.recurrent_capsule",
                    {1, capacity * (heads_v + heads_k * width_k + heads_v * width_v)},
                    sizeof(float)};
  return request;
}

Qwen38ReplayRequest Convolution(int64_t channels = 8192, int64_t history = 3, int capacity = 7, size_t element_size = 2) {
  Qwen38ReplayRequest request;
  request.kind = Qwen38ReplayKind::kConvolution;
  request.capacity = capacity;
  request.state = {"past_key_values.5.conv_state", {1, channels, history}, element_size};
  request.update = {"state_update.5.conv_value", {1, capacity, channels}, element_size};
  return request;
}

Qwen38ReplayRequest Indexer(int64_t entries = 7, int64_t width = 128, int capacity = 7, size_t element_size = 2) {
  Qwen38ReplayRequest request;
  request.kind = Qwen38ReplayKind::kIndexer;
  request.capacity = capacity;
  request.state = {"past_key_values.2.index_buffer", {1, entries, width}, element_size};
  request.update = {"state_update.2.indexer_value", {1, capacity, width}, element_size};
  request.lengths = {"past_key_values.2.index_lengths", {1, 2}, sizeof(int32_t)};
  request.compress_ratio = 4;
  return request;
}

}  // namespace

TEST(Qwen38ReplayGeometryTest, GatedDeltaNetGeometryComesFromTheShapes) {
  // 4 + 2 * 128 + 4 * 64 = 516 values per transition, 7 transitions.
  const Qwen38ReplayRequest request = GatedDeltaNet();
  ASSERT_EQ(request.update.shape[1], 3612);
  const Qwen38ReplayGeometry geometry = ValidateQwen38ReplayDescriptor(request);
  EXPECT_EQ(geometry.channel_count, 4u);
  EXPECT_EQ(geometry.state_width, 64u);
  EXPECT_EQ(geometry.key_width, 128u);
  EXPECT_EQ(geometry.key_head_count, 2u);
  EXPECT_EQ(geometry.element_size, 4u);
  EXPECT_EQ(geometry.blocks, 4 * 2) << "4 value heads of 64 rows, " << QWEN38_PLANE_REPLAY_ROWS_PER_BLOCK << " rows a block";

  const Qwen38ReplayGeometry wide = ValidateQwen38ReplayDescriptor(GatedDeltaNet(32, 100, 132, 16, 7));
  EXPECT_EQ(wide.channel_count, 32u);
  EXPECT_EQ(wide.state_width, 100u);
  EXPECT_EQ(wide.key_width, 132u);
  EXPECT_EQ(wide.key_head_count, 16u);
  EXPECT_EQ(wide.blocks, 32 * 4) << "100 rows take 4 blocks of " << QWEN38_PLANE_REPLAY_ROWS_PER_BLOCK;

  EXPECT_EQ(ValidateQwen38ReplayDescriptor(GatedDeltaNet(4, 1, 128, 2, 7)).blocks, 4);
  EXPECT_EQ(ValidateQwen38ReplayDescriptor(GatedDeltaNet(4, QWEN38_PLANE_REPLAY_ROWS_PER_BLOCK, 128, 2, 7)).blocks, 4);
  EXPECT_EQ(ValidateQwen38ReplayDescriptor(GatedDeltaNet(4, QWEN38_PLANE_REPLAY_ROWS_PER_BLOCK + 1, 128, 2, 7)).blocks, 8);
  EXPECT_NO_THROW(ValidateQwen38ReplayDescriptor(GatedDeltaNet(4, 64, 128, 2, 1)));
  EXPECT_NO_THROW(ValidateQwen38ReplayDescriptor(GatedDeltaNet(4, 64, 128, 4, QWEN38_PLANE_MAX_REPLAY_TRANSITIONS)));
}

// Every dimension may be INT32_MAX, so a state can be that many value rows tall. The kernel counts the blocks of a head
// in int, where the usual (width + rows - 1) / rows wraps for the last rows - 1 widths; the host counts in uint64_t.
TEST(Qwen38ReplayGeometryTest, TheBlocksOfAHeadAreCountedWithoutOverflowAtTheWidestState) {
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  constexpr int64_t kRows = QWEN38_PLANE_REPLAY_ROWS_PER_BLOCK;
  for (const int64_t width : {int64_t{1}, kRows - 1, kRows, kRows + 1, 2 * kRows, int64_t{100}, kInt32Max - kRows,
                              kInt32Max - kRows + 1, kInt32Max - 1, kInt32Max}) {
    const int64_t expected = (width + kRows - 1) / kRows;  // in int64_t, which cannot overflow here
    EXPECT_EQ(QWEN38_PLANE_REPLAY_BLOCKS_PER_HEAD(static_cast<int32_t>(width)), expected)
        << "width " << width << " in int";
    EXPECT_EQ(QWEN38_PLANE_REPLAY_BLOCKS_PER_HEAD(static_cast<uint64_t>(width)), static_cast<uint64_t>(expected))
        << "width " << width << " in uint64_t";
  }

  // The geometry the validator accepts for the widest state: a capsule of INT32_MAX values is one transition of
  // 1 decay, 4 key values and 2147483642 deltas. It takes 2^26 blocks, the last holding 26 of its 32 rows.
  const Qwen38ReplayGeometry widest = ValidateQwen38ReplayDescriptor(GatedDeltaNet(1, 2147483642, 4, 1, 1));
  EXPECT_EQ(widest.state_width, 2147483642u);
  EXPECT_EQ(widest.blocks, int{1} << 26);
}

TEST(Qwen38ReplayGeometryTest, GatedDeltaNetRejectsACapsuleThatIsNotWholeTransitions) {
  Qwen38ReplayRequest request = GatedDeltaNet();
  request.update.shape[1] += 1;
  ExpectRejected(request, {"state_update.3.recurrent_capsule [1, 3613]", "not a multiple of the 7 transitions",
                           "past_key_values.3.recurrent_state"});
}

TEST(Qwen38ReplayGeometryTest, GatedDeltaNetRejectsKeyValuesThatAreNotWholeKeys) {
  // 7 * 516 values: 260 per transition for the decays and the deltas leave 256, which 96-wide keys do not divide.
  Qwen38ReplayRequest request = GatedDeltaNet();
  request.state.shape[3] = 96;
  ExpectRejected(request, {"256 key values", "not a whole number of keys of the key width 96"});
}

TEST(Qwen38ReplayGeometryTest, GatedDeltaNetRejectsACapsuleWithoutKeys) {
  Qwen38ReplayRequest request = GatedDeltaNet();
  request.update.shape[1] = 7 * 260;  // exactly the decays and the deltas
  ExpectRejected(request, {"holds 260 values", "no key is left"});
  request.update.shape[1] = 7 * 100;  // fewer than the decays and the deltas: no unsigned wrap-around into a huge key count
  ExpectRejected(request, {"holds 100 values", "no key is left"});
}

TEST(Qwen38ReplayGeometryTest, GatedDeltaNetRejectsKeyHeadsThatDoNotDivideTheValueHeads) {
  ExpectRejected(GatedDeltaNet(4, 64, 128, 3), {"4 value heads", "not a multiple of the 3 key heads"});
  ExpectRejected(GatedDeltaNet(2, 64, 128, 4), {"2 value heads", "not a multiple of the 4 key heads"});
  EXPECT_EQ(ValidateQwen38ReplayDescriptor(GatedDeltaNet(4, 64, 128, 4)).key_head_count, 4u);
  EXPECT_EQ(ValidateQwen38ReplayDescriptor(GatedDeltaNet(4, 64, 128, 1)).key_head_count, 1u);
}

TEST(Qwen38ReplayGeometryTest, GatedDeltaNetRejectsAKeyWidthTheKernelCannotHold) {
  constexpr int64_t kMaxWidth = QWEN38_PLANE_MAX_REPLAY_KEY_WIDTH;
  EXPECT_EQ(ValidateQwen38ReplayDescriptor(GatedDeltaNet(4, 64, kMaxWidth)).key_width, static_cast<uint64_t>(kMaxWidth));
  ExpectRejected(GatedDeltaNet(4, 64, kMaxWidth + 4),
                 {"the key width " + std::to_string(kMaxWidth + 4), "exceeds the " + std::to_string(kMaxWidth)});
  ExpectRejected(GatedDeltaNet(4, 64, 2 * kMaxWidth), {"exceeds the " + std::to_string(kMaxWidth)});

  // The kernel loads float4 along the key width.
  EXPECT_EQ(ValidateQwen38ReplayDescriptor(GatedDeltaNet(4, 64, 4)).key_width, 4u);
  ExpectRejected(GatedDeltaNet(4, 64, 130), {"key width 130", "not a multiple of 4"});
  ExpectRejected(GatedDeltaNet(4, 64, 2), {"key width 2", "not a multiple of 4"});
  ExpectRejected(GatedDeltaNet(4, 64, 3), {"key width 3", "not a multiple of 4"});

  // The key width is the state's: it is rejected before the capsule is related to it.
  Qwen38ReplayRequest inconsistent = GatedDeltaNet(4, 64, kMaxWidth + 4);
  inconsistent.update.shape[1] += 1;
  ExpectRejected(inconsistent, {"exceeds the " + std::to_string(kMaxWidth)});
}

TEST(Qwen38ReplayGeometryTest, GatedDeltaNetRejectsStateThatIsNotFloat32) {
  for (const size_t size : {size_t{0}, size_t{1}, size_t{2}, size_t{8}}) {
    SCOPED_TRACE(size);
    Qwen38ReplayRequest state = GatedDeltaNet();
    state.state.element_size = size;
    ExpectRejected(state, {"past_key_values.3.recurrent_state has " + std::to_string(size) + "-byte elements", "float32"});
    Qwen38ReplayRequest capsule = GatedDeltaNet();
    capsule.update.element_size = size;
    ExpectRejected(capsule, {"state_update.3.recurrent_capsule has " + std::to_string(size) + "-byte elements", "float32"});
  }
}

TEST(Qwen38ReplayGeometryTest, RejectsAShapeTheDescriptorsCannotDescribe) {
  for (const Qwen38ReplayRequest& valid : {GatedDeltaNet(), Convolution(), Indexer()}) {
    SCOPED_TRACE(valid.update.name);
    ASSERT_NO_THROW(ValidateQwen38ReplayDescriptor(valid));

    Qwen38ReplayRequest rank = valid;
    rank.state.shape.push_back(1);
    ExpectRejected(rank, {valid.state.name, "must have rank " + std::to_string(valid.state.shape.size())});
    rank = valid;
    rank.update.shape.pop_back();
    ExpectRejected(rank, {valid.update.name, "must have rank " + std::to_string(valid.update.shape.size())});

    Qwen38ReplayRequest batch = valid;
    batch.state.shape[0] = 2;
    ExpectRejected(batch, {valid.state.name, "batch of 1"});
    batch = valid;
    batch.update.shape[0] = 2;
    ExpectRejected(batch, {valid.update.name, "batch of 1"});

    for (const int64_t bad : {int64_t{0}, int64_t{-1}, int64_t{1} << 31}) {
      Qwen38ReplayRequest dimension = valid;
      dimension.state.shape.back() = bad;
      ExpectRejected(dimension, {valid.state.name, "dimensions in [1, 2147483647]"});
    }
  }
}

TEST(Qwen38ReplayGeometryTest, RejectsACapacityTheKernelsCannotReplay) {
  for (const int capacity : {0, -1, QWEN38_PLANE_MAX_REPLAY_TRANSITIONS + 1}) {
    SCOPED_TRACE(capacity);
    for (Qwen38ReplayRequest request : {GatedDeltaNet(), Convolution(), Indexer()}) {
      request.capacity = capacity;
      ExpectRejected(request, {"the replay capacity is " + std::to_string(capacity), "1 to 8"});
    }
  }
}

TEST(Qwen38ReplayGeometryTest, ConvolutionGeometryIsChannelFirst) {
  const Qwen38ReplayGeometry bf16 = ValidateQwen38ReplayDescriptor(Convolution());
  EXPECT_EQ(bf16.channel_count, 8192u);
  EXPECT_EQ(bf16.state_width, 3u);
  EXPECT_EQ(bf16.element_size, 2u);

  const Qwen38ReplayGeometry fp32 = ValidateQwen38ReplayDescriptor(Convolution(1024, 3, 7, 4));
  EXPECT_EQ(fp32.channel_count, 1024u);
  EXPECT_EQ(fp32.state_width, 3u);
  EXPECT_EQ(fp32.element_size, 4u);

  // A history longer than the transitions kept, and one shorter.
  EXPECT_EQ(ValidateQwen38ReplayDescriptor(Convolution(512, 15, 7)).state_width, 15u);
  EXPECT_EQ(ValidateQwen38ReplayDescriptor(Convolution(512, 1, 1)).state_width, 1u);

  // The per-layer-embedding convolution is replayed alike.
  Qwen38ReplayRequest ple = Convolution(1024, 3, 7);
  ple.state.name = "past_key_values.5.ple_conv_state";
  ple.update.name = "state_update.5.ple_conv_value";
  EXPECT_EQ(ValidateQwen38ReplayDescriptor(ple).channel_count, 1024u);
}

TEST(Qwen38ReplayGeometryTest, ConvolutionRejectsAHistoryFirstState) {
  // [batch, history, channels]: a replay that took it as [batch, channels, history] would shift 3 of its 8192 channels.
  Qwen38ReplayRequest request = Convolution();
  request.state.shape = {1, 3, 8192};
  ExpectRejected(request, {"past_key_values.5.conv_state [1, 3, 8192]", "state_update.5.conv_value [1, 7, 8192]",
                           "8192 channels", "history-first conv state is not supported by the compact replay"});
}

TEST(Qwen38ReplayGeometryTest, ConvolutionTakesASquareStateAsChannelFirst) {
  // With as many channels as history the two layouts cannot be told apart; the exporter writes channel-first.
  const Qwen38ReplayGeometry geometry = ValidateQwen38ReplayDescriptor(Convolution(16, 16, 7));
  EXPECT_EQ(geometry.channel_count, 16u);
  EXPECT_EQ(geometry.state_width, 16u);
}

TEST(Qwen38ReplayGeometryTest, ConvolutionRejectsAStateOfAnotherChannelCount) {
  Qwen38ReplayRequest request = Convolution();
  request.state.shape = {1, 100, 3};
  ExpectRejected(request, {"neither the second nor the third dimension", "past_key_values.5.conv_state [1, 100, 3]",
                           "8192 channels"});
  request.state.shape = {1, 8192, 3};
  request.update.shape = {1, 7, 4096};
  ExpectRejected(request, {"neither the second nor the third dimension", "4096 channels"});
}

TEST(Qwen38ReplayGeometryTest, ConvolutionRejectsElementsTheKernelDoesNotCopy) {
  // k_replay_generic copies 16- and 32-bit elements and does nothing for any other size: an INT8 or FP8 state stays stale.
  for (const size_t size : {size_t{0}, size_t{1}, size_t{3}, size_t{8}}) {
    SCOPED_TRACE(size);
    ExpectRejected(Convolution(8192, 3, 7, size), {std::to_string(size) + "-byte elements", "2- or 4-byte elements"});
  }
  Qwen38ReplayRequest mixed = Convolution();
  mixed.update.element_size = 4;
  ExpectRejected(mixed, {"state_update.5.conv_value has 4-byte elements", "past_key_values.5.conv_state has 2-byte elements"});
}

TEST(Qwen38ReplayGeometryTest, ConvolutionRejectsAnUpdateOfAnotherCapacity) {
  Qwen38ReplayRequest request = Convolution();
  request.update.shape = {1, 5, 8192};
  ExpectRejected(request, {"state_update.5.conv_value [1, 5, 8192]", "carries 5 transitions", "capacity is 7"});
}

TEST(Qwen38ReplayGeometryTest, IndexerGeometryComesFromTheBuffer) {
  const Qwen38ReplayGeometry geometry = ValidateQwen38ReplayDescriptor(Indexer());
  EXPECT_EQ(geometry.state_width, 128u);
  EXPECT_EQ(geometry.aux_capacity, 7u);
  EXPECT_EQ(geometry.element_size, 2u);
  EXPECT_EQ(ValidateQwen38ReplayDescriptor(Indexer(7, 128, 7, 4)).element_size, 4u);
  // A partly filled block holds at most compress_ratio - 1 = 3 entries.
  EXPECT_EQ(ValidateQwen38ReplayDescriptor(Indexer(3)).aux_capacity, 3u);
}

TEST(Qwen38ReplayGeometryTest, IndexerRejectsAnInconsistentBuffer) {
  ExpectRejected(Indexer(2), {"past_key_values.2.index_buffer [1, 2, 128]", "fewer than the 3 entries"});

  Qwen38ReplayRequest request = Indexer();
  request.update.shape = {1, 7, 64};
  ExpectRejected(request, {"state_update.2.indexer_value [1, 7, 64]", "rows of 64 values", "have 128"});
  request = Indexer();
  request.update.shape = {1, 6, 128};
  ExpectRejected(request, {"carries 6 transitions", "capacity is 7"});
  request = Indexer();
  request.compress_ratio = 0;
  ExpectRejected(request, {"compression ratio is 0"});
  for (const size_t size : {size_t{1}, size_t{8}}) {
    ExpectRejected(Indexer(7, 128, 7, size), {std::to_string(size) + "-byte elements", "2- or 4-byte elements"});
  }
}

TEST(Qwen38ReplayGeometryTest, IndexerRejectsLengthsTheKernelCannotRead) {
  Qwen38ReplayRequest request = Indexer();
  request.lengths.element_size = 8;
  ExpectRejected(request, {"past_key_values.2.index_lengths has 8-byte elements", "int32"});
  request = Indexer();
  request.lengths.shape = {1, 1};
  ExpectRejected(request, {"past_key_values.2.index_lengths [1, 1]", "at least two words"});
  request.lengths.shape = {};
  ExpectRejected(request, {"at least two words"});
  request.lengths.shape = {1, 0};
  ExpectRejected(request, {"dimensions in [1, 2147483647]"});
  request.lengths.shape = {2};
  EXPECT_NO_THROW(ValidateQwen38ReplayDescriptor(request));
}

struct FakeTensor {
  int id{};
};

TEST(Qwen38ReplayGeometryTest, MissingTensorsAreNamedWithTheirLayerAndBank) {
  std::map<std::string, std::unique_ptr<FakeTensor>> bank;
  bank["past_key_values.3.recurrent_state"] = std::make_unique<FakeTensor>(FakeTensor{42});
  bank["past_key_values.4.recurrent_state"] = nullptr;

  EXPECT_EQ(RequireQwen38ReplayTensor(bank, "past_key_values.3.recurrent_state", "3", "bank 1").id, 42);
  for (const std::string name : {"past_key_values.3.conv_state", "past_key_values.4.recurrent_state"}) {
    SCOPED_TRACE(name);
    try {
      RequireQwen38ReplayTensor(bank, name, "3", "bank 1");
      FAIL() << "the missing tensor was returned";
    } catch (const std::out_of_range&) {
      FAIL() << "a bare out_of_range does not name the tensor";
    } catch (const std::runtime_error& error) {
      const std::string message = error.what();
      EXPECT_NE(message.find(name), std::string::npos) << message;
      EXPECT_NE(message.find("layer 3"), std::string::npos) << message;
      EXPECT_NE(message.find("bank 1"), std::string::npos) << message;
      EXPECT_EQ(message, DescribeMissingQwen38ReplayTensor(name, "3", "bank 1"));
    }
  }
}

}  // namespace Generators::test
