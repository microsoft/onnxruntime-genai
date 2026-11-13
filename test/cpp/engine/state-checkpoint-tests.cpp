// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "models/state-checkpoint.h"

namespace Generators::test {
namespace {

class StateCheckpointCacheTest : public testing::Test {
 protected:
  void SetUp() override {
    auto* cpu = GetDeviceInterface(DeviceType::CPU);
    for (const auto& shape : std::vector<std::vector<int64_t>>{{1, 2, 24, 2}, {8, 3, 2}, {8, 2, 2, 2}}) {
      auto tensor = std::make_unique<Tensor>(
          cpu, owned_.size() == 2 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT : ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
      tensor->CreateTensor(shape);
      caches_.push_back(tensor.get());
      owned_.push_back(std::move(tensor));
    }
    aux_ = std::make_unique<Tensor>(cpu, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
    aux_->CreateTensor(std::array<int64_t, 2>{24, 4});
    for (auto* tensor : caches_) Fill(*tensor);
    Fill(*aux_);
    description_ = std::make_unique<StateCheckpointDescription>(this);
    for (size_t i = 0; i < caches_.size(); ++i) {
      auto* tensor = caches_[i];
      const StateCheckpointLayout layout{
          i == 0 ? StateCheckpointLayout::Kind::Sequence : StateCheckpointLayout::Kind::SelectedRow,
          i == 0 ? 2u : 0u};
      description_->BindBuffer(std::to_string(i), [this, tensor, layout] {
        return StateCheckpointDescription::Buffer{
            static_cast<uint8_t*>(tensor->GetMutableRawData()), tensor->GetByteSpan().size(),
            tensor->GetShape(), tensor->GetType(), layout, 0, std::nullopt, static_cast<size_t>(row_)};
      });
    }
    description_->BindBuffer("aux", [this] {
      return StateCheckpointDescription::Buffer{
          static_cast<uint8_t*>(aux_->GetMutableRawData()), aux_->GetByteSpan().size(), aux_->GetShape(), aux_->GetType(), {StateCheckpointLayout::Kind::Sequence, 0}, aux_start_};
    });
    description_->BindValue(row_);
    description_->BindValue(anchor_);
  }
  static void Fill(Tensor& tensor) {
    auto bytes = tensor.GetByteSpan().CpuSpan();
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<uint8_t>(i % 251 + 1);
  }
  static std::vector<uint8_t> Bytes(Tensor& tensor) {
    auto bytes = tensor.GetByteSpan().CpuSpan();
    return {bytes.begin(), bytes.end()};
  }
  std::unique_ptr<StateCheckpointCache> Cache(size_t count = 2, size_t budget = 8192) {
    return std::make_unique<StateCheckpointCache>(
        std::make_shared<PrefixSnapshotCache>(std::array<size_t, 3>{0, budget, 0}, count), 8);
  }
  void Capture(StateCheckpointCache& cache, std::span<const int32_t> tokens,
               int32_t row, int32_t anchor, size_t aux_start) {
    row_ = row;
    anchor_ = anchor;
    aux_start_ = aux_start;
    cache.Capture(tokens, *description_, aux_->p_device_);
  }
  void Restore(StateCheckpointCache& cache, const StateCheckpointCache::Entry& entry, size_t aux_start) {
    aux_start_ = aux_start;
    cache.Restore(entry, *description_, aux_->p_device_);
  }
  std::unique_ptr<StateCheckpointDescription> description_;
  int32_t row_{7}, anchor_{};
  size_t aux_start_{};
  std::vector<std::unique_ptr<Tensor>> owned_;
  std::vector<Tensor*> caches_;
  std::unique_ptr<Tensor> aux_;
};

TEST_F(StateCheckpointCacheTest, RestoresBothKvHeadsSelectedMixedStateRowAndAuxiliarySuffix) {
  for (size_t tier : {size_t{1}, size_t{2}}) {
    SCOPED_TRACE(tier);
    std::array<size_t, 3> limits{};
    limits[tier] = 8192;
    auto storage = std::make_shared<PrefixSnapshotCache>(limits, 1);
    StateCheckpointCache cache{storage, 8};
    const std::vector<int32_t> prompt(16, 5);
    std::vector<std::vector<uint8_t>> original;
    for (auto* tensor : caches_) {
      Fill(*tensor);
      original.push_back(Bytes(*tensor));
    }
    Fill(*aux_);
    const auto original_aux = Bytes(*aux_);
    Capture(cache, prompt, 3, 42, 0);
    auto saved = cache.Find(prompt);
    ASSERT_NE(saved, nullptr);
    EXPECT_EQ(static_cast<size_t>(saved->tier), tier);
    row_ = anchor_ = 0;
    for (auto* tensor : caches_) tensor->GetByteSpan().Zero();
    aux_->GetByteSpan().Zero();
    Restore(cache, *saved, 8);
    EXPECT_EQ(row_, 3);
    EXPECT_EQ(anchor_, 42);
    for (size_t i = 0; i < caches_.size(); ++i) {
      std::vector<uint8_t> expected(original[i].size());
      if (i == 0) {
        constexpr size_t head_bytes = 24 * 2 * sizeof(uint16_t);
        constexpr size_t prefix_bytes = 16 * 2 * sizeof(uint16_t);
        for (size_t head = 0; head < 2; ++head)
          std::copy_n(original[i].begin() + head * head_bytes, prefix_bytes,
                      expected.begin() + head * head_bytes);
      } else {
        const size_t row_bytes = original[i].size() / 8;
        std::copy_n(original[i].begin() + 3 * row_bytes, row_bytes, expected.begin() + 3 * row_bytes);
      }
      EXPECT_EQ(Bytes(*caches_[i]), expected) << i;
    }
    const auto restored_aux = Bytes(*aux_);
    EXPECT_TRUE(std::equal(original_aux.begin() + 64, original_aux.begin() + 128, restored_aux.begin()));
    EXPECT_TRUE(std::all_of(restored_aux.begin() + 64, restored_aux.end(), [](auto byte) { return byte == 0; }));
    EXPECT_EQ(cache.Metrics().matched_tokens, 16u);
    storage->Discard(*saved);
    EXPECT_THROW(Restore(cache, *saved, 8), std::runtime_error);
  }
}

TEST_F(StateCheckpointCacheTest, DuplicateCaptureKeepsTheResidentCheckpointWithoutRepublishing) {
  auto cache = Cache(1);
  const std::vector<int32_t> prompt(8, 3);
  Capture(*cache, prompt, 7, 11, 0);
  auto saved = cache->Find(prompt);
  ASSERT_NE(saved, nullptr);
  Capture(*cache, prompt, 7, 42, 0);
  EXPECT_EQ(cache->Find(prompt), saved);
  EXPECT_EQ(cache->Metrics().registered_blocks, 1u);
  EXPECT_EQ(cache->Metrics().publication_refusals, 0u);
  Restore(*cache, *saved, 0);
  EXPECT_EQ(anchor_, 11);
}

TEST_F(StateCheckpointCacheTest, RefusedStorageAndInvalidBoundariesDoNotCreateHits) {
  for (size_t count : {size_t{0}, size_t{1}}) {
    auto cache = Cache(count, count == 0 ? 8192 : 1);
    const std::vector<int32_t> prompt(8, 3);
    Capture(*cache, prompt, 7, 11, 0);
    EXPECT_EQ(cache->Find(prompt), nullptr);
    EXPECT_EQ(cache->Metrics().retention_refusals, 1u);
  }
  auto cache = Cache();
  EXPECT_THROW(Capture(*cache, {}, 7, 11, 0), std::runtime_error);
  EXPECT_THROW(Capture(*cache, std::vector<int32_t>(7, 3), 6, 11, 0), std::runtime_error);
  EXPECT_EQ(cache->Metrics().registered_blocks, 0u);
}

TEST_F(StateCheckpointCacheTest, RefusedPublicationPrunesAndCountsEvictionsImmediately) {
  auto storage = std::make_shared<PrefixSnapshotCache>(std::array<size_t, 3>{0, 400, 0}, 2);
  StateCheckpointCache cache{storage, 8};
  const std::vector<int32_t> prompt(8, 3);
  Capture(cache, prompt, 7, 11, 0);
  std::weak_ptr<StateCheckpointCache::Entry> evicted = cache.Find(prompt);
  ASSERT_FALSE(evicted.expired());
  auto other = std::make_shared<PrefixSnapshotCache::Entry>();
  other->type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  other->shapes = {{128}};
  ASSERT_TRUE(storage->Allocate(other, aux_->p_device_, "other"));
  auto lease = other;
  Capture(cache, std::vector<int32_t>(16, 4), 7, 12, 0);
  EXPECT_EQ(cache.Metrics().retention_refusals, 1u);
  EXPECT_EQ(cache.Metrics().evictions, 1u);
  EXPECT_TRUE(evicted.expired());
  EXPECT_EQ(cache.Find(prompt), nullptr);
}

TEST(StateCheckpointDescriptionTest, RejectsInvalidAndOverflowingRanges) {
  const StateCheckpointLayout sequence{StateCheckpointLayout::Kind::Sequence, 0};
  const StateCheckpointLayout key{StateCheckpointLayout::Kind::Sequence, 1};
  const StateCheckpointLayout overflow{StateCheckpointLayout::Kind::Sequence, 0, 1, 1};
  EXPECT_THROW(sequence.Ranges(std::array<int64_t, 1>{8}, 8, 64, 9), std::out_of_range);
  EXPECT_THROW(sequence.Ranges(std::array<int64_t, 2>{8, 3}, 2, 48, 9), std::out_of_range);
  EXPECT_THROW(key.Ranges(std::array<int64_t, 3>{1, 0, 3}, 2, 0, 5), std::invalid_argument);
  EXPECT_THROW(key.Ranges(std::array<int64_t, 3>{1, 8, 3}, 2, 47, 5), std::invalid_argument);
  EXPECT_THROW(overflow.Ranges(std::array<int64_t, 1>{8}, 2, 16, std::numeric_limits<size_t>::max()),
               std::overflow_error);
  const StateCheckpointLayout selected{StateCheckpointLayout::Kind::SelectedRow, 0};
  EXPECT_THROW(selected.Ranges(std::array<int64_t, 1>{8}, 1, 8, 4, 8), std::out_of_range);
  const StateCheckpointLayout compressed{StateCheckpointLayout::Kind::Sequence, 0, 0};
  EXPECT_THROW(compressed.Ranges(std::array<int64_t, 1>{8}, 1, 8, 4), std::invalid_argument);
  EXPECT_THROW(key.Ranges(std::array<int64_t, 1>{8}, 1, 8, 4), std::invalid_argument);
}

class StateCheckpointRoundTripTest : public testing::TestWithParam<size_t> {};

TEST_P(StateCheckpointRoundTripTest, RestoresPlannedRowsAndAllBanksWithoutTouchingUnusedCapacity) {
  struct Buffer {
    const char* name;
    std::vector<int64_t> shape;
    size_t element_bytes, bytes, length, stride, kept_per_stride;
    StateCheckpointLayout layout{};
  };
  const std::vector<Buffer> buffers{
      {"past_key_values.0.key", {1, 2, 8, 3}, 2, 96, 5, 48, 30, {StateCheckpointLayout::Kind::Sequence, 2}},
      {"past_key_values.0.value", {1, 2, 8, 3}, 2, 96, 5, 48, 30, {StateCheckpointLayout::Kind::Sequence, 2}},
      {"shared.index_key", {1, 2, 4, 3}, 4, 96, 5, 48, 24, {StateCheckpointLayout::Kind::Sequence, 2, 4, 1, true}},
      {"past_position_ids", {1, 8}, 8, 64, 5, 64, 40, {StateCheckpointLayout::Kind::Sequence, 1}},
      {"bank0", {16}, 1, 16, 5, 16, 16},
      {"bank1", {16}, 1, 16, 5, 16, 16},
      {"bank2", {16}, 1, 16, 5, 16, 16},
      {"ctx_commit", {16}, 1, 16, 5, 16, 16},
      {"last_hidden", {1, 3}, 2, 6, 5, 6, 6},
      {"d_prompt_hidden", {8, 3}, 2, 48, 5, 48, 30, {StateCheckpointLayout::Kind::Sequence, 0}},
      {"d_dflash_prompt", {8, 4}, 2, 64, 3, 64, 24, {StateCheckpointLayout::Kind::Sequence, 0}},
      {"dense.state", {2, 3}, 2, 12, 5, 12, 12},
  };
  std::array<size_t, 3> limits{};
  limits[GetParam()] = 4096;
  PrefixSnapshotCache cache{limits, 1, {}, GetParam() == 0};
  auto snapshot = std::make_shared<StateCheckpointDescription::Snapshot>();
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  StateCheckpointDescription source_description{&cache}, destination_description{&cache};
  std::vector<std::unique_ptr<Tensor>> sources, destinations;
  std::pair<size_t, int> source_cursor{5, 1}, destination_cursor{};
  std::vector<int32_t> source_control{7, 9}, destination_control;
  source_description.BindValue(source_cursor);
  source_description.BindValue(source_control);
  destination_description.BindValue(destination_cursor);
  destination_description.BindValue(destination_control);
  for (size_t i = 0; i < buffers.size(); ++i) {
    const auto& buffer = buffers[i];
    const auto type = buffer.element_bytes == 1   ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
                      : buffer.element_bytes == 2 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
                      : buffer.element_bytes == 4 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                                                  : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    auto source = std::make_unique<Tensor>(cpu, type);
    auto destination = std::make_unique<Tensor>(cpu, type);
    source->CreateTensor(buffer.shape);
    destination->CreateTensor(buffer.shape);
    auto* data = static_cast<uint8_t*>(source->GetMutableRawData());
    std::iota(data, data + buffer.bytes,
              static_cast<uint8_t>(i + 1));
    std::fill_n(static_cast<uint8_t*>(destination->GetMutableRawData()), buffer.bytes, uint8_t{231});
    const auto bind = [&](StateCheckpointDescription& description, Tensor* tensor) {
      description.BindBuffer(buffer.name, [tensor, buffer] {
        return StateCheckpointDescription::Buffer{
            static_cast<uint8_t*>(tensor->GetMutableRawData()), buffer.bytes, buffer.shape,
            tensor->GetType(), buffer.layout, 0, buffer.length};
      });
    };
    bind(source_description, source.get());
    bind(destination_description, destination.get());
    sources.push_back(std::move(source));
    destinations.push_back(std::move(destination));
  }
  ASSERT_TRUE(source_description.Capture(snapshot, cache, cpu, 5, "round-trip"));
  EXPECT_EQ(snapshot->bytes, 344u);
  sources.clear();
  source_control.clear();
  source_cursor = {};
  destination_description.Restore(*snapshot, cpu);
  EXPECT_EQ(destination_cursor, (std::pair<size_t, int>{5, 1}));
  EXPECT_EQ(destination_control, (std::vector<int32_t>{7, 9}));
  for (size_t i = 0; i < buffers.size(); ++i) {
    const auto& buffer = buffers[i];
    const auto* data = static_cast<uint8_t*>(destinations[i]->GetMutableRawData());
    std::vector<uint8_t> expected(buffer.bytes, 231);
    for (size_t j = 0; j < buffer.bytes; ++j)
      if (j % buffer.stride < buffer.kept_per_stride) expected[j] = static_cast<uint8_t>(i + j + 1);
    EXPECT_EQ(std::vector<uint8_t>(data, data + buffer.bytes), expected) << buffer.name;
  }
  // The incomplete index group needs its extra row; KV rows clamp at physical capacity.
  const StateCheckpointLayout index{StateCheckpointLayout::Kind::Sequence, 1, 4, 1, true};
  const StateCheckpointLayout key{StateCheckpointLayout::Kind::Sequence, 1, 1, 0, true};
  EXPECT_EQ(index.Ranges(std::array<int64_t, 3>{1, 4, 3}, 4, 48, 8).front().bytes, 36u);
  EXPECT_EQ(key.Ranges(std::array<int64_t, 3>{1, 8, 3}, 2, 48, 20).front().bytes, 48u);
}

INSTANTIATE_TEST_SUITE_P(StorageTiers, StateCheckpointRoundTripTest, testing::Values(0u, 1u, 2u));

TEST(StateCheckpointDescriptionTest, RecreatesOwnedTensorGroupsAndRestoresTypedControlState) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  std::map<std::string, std::unique_ptr<OrtValue>> tensors;
  tensors.emplace("recurrent", OrtValue::CreateTensor(cpu->GetAllocator(), std::array<int64_t, 2>{2, 3},
                                                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
  auto* data = tensors.at("recurrent")->GetTensorMutableData<float>();
  std::iota(data, data + 6, 1.0f);
  PrefixSnapshotCache storage{{0, 4096, 0}, 1};
  StateCheckpointDescription description{&storage};
  description.BindTensors(tensors, cpu);
  std::optional<std::pair<size_t, size_t>> chunk{{4, 3}};
  description.BindValue(chunk);
  auto snapshot = std::make_shared<StateCheckpointDescription::Snapshot>();
  ASSERT_TRUE(description.Capture(snapshot, storage, cpu, 7, "owned"));
  tensors.clear();
  tensors.emplace("stale", OrtValue::CreateTensor(cpu->GetAllocator(), std::array<int64_t, 1>{1},
                                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
  chunk.reset();
  description.Restore(*snapshot, cpu);
  ASSERT_EQ(tensors.size(), 1u);
  ASSERT_EQ(tensors.count("recurrent"), 1u);
  EXPECT_EQ(tensors.at("recurrent")->GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{2, 3}));
  const auto* restored = tensors.at("recurrent")->GetTensorData<float>();
  EXPECT_EQ(std::vector<float>(restored, restored + 6), (std::vector<float>{1, 2, 3, 4, 5, 6}));
  ASSERT_TRUE(chunk);
  EXPECT_EQ(chunk->first, 4u);
  EXPECT_EQ(chunk->second, 3u);
}

TEST(StateCheckpointDescriptionTest, RejectsIncompatibleBindingsBeforeWritingAnyDestination) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  std::array<uint8_t, 8> first{1, 2, 3, 4, 5, 6, 7, 8}, second{8, 7, 6, 5, 4, 3, 2, 1};
  PrefixSnapshotCache storage{{0, 4096, 0}, 1};
  StateCheckpointDescription description{&storage};
  bool bad_shape = false, bad_layout = false;
  size_t first_position = 0;
  description.BindBuffer("first", [&] {
    return StateCheckpointDescription::Buffer{
        first.data(), first.size(), {8}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8};
  });
  description.BindBuffer("second", [&] {
    return StateCheckpointDescription::Buffer{
        second.data(), second.size(), bad_shape ? std::vector<int64_t>{4, 2} : std::vector<int64_t>{8}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {bad_layout ? StateCheckpointLayout::Kind::Sequence : StateCheckpointLayout::Kind::Whole, 0}, first_position};
  });
  int control = 42;
  description.BindValue(control);
  auto snapshot = std::make_shared<StateCheckpointDescription::Snapshot>();
  ASSERT_TRUE(description.Capture(snapshot, storage, cpu, 4, "preflight"));
  first.fill(0);
  second.fill(0);
  control = 0;
  const auto check = [&] {
    EXPECT_THROW(description.Restore(*snapshot, cpu), std::runtime_error);
    EXPECT_EQ(first, (std::array<uint8_t, 8>{}));
    EXPECT_EQ(second, (std::array<uint8_t, 8>{}));
    EXPECT_EQ(control, 0);
  };
  bad_shape = true;
  check();
  bad_shape = false;
  bad_layout = true;
  check();
  bad_layout = false;
  first_position = 1;
  check();
  first_position = 0;
  snapshot->owner = nullptr;
  check();
  snapshot->owner = &storage;
  snapshot->values[0] = int64_t{42};
  check();
}

TEST(StateCheckpointDescriptionTest, EmptySequenceExtentLeavesCapacityUntouched) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  std::array<uint8_t, 8> sequence{};
  uint8_t state = 17;
  size_t rows = 0;
  PrefixSnapshotCache storage{{0, 4096, 0}, 1};
  StateCheckpointDescription description{&storage};
  description.BindBuffer("sequence", [&] {
    return StateCheckpointDescription::Buffer{
        sequence.data(), sequence.size(), {8}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {StateCheckpointLayout::Kind::Sequence, 0}, 0, rows};
  });
  description.BindBuffer("state", [&] {
    return StateCheckpointDescription::Buffer{
        &state, 1, {1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8};
  });
  auto snapshot = std::make_shared<StateCheckpointDescription::Snapshot>();
  ASSERT_TRUE(description.Capture(snapshot, storage, cpu, 4, "empty-extent"));
  EXPECT_EQ(snapshot->bytes, 1u);
  sequence.fill(231);
  state = 0;
  description.Restore(*snapshot, cpu);
  EXPECT_TRUE(std::all_of(sequence.begin(), sequence.end(), [](auto byte) { return byte == 231; }));
  EXPECT_EQ(state, 17);
  rows = 1;
  state = 0;
  EXPECT_THROW(description.Restore(*snapshot, cpu), std::runtime_error);
  EXPECT_EQ(state, 0);
}

TEST(StateCheckpointDescriptionTest, RejectsUnsupportedWindowRebasingBeforeAnyWrite) {
  for (const StateCheckpointLayout layout : {
           StateCheckpointLayout{StateCheckpointLayout::Kind::Sequence, 1},
           StateCheckpointLayout{StateCheckpointLayout::Kind::Sequence, 0, 2},
           StateCheckpointLayout{StateCheckpointLayout::Kind::Sequence, 0, 1, 1},
           StateCheckpointLayout{StateCheckpointLayout::Kind::Sequence, 0, 1, 0, true}}) {
    std::array<uint8_t, 64> bytes{};
    bytes.fill(17);
    size_t first_position = 0;
    int control = 42;
    auto* cpu = GetDeviceInterface(DeviceType::CPU);
    PrefixSnapshotCache storage{{0, 4096, 0}, 1};
    StateCheckpointDescription description{&storage};
    description.BindBuffer("window", [&] {
      return StateCheckpointDescription::Buffer{
          bytes.data(), bytes.size(), {8, 8}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, layout, first_position};
    });
    description.BindValue(control);
    auto snapshot = std::make_shared<StateCheckpointDescription::Snapshot>();
    ASSERT_TRUE(description.Capture(snapshot, storage, cpu, 4, "window"));
    bytes.fill(0);
    control = 0;
    first_position = 1;
    EXPECT_THROW(description.Restore(*snapshot, cpu), std::runtime_error);
    EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [](auto byte) { return byte == 0; }));
    EXPECT_EQ(control, 0);
  }
}

}  // namespace
}  // namespace Generators::test
