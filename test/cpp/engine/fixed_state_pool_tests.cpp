// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "engine/fixed_state_pool.h"
#include "engine/prefix_cache.h"
#include "dflash2_drafter.h"
#include "engine_test_helpers.h"
#include "models/model_state_manifest.h"
#include "models/session_options.h"
#include "models/utils.h"

namespace Generators {
namespace test {
namespace {

const char kRequestStorageA{};
const char kRequestStorageB{};
const char kRequestStorageC{};
const void* const kRequestA = &kRequestStorageA;
const void* const kRequestB = &kRequestStorageB;
const void* const kRequestC = &kRequestStorageC;

using Request = FixedStateReservationRequest;

template <typename T>
auto ReplayBytes(DeviceSpan<T> view) {
  using Byte = std::conditional_t<std::is_const_v<T>, const uint8_t, uint8_t>;
  return DeviceSpan<Byte>{view.GetBuffer().shared_from_this()}
      .subspan(view.ByteOffset(), view.size() * sizeof(T));
}

template <typename T>
auto HostReplayBytes(T& values) {
  auto view = std::span{values};
  return ReplayBytes(GetDeviceInterface(DeviceType::CPU)->WrapMemory<typename decltype(view)::element_type>(view));
}

class FixedStateTestDevice : public DeviceInterface {
 public:
  explicit FixedStateTestDevice(
      DeviceInterface& inner, DeviceType type = DeviceType::CPU,
      bool supports_transactional_fixed_state = true,
      bool supports_offset_tensor_views = false,
      bool supports_compact_state_replay = true)
      : allow_offset_views{supports_offset_tensor_views},
        inner_{inner},
        type_{type},
        supports_transactional_fixed_state_{supports_transactional_fixed_state},
        supports_compact_state_replay_{supports_compact_state_replay} {}

  DeviceType GetType() const override { return type_; }
  void InitOrt(const OrtApi& api, Ort::Allocator& allocator) override {
    inner_.InitOrt(api, allocator);
  }
  Ort::Allocator& GetAllocator() override { return inner_.GetAllocator(); }
  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override {
    return inner_.GetMemoryInfo();
  }
  std::string GetExecutionProviderName() const override {
    return inner_.GetExecutionProviderName();
  }
  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return inner_.AllocateBase(size);
  }
  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* memory, size_t size) override {
    return inner_.WrapMemoryBase(memory, size);
  }
  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override {
    return inner_.CreateGreedy(params);
  }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override {
    return inner_.CreateBeam(params);
  }
  std::unique_ptr<KeyValueCache> CreateKeyValueCache(State& state) override {
    return inner_.CreateKeyValueCache(state);
  }
  void Synchronize() override { inner_.Synchronize(); }
  bool SupportsOffsetTensorViews() const override { return allow_offset_views; }
  void ReplayStateUpdates(const StateUpdateReplayDesc* descs, size_t count) override {
    if (invalidate_second_gdn) {
      for (size_t index = 0; index < count; ++index) {
        auto descriptor = descs[index];
        const bool recurrent = descriptor.kind == StateUpdateReplayKind::GatedDeltaNet;
        if (recurrent && ++gdn_replay_count == 2) {
          // Deterministic ORT run failure, after the first real GPU replay has completed.
          descriptor.kept_count = 0;
        }
        inner_.ReplayStateUpdates(&descriptor, 1);
        if (recurrent && gdn_replay_count == 1) {
          replayed_destination = descriptor.destination_state;
        }
      }
      return;
    }
    if (fail_replay && count > 0) {
      inner_.ReplayStateUpdates(descs, 1);
      inner_.Synchronize();
      replayed_destination = descs[0].destination_state;
      throw std::runtime_error("Injected failure after first replay write");
    }
    inner_.ReplayStateUpdates(descs, count);
  }
  bool allow_offset_views{false};
  bool fail_replay{false};
  bool invalidate_second_gdn{false};
  size_t gdn_replay_count{};
  DeviceSpan<uint8_t> replayed_destination;
  bool SupportsTransactionalFixedState() const override {
    return supports_transactional_fixed_state_;
  }
  bool SupportsCompactStateReplay() const override {
    return supports_compact_state_replay_;
  }

 private:
  DeviceInterface& inner_;
  DeviceType type_;
  bool supports_transactional_fixed_state_;
  bool supports_compact_state_replay_;
};

struct ReplayTransferBuffer final : DeviceBuffer {
  explicit ReplayTransferBuffer(std::shared_ptr<DeviceBuffer> buffer)
      : inner{std::move(buffer)} {
    p_device_ = inner->p_device_;
    size_in_bytes_ = inner->size_in_bytes_;
  }

  const char* GetType() const override { return inner->GetType(); }
  void AllocateCpu() override {
    const bool allocates = !inner->p_cpu_;
    inner->AllocateCpu();
    p_cpu_ = inner->p_cpu_;
    if (allocates) allocated_bytes += size_in_bytes_;
  }
  void CopyDeviceToCpu() override {
    AllocateCpu();
    inner->CopyDeviceToCpu();
    ++downloads;
  }
  void CopyCpuToDevice() override {
    inner->CopyCpuToDevice();
    ++uploads;
  }
  void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t bytes) override {
    CopyThroughCpu(*this, begin_dest, source, begin_source, bytes);
  }
  void Zero() override { inner->Zero(); }

  std::shared_ptr<DeviceBuffer> inner;
  size_t downloads{}, uploads{}, allocated_bytes{};
};

class ReplayCountingDevice final : public FixedStateTestDevice {
 public:
  explicit ReplayCountingDevice(DeviceInterface& inner)
      : FixedStateTestDevice{inner, inner.GetType()}, inner_{inner} {}

  void ReplayStateUpdates(const StateUpdateReplayDesc* descriptors, size_t count) override {
    buffers.clear();
    std::unordered_map<DeviceBuffer*, std::shared_ptr<ReplayTransferBuffer>> owners;
    const auto instrument = [&](DeviceSpan<const uint8_t> view) {
      if (view.empty()) return DeviceSpan<uint8_t>{};
      auto& buffer = view.GetBuffer();
      auto& owner = owners[&buffer];
      if (!owner) {
        owner = std::make_shared<ReplayTransferBuffer>(buffer.shared_from_this());
        buffers.push_back(owner);
      }
      return DeviceSpan<uint8_t>{std::shared_ptr<DeviceBuffer>{owner}}
          .subspan(view.ByteOffset(), view.size());
    };
    std::vector<StateUpdateReplayDesc> counted(descriptors, descriptors + count);
    for (auto& descriptor : counted) {
      // Recurrent replay binds real WebGPU owners directly; instrument only the CPU fallback.
      if (descriptor.kind != StateUpdateReplayKind::CausalConv) continue;
      descriptor.source_state = instrument(descriptor.source_state);
      descriptor.destination_state = instrument(descriptor.destination_state);
      descriptor.value = instrument(descriptor.value);
      descriptor.decay = instrument(descriptor.decay);
      descriptor.key = instrument(descriptor.key);
      descriptor.delta = instrument(descriptor.delta);
      descriptor.source_aux_state = instrument(descriptor.source_aux_state);
      descriptor.destination_aux_state = instrument(descriptor.destination_aux_state);
      descriptor.source_lengths = instrument(descriptor.source_lengths);
      descriptor.destination_lengths = instrument(descriptor.destination_lengths);
    }
    inner_.ReplayStateUpdates(counted.data(), counted.size());
  }

  std::vector<std::shared_ptr<ReplayTransferBuffer>> buffers;

 private:
  DeviceInterface& inner_;
};

class ScopedKeyValueCacheDevice {
 public:
  ScopedKeyValueCacheDevice(Model& model, DeviceInterface& device)
      : model_{model}, original_{model.p_device_kvcache_} {
    model_.p_device_kvcache_ = &device;
  }
  ScopedKeyValueCacheDevice(const ScopedKeyValueCacheDevice&) = delete;
  ScopedKeyValueCacheDevice& operator=(const ScopedKeyValueCacheDevice&) = delete;
  ~ScopedKeyValueCacheDevice() { model_.p_device_kvcache_ = original_; }

 private:
  Model& model_;
  DeviceInterface* original_;
};

// Builds a one-request reservation input in scheduled row order.
std::array<Request, 1> One(const void* id, uint64_t target_tokens = 1,
                           size_t capture_count = 0) {
  return {Request{id, target_tokens, capture_count}};
}

size_t RowElements(const OrtValue& tensor) {
  const auto shape = tensor.GetTensorTypeAndShapeInfo()->GetShape();
  size_t row_elements = 1;
  for (size_t axis = 1; axis < shape.size(); ++axis) {
    row_elements *= static_cast<size_t>(shape[axis]);
  }
  return row_elements;
}

void FillStagedRow(const FixedStateBinding& binding, size_t row, float value) {
  const auto row_elements = RowElements(*binding.output);
  auto* data = binding.output->GetTensorMutableData<float>();
  std::fill_n(data + row * row_elements, row_elements, value);
}

void FillStagedRows(FixedStateReservation& reservation, size_t row, float value) {
  for (const auto& binding : reservation.Bindings()) {
    FillStagedRow(binding, row, value);
  }
}

void ExpectInputRow(const FixedStateBinding& binding, size_t row, float expected) {
  const auto row_elements = RowElements(*binding.input);
  const auto* data = binding.input->GetTensorData<float>();
  for (size_t index = 0; index < row_elements; ++index) {
    EXPECT_FLOAT_EQ(data[row * row_elements + index], expected);
  }
}

void ExpectInputRow(const FixedStateBinding& binding, size_t row,
                    std::span<const float> expected) {
  ASSERT_EQ(RowElements(*binding.input), expected.size());
  const auto* data = binding.input->GetTensorData<float>() + row * expected.size();
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_FLOAT_EQ(data[index], expected[index]);
  }
}

void FillHalfTensor(OrtValue& tensor, float value) {
  std::fill_n(tensor.GetTensorMutableData<Ort::Float16_t>(), RowElements(tensor),
              Ort::Float16_t{FastFloat32ToFloat16(value)});
}

void ExpectHalfInputRow(const FixedStateBinding& binding,
                        std::span<const float> expected) {
  ASSERT_EQ(RowElements(*binding.input), expected.size());
  const auto* data = binding.input->GetTensorData<Ort::Float16_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_FLOAT_EQ(ToFloat32(data[index]), expected[index]);
  }
}

void FillConvUpdates(const FixedStateBinding& binding, size_t row,
                     std::span<const float> values) {
  ASSERT_NE(binding.state_update_value, nullptr);
  const size_t row_elements = RowElements(*binding.state_update_value);
  ASSERT_EQ(row_elements, values.size());
  std::copy(values.begin(), values.end(),
            binding.state_update_value->GetTensorMutableData<float>() + row * row_elements);
}

void FillGdnUpdates(const FixedStateBinding& binding, size_t row,
                    std::span<const float> decay,
                    std::span<const float> key,
                    std::span<const float> delta) {
  ASSERT_NE(binding.state_update_capsule, nullptr);
  const size_t row_elements = RowElements(*binding.state_update_capsule);
  ASSERT_EQ(row_elements, decay.size() + key.size() + delta.size());
  auto* destination = binding.state_update_capsule->GetTensorMutableData<float>() +
                      row * row_elements;
  destination = std::copy(decay.begin(), decay.end(), destination);
  destination = std::copy(key.begin(), key.end(), destination);
  std::copy(delta.begin(), delta.end(), destination);
}

void ExpectInputRows(const FixedStateReservation& reservation, size_t row, float expected) {
  for (const auto& binding : reservation.Bindings()) {
    ExpectInputRow(binding, row, expected);
  }
}

// ONNX element type shorthands for the direct geometry tests.
constexpr auto kFloat = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
constexpr auto kDouble = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;

class FixedStatePoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_ = LoadSyntheticHybridModel();
  }

  std::unique_ptr<FixedStatePool> MakePool(size_t capacity = 4) {
    return std::make_unique<FixedStatePool>(model_, capacity);
  }

  // Admits a fresh request and commits it with every state row filled with `value`, returning the
  // now-committed slot handle. This is how the tests create resident state without a standalone
  // Allocate surface.
  FixedStateSlotHandle MakeResident(FixedStatePool& pool, const void* request_id, float value,
                                    uint64_t target_tokens = 1) {
    auto requests = One(request_id, target_tokens);
    auto reservation = pool.Reserve(requests);
    FillStagedRows(reservation, 0, value);
    reservation.Commit();
    return pool.HandleFor(request_id);
  }

  std::shared_ptr<Model> model_;
};

TEST_F(FixedStatePoolTest, UsesManifestBindingOrderAndSessionGeometry) {
  auto pool = MakePool();
  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);

  ASSERT_EQ(reservation.Bindings().size(), 4u);
  using Kind = Config::Model::Decoder::StateGroupKind;
  EXPECT_EQ(reservation.Bindings()[0].kind, Kind::FixedConv);
  EXPECT_EQ(reservation.Bindings()[1].kind, Kind::FixedConv);
  EXPECT_EQ(reservation.Bindings()[2].kind, Kind::FixedRecurrent);
  EXPECT_EQ(reservation.Bindings()[3].kind, Kind::FixedRecurrent);
  // Convolution group first, then recurrent group, each in layer order.
  EXPECT_EQ(reservation.Bindings()[0].layer_id, 0);
  EXPECT_STREQ(reservation.Bindings()[0].input_name, "past_conv.0");
  EXPECT_STREQ(reservation.Bindings()[0].output_name, "present_conv.0");
  EXPECT_EQ(reservation.Bindings()[1].layer_id, 3);
  EXPECT_STREQ(reservation.Bindings()[2].input_name, "past_recurrent.2");
  EXPECT_STREQ(reservation.Bindings()[3].output_name, "present_recurrent.5");

  for (const auto& binding : reservation.Bindings()) {
    EXPECT_NE(binding.input, binding.output);
    EXPECT_NE(binding.input->GetTensorMutableData<float>(),
              binding.output->GetTensorMutableData<float>());
  }

  EXPECT_EQ(reservation.Bindings()[0].input->GetTensorTypeAndShapeInfo()->GetShape(),
            (std::vector<int64_t>{1, 2, 3}));
  EXPECT_EQ(reservation.Bindings()[2].output->GetTensorTypeAndShapeInfo()->GetShape(),
            (std::vector<int64_t>{1, 2, 2, 2}));
  EXPECT_EQ(reservation.Handles()[0].request_id, kRequestA);
  ASSERT_EQ(reservation.TargetTokens().size(), 1u);
  EXPECT_EQ(reservation.TargetTokens()[0], 1u);
}

TEST_F(FixedStatePoolTest, StateBankBytesPricesPersistentMtpScratchWithoutAllocating) {
  EXPECT_EQ(FixedStatePool::StateBankBytes(*model_, 1), 112u);
  EXPECT_EQ(FixedStatePool::StateBankBytes(*model_, 4), 448u);
  EXPECT_EQ(FixedStatePool::StateBankBytes(*model_, 0), 0u);
  EXPECT_THROW(FixedStatePool::StateBankBytes(*model_, std::numeric_limits<size_t>::max()),
               std::runtime_error);
}

TEST_F(FixedStatePoolTest, ReusesPreallocatedStagingAcrossReservationBatchSizes) {
  auto pool = MakePool(2);
  std::vector<void*> input_addresses;
  std::vector<void*> output_addresses;
  std::vector<void*> update_addresses;
  void* capture_count_address{};
  void* active_address{};
  {
    auto first_requests = One(kRequestA, /*target_tokens=*/4, /*capture_count=*/3);
    auto first = pool->Reserve(first_requests);
    for (const auto& binding : first.Bindings()) {
      input_addresses.push_back(
          binding.input->GetTensorMutableData<void>());
      output_addresses.push_back(
          binding.output->GetTensorMutableData<void>());
      OrtValue* update = binding.state_update_value
                             ? binding.state_update_value
                             : binding.state_update_capsule;
      ASSERT_NE(update, nullptr);
      update_addresses.push_back(update->GetTensorMutableData<void>());
    }
    capture_count_address = first.Bindings()[0].state_update_capture_count->GetTensorMutableData<void>();
    active_address = first.Bindings()[0].state_update_active->GetTensorMutableData<void>();
    first.Discard();
  }

  const std::array<Request, 2> second_requests{
      Request{kRequestB, 4, 3}, Request{kRequestC, 3, 2}};
  auto second = pool->Reserve(second_requests);
  ASSERT_EQ(second.Bindings().size(), input_addresses.size());
  for (size_t index = 0; index < second.Bindings().size(); ++index) {
    EXPECT_EQ(second.Bindings()[index].input->GetTensorMutableData<void>(),
              input_addresses[index]);
    EXPECT_EQ(second.Bindings()[index].output->GetTensorMutableData<void>(),
              output_addresses[index]);
    OrtValue* update = second.Bindings()[index].state_update_value
                           ? second.Bindings()[index].state_update_value
                           : second.Bindings()[index].state_update_capsule;
    ASSERT_NE(update, nullptr);
    EXPECT_EQ(update->GetTensorMutableData<void>(), update_addresses[index]);
  }
  EXPECT_EQ(second.Bindings()[0].state_update_capture_count->GetTensorMutableData<void>(),
            capture_count_address);
  EXPECT_EQ(second.Bindings()[0].state_update_active->GetTensorMutableData<void>(),
            active_address);
  EXPECT_EQ(second.Bindings()[0].state_update_value->GetTensorTypeAndShapeInfo()->GetShape(),
            (std::vector<int64_t>{2, 3, 2}));
  EXPECT_EQ(second.Bindings()[2].state_update_capsule->GetTensorTypeAndShapeInfo()->GetShape(),
            (std::vector<int64_t>{2, 24}));
}

TEST_F(FixedStatePoolTest, FreshRowsGatherZeroAndCommitPublishes) {
  auto pool = MakePool(1);
  {
    auto requests = One(kRequestA);
    auto reservation = pool->Reserve(requests);
    for (const auto& binding : reservation.Bindings()) {
      ExpectInputRow(binding, 0, 0.0f);  // Fresh admission gathers the reusable zero row.
    }
    FillStagedRows(reservation, 0, 7.0f);
    reservation.Commit();
  }

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  ExpectInputRows(reservation, 0, 7.0f);  // Committed state is now gathered for the resident request.
}

TEST(FixedStatePoolComponentsTest, InitializesAndCommitsPleAndIndexerState) {
  auto model = LoadSyntheticFixedComponentsModel();
  FixedStateTestDevice cuda_device{*model->p_device_kvcache_, DeviceType::CUDA,
                                   /*supports_transactional_fixed_state=*/true,
                                   /*supports_offset_tensor_views=*/true};
  ScopedKeyValueCacheDevice scoped_device{*model, cuda_device};
  FixedStatePool pool{model, 1};

  auto requests = One(kRequestA);
  {
    auto reservation = pool.Reserve(requests);
    ASSERT_EQ(reservation.Bindings().size(), 5u);
    using Kind = Config::Model::Decoder::StateGroupKind;
    EXPECT_EQ(reservation.Bindings()[0].kind, Kind::FixedPle);
    EXPECT_STREQ(reservation.Bindings()[0].input_name, "past.0.ple_tokens");
    EXPECT_STREQ(reservation.Bindings()[1].input_name, "past.0.ple_conv");
    EXPECT_EQ(reservation.Bindings()[2].kind, Kind::FixedIndexer);
    EXPECT_STREQ(reservation.Bindings()[2].input_name, "past.1.indexer_key");
    EXPECT_STREQ(reservation.Bindings()[3].input_name, "past.1.indexer_kv_buffer");
    EXPECT_STREQ(reservation.Bindings()[4].input_name, "past.1.indexer_state_lengths");
    EXPECT_EQ(reservation.Bindings()[2].input->GetTensorMutableData<void>(),
              reservation.Bindings()[2].output->GetTensorMutableData<void>());
    EXPECT_NE(reservation.Bindings()[3].input->GetTensorMutableData<void>(),
              reservation.Bindings()[3].output->GetTensorMutableData<void>());
    EXPECT_NE(reservation.Bindings()[4].input->GetTensorMutableData<void>(),
              reservation.Bindings()[4].output->GetTensorMutableData<void>());

    const auto& token_binding = reservation.Bindings()[0];
    const auto* token_input = token_binding.input->GetTensorData<int64_t>();
    EXPECT_EQ(token_input[0], 7);
    EXPECT_EQ(token_input[1], 7);
    std::fill_n(token_binding.output->GetTensorMutableData<int64_t>(), 2, 11);

    const auto& conv_binding = reservation.Bindings()[1];
    const auto* conv_input = conv_binding.input->GetTensorData<Ort::Float16_t>();
    EXPECT_TRUE(std::all_of(conv_input, conv_input + RowElements(*conv_binding.input),
                            [](Ort::Float16_t value) { return ToFloat32(value) == 0.0f; }));
    FillHalfTensor(*conv_binding.output, 2.0f);
    for (size_t index = 2; index < 4; ++index) {
      const auto& binding = reservation.Bindings()[index];
      const auto elements = RowElements(*binding.input);
      const auto* input = binding.input->GetTensorData<float>();
      EXPECT_TRUE(std::all_of(input, input + elements, [](float value) { return value == 0.0f; }));
      std::fill_n(binding.output->GetTensorMutableData<float>(), elements, 2.0f);
    }
    const auto& lengths_binding = reservation.Bindings()[4];
    const auto* lengths_input = lengths_binding.input->GetTensorData<int32_t>();
    EXPECT_EQ(lengths_input[0], 0);
    EXPECT_EQ(lengths_input[1], 0);
    std::fill_n(lengths_binding.output->GetTensorMutableData<int32_t>(), 2, 3);
    reservation.Commit();
  }

  auto resident = pool.Reserve(requests);
  EXPECT_EQ(resident.Bindings()[2].input->GetTensorMutableData<void>(),
            resident.Bindings()[2].output->GetTensorMutableData<void>());
  EXPECT_NE(resident.Bindings()[3].input->GetTensorMutableData<void>(),
            resident.Bindings()[3].output->GetTensorMutableData<void>());
  EXPECT_NE(resident.Bindings()[4].input->GetTensorMutableData<void>(),
            resident.Bindings()[4].output->GetTensorMutableData<void>());
  const auto* token_input = resident.Bindings()[0].input->GetTensorData<int64_t>();
  EXPECT_EQ(token_input[0], 11);
  EXPECT_EQ(token_input[1], 11);
  ExpectHalfInputRow(resident.Bindings()[1], std::array<float, 12>{
                                                 2, 2, 2, 2, 2, 2,
                                                 2, 2, 2, 2, 2, 2});
  for (size_t index = 2; index < 4; ++index) {
    const auto& binding = resident.Bindings()[index];
    const auto elements = RowElements(*binding.input);
    const auto* input = binding.input->GetTensorData<float>();
    EXPECT_TRUE(std::all_of(input, input + elements, [](float value) { return value == 2.0f; }));
  }
  const auto* lengths_input = resident.Bindings()[4].input->GetTensorData<int32_t>();
  EXPECT_EQ(lengths_input[0], 3);
  EXPECT_EQ(lengths_input[1], 3);
}

TEST(FixedStatePoolComponentsTest, KeepsSeparateIndexerKeyBuffersWhenSharingIsDisabled) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-fixed-components");
  config->search.past_present_share_buffer = false;
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  FixedStateTestDevice cuda_device{*model->p_device_kvcache_, DeviceType::CUDA,
                                   /*supports_transactional_fixed_state=*/true,
                                   /*supports_offset_tensor_views=*/true};
  ScopedKeyValueCacheDevice scoped_device{*model, cuda_device};
  FixedStatePool pool{model, 1};

  auto reservation = pool.Reserve(One(kRequestA));
  ASSERT_EQ(reservation.Bindings().size(), 5u);
  EXPECT_NE(reservation.Bindings()[2].input->GetTensorMutableData<void>(),
            reservation.Bindings()[2].output->GetTensorMutableData<void>());
}

TEST(FixedStatePoolComponentsTest, ReplaysPleAndIndexerUpdatesAcrossBlockBoundaryFromZero) {
  auto model = LoadSyntheticFixedComponentsModel();
  FixedStatePool pool{model, 1};
  EXPECT_TRUE(pool.SupportsStateUpdates());
  EXPECT_EQ(pool.StateUpdateCapacity(), 7u);

  {
    auto reservation = pool.Reserve(One(kRequestA, 8, 7));
    auto bindings = reservation.Bindings();
    ASSERT_EQ(bindings.size(), 5u);
    auto* token_updates = bindings[0].state_update_value->GetTensorMutableData<int64_t>();
    for (size_t token = 0; token < 7; ++token) {
      token_updates[token * 2] = static_cast<int64_t>(100 + token * 2);
      token_updates[token * 2 + 1] = static_cast<int64_t>(101 + token * 2);
    }
    auto* conv_updates = bindings[1].state_update_value->GetTensorMutableData<Ort::Float16_t>();
    auto* indexer_updates = bindings[2].state_update_value->GetTensorMutableData<float>();
    for (size_t token = 0; token < 7; ++token) {
      for (size_t channel = 0; channel < 4; ++channel) {
        conv_updates[token * 4 + channel] =
            Ort::Float16_t{FastFloat32ToFloat16(static_cast<float>(token * 10 + channel))};
      }
      indexer_updates[token * 2] = static_cast<float>(token * 10);
      indexer_updates[token * 2 + 1] = static_cast<float>(token * 10 + 1);
    }
    reservation.CommitPrefix(0, 8, 5);
    reservation.Commit();
  }

  auto resident = pool.Reserve(One(kRequestA, 5));
  EXPECT_EQ(resident.Bindings()[0].input->GetTensorData<int64_t>()[0], 108);
  EXPECT_EQ(resident.Bindings()[0].input->GetTensorData<int64_t>()[1], 109);
  const std::array<float, 12> expected_conv{
      20, 30, 40, 21, 31, 41, 22, 32, 42, 23, 33, 43};
  ExpectHalfInputRow(resident.Bindings()[1], expected_conv);
  const auto* key = resident.Bindings()[2].input->GetTensorData<float>();
  EXPECT_FLOAT_EQ(key[0], 30);
  EXPECT_FLOAT_EQ(key[1], 31);
  const auto* buffer = resident.Bindings()[3].input->GetTensorData<float>();
  EXPECT_FLOAT_EQ(buffer[0], 40);
  EXPECT_FLOAT_EQ(buffer[1], 41);
  const auto* lengths = resident.Bindings()[4].input->GetTensorData<int32_t>();
  EXPECT_EQ(lengths[0], 1);
  EXPECT_EQ(lengths[1], 1);
}

TEST(FixedStatePoolComponentsTest, IndexerRowsShareReplayBackingOwners) {
  struct ReplayDevice final : FixedStateTestDevice {
    using FixedStateTestDevice::FixedStateTestDevice;
    void ReplayStateUpdates(const StateUpdateReplayDesc* descriptors, size_t count) override {
      replayed.assign(descriptors, descriptors + count);
      FixedStateTestDevice::ReplayStateUpdates(descriptors, count);
    }
    std::vector<StateUpdateReplayDesc> replayed;
  };
  auto model = LoadSyntheticFixedComponentsModel();
  ReplayDevice device{*model->p_device_kvcache_};
  ScopedKeyValueCacheDevice scoped_device{*model, device};
  FixedStatePool pool{model, 2};
  {
    const std::array<Request, 2> requests{Request{kRequestA, 8, 7}, Request{kRequestB, 8, 7}};
    auto reservation = pool.Reserve(requests);
    auto bindings = reservation.Bindings();
    std::fill_n(bindings[0].state_update_value->GetTensorMutableData<int64_t>(), 28, 1);
    FillHalfTensor(*bindings[1].state_update_value, 2.0f);
    std::fill_n(bindings[2].state_update_value->GetTensorMutableData<float>(), 28, 3.0f);
    reservation.CommitPrefix(0, 8, 5);
    reservation.CommitPrefix(1, 8, 3);
    reservation.Commit();
  }
  auto resident = pool.Reserve(One(kRequestA, 5));
  std::vector<const StateUpdateReplayDesc*> indexers;
  for (const auto& descriptor : device.replayed) {
    if (descriptor.kind == StateUpdateReplayKind::Indexer) indexers.push_back(&descriptor);
  }
  ASSERT_EQ(indexers.size(), 2u);
  EXPECT_EQ(&indexers[0]->source_aux_state.GetBuffer(), &indexers[1]->source_aux_state.GetBuffer());
  EXPECT_EQ(&indexers[0]->destination_aux_state.GetBuffer(), &indexers[1]->destination_aux_state.GetBuffer());
  EXPECT_EQ(&indexers[0]->source_lengths.GetBuffer(), &indexers[1]->source_lengths.GetBuffer());
  EXPECT_EQ(&indexers[0]->destination_lengths.GetBuffer(), &indexers[1]->destination_lengths.GetBuffer());
  EXPECT_NE(indexers[0]->destination_lengths.ByteOffset(), indexers[1]->destination_lengths.ByteOffset());
}

TEST(FixedStatePoolComponentsTest, ReplaysIndexerUpdateAcrossBlockBoundaryFromThree) {
  auto model = LoadSyntheticFixedComponentsModel();
  FixedStatePool pool{model, 1};
  {
    auto initial = pool.Reserve(One(kRequestA, 3));
    std::fill_n(initial.Bindings()[0].output->GetTensorMutableData<int64_t>(), 2, 7);
    FillHalfTensor(*initial.Bindings()[1].output, 0.0f);
    std::fill_n(initial.Bindings()[2].output->GetTensorMutableData<float>(), 16, 9.0f);
    auto* buffer = initial.Bindings()[3].output->GetTensorMutableData<float>();
    std::copy_n(std::array<float, 6>{1, 2, 3, 4, 5, 6}.begin(), 6, buffer);
    auto* lengths = initial.Bindings()[4].output->GetTensorMutableData<int32_t>();
    lengths[0] = 2;
    lengths[1] = 3;
    initial.Commit();
  }
  {
    auto reservation = pool.Reserve(One(kRequestA, 11, 7));
    std::fill_n(reservation.Bindings()[0].state_update_value->GetTensorMutableData<int64_t>(), 14, 8);
    FillHalfTensor(*reservation.Bindings()[1].state_update_value, 0.0f);
    auto* updates = reservation.Bindings()[2].state_update_value->GetTensorMutableData<float>();
    for (size_t token = 0; token < 7; ++token) {
      updates[token * 2] = static_cast<float>(100 + token * 10);
      updates[token * 2 + 1] = static_cast<float>(101 + token * 10);
    }
    reservation.CommitPrefix(0, 8, 3);
    reservation.Commit();
  }

  auto resident = pool.Reserve(One(kRequestA, 6));
  const auto* key = resident.Bindings()[2].input->GetTensorData<float>();
  EXPECT_FLOAT_EQ(key[4], 100);
  EXPECT_FLOAT_EQ(key[5], 101);
  const auto* buffer = resident.Bindings()[3].input->GetTensorData<float>();
  EXPECT_FLOAT_EQ(buffer[0], 110);
  EXPECT_FLOAT_EQ(buffer[1], 111);
  EXPECT_FLOAT_EQ(buffer[2], 120);
  EXPECT_FLOAT_EQ(buffer[3], 121);
  const auto* lengths = resident.Bindings()[4].input->GetTensorData<int32_t>();
  EXPECT_EQ(lengths[0], 3);
  EXPECT_EQ(lengths[1], 2);
}

TEST(FixedStatePoolComponentsTest, RejectsInvalidIndexerReplayLengthsBeforePublication) {
  const std::array<std::array<int32_t, 2>, 6> invalid_lengths{
      {{-1, 0}, {0, -1}, {9, 0}, {0, 4}, {8, 0}, {7, 3}}};
  for (const auto& lengths : invalid_lengths) {
    SCOPED_TRACE(::testing::Message() << "keys " << lengths[0] << " buffer " << lengths[1]);
    auto model = LoadSyntheticFixedComponentsModel();
    FixedStatePool pool{model, 1};
    {
      auto initial = pool.Reserve(One(kRequestA, 3));
      std::copy(lengths.begin(), lengths.end(),
                initial.Bindings()[4].output->GetTensorMutableData<int32_t>());
      initial.Commit();
    }
    auto reservation = pool.Reserve(One(kRequestA, 11, 7));
    reservation.CommitPrefix(0, 8, 7);
    EXPECT_THROW(reservation.Commit(), std::runtime_error);
    EXPECT_EQ(reservation.State(), FixedStateReservationState::Failed);
    EXPECT_EQ(reservation.Bindings()[4].input->GetTensorData<int32_t>()[0], lengths[0]);
    EXPECT_EQ(reservation.Bindings()[4].input->GetTensorData<int32_t>()[1], lengths[1]);
  }
}

TEST(FixedStatePoolComponentsTest, IndexerReplayCanFillKeyCapacityExactly) {
  auto model = LoadSyntheticFixedComponentsModel();
  FixedStatePool pool{model, 1};
  {
    auto initial = pool.Reserve(One(kRequestA, 3));
    auto* lengths = initial.Bindings()[4].output->GetTensorMutableData<int32_t>();
    lengths[0] = 7;
    lengths[1] = 3;
    initial.Commit();
  }
  {
    auto reservation = pool.Reserve(One(kRequestA, 11, 7));
    std::fill_n(reservation.Bindings()[0].state_update_value->GetTensorMutableData<int64_t>(), 14, 8);
    FillHalfTensor(*reservation.Bindings()[1].state_update_value, 0.0f);
    auto* updates = reservation.Bindings()[2].state_update_value->GetTensorMutableData<float>();
    std::fill_n(updates, 14, 42.0f);
    reservation.CommitPrefix(0, 8, 1);
    EXPECT_NO_THROW(reservation.Commit());
  }
  auto resident = pool.Reserve(One(kRequestA, 4));
  const auto* lengths = resident.Bindings()[4].input->GetTensorData<int32_t>();
  EXPECT_EQ(lengths[0], 8);
  EXPECT_EQ(lengths[1], 0);
  const auto* keys = resident.Bindings()[2].input->GetTensorData<float>();
  EXPECT_EQ(keys[14], 42.0f);
  EXPECT_EQ(keys[15], 42.0f);
}

TEST(FixedStateReplayTest, RejectsInvalidIndexerDescriptorWithoutWriting) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  std::array<float, 16> source{}, destination;
  std::array<float, 6> buffer{}, destination_buffer;
  std::array<float, 14> updates{};
  const std::array<std::array<int32_t, 2>, 6> invalid_lengths{
      {{-1, 0}, {0, -1}, {9, 0}, {0, 4}, {8, 0}, {7, 3}}};
  for (const auto& lengths : invalid_lengths) {
    destination.fill(123.0f);
    destination_buffer.fill(456.0f);
    std::array<int32_t, 2> destination_lengths{99, 99};
    StateUpdateReplayDesc descriptor{};
    descriptor.kind = StateUpdateReplayKind::Indexer;
    descriptor.source_state = HostReplayBytes(source);
    descriptor.destination_state = HostReplayBytes(destination);
    descriptor.source_aux_state = HostReplayBytes(buffer);
    descriptor.destination_aux_state = HostReplayBytes(destination_buffer);
    descriptor.source_lengths = HostReplayBytes(lengths);
    descriptor.destination_lengths = HostReplayBytes(destination_lengths);
    descriptor.value = HostReplayBytes(updates);
    descriptor.state_width = 2;
    descriptor.state_capacity = 8;
    descriptor.aux_capacity = 3;
    descriptor.compress_ratio = 4;
    descriptor.kept_count = 7;
    descriptor.element_size = sizeof(float);
    EXPECT_THROW(cpu->ReplayStateUpdates(&descriptor, 1), std::runtime_error);
    EXPECT_TRUE(std::all_of(destination.begin(), destination.end(),
                            [](float value) { return value == 123.0f; }));
    EXPECT_TRUE(std::all_of(destination_buffer.begin(), destination_buffer.end(),
                            [](float value) { return value == 456.0f; }));
    EXPECT_EQ(destination_lengths, (std::array<int32_t, 2>{99, 99}));
  }
}

TEST(FixedStateReplayTest, ValidatesPortableElementWidths) {
  using Kind = Config::Model::Decoder::StateUpdateKind;
  for (size_t width : {size_t{1}, size_t{2}, size_t{4}, size_t{8}, size_t{16}}) {
    for (auto kind : {Kind::CausalConv, Kind::Indexer, Kind::Ple}) {
      const bool supported = width == 2 || width == 4 || (kind == Kind::Ple && width == 8);
      if (supported) {
        EXPECT_NO_THROW(ValidateFixedStateUpdateElementSize(kind, width));
      } else {
        EXPECT_THROW(ValidateFixedStateUpdateElementSize(kind, width), std::runtime_error);
      }
    }
  }
}

TEST_F(FixedStatePoolTest, SlotReuseGathersZeroAfterRelease) {
  auto pool = MakePool(1);
  const auto handle_a = MakeResident(*pool, kRequestA, 5.0f);
  pool->Release(handle_a);
  EXPECT_THROW(pool->Release(handle_a), std::runtime_error);  // Stale handle after release.

  auto requests = One(kRequestB);
  auto reservation = pool->Reserve(requests);
  EXPECT_EQ(reservation.Handles()[0].slot, handle_a.slot);
  EXPECT_GT(reservation.Handles()[0].generation, handle_a.generation);
  ExpectInputRows(reservation, 0, 0.0f);  // Reused slot must not leak the released request's state.
}

TEST_F(FixedStatePoolTest, PrefixCheckpointRestoresEveryFixedStateRow) {
  FixedStatePool pool{model_, /*capacity=*/1,
                      /*prefix_checkpoint_capacity=*/1};
  const auto source = MakeResident(
      pool, kRequestA, 7.0f, /*target_tokens=*/4);
  auto checkpoint = pool.CapturePrefixCheckpoint(kRequestA);
  ASSERT_NE(checkpoint, nullptr);
  EXPECT_EQ(checkpoint->TokenCount(), 4u);
  EXPECT_EQ(pool.AvailablePrefixCheckpoints(), 0u);
  pool.Release(source);

  const std::array<FixedStateReservationRequest, 1> requests{
      FixedStateReservationRequest{
          kRequestB, /*target_tokens=*/5, /*capture_count=*/0, checkpoint}};
  {
    auto reservation = pool.Reserve(requests);
    EXPECT_FALSE(reservation.UsesDirectBindings());
    ExpectInputRows(reservation, 0, 7.0f);
    FillStagedRows(reservation, 0, 9.0f);
    reservation.Commit();
  }
  EXPECT_EQ(pool.CommittedTokens(pool.HandleFor(kRequestB)), 5u);

  checkpoint.reset();
  EXPECT_EQ(pool.AvailablePrefixCheckpoints(), 0u);
}

TEST_F(FixedStatePoolTest, PrefixCheckpointLeaseReleasesItsPreallocatedSlot) {
  FixedStatePool pool{model_, /*capacity=*/1,
                      /*prefix_checkpoint_capacity=*/1};
  MakeResident(pool, kRequestA, 3.0f, /*target_tokens=*/4);
  auto checkpoint = pool.CapturePrefixCheckpoint(kRequestA);
  ASSERT_NE(checkpoint, nullptr);
  EXPECT_EQ(pool.Snapshot().checkpoint_count, 1u);

  checkpoint.reset();

  EXPECT_EQ(pool.AvailablePrefixCheckpoints(), 1u);
  EXPECT_EQ(pool.Snapshot().checkpoint_count, 0u);
}

TEST_F(FixedStatePoolTest, DraftAttachmentRequiresExactFixedBoundaryAndRetainsLeasedReaders) {
  constexpr size_t block_size = 4;
  BlockPool blocks{block_size, 2};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 2;
  options.requires_checkpoint = true;
  options.max_checkpoints = 2;
  FixedStatePool pool{model_, /*capacity=*/2, /*prefix_checkpoint_capacity=*/2};
  PrefixCache index{blocks, options};
  MakeResident(pool, kRequestA, 7.0f, /*target_tokens=*/block_size);
  auto fixed = pool.CapturePrefixCheckpoint(kRequestA);
  ASSERT_NE(fixed, nullptr);

  const std::array<int32_t, 5> tokens{1, 2, 3, 4, 5};
  auto owned = blocks.AllocateBlocks(block_size);
  ASSERT_EQ(owned.size(), 1u);
  auto registration = index.Register(owned.front(), std::span<const int32_t>(tokens).first(block_size), {});
  ASSERT_NE(registration.identity, nullptr);
  ASSERT_TRUE(index.AttachCheckpoint(registration.identity, fixed));

  auto draft = std::make_shared<Dflash2PrefixCheckpoint>();
  draft->token_count = block_size;
  ASSERT_TRUE(index.CanAttachDraftCheckpoint(registration.identity, block_size));
  EXPECT_FALSE(index.CanAttachDraftCheckpoint(registration.identity, block_size * 2));
  auto other = pool.CapturePrefixCheckpoint(kRequestA);
  ASSERT_NE(other, nullptr);
  EXPECT_FALSE(index.AttachDraftCheckpoint(registration.identity, other, draft));
  ASSERT_TRUE(index.AttachDraftCheckpoint(registration.identity, fixed, draft));
  draft.reset();

  {
    auto match = index.Match(tokens, tokens.size() - 1);
    EXPECT_EQ(match.token_count, block_size);
    ASSERT_NE(match.draft_checkpoint, nullptr);
    index.DropUnleasedDraftCheckpoints();
    EXPECT_EQ(index.Match(tokens, tokens.size() - 1).draft_checkpoint, match.draft_checkpoint);
  }
  index.DropUnleasedDraftCheckpoints();
  auto match = index.Match(tokens, tokens.size() - 1);
  EXPECT_EQ(match.token_count, block_size);
  EXPECT_EQ(match.draft_checkpoint, nullptr);
  EXPECT_EQ(match.fixed_state_checkpoint, fixed);
  match = {};
  fixed.reset();
  other.reset();
  blocks.Free(owned);
  EXPECT_EQ(index.Reclaim(1), 1u);
  EXPECT_FALSE(index.CanAttachDraftCheckpoint(registration.identity, block_size));
  auto detached = pool.CapturePrefixCheckpoint(kRequestA);
  ASSERT_NE(detached, nullptr);
  EXPECT_FALSE(index.AttachDraftCheckpoint(registration.identity, detached,
                                           std::make_shared<Dflash2PrefixCheckpoint>()));
}

TEST_F(FixedStatePoolTest, DraftSnapshotReclamationDropsOnlyOneUnleasedSnapshot) {
  constexpr size_t block_size = 4;
  BlockPool blocks{block_size, 3};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 3;
  options.requires_checkpoint = true;
  options.max_checkpoints = 3;
  FixedStatePool pool{model_, /*capacity=*/1, /*prefix_checkpoint_capacity=*/3};
  PrefixCache index{blocks, options};
  MakeResident(pool, kRequestA, 7.0f, block_size);
  const std::array<std::array<int32_t, 5>, 3> histories{
      std::array<int32_t, 5>{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
  std::vector<std::shared_ptr<Block>> owned;
  for (const auto& tokens : histories) {
    auto allocated = blocks.AllocateBlocks(block_size);
    ASSERT_EQ(allocated.size(), 1u);
    auto registration = index.Register(allocated.front(), std::span<const int32_t>(tokens).first(block_size), {});
    auto fixed = pool.CapturePrefixCheckpoint(kRequestA);
    ASSERT_NE(fixed, nullptr);
    ASSERT_TRUE(index.AttachCheckpoint(registration.identity, fixed));
    auto draft = std::make_shared<Dflash2PrefixCheckpoint>();
    draft->token_count = block_size;
    ASSERT_TRUE(index.AttachDraftCheckpoint(registration.identity, fixed, std::move(draft)));
    owned.push_back(std::move(allocated.front()));
  }
  auto leased = index.Match(histories[0], block_size);
  ASSERT_NE(leased.draft_checkpoint, nullptr);
  EXPECT_TRUE(index.ReclaimDraftCheckpoint());
  size_t retained = 0;
  for (const auto& tokens : histories) {
    const auto match = index.Match(tokens, block_size);
    EXPECT_EQ(match.token_count, block_size);
    ASSERT_NE(match.fixed_state_checkpoint, nullptr);
    if (match.draft_checkpoint) {
      ++retained;
    }
  }
  EXPECT_EQ(retained, 2u);
  EXPECT_EQ(index.Match(histories[0], block_size).draft_checkpoint, leased.draft_checkpoint);
  EXPECT_TRUE(index.ReclaimDraftCheckpoint());
  EXPECT_FALSE(index.ReclaimDraftCheckpoint());
  leased = {};
  EXPECT_TRUE(index.ReclaimDraftCheckpoint());
  EXPECT_FALSE(index.ReclaimDraftCheckpoint());
  blocks.Free(owned);
}

TEST_F(FixedStatePoolTest, DraftSnapshotReclamationPrefersDescendantOverEarlierAncestor) {
  constexpr size_t block_size = 4;
  BlockPool blocks{block_size, 2};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 2;
  options.requires_checkpoint = true;
  options.max_checkpoints = 2;
  FixedStatePool pool{model_, /*capacity=*/1, /*prefix_checkpoint_capacity=*/2};
  PrefixCache index{blocks, options};
  const std::array<int32_t, 9> tokens{2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto owned = blocks.AllocateBlocks(2 * block_size);
  ASSERT_EQ(owned.size(), 2u);
  std::shared_ptr<const BlockIdentity> parent;
  for (size_t i = 0; i < owned.size(); ++i) {
    const size_t boundary = (i + 1) * block_size;
    MakeResident(pool, kRequestA, 7.0f, boundary);
    auto fixed = pool.CapturePrefixCheckpoint(kRequestA);
    ASSERT_NE(fixed, nullptr);
    auto registration = index.Register(
        owned[i], std::span<const int32_t>{tokens}.subspan(i * block_size, block_size), parent);
    ASSERT_TRUE(index.AttachCheckpoint(registration.identity, fixed));
    auto draft = std::make_shared<Dflash2PrefixCheckpoint>();
    draft->token_count = boundary;
    ASSERT_TRUE(index.AttachDraftCheckpoint(registration.identity, fixed, std::move(draft)));
    parent = registration.identity;
  }

  ASSERT_NE(index.Match(tokens, block_size).draft_checkpoint, nullptr);
  ASSERT_NE(index.Match(tokens, 2 * block_size).draft_checkpoint, nullptr);
  ASSERT_TRUE(index.ReclaimDraftCheckpoint());
  {
    const auto shorter = index.Match(tokens, block_size);
    const auto longer = index.Match(tokens, 2 * block_size);
    EXPECT_EQ(shorter.token_count, block_size);
    EXPECT_NE(shorter.draft_checkpoint, nullptr);
    EXPECT_EQ(longer.token_count, 2 * block_size);
    EXPECT_NE(longer.fixed_state_checkpoint, nullptr);
    EXPECT_EQ(longer.draft_checkpoint, nullptr);
    EXPECT_EQ(index.CheckpointCount(), 2u);
    EXPECT_EQ(index.IndexedBlocks(), 2u);
    EXPECT_FALSE(index.ReclaimDraftCheckpoint());
  }
  EXPECT_TRUE(index.ReclaimDraftCheckpoint());
  EXPECT_FALSE(index.ReclaimDraftCheckpoint());
  blocks.Free(owned);
}

TEST_F(FixedStatePoolTest, HybridPublicationRollsBackCapacityRefusalAndRetries) {
  BlockPool blocks{4, 3};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 2;
  options.requires_checkpoint = true;
  options.max_checkpoints = 1;
  PrefixCache index{blocks, options};
  FixedStatePool pool{model_, /*capacity=*/1, /*prefix_checkpoint_capacity=*/1};
  MakeResident(pool, kRequestA, 7.0f, /*target_tokens=*/8);
  auto checkpoint = pool.CapturePrefixCheckpoint(kRequestA);

  const std::array<int32_t, 4> unrelated{20, 21, 22, 23};
  auto occupied = blocks.AllocateBlocks(4);
  ASSERT_NE(index.Register(occupied.front(), unrelated, {}).identity, nullptr);
  const std::array<int32_t, 8> tokens{1, 2, 3, 4, 5, 6, 7, 8};
  auto suffix = blocks.AllocateBlocks(8);
  auto refused = index.RegisterCheckpointedPrefix(suffix, tokens, {}, checkpoint);
  EXPECT_EQ(refused.status, PrefixCacheRegistrationStatus::CapacityRefused);
  EXPECT_EQ(index.IndexedBlocks(), 1u);
  EXPECT_EQ(index.CheckpointCount(), 0u);
  for (const auto& block : suffix) {
    EXPECT_FALSE(block->HasIdentity());
    EXPECT_EQ(block->RefCount(), 1u);
  }

  blocks.Free(occupied);
  ASSERT_EQ(index.Reclaim(1), 1u);
  ASSERT_NE(index.RegisterCheckpointedPrefix(suffix, tokens, {}, checkpoint).identity, nullptr);
  EXPECT_EQ(index.Match(tokens, tokens.size()).token_count, 8u);
  EXPECT_EQ(index.CheckpointCount(), 1u);
  blocks.Free(suffix);
}

TEST_F(FixedStatePoolTest, HybridPublicationDoesNotReplaceLeasedCheckpoint) {
  BlockPool blocks{4, 3};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 3;
  options.requires_checkpoint = true;
  options.max_checkpoints = 1;
  PrefixCache index{blocks, options};
  FixedStatePool pool{model_, /*capacity=*/2, /*prefix_checkpoint_capacity=*/2};
  MakeResident(pool, kRequestA, 3.0f, /*target_tokens=*/4);
  MakeResident(pool, kRequestB, 7.0f, /*target_tokens=*/8);
  const std::array<int32_t, 4> first_tokens{20, 21, 22, 23};
  auto first = blocks.AllocateBlocks(4);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     first, first_tokens, {}, pool.CapturePrefixCheckpoint(kRequestA))
                .identity,
            nullptr);
  auto adopter = index.Match(first_tokens, first_tokens.size());
  ASSERT_NE(adopter.fixed_state_checkpoint, nullptr);
  blocks.AddRef(adopter.blocks);

  const std::array<int32_t, 8> tokens{1, 2, 3, 4, 5, 6, 7, 8};
  auto suffix = blocks.AllocateBlocks(8);
  auto checkpoint = pool.CapturePrefixCheckpoint(kRequestB);
  EXPECT_EQ(index.RegisterCheckpointedPrefix(suffix, tokens, {}, checkpoint).status,
            PrefixCacheRegistrationStatus::CapacityRefused);
  EXPECT_EQ(index.IndexedBlocks(), 1u);
  EXPECT_EQ(index.Match(first_tokens, first_tokens.size()).fixed_state_checkpoint,
            adopter.fixed_state_checkpoint);
  for (const auto& block : suffix) {
    EXPECT_FALSE(block->HasIdentity());
    EXPECT_EQ(block->RefCount(), 1u);
  }

  blocks.Free(adopter.blocks);
  adopter = {};
  ASSERT_NE(index.RegisterCheckpointedPrefix(suffix, tokens, {}, checkpoint).identity, nullptr);
  EXPECT_EQ(index.Match(tokens, tokens.size()).token_count, 8u);
  blocks.Free(first);
  blocks.Free(suffix);
}

TEST_F(FixedStatePoolTest, HybridBudgetPressureProtectsLeasedCheckpointAncestry) {
  for (int lease_kind = 0; lease_kind < 2; ++lease_kind) {
    SCOPED_TRACE(lease_kind);
    BlockPool blocks{4, 4};
    PrefixCacheOptions options;
    options.enabled = true;
    options.max_blocks = 2;
    options.requires_checkpoint = true;
    options.max_checkpoints = 2;
    PrefixCache index{blocks, options};
    FixedStatePool pool{model_, /*capacity=*/3, /*prefix_checkpoint_capacity=*/2};
    MakeResident(pool, kRequestA, 7.0f, 8);
    MakeResident(pool, kRequestB, 9.0f, 8);
    const std::array<int32_t, 8> old_tokens{1, 2, 3, 4, 5, 6, 7, 8};
    const std::array<int32_t, 8> new_tokens{20, 21, 22, 23, 24, 25, 26, 27};
    auto old_history = blocks.AllocateBlocks(8);
    auto old_registration = index.RegisterCheckpointedPrefix(
        old_history, old_tokens, {}, pool.CapturePrefixCheckpoint(kRequestA));
    ASSERT_NE(old_registration.identity, nullptr);
    blocks.Free(old_history);

    PrefixCacheMatch held_match;
    std::shared_ptr<Dflash2PrefixCheckpoint> held_draft;
    if (lease_kind == 0) {
      // Match itself leases only checkpoint metadata; it has not added BlockPool references.
      held_match = index.Match(old_tokens, old_tokens.size());
      ASSERT_NE(held_match.fixed_state_checkpoint, nullptr);
    } else {
      auto boundary = index.Match(old_tokens, old_tokens.size());
      held_draft = std::make_shared<Dflash2PrefixCheckpoint>();
      held_draft->token_count = old_tokens.size();
      ASSERT_TRUE(index.AttachDraftCheckpoint(
          old_registration.identity, boundary.fixed_state_checkpoint, held_draft));
      boundary = {};
    }

    EXPECT_EQ(index.ReclaimableBlocks(), 0u);
    EXPECT_EQ(index.Reclaim(1), 0u);
    EXPECT_EQ(index.ReclaimableCheckpoints(), 0u);
    EXPECT_EQ(index.ReclaimCheckpoints(1), 0u);
    auto new_history = blocks.AllocateBlocks(8);
    bool captured = false;
    EXPECT_EQ(index.RegisterCheckpointedPrefix(
                       new_history, new_tokens, {}, [&] {
                         captured = true;
                         return pool.CapturePrefixCheckpoint(kRequestB);
                       })
                  .status,
              PrefixCacheRegistrationStatus::CapacityRefused);
    EXPECT_FALSE(captured);
    for (const auto& block : new_history) {
      EXPECT_FALSE(block->HasIdentity());
      EXPECT_EQ(block->RefCount(), 1u);
    }

    auto retained = index.Match(old_tokens, old_tokens.size());
    ASSERT_EQ(retained.token_count, 8u);
    {
      auto restored = pool.Reserve(
          std::array{Request{kRequestC, 9, 0, retained.fixed_state_checkpoint}});
      ExpectInputRows(restored, 0, 7.0f);
    }
    retained = {};
    held_match = {};
    held_draft.reset();
    index.DropUnleasedDraftCheckpoints();
    EXPECT_EQ(index.ReclaimableBlocks(), 2u);
    blocks.Free(new_history);
  }
}

TEST_F(FixedStatePoolTest, LeasedAncestryCountsOverlappingEndpointsAndRefreshesEveryQuery) {
  BlockPool blocks{4, 5};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 5;
  options.requires_checkpoint = true;
  options.max_checkpoints = 3;
  FixedStatePool pool{model_, 2, 3};
  PrefixCache index{blocks, options};
  MakeResident(pool, kRequestA, 7.0f, 8);
  MakeResident(pool, kRequestB, 9.0f, 8);
  const std::array<int32_t, 12> tokens{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const std::array<int32_t, 8> unrelated{20, 21, 22, 23, 24, 25, 26, 27};
  auto head = blocks.AllocateBlocks(8);
  auto first = index.RegisterCheckpointedPrefix(
      head, std::span<const int32_t>{tokens}.first(8), {}, pool.CapturePrefixCheckpoint(kRequestA));
  ASSERT_NE(first.identity, nullptr);
  MakeResident(pool, kRequestA, 8.0f, 12);
  auto tail = blocks.AllocateBlocks(4);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     tail, std::span<const int32_t>{tokens}.subspan(8), first.identity,
                     pool.CapturePrefixCheckpoint(kRequestA))
                .identity,
            nullptr);
  auto other = blocks.AllocateBlocks(8);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     other, unrelated, {}, pool.CapturePrefixCheckpoint(kRequestB))
                .identity,
            nullptr);
  blocks.Free(head);
  blocks.Free(tail);
  blocks.Free(other);
  EXPECT_EQ(index.ReclaimableBlocks(), 5u);

  auto short_lease = index.Match(tokens, 8);
  EXPECT_EQ(index.ReclaimableBlocks(), 3u);
  auto long_lease = index.Match(tokens, 12);
  EXPECT_EQ(index.ReclaimableBlocks(), 2u);
  blocks.AddRef(head.front());
  EXPECT_EQ(index.ReclaimableBlocks(), 2u);
  blocks.Release(head.front());
  auto other_lease = index.Match(unrelated, 8);
  EXPECT_EQ(index.ReclaimableBlocks(), 0u);
  EXPECT_EQ(index.Reclaim(5), 0u);

  long_lease = {};
  EXPECT_EQ(index.ReclaimableBlocks(), 1u);
  EXPECT_EQ(index.Reclaim(5), 1u);
  EXPECT_EQ(index.Match(tokens, 12).token_count, 8u);
  short_lease = {};
  EXPECT_EQ(index.ReclaimableBlocks(), 2u);
  other_lease = {};
  EXPECT_EQ(index.ReclaimableBlocks(), 4u);
  EXPECT_EQ(index.Reclaim(4), 4u);
}

TEST_F(FixedStatePoolTest, LeasedOrphanAncestryDoesNotProtectAReusedPhysicalBlockId) {
  BlockPool blocks{4, 2};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 2;
  options.requires_checkpoint = true;
  options.max_checkpoints = 1;
  FixedStatePool pool{model_, 1, 1};
  PrefixCache index{blocks, options};
  MakeResident(pool, kRequestA, 7.0f, 8);
  const std::array<int32_t, 8> tokens{1, 2, 3, 4, 5, 6, 7, 8};
  auto history = blocks.AllocateBlocks(8);
  auto endpoint = index.RegisterCheckpointedPrefix(
      history, tokens, {}, pool.CapturePrefixCheckpoint(kRequestA));
  ASSERT_NE(endpoint.identity, nullptr);
  const auto old_id = history.front()->Id();
  blocks.Release(history.front());
  ASSERT_EQ(index.Reclaim(1), 1u);
  auto lease = index.DraftBoundary(endpoint.identity, 8);
  ASSERT_NE(lease, nullptr);

  auto reused = blocks.AllocateBlocks(4);
  ASSERT_EQ(reused.front()->Id(), old_id);
  const std::array<int32_t, 4> unrelated{20, 21, 22, 23};
  ASSERT_NE(index.Register(reused.front(), unrelated, {}).identity, nullptr);
  blocks.Free(reused);
  EXPECT_EQ(index.ReclaimableBlocks(), 1u);
  EXPECT_EQ(index.Reclaim(1), 1u);
  blocks.Release(history.back());
  EXPECT_EQ(index.ReclaimableBlocks(), 0u);
  lease.reset();
  EXPECT_EQ(index.ReclaimableBlocks(), 1u);
  EXPECT_EQ(index.Reclaim(1), 1u);
}

TEST_F(FixedStatePoolTest, ReclaimableBlockQueryRefreshesAfterLeaseAndCheckpointRelease) {
  BlockPool blocks{4, 1};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 1;
  options.requires_checkpoint = true;
  options.max_checkpoints = 1;
  FixedStatePool pool{model_, 1, 1};
  PrefixCache index{blocks, options};
  MakeResident(pool, kRequestA, 7.0f, 4);
  const std::array<int32_t, 4> tokens{1, 2, 3, 4};
  auto history = blocks.AllocateBlocks(4);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     history, tokens, {}, pool.CapturePrefixCheckpoint(kRequestA))
                .identity,
            nullptr);
  blocks.Free(history);

  auto lease = index.Match(tokens, tokens.size());
  ASSERT_EQ(lease.blocks.size(), 1u);
  const auto retained = lease.blocks;
  const auto retained_id = retained.front()->Id();
  const auto* checkpoint = lease.fixed_state_checkpoint.get();
  EXPECT_TRUE(index.ReclaimableBlockIds(retained).empty());

  lease = {};
  EXPECT_EQ(index.ReclaimableBlocks(), 1u);
  EXPECT_EQ(index.ReclaimableBlockIds(retained),
            std::vector<size_t>{retained_id});
  ASSERT_TRUE(index.ReclaimCheckpoint(checkpoint));
  EXPECT_EQ(index.ReclaimableBlockIds(retained),
            std::vector<size_t>{retained_id});

  ASSERT_EQ(index.Reclaim(1), 1u);
  auto unrelated = blocks.AllocateBlocks(4);
  ASSERT_EQ(unrelated.front()->Id(), retained_id);
  const std::array<int32_t, 4> unrelated_tokens{20, 21, 22, 23};
  ASSERT_NE(index.Register(unrelated.front(), unrelated_tokens, {}).identity,
            nullptr);
  blocks.Free(unrelated);
  EXPECT_TRUE(index.ReclaimableBlockIds(retained).empty());
  EXPECT_EQ(index.ReclaimableBlockIds(unrelated),
            std::vector<size_t>{retained_id});
  EXPECT_TRUE(index.Match(tokens, tokens.size()).Empty());
  EXPECT_TRUE(index.Match(unrelated_tokens, unrelated_tokens.size()).Empty());
}

TEST_F(FixedStatePoolTest, ManyLongHistoriesProtectOnlyTheirCurrentFixedOrDraftLeases) {
  constexpr size_t histories = 32;
  constexpr size_t history_blocks = 128;
  constexpr size_t token_count = history_blocks * 4;
  BlockPool blocks{4, histories * history_blocks};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = blocks.Capacity();
  options.requires_checkpoint = true;
  options.max_checkpoints = histories;
  FixedStatePool pool{model_, 1, histories};
  PrefixCache index{blocks, options};
  std::vector<std::shared_ptr<const FixedStatePrefixCheckpoint>> fixed_leases(histories);
  std::vector<std::shared_ptr<Dflash2PrefixCheckpoint>> draft_leases(histories);
  for (size_t history = 0; history < histories; ++history) {
    const auto handle = MakeResident(pool, kRequestA, 7.0f, token_count);
    std::vector<int32_t> tokens(token_count);
    for (size_t token = 0; token < token_count; ++token) {
      tokens[token] = static_cast<int32_t>(history * token_count + token + 1);
    }
    auto owned = blocks.AllocateBlocks(token_count);
    const auto registration = index.RegisterCheckpointedPrefix(
        owned, tokens, {}, pool.CapturePrefixCheckpoint(kRequestA));
    ASSERT_NE(registration.identity, nullptr);
    if (history % 2 == 0) {
      fixed_leases[history] = index.DraftBoundary(registration.identity, token_count);
      ASSERT_NE(fixed_leases[history], nullptr);
    }
    if (history % 3 == 0) {
      draft_leases[history] = std::make_shared<Dflash2PrefixCheckpoint>();
      draft_leases[history]->token_count = token_count;
      ASSERT_TRUE(index.AttachDraftCheckpoint(
          registration.identity, index.DraftBoundary(registration.identity, token_count),
          draft_leases[history]));
    }
    blocks.Free(owned);
    pool.Release(handle);
  }
  const auto expected = [&] {
    size_t count = 0;
    for (size_t history = 0; history < histories; ++history) {
      if (!fixed_leases[history] && !draft_leases[history]) {
        count += history_blocks;
      }
    }
    return count;
  };
  EXPECT_EQ(index.ReclaimableBlocks(), expected());
  for (size_t history = 0; history < histories; ++history) {
    fixed_leases[history].reset();
    EXPECT_EQ(index.ReclaimableBlocks(), expected());
  }
  const size_t reclaimable = expected();
  EXPECT_EQ(index.Reclaim(blocks.Capacity()), reclaimable);
  EXPECT_EQ(index.ReclaimableBlocks(), 0u);
  for (auto& lease : draft_leases) {
    lease.reset();
  }
  index.DropUnleasedDraftCheckpoints();
  EXPECT_EQ(index.ReclaimableBlocks(), blocks.Capacity() - reclaimable);
  EXPECT_EQ(index.Reclaim(blocks.Capacity()), blocks.Capacity() - reclaimable);
}

TEST_F(FixedStatePoolTest, HybridPublicationRollsBackAfterAllocationFailure) {
  BlockPool blocks{4, 2};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 2;
  options.requires_checkpoint = true;
  options.max_checkpoints = 1;
  options.hash = [](uint64_t parent, std::span<const int32_t> tokens) {
    static size_t second_block_calls = 0;
    if (tokens.front() == 5 && ++second_block_calls % 2 == 0) {
      throw std::bad_alloc{};
    }
    return PrefixCache::ChainHash(parent, tokens);
  };
  PrefixCache index{blocks, options};
  FixedStatePool pool{model_, /*capacity=*/1, /*prefix_checkpoint_capacity=*/1};
  MakeResident(pool, kRequestA, 7.0f, /*target_tokens=*/8);
  const std::array<int32_t, 8> tokens{1, 2, 3, 4, 5, 6, 7, 8};
  auto suffix = blocks.AllocateBlocks(8);
  EXPECT_THROW(index.RegisterCheckpointedPrefix(
                   suffix, tokens, {}, pool.CapturePrefixCheckpoint(kRequestA)),
               std::bad_alloc);
  EXPECT_EQ(index.IndexedBlocks(), 0u);
  EXPECT_EQ(index.CheckpointCount(), 0u);
  EXPECT_EQ(pool.AvailablePrefixCheckpoints(), 1u);
  for (const auto& block : suffix) {
    EXPECT_FALSE(block->HasIdentity());
    EXPECT_EQ(block->RefCount(), 1u);
  }
  blocks.Free(suffix);
}

TEST_F(FixedStatePoolTest, CompetingHybridPublicationKeepsCanonicalPhysicalHistory) {
  BlockPool blocks{4, 4};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 4;
  options.requires_checkpoint = true;
  options.max_checkpoints = 2;
  PrefixCache index{blocks, options};
  FixedStatePool pool{model_, /*capacity=*/2, /*prefix_checkpoint_capacity=*/2};
  MakeResident(pool, kRequestA, 3.0f, /*target_tokens=*/8);
  MakeResident(pool, kRequestB, 7.0f, /*target_tokens=*/8);
  const std::array<int32_t, 8> tokens{1, 2, 3, 4, 5, 6, 7, 8};
  auto first = blocks.AllocateBlocks(8);
  auto second = blocks.AllocateBlocks(8);
  auto canonical = pool.CapturePrefixCheckpoint(kRequestA);
  ASSERT_NE(index.RegisterCheckpointedPrefix(first, tokens, {}, canonical).identity, nullptr);
  EXPECT_EQ(index.RegisterCheckpointedPrefix(
                     second, tokens, {}, pool.CapturePrefixCheckpoint(kRequestB))
                .status,
            PrefixCacheRegistrationStatus::Duplicate);
  const auto match = index.Match(tokens, tokens.size());
  EXPECT_EQ(match.blocks, first);
  EXPECT_EQ(match.fixed_state_checkpoint, canonical);
  EXPECT_EQ(index.IndexedBlocks(), 2u);
  for (const auto& block : second) {
    EXPECT_FALSE(block->HasIdentity());
    EXPECT_EQ(block->RefCount(), 1u);
  }
  blocks.Free(first);
  blocks.Free(second);
}

TEST_F(FixedStatePoolTest, HybridLogicalVariantsRetainExactRootPhysicalHistories) {
  BlockPool blocks{4, 8};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 8;
  options.requires_checkpoint = true;
  options.max_checkpoints = 2;
  PrefixCache index{blocks, options};
  FixedStatePool pool{model_, /*capacity=*/2, /*prefix_checkpoint_capacity=*/2};
  MakeResident(pool, kRequestA, 7.0f, 8);
  MakeResident(pool, kRequestB, 9.0f, 8);
  const std::array<int32_t, 8> first_tokens{1, 2, 3, 4, 5, 6, 7, 8};
  const std::array<int32_t, 8> second_tokens{1, 2, 3, 4, 9, 10, 11, 12};
  auto first = blocks.AllocateBlocks(8);
  auto second = blocks.AllocateBlocks(8);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     first, first_tokens, {}, pool.CapturePrefixCheckpoint(kRequestA))
                .identity,
            nullptr);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     second, second_tokens, {}, pool.CapturePrefixCheckpoint(kRequestB))
                .identity,
            nullptr);

  EXPECT_NE(first.front()->IdentityPtr(), second.front()->IdentityPtr());
  ASSERT_NE(first.front()->Identity().logical, nullptr);
  EXPECT_EQ(first.front()->Identity().logical, second.front()->Identity().logical);
  for (const auto* history : {&first, &second}) {
    for (const auto& block : *history) {
      const auto& identity = block->Identity();
      ASSERT_NE(identity.logical, nullptr);
      EXPECT_TRUE(identity.tokens.empty());
      EXPECT_EQ(identity.Tokens().data(), identity.logical->tokens.data());
      EXPECT_EQ(identity.Tokens().size(), block->Capacity());
    }
  }

  auto first_match = index.Match(first_tokens, first_tokens.size());
  auto second_match = index.Match(second_tokens, second_tokens.size());
  EXPECT_EQ(first_match.blocks, first);
  EXPECT_EQ(second_match.blocks, second);
  ASSERT_NE(first_match.fixed_state_checkpoint, nullptr);
  ASSERT_NE(second_match.fixed_state_checkpoint, nullptr);
  EXPECT_NE(first_match.fixed_state_checkpoint, second_match.fixed_state_checkpoint);
  EXPECT_EQ(index.IndexedBlocks(), 4u);
  EXPECT_EQ(index.CheckpointCount(), 2u);
  EXPECT_EQ(index.Metrics().hash_collisions, 0u);
  EXPECT_EQ(index.Metrics().duplicate_registrations, 0u);
  first_match = {};
  second_match = {};
  blocks.Free(first);
  blocks.Free(second);
}

TEST_F(FixedStatePoolTest, HybridMatchChoosesLongestExactEndpointAndReclaimsSelectedPointer) {
  BlockPool blocks{4, 4};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 4;
  options.requires_checkpoint = true;
  options.max_checkpoints = 2;
  PrefixCache index{blocks, options};
  FixedStatePool pool{model_, /*capacity=*/2, /*prefix_checkpoint_capacity=*/2};
  MakeResident(pool, kRequestA, 7.0f, 4);
  MakeResident(pool, kRequestB, 9.0f, 8);
  const std::array<int32_t, 8> tokens{1, 2, 3, 4, 5, 6, 7, 8};
  auto head = blocks.AllocateBlocks(4);
  auto head_registration = index.RegisterCheckpointedPrefix(
      head, std::span{tokens}.first(4), {}, pool.CapturePrefixCheckpoint(kRequestA));
  ASSERT_NE(head_registration.identity, nullptr);
  auto tail = blocks.AllocateBlocks(4);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     tail, std::span{tokens}.subspan(4), head_registration.identity,
                     pool.CapturePrefixCheckpoint(kRequestB))
                .identity,
            nullptr);

  auto longest = index.Match(tokens, tokens.size());
  ASSERT_EQ(longest.token_count, 8u);
  ASSERT_EQ(longest.blocks.size(), 2u);
  EXPECT_EQ(longest.blocks[0], head[0]);
  EXPECT_EQ(longest.blocks[1], tail[0]);
  longest = {};

  const auto* selected = index.ReclaimableCheckpoint(head_registration.identity);
  ASSERT_NE(selected, nullptr);
  EXPECT_EQ(selected->TokenCount(), 4u);
  ASSERT_TRUE(index.ReclaimCheckpoint(selected));
  EXPECT_EQ(index.CheckpointCount(), 1u);
  EXPECT_EQ(index.Match(tokens, tokens.size()).token_count, 8u);
  EXPECT_FALSE(index.ReclaimCheckpoint(selected));
  blocks.Free(head);
  blocks.Free(tail);
}

TEST_F(FixedStatePoolTest, HybridEndpointRejectsAnOrphanedPhysicalAncestor) {
  BlockPool blocks{4, 3};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 3;
  options.requires_checkpoint = true;
  options.max_checkpoints = 1;
  PrefixCache index{blocks, options};
  FixedStatePool pool{model_, /*capacity=*/1, /*prefix_checkpoint_capacity=*/1};
  MakeResident(pool, kRequestA, 7.0f, 8);
  const std::array<int32_t, 8> tokens{1, 2, 3, 4, 5, 6, 7, 8};
  auto history = blocks.AllocateBlocks(8);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     history, tokens, {}, pool.CapturePrefixCheckpoint(kRequestA))
                .identity,
            nullptr);

  // Keep the endpoint physically referenced while making only its ancestor reclaimable. A stale
  // endpoint must not splice the surviving tail onto a missing physical history.
  blocks.Release(history.front());
  ASSERT_EQ(index.Reclaim(1), 1u);
  EXPECT_TRUE(index.Match(tokens, tokens.size()).Empty());
  EXPECT_EQ(index.CheckpointCount(), 1u);
  blocks.Release(history.back());
}

TEST_F(FixedStatePoolTest, HybridIdenticalRepublishReplacesAnOrphanedLogicalHistory) {
  BlockPool blocks{4, 6};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 6;
  options.requires_checkpoint = true;
  options.max_checkpoints = 2;
  PrefixCache index{blocks, options};
  FixedStatePool pool{model_, /*capacity=*/2, /*prefix_checkpoint_capacity=*/2};
  MakeResident(pool, kRequestA, 7.0f, 8);
  MakeResident(pool, kRequestB, 9.0f, 8);
  const std::array<int32_t, 8> tokens{1, 2, 3, 4, 5, 6, 7, 8};
  auto orphaned = blocks.AllocateBlocks(8);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     orphaned, tokens, {}, pool.CapturePrefixCheckpoint(kRequestA))
                .identity,
            nullptr);

  // Leave the checkpoint endpoint and tail block retained while removing its exact physical root.
  blocks.Release(orphaned.front());
  ASSERT_EQ(index.Reclaim(1), 1u);
  ASSERT_TRUE(index.Match(tokens, tokens.size()).Empty());

  auto replacement = blocks.AllocateBlocks(8);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     replacement, tokens, {}, pool.CapturePrefixCheckpoint(kRequestB))
                .identity,
            nullptr);
  auto match = index.Match(tokens, tokens.size());
  ASSERT_EQ(match.token_count, 8u);
  EXPECT_EQ(match.blocks, replacement);
  EXPECT_EQ(index.Metrics().hash_collisions, 0u);
  EXPECT_EQ(index.Metrics().duplicate_registrations, 0u);
  match = {};
  blocks.Release(orphaned.back());
  blocks.Free(replacement);
}

TEST_F(FixedStatePoolTest, HybridMetadataFailurePreservesTwoIndependentEndpoints) {
  BlockPool blocks{4, 8};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 8;
  options.requires_checkpoint = true;
  options.max_checkpoints = 3;
  PrefixCache index{blocks, options};
  FixedStatePool pool{model_, /*capacity=*/3, /*prefix_checkpoint_capacity=*/3};
  MakeResident(pool, kRequestA, 7.0f, 8);
  MakeResident(pool, kRequestB, 9.0f, 8);
  MakeResident(pool, kRequestC, 11.0f, 8);
  const std::array<int32_t, 8> first_tokens{1, 2, 3, 4, 5, 6, 7, 8};
  const std::array<int32_t, 8> second_tokens{1, 2, 3, 4, 9, 10, 11, 12};
  const std::array<int32_t, 8> third_tokens{1, 2, 3, 4, 13, 14, 15, 16};
  auto first = blocks.AllocateBlocks(8);
  auto second = blocks.AllocateBlocks(8);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     first, first_tokens, {}, pool.CapturePrefixCheckpoint(kRequestA))
                .identity,
            nullptr);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     second, second_tokens, {}, pool.CapturePrefixCheckpoint(kRequestB))
                .identity,
            nullptr);
  blocks.Free(first);
  blocks.Free(second);
  auto third = blocks.AllocateBlocks(8);
  EXPECT_THROW(index.RegisterCheckpointedPrefix(
                   third, third_tokens, {},
                   []() -> std::shared_ptr<const FixedStatePrefixCheckpoint> {
                     throw std::bad_alloc{};
                   }),
               std::bad_alloc);
  EXPECT_EQ(index.Match(first_tokens, first_tokens.size()).token_count, 8u);
  EXPECT_EQ(index.Match(second_tokens, second_tokens.size()).token_count, 8u);
  EXPECT_EQ(index.Match(third_tokens, third_tokens.size()).token_count, 0u);
  EXPECT_EQ(index.IndexedBlocks(), 4u);
  EXPECT_EQ(index.CheckpointCount(), 2u);
  EXPECT_EQ(index.Metrics().evictions, 0u);
  for (const auto& block : third) {
    EXPECT_FALSE(block->HasIdentity());
    EXPECT_EQ(block->RefCount(), 1u);
  }
  blocks.Free(third);
}

TEST_F(FixedStatePoolTest, HybridBranchReplacementPreservesOldHitOnCaptureFailure) {
  BlockPool blocks{4, 4};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 4;
  options.requires_checkpoint = true;
  options.max_checkpoints = 1;
  PrefixCache index{blocks, options};
  FixedStatePool pool{model_, /*capacity=*/3, /*prefix_checkpoint_capacity=*/2};
  MakeResident(pool, kRequestA, 7.0f, 8);
  MakeResident(pool, kRequestB, 9.0f, 8);
  const std::array<int32_t, 8> original{1, 2, 3, 4, 5, 6, 7, 8};
  const std::array<int32_t, 8> branch{1, 2, 3, 4, 5, 6, 7, 9};
  auto first = blocks.AllocateBlocks(8);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     first, original, {}, pool.CapturePrefixCheckpoint(kRequestA))
                .identity,
            nullptr);
  blocks.Free(first);
  auto suffix = blocks.AllocateBlocks(8);
  EXPECT_THROW(index.RegisterCheckpointedPrefix(suffix, branch, {}, []() -> std::shared_ptr<const FixedStatePrefixCheckpoint> {
    throw std::bad_alloc{};
  }),
               std::bad_alloc);
  EXPECT_EQ(index.IndexedBlocks(), 2u);
  EXPECT_EQ(index.CheckpointCount(), 1u);
  EXPECT_EQ(index.Metrics().evictions, 0u);
  auto retained = index.Match(original, original.size());
  ASSERT_EQ(retained.token_count, 8u);
  {
    auto restored = pool.Reserve(
        std::array{Request{kRequestC, 9, 0, retained.fixed_state_checkpoint}});
    ExpectInputRows(restored, 0, 7.0f);
  }
  retained = {};
  for (const auto& block : suffix) {
    EXPECT_FALSE(block->HasIdentity());
    EXPECT_EQ(block->RefCount(), 1u);
  }

  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     suffix, branch, {}, pool.CapturePrefixCheckpoint(kRequestB))
                .identity,
            nullptr);
  auto match = index.Match(branch, branch.size());
  ASSERT_EQ(match.token_count, 8u);
  EXPECT_EQ(match.blocks, suffix);
  {
    auto replacement = pool.Reserve(
        std::array{Request{kRequestC, 9, 0, match.fixed_state_checkpoint}});
    ExpectInputRows(replacement, 0, 9.0f);
  }
  EXPECT_EQ(index.Match(original, original.size()).token_count, 0u);
  EXPECT_EQ(index.Metrics().evictions, 2u);
  blocks.Free(suffix);
}

TEST_F(FixedStatePoolTest, HybridBranchMetadataFailurePreservesOldHit) {
  BlockPool blocks{4, 4};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 4;
  options.requires_checkpoint = true;
  options.max_checkpoints = 1;
  static thread_local bool fail_staging = false;
  static thread_local size_t hash_calls = 0;
  hash_calls = 0;
  options.hash = [](uint64_t parent, std::span<const int32_t> tokens) {
    if (fail_staging && ++hash_calls == 4) {
      throw std::bad_alloc{};
    }
    return PrefixCache::ChainHash(parent, tokens);
  };
  PrefixCache index{blocks, options};
  FixedStatePool pool{model_, /*capacity=*/2, /*prefix_checkpoint_capacity=*/2};
  MakeResident(pool, kRequestA, 7.0f, 8);
  MakeResident(pool, kRequestB, 9.0f, 8);
  const std::array<int32_t, 8> original{1, 2, 3, 4, 5, 6, 7, 8};
  const std::array<int32_t, 8> branch{1, 2, 3, 4, 5, 6, 7, 9};
  auto first = blocks.AllocateBlocks(8);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     first, original, {}, pool.CapturePrefixCheckpoint(kRequestA))
                .identity,
            nullptr);
  blocks.Free(first);
  auto suffix = blocks.AllocateBlocks(8);
  bool captured = false;
  fail_staging = true;
  EXPECT_THROW(index.RegisterCheckpointedPrefix(suffix, branch, {}, [&] {
    captured = true;
    return pool.CapturePrefixCheckpoint(kRequestB);
  }),
               std::bad_alloc);
  fail_staging = false;
  EXPECT_FALSE(captured);
  EXPECT_EQ(index.Match(original, original.size()).token_count, 8u);
  EXPECT_EQ(index.Metrics().evictions, 0u);
  for (const auto& block : suffix) {
    EXPECT_FALSE(block->HasIdentity());
    EXPECT_EQ(block->RefCount(), 1u);
  }
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     suffix, branch, {}, pool.CapturePrefixCheckpoint(kRequestB))
                .identity,
            nullptr);
  EXPECT_EQ(index.Match(branch, branch.size()).blocks, suffix);
  blocks.Free(suffix);
}

TEST_F(FixedStatePoolTest, HybridBranchCapacityRefusalPreservesOldHit) {
  BlockPool blocks{4, 5};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 2;
  options.requires_checkpoint = true;
  options.max_checkpoints = 1;
  PrefixCache index{blocks, options};
  FixedStatePool pool{model_, /*capacity=*/2, /*prefix_checkpoint_capacity=*/2};
  MakeResident(pool, kRequestA, 7.0f, 8);
  MakeResident(pool, kRequestB, 9.0f, 12);
  const std::array<int32_t, 8> original{1, 2, 3, 4, 5, 6, 7, 8};
  const std::array<int32_t, 12> branch{1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12};
  auto first = blocks.AllocateBlocks(8);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     first, original, {}, pool.CapturePrefixCheckpoint(kRequestA))
                .identity,
            nullptr);
  blocks.Free(first);
  auto suffix = blocks.AllocateBlocks(12);
  bool captured = false;
  EXPECT_EQ(index.RegisterCheckpointedPrefix(suffix, branch, {}, [&] {
                   captured = true;
                   return pool.CapturePrefixCheckpoint(kRequestB);
                 })
                .status,
            PrefixCacheRegistrationStatus::CapacityRefused);
  EXPECT_FALSE(captured);
  EXPECT_EQ(index.Match(original, original.size()).token_count, 8u);
  EXPECT_EQ(index.CheckpointCount(), 1u);
  EXPECT_EQ(index.Metrics().evictions, 0u);
  for (const auto& block : suffix) {
    EXPECT_FALSE(block->HasIdentity());
    EXPECT_EQ(block->RefCount(), 1u);
  }
  blocks.Free(suffix);
}

TEST_F(FixedStatePoolTest, HybridBranchReplacementWaitsForEveryOldHistoryLease) {
  for (int lease_kind = 0; lease_kind < 3; ++lease_kind) {
    SCOPED_TRACE(lease_kind);
    BlockPool blocks{4, 4};
    PrefixCacheOptions options;
    options.enabled = true;
    options.max_blocks = 4;
    options.requires_checkpoint = true;
    options.max_checkpoints = 1;
    PrefixCache index{blocks, options};
    FixedStatePool pool{model_, /*capacity=*/2, /*prefix_checkpoint_capacity=*/2};
    MakeResident(pool, kRequestA, 7.0f, 8);
    MakeResident(pool, kRequestB, 9.0f, 8);
    const std::array<int32_t, 8> original{1, 2, 3, 4, 5, 6, 7, 8};
    const std::array<int32_t, 8> branch{1, 2, 3, 4, 5, 6, 7, 9};
    auto first = blocks.AllocateBlocks(8);
    auto registration = index.RegisterCheckpointedPrefix(
        first, original, {}, pool.CapturePrefixCheckpoint(kRequestA));
    ASSERT_NE(registration.identity, nullptr);
    PrefixCacheMatch held_match;
    std::shared_ptr<Dflash2PrefixCheckpoint> held_draft;
    if (lease_kind != 0) {
      blocks.Free(first);
    }
    if (lease_kind == 1) {
      held_match = index.Match(original, original.size());
    } else if (lease_kind == 2) {
      held_draft = std::make_shared<Dflash2PrefixCheckpoint>();
      held_draft->token_count = original.size();
      ASSERT_TRUE(index.AttachDraftCheckpoint(
          registration.identity, index.Match(original, original.size()).fixed_state_checkpoint,
          held_draft));
    }
    auto suffix = blocks.AllocateBlocks(8);
    bool captured = false;
    EXPECT_EQ(index.RegisterCheckpointedPrefix(suffix, branch, {}, [&] {
                     captured = true;
                     return pool.CapturePrefixCheckpoint(kRequestB);
                   })
                  .status,
              PrefixCacheRegistrationStatus::CapacityRefused);
    EXPECT_FALSE(captured);
    EXPECT_EQ(index.Match(original, original.size()).token_count, 8u);
    EXPECT_EQ(index.Metrics().evictions, 0u);
    if (lease_kind == 0) {
      blocks.Free(first);
    }
    held_match = {};
    held_draft.reset();
    ASSERT_NE(index.RegisterCheckpointedPrefix(
                       suffix, branch, {}, pool.CapturePrefixCheckpoint(kRequestB))
                  .identity,
              nullptr);
    EXPECT_EQ(index.Match(branch, branch.size()).blocks, suffix);
    blocks.Free(suffix);
  }
}

TEST_F(FixedStatePoolTest, HybridOrphanedHistoryCanPublishANewCheckpoint) {
  BlockPool blocks{4, 5};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 5;
  options.requires_checkpoint = true;
  options.max_checkpoints = 1;
  PrefixCache index{blocks, options};
  FixedStatePool pool{model_, /*capacity=*/2, /*prefix_checkpoint_capacity=*/1};
  MakeResident(pool, kRequestA, 7.0f, 8);
  MakeResident(pool, kRequestB, 9.0f, 12);
  const std::array<int32_t, 12> tokens{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  auto first = blocks.AllocateBlocks(8);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     first, std::span{tokens}.first(8), {}, pool.CapturePrefixCheckpoint(kRequestA))
                .identity,
            nullptr);
  blocks.Free(first);
  ASSERT_EQ(index.ReclaimCheckpoints(1), 1u);
  EXPECT_EQ(index.Match(tokens, tokens.size()).token_count, 0u);
  auto suffix = blocks.AllocateBlocks(12);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     suffix, tokens, {}, pool.CapturePrefixCheckpoint(kRequestB))
                .identity,
            nullptr);
  EXPECT_EQ(index.Match(tokens, tokens.size()).token_count, 12u);
  EXPECT_EQ(index.Match(tokens, tokens.size()).blocks, suffix);
  EXPECT_EQ(index.Metrics().hash_collisions, 0u);
  EXPECT_EQ(index.Metrics().duplicate_registrations, 0u);
  blocks.Free(suffix);
}

TEST_F(FixedStatePoolTest, HybridBranchHashCollisionPreservesCanonicalHistory) {
  BlockPool blocks{4, 4};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 4;
  options.requires_checkpoint = true;
  options.max_checkpoints = 1;
  options.hash = [](uint64_t, std::span<const int32_t> tokens) -> uint64_t {
    return static_cast<uint64_t>(tokens.front());
  };
  PrefixCache index{blocks, options};
  FixedStatePool pool{model_, /*capacity=*/1, /*prefix_checkpoint_capacity=*/1};
  MakeResident(pool, kRequestA, 7.0f, 8);
  const std::array<int32_t, 8> original{1, 2, 3, 4, 5, 6, 7, 8};
  const std::array<int32_t, 8> branch{1, 2, 3, 4, 5, 6, 7, 9};
  auto first = blocks.AllocateBlocks(8);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     first, original, {}, pool.CapturePrefixCheckpoint(kRequestA))
                .identity,
            nullptr);
  blocks.Free(first);
  auto suffix = blocks.AllocateBlocks(8);
  EXPECT_EQ(index.CheckCheckpointedPrefix(suffix, branch, {}),
            PrefixCacheRegistrationStatus::HashCollision);
  EXPECT_EQ(index.Match(original, original.size()).token_count, 8u);
  EXPECT_EQ(index.Metrics().evictions, 0u);
  blocks.Free(suffix);
}

TEST_F(FixedStatePoolTest, FailedCheckpointReplacementKeepsOldRowAndRetries) {
  BlockPool blocks{4, 2};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = 2;
  options.requires_checkpoint = true;
  options.max_checkpoints = 1;
  PrefixCache index{blocks, options};
  FixedStatePool pool{model_, /*capacity=*/2, /*prefix_checkpoint_capacity=*/1};
  MakeResident(pool, kRequestA, 7.0f, /*target_tokens=*/4);
  MakeResident(pool, kRequestB, 9.0f, /*target_tokens=*/4);
  const std::array<int32_t, 4> first_tokens{1, 2, 3, 4};
  const std::array<int32_t, 4> second_tokens{5, 6, 7, 8};
  auto first = blocks.AllocateBlocks(4);
  ASSERT_NE(index.RegisterCheckpointedPrefix(
                     first, first_tokens, {}, pool.CapturePrefixCheckpoint(kRequestA))
                .identity,
            nullptr);
  const auto* replacement = index.ReclaimableCheckpoint();
  ASSERT_NE(replacement, nullptr);
  EXPECT_THROW(pool.CapturePrefixCheckpoint(kRequestB, replacement, [] {
    throw std::bad_alloc{};
  }),
               std::bad_alloc);
  EXPECT_EQ(index.Match(first_tokens, first_tokens.size()).fixed_state_checkpoint.get(), replacement);
  EXPECT_EQ(pool.AvailablePrefixCheckpoints(), 0u);
  EXPECT_EQ(index.CheckpointCount(), 1u);

  auto second = blocks.AllocateBlocks(4);
  auto checkpoint = pool.CapturePrefixCheckpoint(kRequestB, replacement, [&] {
    EXPECT_EQ(index.ReclaimCheckpoints(1), 1u);
  });
  ASSERT_NE(checkpoint, nullptr);
  ASSERT_NE(index.RegisterCheckpointedPrefix(second, second_tokens, {}, checkpoint).identity, nullptr);
  EXPECT_EQ(index.Match(second_tokens, second_tokens.size()).token_count, 4u);
  blocks.Free(first);
  blocks.Free(second);
}

TEST_F(FixedStatePoolTest, PrefixCheckpointMustBelongToTheAdoptingPool) {
  FixedStatePool source_pool{model_, /*capacity=*/1,
                             /*prefix_checkpoint_capacity=*/1};
  FixedStatePool destination_pool{model_, /*capacity=*/1,
                                  /*prefix_checkpoint_capacity=*/1};
  MakeResident(source_pool, kRequestA, 3.0f, /*target_tokens=*/4);
  auto checkpoint = source_pool.CapturePrefixCheckpoint(kRequestA);
  const std::array<FixedStateReservationRequest, 1> requests{
      FixedStateReservationRequest{
          kRequestB, /*target_tokens=*/5, /*capture_count=*/0, checkpoint}};

  EXPECT_THROW(destination_pool.Reserve(requests), std::runtime_error);
}

TEST_F(FixedStatePoolTest, ValidateReleaseIsPureAndPublicationIsNoexcept) {
  auto pool = MakePool(1);
  const auto handle = MakeResident(*pool, kRequestA, 1.0f);

  pool->ValidateRelease(handle);

  EXPECT_TRUE(pool->OwnsCommittedSlot(kRequestA));
  EXPECT_EQ(pool->AvailableSlots(), 0u);
  static_assert(noexcept(pool->ReleaseValidated(handle)));
  pool->ReleaseValidated(handle);
  EXPECT_FALSE(pool->OwnsCommittedSlot(kRequestA));
  EXPECT_EQ(pool->AvailableSlots(), 1u);
}

TEST_F(FixedStatePoolTest, ReleaseValidatedMisuseFailsFast) {
  auto pool = MakePool(1);
  auto other_pool = MakePool(1);
  const auto handle = MakeResident(*pool, kRequestA, 1.0f);
  auto out_of_range = handle;
  out_of_range.slot = pool->Capacity();
  auto wrong_pool = handle;
  wrong_pool.pool = other_pool.get();
  auto wrong_request = handle;
  wrong_request.request_id = kRequestB;
  auto stale_generation = handle;
  ++stale_generation.generation;

  EXPECT_DEATH_IF_SUPPORTED(pool->ReleaseValidated(out_of_range), "");
  EXPECT_DEATH_IF_SUPPORTED(pool->ReleaseValidated(wrong_pool), "");
  EXPECT_DEATH_IF_SUPPORTED(pool->ReleaseValidated(wrong_request), "");
  EXPECT_DEATH_IF_SUPPORTED(pool->ReleaseValidated(stale_generation), "");

  auto requests = One(kRequestA, 2);
  auto reservation = pool->Reserve(requests);
  EXPECT_DEATH_IF_SUPPORTED(pool->ReleaseValidated(handle), "");
}

TEST_F(FixedStatePoolTest, StagedOutputsBecomeVisibleOnlyAfterCommit) {
  auto pool = MakePool(1);
  MakeResident(*pool, kRequestA, 4.0f);
  {
    auto requests = One(kRequestA);
    auto reservation = pool->Reserve(requests);
    ASSERT_TRUE(reservation.UsesDirectBindings());
    for (const auto& binding : reservation.Bindings()) {
      ExpectInputRow(binding, 0, 4.0f);
      FillStagedRow(binding, 0, 9.0f);
      ExpectInputRow(binding, 0, 4.0f);  // Staging the output does not touch committed state.
    }
    reservation.Discard();
  }

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  ExpectInputRows(reservation, 0, 4.0f);  // Discard left the committed state unchanged.
}

TEST_F(FixedStatePoolTest, PreparedOutputIsInvisibleUntilPublish) {
  auto pool = MakePool(1);
  MakeResident(*pool, kRequestA, 4.0f, /*target_tokens=*/2);
  {
    auto requests = One(kRequestA, /*target_tokens=*/5);
    auto reservation = pool->Reserve(requests);
    FillStagedRows(reservation, 0, 9.0f);
    reservation.ValidateCommit();
    reservation.PrepareCommit();  // Device copy into the inactive bank only.
    EXPECT_EQ(reservation.State(), FixedStateReservationState::Prepared);
    // Prepare staged into the inactive bank; the visible committed state is untouched.
    EXPECT_EQ(pool->CommittedTokens(pool->HandleFor(kRequestA)), 2u);
    reservation.Discard();  // Discarding a prepared reservation preserves active state.
  }

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  ExpectInputRows(reservation, 0, 4.0f);  // The prepared-but-unpublished 9.0 never became visible.
  EXPECT_EQ(pool->CommittedTokens(pool->HandleFor(kRequestA)), 2u);
}

TEST_F(FixedStatePoolTest, ThreePhaseCommitPublishesStateAndTokens) {
  auto pool = MakePool(1);
  {
    auto requests = One(kRequestA, /*target_tokens=*/3);
    auto reservation = pool->Reserve(requests);
    FillStagedRows(reservation, 0, 7.0f);
    reservation.ValidateCommit();
    reservation.PrepareCommit();
    EXPECT_THROW(pool->HandleFor(kRequestA), std::runtime_error);  // Not committed until publish.
    reservation.PublishCommit();
    EXPECT_EQ(reservation.State(), FixedStateReservationState::Committed);
  }
  const auto handle = pool->HandleFor(kRequestA);
  EXPECT_EQ(pool->StateGeneration(handle), 1u);
  EXPECT_EQ(pool->CommittedTokens(handle), 3u);

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  ExpectInputRows(reservation, 0, 7.0f);
}

TEST_F(FixedStatePoolTest, RepeatedCommitsAdvanceGenerationAndTokens) {
  auto pool = MakePool(1);
  MakeResident(*pool, kRequestA, 1.0f, /*target_tokens=*/4);
  EXPECT_EQ(pool->CommittedTokens(pool->HandleFor(kRequestA)), 4u);

  {
    auto requests = One(kRequestA, /*target_tokens=*/9);
    auto reservation = pool->Reserve(requests);
    ExpectInputRows(reservation, 0, 1.0f);  // Gathered the first committed value.
    FillStagedRows(reservation, 0, 2.0f);
    reservation.Commit();
  }
  const auto handle = pool->HandleFor(kRequestA);
  EXPECT_EQ(pool->StateGeneration(handle), 2u);
  EXPECT_EQ(pool->CommittedTokens(handle), 9u);

  // The second commit must have landed in the other bank; gather now sees the new value.
  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  ExpectInputRows(reservation, 0, 2.0f);
}

TEST_F(FixedStatePoolTest, ResidentRowsBindDirectlyAndAlternateBanks) {
  auto pool = MakePool(2);
  MakeResident(*pool, kRequestA, 11.0f);
  MakeResident(*pool, kRequestB, 22.0f);

  const std::array<Request, 2> requests{Request{kRequestA, 2}, Request{kRequestB, 2}};
  void* first_input{};
  void* first_output{};
  {
    auto reservation = pool->Reserve(requests);
    ASSERT_TRUE(reservation.UsesDirectBindings());
    ASSERT_EQ(reservation.Bindings().size(), 4u);
    for (const auto& binding : reservation.Bindings()) {
      ExpectInputRow(binding, 0, 11.0f);
      ExpectInputRow(binding, 1, 22.0f);
    }
    first_input = reservation.Bindings()[0].input->GetTensorMutableData<void>();
    first_output = reservation.Bindings()[0].output->GetTensorMutableData<void>();
    EXPECT_NE(first_input, first_output);
    FillStagedRows(reservation, 0, 33.0f);
    FillStagedRows(reservation, 1, 44.0f);
    reservation.Commit();
  }

  auto reservation = pool->Reserve(requests);
  ASSERT_TRUE(reservation.UsesDirectBindings());
  EXPECT_EQ(reservation.Bindings()[0].input->GetTensorMutableData<void>(), first_output);
  EXPECT_EQ(reservation.Bindings()[0].output->GetTensorMutableData<void>(), first_input);
  ExpectInputRows(reservation, 0, 33.0f);
  ExpectInputRows(reservation, 1, 44.0f);
}

TEST_F(FixedStatePoolTest, DirectBindingsFallBackForUnsupportedRowLayouts) {
  auto pool = MakePool(3);
  MakeResident(*pool, kRequestA, 11.0f);
  MakeResident(*pool, kRequestB, 22.0f);

  {
    const std::array<Request, 2> reordered{Request{kRequestB, 2}, Request{kRequestA, 2}};
    auto reservation = pool->Reserve(reordered);
    EXPECT_FALSE(reservation.UsesDirectBindings());
    reservation.Discard();
  }
  {
    const std::array<Request, 2> with_admission{Request{kRequestA, 2}, Request{kRequestC, 1}};
    auto reservation = pool->Reserve(with_admission);
    EXPECT_FALSE(reservation.UsesDirectBindings());
    reservation.Discard();
  }
  {
    auto reservation = pool->Reserve(One(kRequestA, 2));
    ASSERT_TRUE(reservation.UsesDirectBindings());
    FillStagedRows(reservation, 0, 33.0f);
    reservation.Commit();
  }
  {
    const std::array<Request, 2> mixed_banks{Request{kRequestA, 3}, Request{kRequestB, 2}};
    auto reservation = pool->Reserve(mixed_banks);
    ASSERT_TRUE(reservation.UsesDirectBindings());
    ExpectInputRows(reservation, 0, 33.0f);
    ExpectInputRows(reservation, 1, 22.0f);
  }
}

TEST_F(FixedStatePoolTest, DirectBindingsSupportContiguousRowsAtNonzeroOffset) {
  auto pool = MakePool(3);
  const auto handle_a = MakeResident(*pool, kRequestA, 11.0f);
  MakeResident(*pool, kRequestB, 22.0f);
  MakeResident(*pool, kRequestC, 33.0f);
  pool->Release(handle_a);

  const std::array<Request, 2> requests{Request{kRequestB, 2}, Request{kRequestC, 2}};
  auto reservation = pool->Reserve(requests);
  ASSERT_TRUE(reservation.UsesDirectBindings());
  ASSERT_EQ(reservation.Handles()[0].slot, 1u);
  ASSERT_EQ(reservation.Handles()[1].slot, 2u);
  ExpectInputRows(reservation, 0, 22.0f);
  ExpectInputRows(reservation, 1, 33.0f);
}

TEST_F(FixedStatePoolTest, UnsupportedOffsetViewsUseEquivalentStagingBindings) {
  auto run_scenario = [this](FixedStatePool& pool, bool expect_direct) {
    const auto handle_a = MakeResident(pool, kRequestA, 11.0f);
    MakeResident(pool, kRequestB, 22.0f);
    MakeResident(pool, kRequestC, 33.0f);
    pool.Release(handle_a);

    const std::array<Request, 2> requests{
        Request{kRequestB, 2}, Request{kRequestC, 2}};
    {
      auto reservation = pool.Reserve(requests);
      EXPECT_EQ(reservation.UsesDirectBindings(), expect_direct);
      ExpectInputRows(reservation, 0, 22.0f);
      ExpectInputRows(reservation, 1, 33.0f);
      FillStagedRows(reservation, 0, 44.0f);
      FillStagedRows(reservation, 1, 55.0f);
      reservation.Commit();
    }

    auto reservation = pool.Reserve(requests);
    EXPECT_EQ(reservation.UsesDirectBindings(), expect_direct);
    ExpectInputRows(reservation, 0, 44.0f);
    ExpectInputRows(reservation, 1, 55.0f);
  };

  {
    auto direct_pool = MakePool(3);
    run_scenario(*direct_pool, true);
  }

  auto fallback_model = LoadSyntheticHybridModel();
  FixedStateTestDevice fallback_device{
      *fallback_model->p_device_kvcache_};
  ScopedKeyValueCacheDevice scoped_device{*fallback_model, fallback_device};
  FixedStatePool fallback_pool{fallback_model, 3};
  run_scenario(fallback_pool, false);
}

TEST_F(FixedStatePoolTest, GenericDeviceSupportsOrdinaryStagingWithoutStateUpdates) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-hybrid");
  ASSERT_TRUE(config->model.decoder.state_groups.has_value());
  for (auto& group : *config->model.decoder.state_groups) {
    group.state_update.reset();
  }
  auto generic_model = CreateModel(GetOrtEnv(), std::move(config));
  FixedStateTestDevice generic_device{
      *generic_model->p_device_kvcache_, DeviceType::DML};
  ScopedKeyValueCacheDevice scoped_device{*generic_model, generic_device};
  FixedStatePool pool{generic_model, 2};

  EXPECT_FALSE(pool.SupportsStateUpdates());
  EXPECT_EQ(pool.StateUpdateCapacity(), 0u);
  const std::array<Request, 2> requests{
      Request{kRequestA, 1}, Request{kRequestB, 1}};
  {
    auto reservation = pool.Reserve(requests);
    EXPECT_FALSE(reservation.UsesDirectBindings());
    ExpectInputRows(reservation, 0, 0.0f);
    ExpectInputRows(reservation, 1, 0.0f);
    FillStagedRows(reservation, 0, 44.0f);
    FillStagedRows(reservation, 1, 55.0f);
    reservation.Commit();
  }

  auto reservation = pool.Reserve(requests);
  EXPECT_FALSE(reservation.UsesDirectBindings());
  ExpectInputRows(reservation, 0, 44.0f);
  ExpectInputRows(reservation, 1, 55.0f);
}

TEST_F(FixedStatePoolTest, RejectsGenericDeviceWithoutTransactionalFixedStateSupport) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-hybrid");
  ASSERT_TRUE(config->model.decoder.state_groups.has_value());
  for (auto& group : *config->model.decoder.state_groups) {
    group.state_update.reset();
  }
  auto generic_model = CreateModel(GetOrtEnv(), std::move(config));
  FixedStateTestDevice generic_device{
      *generic_model->p_device_kvcache_, DeviceType::DML, false};
  ScopedKeyValueCacheDevice scoped_device{*generic_model, generic_device};

  try {
    FixedStatePool pool{generic_model, 2};
    FAIL() << "Expected unqualified fixed-state device to be rejected.";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(
        error.what(),
        "Fixed state pools require qualified transactional device semantics.");
  }
}

TEST_F(FixedStatePoolTest, GenericDeviceRejectsCompactStateReplay) {
  auto replay_model = LoadSyntheticHybridModel();
  FixedStateTestDevice generic_device{
      *replay_model->p_device_kvcache_, DeviceType::DML, true, false, false};
  ScopedKeyValueCacheDevice scoped_device{*replay_model, generic_device};

  try {
    FixedStatePool pool{replay_model, 2};
    FAIL() << "Expected compact replay configuration to be rejected.";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(
        error.what(),
        "Compact fixed state replay is not supported by this device.");
  }
}

TEST_F(FixedStatePoolTest, AdmissionAlignsReusedSlotWithResidentBank) {
  auto pool = MakePool(2);
  const auto handle_a = MakeResident(*pool, kRequestA, 11.0f);
  MakeResident(*pool, kRequestB, 22.0f);
  pool->Release(handle_a);
  {
    auto reservation = pool->Reserve(One(kRequestB, 2));
    ASSERT_TRUE(reservation.UsesDirectBindings());
    FillStagedRows(reservation, 0, 23.0f);
    reservation.Commit();
  }
  {
    const std::array<Request, 2> admission{Request{kRequestC, 1}, Request{kRequestB, 3}};
    auto reservation = pool->Reserve(admission);
    ASSERT_FALSE(reservation.UsesDirectBindings());
    FillStagedRows(reservation, 0, 33.0f);
    FillStagedRows(reservation, 1, 24.0f);
    reservation.Commit();
  }

  const std::array<Request, 2> resident{Request{kRequestC, 2}, Request{kRequestB, 4}};
  auto reservation = pool->Reserve(resident);
  ASSERT_TRUE(reservation.UsesDirectBindings());
  ExpectInputRows(reservation, 0, 33.0f);
  ExpectInputRows(reservation, 1, 24.0f);
}

TEST_F(FixedStatePoolTest, RejectsCommitThatRegressesCommittedTokens) {
  auto pool = MakePool(1);
  MakeResident(*pool, kRequestA, 1.0f, /*target_tokens=*/10);

  auto requests = One(kRequestA, /*target_tokens=*/3);  // 3 < committed 10.
  auto reservation = pool->Reserve(requests);
  FillStagedRows(reservation, 0, 2.0f);
  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);
  EXPECT_THROW(reservation.PrepareCommit(), std::logic_error);
  reservation.Discard();

  // The rejected commit left the resident state and tokens untouched.
  const auto handle = pool->HandleFor(kRequestA);
  EXPECT_EQ(pool->CommittedTokens(handle), 10u);
  EXPECT_EQ(pool->StateGeneration(handle), 1u);
  EXPECT_TRUE(pool->Snapshot().healthy);
}

TEST_F(FixedStatePoolTest, GathersCommittedSlotsInScheduledRowOrder) {
  auto pool = MakePool(2);
  MakeResident(*pool, kRequestA, 11.0f);
  MakeResident(*pool, kRequestB, 22.0f);

  const std::array<Request, 2> reordered{Request{kRequestB, 1}, Request{kRequestA, 1}};
  auto reservation = pool->Reserve(reordered);
  ASSERT_EQ(reservation.Handles()[0].request_id, kRequestB);
  ASSERT_EQ(reservation.Handles()[1].request_id, kRequestA);
  for (const auto& binding : reservation.Bindings()) {
    ExpectInputRow(binding, 0, 22.0f);
    ExpectInputRow(binding, 1, 11.0f);
  }
}

TEST_F(FixedStatePoolTest, NewSlotOwnershipPublishesAtCommit) {
  auto pool = MakePool(1);
  FixedStateSlotHandle reserved_handle;
  {
    auto requests = One(kRequestA);
    auto reservation = pool->Reserve(requests);
    reserved_handle = reservation.Handles()[0];
    EXPECT_THROW(pool->HandleFor(kRequestA), std::runtime_error);  // Not yet committed.
    EXPECT_EQ(pool->Snapshot().reserved_slots, 1u);
    ExpectInputRows(reservation, 0, 0.0f);
    FillStagedRows(reservation, 0, 5.0f);
    reservation.Commit();
    EXPECT_EQ(pool->HandleFor(kRequestA), reserved_handle);
    EXPECT_EQ(pool->StateGeneration(reserved_handle), 1u);
  }

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  ExpectInputRows(reservation, 0, 5.0f);
}

TEST_F(FixedStatePoolTest, DiscardReleasesProvisionalSlotWithoutPublishing) {
  auto pool = MakePool(1);
  FixedStateSlotHandle discarded_handle;
  {
    auto requests = One(kRequestA);
    auto reservation = pool->Reserve(requests);
    discarded_handle = reservation.Handles()[0];
    FillStagedRows(reservation, 0, 13.0f);
    reservation.Discard();
  }

  EXPECT_THROW(pool->HandleFor(kRequestA), std::runtime_error);
  EXPECT_EQ(pool->AvailableSlots(), 1u);

  auto requests = One(kRequestB);
  auto replacement = pool->Reserve(requests);
  EXPECT_EQ(replacement.Handles()[0].slot, discarded_handle.slot);
  EXPECT_GT(replacement.Handles()[0].generation, discarded_handle.generation);
}

TEST_F(FixedStatePoolTest, DestructorDiscardsUncommittedReservation) {
  auto pool = MakePool(1);
  {
    auto requests = One(kRequestA);
    auto reservation = pool->Reserve(requests);
    EXPECT_EQ(pool->Snapshot().reserved_slots, 1u);
    // Reservation leaves scope without Commit/Discard: the destructor must free the provisional slot.
  }
  const auto snapshot = pool->Snapshot();
  EXPECT_EQ(snapshot.reserved_slots, 0u);
  EXPECT_EQ(snapshot.free_slots, 1u);
  EXPECT_THROW(pool->HandleFor(kRequestA), std::runtime_error);
}

TEST_F(FixedStatePoolTest, DestructorDiscardsPreparedReservation) {
  auto pool = MakePool(1);
  MakeResident(*pool, kRequestA, 4.0f, /*target_tokens=*/2);
  {
    auto requests = One(kRequestA, /*target_tokens=*/6);
    auto reservation = pool->Reserve(requests);
    FillStagedRows(reservation, 0, 8.0f);
    reservation.PrepareCommit();
    EXPECT_EQ(reservation.State(), FixedStateReservationState::Prepared);
    // Prepared reservation leaves scope without PublishCommit: the destructor discards it, and the
    // active committed state and tokens are preserved.
  }
  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  ExpectInputRows(reservation, 0, 4.0f);
  EXPECT_EQ(pool->CommittedTokens(pool->HandleFor(kRequestA)), 2u);
}

TEST_F(FixedStatePoolTest, MoveTransfersLiveReservationOwnership) {
  auto pool = MakePool(1);
  auto requests = One(kRequestA);
  auto original = pool->Reserve(requests);
  auto moved = std::move(original);

  // The moved-from reservation is inert; the moved-to reservation still governs the live slot.
  EXPECT_EQ(moved.State(), FixedStateReservationState::Reserved);
  ASSERT_EQ(moved.Handles().size(), 1u);
  FillStagedRows(moved, 0, 6.0f);
  moved.Commit();
  EXPECT_EQ(pool->StateGeneration(pool->HandleFor(kRequestA)), 1u);
}

TEST_F(FixedStatePoolTest, ReportsPersistentStagingAndReleaseAccounting) {
  auto pool = MakePool(3);
  constexpr size_t bytes_per_request =
      2 * (2 * 3) * sizeof(float) +     // convolution: 2 layers, row [2, 3]
      2 * (2 * 2 * 2) * sizeof(float);  // recurrent: 2 layers, row [2, 2, 2]
  constexpr size_t state_update_bytes_per_request =
      2 * (3 * 2) * sizeof(float) +               // convolution: 2 layers, row [3, 2]
      2 * (3 * (2 + 2 + 2 * 2)) * sizeof(float);  // recurrent: 2 layers, row [24]
  constexpr size_t persistent_state_update_control_bytes =
      3 * sizeof(int32_t) + sizeof(int32_t);  // capacity-sized counts plus one active flag
  constexpr size_t state_update_control_bytes = 2 * sizeof(int32_t) + sizeof(int32_t);
  // Two state banks, capacity-sized state/output/update staging, and update controls.
  EXPECT_EQ(pool->PersistentBytes(),
            4 * 3 * bytes_per_request +
                3 * state_update_bytes_per_request +
                persistent_state_update_control_bytes);
  EXPECT_EQ(pool->ZeroingScratchBytes(), bytes_per_request);
  EXPECT_EQ(pool->ActiveStagingBytes(), 0u);

  const auto handle_a = MakeResident(*pool, kRequestA, 1.0f);
  {
    const std::array<Request, 2> requests{Request{kRequestA, 1}, Request{kRequestB, 1}};  // A resident, B provisional.
    auto reservation = pool->Reserve(requests);
    EXPECT_EQ(reservation.PlannedStagingBytes(),
              4 * bytes_per_request + state_update_control_bytes);
    EXPECT_EQ(pool->ActiveStagingBytes(),
              4 * bytes_per_request + state_update_control_bytes);
    const auto snapshot = pool->Snapshot();
    EXPECT_EQ(snapshot.free_slots, 1u);
    EXPECT_EQ(snapshot.reserved_slots, 1u);
    EXPECT_EQ(snapshot.committed_slots, 1u);
    reservation.Discard();
  }
  EXPECT_EQ(pool->ActiveStagingBytes(), 0u);
  EXPECT_EQ(pool->AvailableSlots(), 2u);

  pool->Release(handle_a);
  const auto snapshot = pool->Snapshot();
  EXPECT_EQ(snapshot.free_slots, 3u);
  EXPECT_EQ(snapshot.reserved_slots, 0u);
  EXPECT_EQ(snapshot.committed_slots, 0u);
}

TEST_F(FixedStatePoolTest, BindsCompactOutputsOnlyWhenCaptureIsRequested) {
  auto pool = MakePool(1);
  EXPECT_TRUE(pool->SupportsStateUpdates());
  EXPECT_EQ(pool->StateUpdateCapacity(), 3u);

  {
    auto reservation = pool->Reserve(One(kRequestA));
    EXPECT_FALSE(reservation.CapturesStateUpdates());
    for (const auto& binding : reservation.Bindings()) {
      EXPECT_STREQ(binding.state_update_capture_count_name, "state_update_capture_count");
      ASSERT_NE(binding.state_update_capture_count, nullptr);
      EXPECT_EQ(binding.state_update_capture_count->GetTensorData<int32_t>()[0], 0);
      EXPECT_STREQ(binding.state_update_active_name, "state_update_active");
      ASSERT_NE(binding.state_update_active, nullptr);
      EXPECT_EQ(binding.state_update_active->GetTensorData<int32_t>()[0], 0);
      EXPECT_EQ(binding.state_update_value, nullptr);
      EXPECT_EQ(binding.state_update_capsule, nullptr);
    }
    reservation.Discard();
  }

  auto reservation = pool->Reserve(One(kRequestA, 4, 3));
  EXPECT_TRUE(reservation.CapturesStateUpdates());
  EXPECT_EQ(reservation.Bindings()[0].state_update_capture_count->GetTensorData<int32_t>()[0], 3);
  EXPECT_EQ(reservation.Bindings()[0].state_update_active->GetTensorData<int32_t>()[0], 1);
  EXPECT_NE(reservation.Bindings()[0].state_update_value, nullptr);
  EXPECT_NE(reservation.Bindings()[2].state_update_capsule, nullptr);
}

TEST_F(FixedStatePoolTest, CompactPartialAcceptanceReplaysConvAndGdn) {
  auto pool = MakePool(1);
  MakeResident(*pool, kRequestA, 4.0f);
  {
    auto reservation = pool->Reserve(One(kRequestA, 4, 3));
    ASSERT_TRUE(reservation.UsesDirectBindings());
    FillStagedRows(reservation, 0, 99.0f);
    const std::array<float, 6> conv_values{10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f};
    const std::array<float, 6> decay{0.5f, 0.25f, 1.0f, 0.5f, 1.0f, 1.0f};
    const std::array<float, 6> key{2.0f, 3.0f, 4.0f, 5.0f, 1.0f, 1.0f};
    const std::array<float, 12> delta{
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (const auto& binding : reservation.Bindings()) {
      if (binding.state_update_kind ==
          Config::Model::Decoder::StateUpdateKind::CausalConv) {
        FillConvUpdates(binding, 0, conv_values);
      } else {
        FillGdnUpdates(binding, 0, decay, key, delta);
      }
    }
    reservation.CommitPrefix(0, 4, 2);
    reservation.Commit();
  }

  EXPECT_EQ(pool->CommittedTokens(pool->HandleFor(kRequestA)), 2u);
  auto reservation = pool->Reserve(One(kRequestA, 2));
  const std::array<float, 6> expected_conv{4.0f, 10.0f, 20.0f, 4.0f, 11.0f, 21.0f};
  const std::array<float, 8> expected_gdn{
      24.0f, 30.0f, 30.0f, 38.0f, 31.5f, 40.0f, 36.5f, 46.5f};
  ExpectInputRow(reservation.Bindings()[0], 0, expected_conv);
  ExpectInputRow(reservation.Bindings()[1], 0, expected_conv);
  ExpectInputRow(reservation.Bindings()[2], 0, expected_gdn);
  ExpectInputRow(reservation.Bindings()[3], 0, expected_gdn);
}

// The compact replay of a partial commit is launched by the next pool operation, so a checkpoint
// captured right after the commit must still observe the replayed state.
TEST_F(FixedStatePoolTest, PrefixCheckpointAfterPartialAcceptanceSeesReplayedState) {
  FixedStatePool pool{model_, /*capacity=*/2, /*prefix_checkpoint_capacity=*/1};
  MakeResident(pool, kRequestA, 4.0f);
  {
    auto reservation = pool.Reserve(One(kRequestA, 4, 3));
    FillStagedRows(reservation, 0, 99.0f);
    const std::array<float, 6> conv_values{10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f};
    const std::array<float, 6> decay{0.5f, 0.25f, 1.0f, 0.5f, 1.0f, 1.0f};
    const std::array<float, 6> key{2.0f, 3.0f, 4.0f, 5.0f, 1.0f, 1.0f};
    const std::array<float, 12> delta{
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (const auto& binding : reservation.Bindings()) {
      if (binding.state_update_kind ==
          Config::Model::Decoder::StateUpdateKind::CausalConv) {
        FillConvUpdates(binding, 0, conv_values);
      } else {
        FillGdnUpdates(binding, 0, decay, key, delta);
      }
    }
    reservation.CommitPrefix(0, 4, 2);
    reservation.Commit();
  }

  auto checkpoint = pool.CapturePrefixCheckpoint(kRequestA);
  ASSERT_NE(checkpoint, nullptr);
  const std::array<FixedStateReservationRequest, 1> requests{
      FixedStateReservationRequest{
          kRequestB, /*target_tokens=*/3, /*capture_count=*/0, checkpoint}};
  auto reservation = pool.Reserve(requests);
  const std::array<float, 6> expected_conv{4.0f, 10.0f, 20.0f, 4.0f, 11.0f, 21.0f};
  const std::array<float, 8> expected_gdn{
      24.0f, 30.0f, 30.0f, 38.0f, 31.5f, 40.0f, 36.5f, 46.5f};
  ExpectInputRow(reservation.Bindings()[0], 0, expected_conv);
  ExpectInputRow(reservation.Bindings()[2], 0, expected_gdn);
}

// Discarding a prepared reservation drops the replay it deferred: the published state is untouched.
TEST_F(FixedStatePoolTest, DiscardedPartialAcceptanceLeavesStateUnchanged) {
  auto pool = MakePool(1);
  MakeResident(*pool, kRequestA, 4.0f);
  {
    auto reservation = pool->Reserve(One(kRequestA, 4, 3));
    FillStagedRows(reservation, 0, 99.0f);
    const std::array<float, 6> conv_values{10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f};
    const std::array<float, 6> decay{0.5f, 0.25f, 1.0f, 0.5f, 1.0f, 1.0f};
    const std::array<float, 6> key{2.0f, 3.0f, 4.0f, 5.0f, 1.0f, 1.0f};
    const std::array<float, 12> delta{
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (const auto& binding : reservation.Bindings()) {
      if (binding.state_update_kind ==
          Config::Model::Decoder::StateUpdateKind::CausalConv) {
        FillConvUpdates(binding, 0, conv_values);
      } else {
        FillGdnUpdates(binding, 0, decay, key, delta);
      }
    }
    reservation.CommitPrefix(0, 4, 2);
    reservation.PrepareCommit();
    reservation.Discard();
  }

  auto reservation = pool->Reserve(One(kRequestA, 2));
  ExpectInputRows(reservation, 0, 4.0f);
}

TEST_F(FixedStatePoolTest, DirectBindingsMixPartialAndFullAcceptance) {
  auto pool = MakePool(2);
  MakeResident(*pool, kRequestA, 4.0f);
  MakeResident(*pool, kRequestB, 5.0f);
  {
    auto reservation = pool->Reserve(One(kRequestA, 2));
    ASSERT_TRUE(reservation.UsesDirectBindings());
    FillStagedRows(reservation, 0, 4.0f);
    reservation.Commit();
  }

  const std::array<Request, 2> requests{
      Request{kRequestA, 4, 3}, Request{kRequestB, 4, 3}};
  {
    auto reservation = pool->Reserve(requests);
    ASSERT_TRUE(reservation.UsesDirectBindings());
    FillStagedRows(reservation, 0, 99.0f);
    FillStagedRows(reservation, 1, 77.0f);
    const std::array<float, 6> conv_values{10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f};
    const std::array<float, 6> decay{0.5f, 0.25f, 1.0f, 0.5f, 1.0f, 1.0f};
    const std::array<float, 6> key{2.0f, 3.0f, 4.0f, 5.0f, 1.0f, 1.0f};
    const std::array<float, 12> delta{
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (const auto& binding : reservation.Bindings()) {
      if (binding.state_update_kind ==
          Config::Model::Decoder::StateUpdateKind::CausalConv) {
        FillConvUpdates(binding, 0, conv_values);
      } else {
        FillGdnUpdates(binding, 0, decay, key, delta);
      }
    }
    reservation.CommitPrefix(0, 4, 2);
    reservation.CommitPrefix(1, 4, 4);
    reservation.Commit();
  }

  EXPECT_EQ(pool->CommittedTokens(pool->HandleFor(kRequestA)), 2u);
  EXPECT_EQ(pool->CommittedTokens(pool->HandleFor(kRequestB)), 4u);
  const std::array<Request, 2> committed{
      Request{kRequestA, 2}, Request{kRequestB, 4}};
  auto reservation = pool->Reserve(committed);
  ASSERT_TRUE(reservation.UsesDirectBindings());
  const std::array<float, 6> expected_conv{4.0f, 10.0f, 20.0f, 4.0f, 11.0f, 21.0f};
  const std::array<float, 8> expected_gdn{
      24.0f, 30.0f, 30.0f, 38.0f, 31.5f, 40.0f, 36.5f, 46.5f};
  ExpectInputRow(reservation.Bindings()[0], 0, expected_conv);
  ExpectInputRow(reservation.Bindings()[2], 0, expected_gdn);
  ExpectInputRows(reservation, 1, 77.0f);
}

TEST_F(FixedStatePoolTest, DirectBindingStorageOutlivesPool) {
  std::optional<FixedStateReservation> reservation;
  {
    auto pool = MakePool(1);
    MakeResident(*pool, kRequestA, 4.0f);
    reservation.emplace(pool->Reserve(One(kRequestA, 2)));
    ASSERT_TRUE(reservation->UsesDirectBindings());
    ExpectInputRows(*reservation, 0, 4.0f);
    FillStagedRows(*reservation, 0, 7.0f);
  }

  EXPECT_EQ(reservation->State(), FixedStateReservationState::Failed);
  ExpectInputRows(*reservation, 0, 4.0f);
  for (const auto& binding : reservation->Bindings()) {
    EXPECT_FLOAT_EQ(binding.output->GetTensorData<float>()[0], 7.0f);
  }
}

#if USE_CUDA
template <typename T>
void ExpectIndexerReplayMatchesCpu(DeviceInterface& device) {
  constexpr size_t width = 3;
  constexpr size_t key_capacity = 8;
  constexpr size_t buffer_capacity = 3;
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  for (int32_t old_buffer_length = 0; old_buffer_length < 4; ++old_buffer_length) {
    for (uint32_t kept = 1; kept <= 7; ++kept) {
      for (bool in_place : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "bytes " << sizeof(T) << " buffer "
                                          << old_buffer_length << " kept " << kept
                                          << " in_place " << in_place);
        std::vector<T> source(key_capacity * width), buffer(buffer_capacity * width), values(7 * width);
        for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<T>(100 + i);
        for (size_t i = 0; i < buffer.size(); ++i) buffer[i] = static_cast<T>(200 + i);
        for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<T>(300 + i);
        std::array<int32_t, 2> lengths{2, old_buffer_length}, expected_lengths{};
        auto expected = source;
        auto expected_buffer = buffer;
        StateUpdateReplayDesc descriptor{};
        descriptor.kind = StateUpdateReplayKind::Indexer;
        descriptor.source_state = HostReplayBytes(in_place ? expected : source);
        descriptor.destination_state = HostReplayBytes(expected);
        descriptor.source_aux_state = HostReplayBytes(in_place ? expected_buffer : buffer);
        descriptor.destination_aux_state = HostReplayBytes(expected_buffer);
        descriptor.source_lengths = HostReplayBytes(lengths);
        descriptor.destination_lengths = HostReplayBytes(expected_lengths);
        descriptor.value = HostReplayBytes(values);
        descriptor.state_width = width;
        descriptor.state_capacity = key_capacity;
        descriptor.aux_capacity = buffer_capacity;
        descriptor.compress_ratio = 4;
        descriptor.capacity = 7;
        descriptor.kept_count = kept;
        descriptor.element_size = sizeof(T);
        cpu->ReplayStateUpdates(&descriptor, 1);

        auto gpu_source = device.Allocate<T>(source.size());
        auto gpu_buffer = device.Allocate<T>(buffer.size());
        auto gpu_values = device.Allocate<T>(values.size());
        auto gpu_destination = in_place ? gpu_source : device.Allocate<T>(source.size());
        auto gpu_destination_buffer = in_place ? gpu_buffer : device.Allocate<T>(buffer.size());
        auto gpu_lengths = device.Allocate<int32_t>(2);
        auto gpu_destination_lengths = device.Allocate<int32_t>(2);
        gpu_source.CopyFromCpu(source);
        gpu_buffer.CopyFromCpu(buffer);
        gpu_values.CopyFromCpu(values);
        gpu_lengths.CopyFromCpu(lengths);
        descriptor.source_state = ReplayBytes(gpu_source);
        descriptor.destination_state = ReplayBytes(gpu_destination);
        descriptor.source_aux_state = ReplayBytes(gpu_buffer);
        descriptor.destination_aux_state = ReplayBytes(gpu_destination_buffer);
        descriptor.source_lengths = ReplayBytes(gpu_lengths);
        descriptor.destination_lengths = ReplayBytes(gpu_destination_lengths);
        descriptor.value = ReplayBytes(gpu_values);
        device.ReplayStateUpdates(&descriptor, 1);
        device.Synchronize();
        const auto actual_lengths = gpu_destination_lengths.CopyDeviceToCpu();
        ASSERT_EQ(actual_lengths[0], expected_lengths[0]);
        ASSERT_EQ(actual_lengths[1], expected_lengths[1]);
        const auto actual = gpu_destination.CopyDeviceToCpu();
        EXPECT_TRUE(std::equal(actual.begin(), actual.end(), expected.begin()));
        const auto actual_buffer = gpu_destination_buffer.CopyDeviceToCpu();
        // Entries beyond buffer_length are scratch, not part of the published state.
        EXPECT_TRUE(std::equal(actual_buffer.begin(), actual_buffer.begin() + expected_lengths[1] * width,
                               expected_buffer.begin()));
      }
    }
  }
}

TEST(CudaFixedStatePoolTest, IndexerReplayMatchesCpuAcrossEveryCompressionBoundary) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-hybrid");
  ClearProviders(*config);
  SetProviderOption(*config, "cuda", {}, {});
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  ExpectIndexerReplayMatchesCpu<uint16_t>(*model->p_device_kvcache_);
  ExpectIndexerReplayMatchesCpu<uint32_t>(*model->p_device_kvcache_);
}

TEST(CudaFixedStatePoolTest, ReplaysHalfSnapshotAtAcceptedPrefix) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-hybrid");
  ClearProviders(*config);
  SetProviderOption(*config, "cuda", {}, {});
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  auto& device = *model->p_device_kvcache_;
  const std::array<uint16_t, 12> values{
      0x3c00, 0x4000, 0x4200, 0x4400,
      0x4500, 0x4600, 0x4700, 0x4800,
      0x4880, 0x4900, 0x4980, 0x4a00};
  auto updates = device.Allocate<uint16_t>(values.size());
  auto destination = device.Allocate<uint16_t>(4);
  updates.CopyFromCpu(values);
  destination.CopyFromCpu(std::array<uint16_t, 4>{});

  StateUpdateReplayDesc descriptor{};
  descriptor.kind = StateUpdateReplayKind::Snapshot;
  descriptor.value = ReplayBytes(updates);
  descriptor.destination_state = ReplayBytes(destination);
  descriptor.state_width = 4;
  descriptor.capacity = 3;
  descriptor.kept_count = 2;
  descriptor.element_size = sizeof(uint16_t);
  device.ReplayStateUpdates(&descriptor, 1);

  const auto actual = destination.CopyDeviceToCpu();
  ASSERT_EQ(actual.size(), 4u);
  for (size_t index = 0; index < actual.size(); ++index) {
    EXPECT_EQ(actual[index], values[4 + index]);
  }
}

#endif

TEST_F(FixedStatePoolTest, DeferredReplayFailurePoisonsPool) {
  FixedStateTestDevice device{*model_->p_device_kvcache_};
  device.allow_offset_views = true;
  ScopedKeyValueCacheDevice guard{*model_, device};
  FixedStatePool pool{model_, 2};
  MakeResident(pool, kRequestA, 4.0f, 2);
  const auto handle = pool.HandleFor(kRequestA);
  const auto generation = pool.StateGeneration(handle);
  std::vector<DeviceSpan<float>> authoritative_state;
  {
    auto reservation = pool.Reserve(One(kRequestA, 2));
    ASSERT_TRUE(reservation.UsesDirectBindings());
    for (const auto& binding : reservation.Bindings()) {
      authoritative_state.push_back(WrapTensor<float>(device, *binding.input));
    }
  }
  {
    const std::array<Request, 2> requests{Request{kRequestA, 6, 3}, Request{kRequestB, 4, 3}};
    auto reservation = pool.Reserve(requests);
    FillStagedRows(reservation, 0, 99.0f);
    FillStagedRows(reservation, 1, 77.0f);
    const std::array<float, 6> updates{10, 11, 20, 21, 30, 31};
    for (const auto& binding : reservation.Bindings()) {
      if (binding.state_update_value) FillConvUpdates(binding, 0, updates);
    }
    reservation.CommitPrefix(0, 4, 2);
    device.fail_replay = true;
    reservation.PrepareCommit();
    EXPECT_EQ(reservation.State(), FixedStateReservationState::Prepared);
    for (auto& state : authoritative_state) {
      for (float value : state.CopyDeviceToCpu()) EXPECT_FLOAT_EQ(value, 4.0f);
    }
    EXPECT_EQ(pool.StateGeneration(handle), generation);
    EXPECT_EQ(pool.CommittedTokens(handle), 2u);
    EXPECT_FALSE(pool.OwnsCommittedSlot(kRequestB));
    reservation.PublishCommit();
    EXPECT_EQ(pool.StateGeneration(handle), generation + 1);
    EXPECT_EQ(pool.CommittedTokens(handle), 4u);
    EXPECT_TRUE(pool.OwnsCommittedSlot(kRequestB));
    EXPECT_EQ(pool.Snapshot().reserved_slots, 0u);
  }
  EXPECT_THROW(pool.Reserve(One(kRequestA, 5)), std::runtime_error);
  EXPECT_FALSE(pool.Snapshot().healthy);
  // The next reservation flushes deferred replay, which fails after a real bank write.
  const std::array<float, 6> replayed{4, 10, 20, 4, 11, 21};
  ASSERT_FALSE(device.replayed_destination.empty());
  const auto actual = device.replayed_destination.CopyDeviceToCpu();
  ASSERT_EQ(actual.size_bytes(), sizeof(replayed));
  EXPECT_EQ(std::memcmp(actual.data(), replayed.data(), sizeof(replayed)), 0);
  EXPECT_THROW(pool.Reserve(One(kRequestA, 5)), std::logic_error);
}

void RunDeviceCompactPartialAcceptanceReplay(
    const std::shared_ptr<Model>& model, bool expect_direct_bindings) {
  auto& device = *model->p_device_kvcache_;
  FixedStatePool pool{model, 1};
  EXPECT_TRUE(device.SupportsCompactStateReplay());

  const auto fill_tensor = [&](OrtValue& tensor, std::span<const float> values) {
    auto tensor_span = WrapTensor<float>(device, tensor);
    ASSERT_EQ(tensor_span.size(), values.size());
    tensor_span.CopyFromCpu(values);
  };
  const auto fill_tensor_value = [&](OrtValue& tensor, float value) {
    auto tensor_span = WrapTensor<float>(device, tensor);
    std::vector<float> values(tensor_span.size(), value);
    tensor_span.CopyFromCpu(values);
  };
  const auto expect_tensor = [&](OrtValue& tensor, std::span<const float> expected) {
    auto tensor_span = WrapTensor<float>(device, tensor);
    const auto actual = tensor_span.CopyDeviceToCpu();
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t index = 0; index < expected.size(); ++index) {
      EXPECT_FLOAT_EQ(actual[index], expected[index]);
    }
  };

  {
    auto reservation = pool.Reserve(One(kRequestA, 1));
    for (const auto& binding : reservation.Bindings()) {
      fill_tensor_value(*binding.output, 4.0f);
    }
    reservation.Commit();
  }
  {
    auto reservation = pool.Reserve(One(kRequestA, 4, 3));
    EXPECT_EQ(reservation.UsesDirectBindings(), expect_direct_bindings);
    for (const auto& binding : reservation.Bindings()) {
      fill_tensor_value(*binding.output, 99.0f);
    }
    const std::array<float, 6> conv_values{10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f};
    const std::array<float, 6> decay{0.5f, 0.25f, 1.0f, 0.5f, 1.0f, 1.0f};
    const std::array<float, 6> key{2.0f, 3.0f, 4.0f, 5.0f, 1.0f, 1.0f};
    const std::array<float, 12> delta{
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 24> capsule;
    auto capsule_end = std::copy(decay.begin(), decay.end(), capsule.begin());
    capsule_end = std::copy(key.begin(), key.end(), capsule_end);
    std::copy(delta.begin(), delta.end(), capsule_end);
    for (const auto& binding : reservation.Bindings()) {
      if (binding.state_update_kind ==
          Config::Model::Decoder::StateUpdateKind::CausalConv) {
        fill_tensor(*binding.state_update_value, conv_values);
      } else {
        fill_tensor(*binding.state_update_capsule, capsule);
      }
    }
    reservation.CommitPrefix(0, 4, 2);
    reservation.Commit();
  }

  EXPECT_EQ(pool.CommittedTokens(pool.HandleFor(kRequestA)), 2u);
  auto reservation = pool.Reserve(One(kRequestA, 2));
  const std::array<float, 6> expected_conv{4.0f, 10.0f, 20.0f, 4.0f, 11.0f, 21.0f};
  const std::array<float, 8> expected_gdn{
      24.0f, 30.0f, 30.0f, 38.0f, 31.5f, 40.0f, 36.5f, 46.5f};
  expect_tensor(*reservation.Bindings()[0].input, expected_conv);
  expect_tensor(*reservation.Bindings()[1].input, expected_conv);
  expect_tensor(*reservation.Bindings()[2].input, expected_gdn);
  expect_tensor(*reservation.Bindings()[3].input, expected_gdn);
}

void RunDeviceAdversarialReplay(const std::shared_ptr<Model>& model) {
  auto& device = *model->p_device_kvcache_;
  FixedStatePool pool{model, 3};
  const std::array<const void*, 3> ids{kRequestA, kRequestB, kRequestC};
  std::array<std::array<std::vector<float>, 4>, 3> expected;
  const auto write_row = [&](OrtValue& tensor, size_t row, std::span<const float> values) {
    auto data = WrapTensor<float>(device, tensor);
    ASSERT_EQ(RowElements(tensor), values.size());
    auto cpu = data.CopyDeviceToCpu();
    std::copy(values.begin(), values.end(), cpu.begin() + row * values.size());
    data.CopyCpuToDevice();
  };
  const auto check_rows = [&](FixedStateReservation& reservation, std::span<const size_t> order) {
    for (size_t row = 0; row < order.size(); ++row) {
      for (size_t layer = 0; layer < reservation.Bindings().size(); ++layer) {
        const auto& values = expected[order[row]][layer];
        auto data = WrapTensor<float>(device, *reservation.Bindings()[layer].input);
        const auto actual = data.subspan(row * values.size(), values.size()).CopyDeviceToCpu();
        ASSERT_EQ(actual.size(), values.size());
        for (size_t index = 0; index < values.size(); ++index) {
          EXPECT_FLOAT_EQ(actual[index], values[index])
              << "request=" << order[row] << " binding=" << layer << " element=" << index;
        }
      }
    }
  };
  for (size_t id = 0; id < ids.size(); ++id) {
    auto reservation = pool.Reserve(One(ids[id], 1));
    for (size_t layer = 0; layer < reservation.Bindings().size(); ++layer) {
      auto& values = expected[id][layer];
      values.resize(RowElements(*reservation.Bindings()[layer].output));
      for (size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<float>(1 + 32 * id + 8 * layer + index) / 8;
      }
      write_row(*reservation.Bindings()[layer].output, 0, values);
    }
    reservation.Commit();
  }
  ASSERT_GT(pool.HandleFor(kRequestB).slot, 0u);
  ASSERT_GT(pool.HandleFor(kRequestC).slot, 0u);

  const std::array<size_t, 2> reversed{2, 1};
  const std::array<Request, 2> requests{Request{kRequestC, 5, 3}, Request{kRequestB, 5, 3}};
  {
    auto reservation = pool.Reserve(requests);
    ASSERT_FALSE(reservation.UsesDirectBindings());
    check_rows(reservation, reversed);
    for (size_t row = 0; row < reversed.size(); ++row) {
      const size_t id = reversed[row];
      for (size_t layer = 0; layer < reservation.Bindings().size(); ++layer) {
        const auto& binding = reservation.Bindings()[layer];
        auto& values = expected[id][layer];
        std::vector<float> full_output(values.size());
        for (size_t index = 0; index < values.size(); ++index) {
          full_output[index] = static_cast<float>(80 + id * 16 + layer * 8 + index);
        }
        write_row(*binding.output, row, full_output);
        if (binding.state_update_value) {
          std::array<float, 6> updates;
          for (size_t index = 0; index < updates.size(); ++index) {
            updates[index] = static_cast<float>(10 + id * 8 + layer * 4 + index);
          }
          write_row(*binding.state_update_value, row, updates);
          if (row == 0) {
            for (size_t token = 0; token < 2; ++token) {
              for (size_t channel = 0; channel < 2; ++channel) {
                values[channel * 3] = values[channel * 3 + 1];
                values[channel * 3 + 1] = values[channel * 3 + 2];
                values[channel * 3 + 2] = updates[token * 2 + channel];
              }
            }
          }
        } else {
          std::array<float, 24> capsule;
          for (size_t token = 0; token < 3; ++token) {
            for (size_t head = 0; head < 2; ++head) {
              capsule[token * 2 + head] = head == 0 ? 0.5f : 0.25f;
              capsule[6 + token * 2 + head] = static_cast<float>(1 + token + head + id);
            }
            for (size_t index = 0; index < 4; ++index) {
              capsule[12 + token * 4 + index] = static_cast<float>(1 + layer + token + index) / 8;
            }
          }
          write_row(*binding.state_update_capsule, row, capsule);
          if (row == 0) {
            for (size_t token = 0; token < 2; ++token) {
              for (size_t head = 0; head < 2; ++head) {
                for (size_t key = 0; key < 2; ++key) {
                  for (size_t value = 0; value < 2; ++value) {
                    auto& state = values[head * 4 + value * 2 + key];
                    state = state * capsule[token * 2 + head] +
                            capsule[6 + token * 2 + key] * capsule[12 + token * 4 + head * 2 + value];
                  }
                }
              }
            }
          }
        }
        if (row == 1) values = full_output;
      }
    }
    reservation.CommitPrefix(0, 4, 2);
    reservation.CommitPrefix(1, 4, 4);
    reservation.Commit();
  }
  EXPECT_EQ(pool.CommittedTokens(pool.HandleFor(kRequestC)), 3u);
  EXPECT_EQ(pool.CommittedTokens(pool.HandleFor(kRequestB)), 5u);
  EXPECT_EQ(pool.CommittedTokens(pool.HandleFor(kRequestA)), 1u);
  EXPECT_EQ(pool.StateGeneration(pool.HandleFor(kRequestA)), 1u);
  EXPECT_EQ(pool.StateGeneration(pool.HandleFor(kRequestB)), 2u);
  EXPECT_EQ(pool.StateGeneration(pool.HandleFor(kRequestC)), 2u);
  const std::array<size_t, 3> order{0, 1, 2};
  const std::array<Request, 3> all{Request{kRequestA, 2}, Request{kRequestB, 6}, Request{kRequestC, 4}};
  {
    auto reservation = pool.Reserve(all);
    check_rows(reservation, order);
    for (size_t row = 0; row < order.size(); ++row) {
      for (size_t layer = 0; layer < reservation.Bindings().size(); ++layer) {
        auto& values = expected[row][layer];
        for (auto& value : values) value += 0.5f;
        write_row(*reservation.Bindings()[layer].output, row, values);
      }
    }
    reservation.Commit();
  }
  {
    auto reservation = pool.Reserve(all);
    check_rows(reservation, order);
    for (size_t row = 0; row < order.size(); ++row) {
      for (size_t layer = 0; layer < reservation.Bindings().size(); ++layer) {
        std::vector<float> poison(expected[row][layer].size(), -999.0f);
        write_row(*reservation.Bindings()[layer].output, row, poison);
      }
    }
    reservation.PrepareCommit();
    reservation.Discard();
  }
  {
    auto reservation = pool.Reserve(all);
    check_rows(reservation, order);

    // Two-byte-aligned subviews exercise FP16 convolution replay without changing the FP32 fixture.
    std::array<Ort::Float16_t, 10> source, updates, destination;
    for (size_t index = 0; index < source.size(); ++index) {
      source[index] = Ort::Float16_t{FastFloat32ToFloat16(static_cast<float>(index))};
      updates[index] = Ort::Float16_t{FastFloat32ToFloat16(-20.0f)};
      destination[index] = Ort::Float16_t{FastFloat32ToFloat16(-10.0f - static_cast<float>(index))};
    }
    for (size_t token = 0; token < 3; ++token) {
      for (size_t channel = 0; channel < 2; ++channel) {
        updates[2 + token * 2 + channel] = Ort::Float16_t{FastFloat32ToFloat16(static_cast<float>(10 + token * 10 + channel))};
      }
    }
    const auto upload = [&](const auto& values) {
      auto buffer = device.Allocate<uint8_t>(sizeof(values));
      std::memcpy(buffer.CpuSpan().data(), values.data(), sizeof(values));
      buffer.CopyCpuToDevice();
      return buffer;
    };
    auto source_buffer = upload(source);
    auto update_buffer = upload(updates);
    auto destination_buffer = upload(destination);
    StateUpdateReplayDesc half_replay{
        source_buffer.subspan(2, 12), destination_buffer.subspan(6, 12), update_buffer.subspan(4, 12), {}, {}, {}, 2, 3, 0, 0, 3, 2, sizeof(Ort::Float16_t), StateUpdateReplayKind::CausalConv};
    device.ReplayStateUpdates(&half_replay, 1);
    device.Synchronize();
    const std::array<float, 6> half_expected{3, 10, 20, 6, 11, 21};
    for (size_t index = 0; index < half_expected.size(); ++index) {
      destination[3 + index] = Ort::Float16_t{FastFloat32ToFloat16(half_expected[index])};
    }
    const auto actual_destination = destination_buffer.CopyDeviceToCpu();
    EXPECT_EQ(std::memcmp(actual_destination.data(), destination.data(), sizeof(destination)), 0);
    const auto actual_source = source_buffer.CopyDeviceToCpu();
    EXPECT_EQ(std::memcmp(actual_source.data(), source.data(), sizeof(source)), 0);
    const auto actual_updates = update_buffer.CopyDeviceToCpu();
    EXPECT_EQ(std::memcmp(actual_updates.data(), updates.data(), sizeof(updates)), 0);
  }
  const auto old_handle = pool.HandleFor(kRequestC);
  pool.Release(old_handle);
  {
    auto reservation = pool.Reserve(One(kRequestC));
    EXPECT_EQ(reservation.Handles()[0].slot, old_handle.slot);
    EXPECT_NE(reservation.Handles()[0].generation, old_handle.generation);
    for (auto& values : expected[2]) std::fill(values.begin(), values.end(), 0.0f);
    const std::array<size_t, 1> reused{2};
    check_rows(reservation, reused);
    for (size_t layer = 0; layer < reservation.Bindings().size(); ++layer) {
      for (size_t index = 0; index < expected[2][layer].size(); ++index) {
        expected[2][layer][index] = static_cast<float>(layer * 8 + index + 1);
      }
      write_row(*reservation.Bindings()[layer].output, 0, expected[2][layer]);
    }
    reservation.Commit();
  }
  auto reservation = pool.Reserve(all);
  check_rows(reservation, order);
}

#if USE_CUDA
// Realistic gated-delta-net geometries take the fast replay kernel, and a key width that is not a
// multiple of 4 keeps the generic one; both must match the sequential host recurrence exactly.
TEST(CudaFixedStatePoolTest, GatedDeltaNetReplayMatchesHostRecurrence) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-hybrid");
  ClearProviders(*config);
  SetProviderOption(*config, "cuda", {}, {});
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  auto& device = *model->p_device_kvcache_;

  struct Geometry {
    size_t heads, key_heads, value_width, key_width, capacity, kept;
  };
  const std::array<Geometry, 2> geometries{
      Geometry{4, 2, 40, 128, 7, 3},  // fast kernel, with a partial row chunk
      Geometry{3, 3, 5, 6, 4, 2},     // generic kernel
  };
  std::vector<DeviceSpan<float>> keepalive;
  std::vector<std::vector<float>> expected;
  std::vector<DeviceSpan<float>> outputs;
  std::vector<StateUpdateReplayDesc> descriptors;
  uint32_t seed = 1;
  const auto next = [&seed] {
    seed = seed * 1664525u + 1013904223u;
    return static_cast<float>(seed >> 8) / static_cast<float>(1u << 24) - 0.5f;
  };
  for (const auto& g : geometries) {
    const size_t state_count = g.heads * g.value_width * g.key_width;
    const size_t capsule_count =
        g.capacity * (g.heads + g.key_heads * g.key_width + g.heads * g.value_width);
    std::vector<float> state(state_count), capsule(capsule_count);
    for (auto& value : state) value = next();
    for (auto& value : capsule) value = next();
    const float* decay = capsule.data();
    const float* key = decay + g.capacity * g.heads;
    const float* delta = key + g.capacity * g.key_heads * g.key_width;

    std::vector<float> reference = state;
    for (size_t h = 0; h < g.heads; ++h) {
      const size_t kh = h * g.key_heads / g.heads;
      for (size_t v = 0; v < g.value_width; ++v) {
        for (size_t k = 0; k < g.key_width; ++k) {
          float s = reference[(h * g.value_width + v) * g.key_width + k];
          for (size_t t = 0; t < g.kept; ++t) {
            const volatile float scaled_state = s * decay[t * g.heads + h];
            s = std::fma(key[(t * g.key_heads + kh) * g.key_width + k],
                         delta[(t * g.heads + h) * g.value_width + v], scaled_state);
          }
          reference[(h * g.value_width + v) * g.key_width + k] = s;
        }
      }
    }

    auto source = device.Allocate<float>(state_count);
    auto destination = device.Allocate<float>(state_count);
    auto capsule_device = device.Allocate<float>(capsule_count);
    source.CopyFromCpu(state);
    capsule_device.CopyFromCpu(capsule);
    auto source_bytes = device.WrapMemory<uint8_t>(
        {reinterpret_cast<uint8_t*>(source.Span().data()), state_count * sizeof(float)});
    auto destination_bytes = device.WrapMemory<uint8_t>(
        {reinterpret_cast<uint8_t*>(destination.Span().data()), state_count * sizeof(float)});
    auto capsule_bytes = device.WrapMemory<uint8_t>(
        {reinterpret_cast<uint8_t*>(capsule_device.Span().data()), capsule_count * sizeof(float)});
    const size_t decay_bytes = g.capacity * g.heads * sizeof(float);
    const size_t key_bytes = g.capacity * g.key_heads * g.key_width * sizeof(float);
    descriptors.push_back(StateUpdateReplayDesc{
        source_bytes,
        destination_bytes,
        {},
        capsule_bytes.subspan(0, decay_bytes),
        capsule_bytes.subspan(decay_bytes, key_bytes),
        capsule_bytes.subspan(decay_bytes + key_bytes, g.capacity * g.heads * g.value_width * sizeof(float)),
        g.heads,
        g.value_width,
        g.key_width,
        g.key_heads,
        static_cast<uint32_t>(g.capacity),
        static_cast<uint32_t>(g.kept),
        static_cast<uint32_t>(sizeof(float)),
        StateUpdateReplayKind::GatedDeltaNet,
    });
    keepalive.push_back(source);
    keepalive.push_back(capsule_device);
    outputs.push_back(destination);
    expected.push_back(std::move(reference));
  }

  device.ReplayStateUpdates(descriptors.data(), descriptors.size());
  device.Synchronize();
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto actual = outputs[i].CopyDeviceToCpu();
    ASSERT_EQ(actual.size(), expected[i].size());
    for (size_t j = 0; j < actual.size(); ++j) {
      ASSERT_EQ(actual[j], expected[i][j]) << "descriptor " << i << " element " << j;
    }
  }
}

TEST(CudaFixedStatePoolTest, GatedDeltaNetReplayMatchesActualOrtAcceptedPrefix) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-hybrid");
  ClearProviders(*config);
  SetProviderOption(*config, "cuda", {}, {});
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  auto& device = *model->p_device_kvcache_;
  struct Geometry {
    const char* filename;
    size_t heads, key_heads, value_width, key_width;
  };
  const std::array<Geometry, 2> geometries{{
      {"gdn-replay-fast.onnx", 4, 2, 40, 128},
      {"gdn-replay-generic.onnx", 3, 3, 5, 6},
  }};
  uint32_t seed = 17;
  const auto next = [&seed] {
    seed = seed * 1664525u + 1013904223u;
    return static_cast<float>(seed >> 8) / static_cast<float>(1u << 24) - 0.5f;
  };
  for (const auto& g : geometries) {
    SCOPED_TRACE(g.filename);
    constexpr size_t capacity = 7;
    const size_t state_count = g.heads * g.value_width * g.key_width;
    std::vector<float> initial(state_count), query(capacity * g.key_heads * g.key_width),
        key(query.size()), value(capacity * g.heads * g.value_width),
        decay(capacity * g.heads), beta(decay.size());
    for (auto* values : {&initial, &query, &key, &value, &beta}) {
      for (auto& element : *values) element = next();
    }
    for (auto& element : decay) element = -std::abs(next());
    auto session = model->CreateSession(GetOrtEnv(), g.filename, model->session_options_.get());
    const auto run = [&](size_t tokens) {
      const auto floats = [&](std::span<const float> values, std::initializer_list<int64_t> shape) {
        auto tensor = OrtValue::CreateTensor<float>(
            model->allocator_cpu_, std::span<const int64_t>{shape.begin(), shape.size()});
        const size_t count = tensor->GetTensorTypeAndShapeInfo()->GetElementCount();
        std::copy_n(values.data(), count, tensor->GetTensorMutableData<float>());
        return tensor;
      };
      std::vector<std::unique_ptr<OrtValue>> inputs;
      const int64_t n = static_cast<int64_t>(tokens), h = static_cast<int64_t>(g.heads),
                    kh = static_cast<int64_t>(g.key_heads), k = static_cast<int64_t>(g.key_width),
                    v = static_cast<int64_t>(g.value_width);
      inputs.push_back(floats(query, {n, kh, k}));
      inputs.push_back(floats(key, {n, kh, k}));
      inputs.push_back(floats(value, {n, h, v}));
      inputs.push_back(floats(decay, {n, h}));
      inputs.push_back(floats(beta, {n, h}));
      inputs.push_back(floats(initial, {1, h, v, k}));
      inputs.push_back(OrtValue::CreateTensor<int32_t>(model->allocator_cpu_, std::array<int64_t, 1>{1}));
      inputs.back()->GetTensorMutableData<int32_t>()[0] = static_cast<int32_t>(tokens);
      const std::array<const char*, 7> names{
          "query", "key", "value", "decay", "beta", "initial_state", "capture_count"};
      std::array<const OrtValue*, 7> pointers;
      for (size_t i = 0; i < inputs.size(); ++i) pointers[i] = inputs[i].get();
      const std::array<const char*, 2> output_names{"final_state", "state_update"};
      return session->Run(nullptr, names.data(), pointers.data(), pointers.size(),
                          output_names.data(), output_names.size());
    };
    auto full = run(capacity);
    const size_t capsule_count = capacity * (g.heads + g.key_heads * g.key_width + g.heads * g.value_width);
    const auto* capsule = full[1]->GetTensorData<float>();
    auto source = device.Allocate<float>(state_count);
    auto destination = device.Allocate<float>(state_count);
    auto gpu_capsule = device.Allocate<float>(capsule_count);
    source.CopyFromCpu(initial);
    gpu_capsule.CopyFromCpu(std::span<const float>{capsule, capsule_count});
    for (uint32_t kept = 1; kept < capacity; ++kept) {
      SCOPED_TRACE(kept);
      auto prefix = run(kept);
      const auto* expected = prefix[0]->GetTensorData<float>();
      StateUpdateReplayDesc descriptor{};
      descriptor.kind = StateUpdateReplayKind::GatedDeltaNet;
      descriptor.source_state = ReplayBytes(source);
      descriptor.destination_state = ReplayBytes(destination);
      auto capsule_bytes = ReplayBytes(gpu_capsule);
      const size_t decay_bytes = capacity * g.heads * sizeof(float);
      const size_t key_bytes = capacity * g.key_heads * g.key_width * sizeof(float);
      const size_t delta_bytes = capacity * g.heads * g.value_width * sizeof(float);
      descriptor.decay = capsule_bytes.subspan(0, decay_bytes);
      descriptor.key = capsule_bytes.subspan(decay_bytes, key_bytes);
      descriptor.delta = capsule_bytes.subspan(decay_bytes + key_bytes, delta_bytes);
      descriptor.channel_count = g.heads;
      descriptor.state_width = g.value_width;
      descriptor.key_width = g.key_width;
      descriptor.key_head_count = g.key_heads;
      descriptor.capacity = capacity;
      descriptor.kept_count = kept;
      descriptor.element_size = sizeof(float);
      device.ReplayStateUpdates(&descriptor, 1);
      device.Synchronize();
      const auto actual = destination.CopyDeviceToCpu();
      std::vector<float> cpu_state(state_count);
      descriptor.source_state = HostReplayBytes(initial);
      descriptor.destination_state = HostReplayBytes(cpu_state);
      auto host_capsule = std::span<const float>{capsule, capsule_count};
      auto host_capsule_bytes = HostReplayBytes(host_capsule);
      descriptor.decay = host_capsule_bytes.subspan(0, decay_bytes);
      descriptor.key = host_capsule_bytes.subspan(decay_bytes, key_bytes);
      descriptor.delta = host_capsule_bytes.subspan(decay_bytes + key_bytes, delta_bytes);
      GetDeviceInterface(DeviceType::CPU)->ReplayStateUpdates(&descriptor, 1);
      for (size_t i = 0; i < state_count; ++i) {
        ASSERT_EQ(actual[i], expected[i]) << "CUDA element " << i;
        ASSERT_EQ(cpu_state[i], expected[i]) << "CPU element " << i;
      }
    }
  }
}

TEST(CudaFixedStatePoolTest, CompactPartialAcceptanceReplaysConvAndGdn) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-hybrid");
  ClearProviders(*config);
  SetProviderOption(*config, "cuda", {}, {});
  RunDeviceCompactPartialAcceptanceReplay(
      CreateModel(GetOrtEnv(), std::move(config)), true);
}

TEST(CudaFixedStatePoolTest, AdversarialReplayPreservesRequestState) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-hybrid");
  ClearProviders(*config);
  SetProviderOption(*config, "cuda", {}, {});
  RunDeviceAdversarialReplay(CreateModel(GetOrtEnv(), std::move(config)));
}
#endif

bool HasWebGpuProvider() {
  GetOrtEnv();
  const auto providers = Ort::GetAvailableProviders();
  return std::find(providers.begin(), providers.end(), "WebGpuExecutionProvider") != providers.end() ||
         !FindRegisteredEpDevices("WebGpuExecutionProvider").empty();
}

TEST(WebGpuFixedStatePoolTest, CompactPartialAcceptanceReplaysConvAndGdn) {
  if (!HasWebGpuProvider()) {
    GTEST_SKIP() << "No native or registered WebGPU EP is available.";
  }
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-hybrid");
  ClearProviders(*config);
  SetProviderOption(*config, "webgpu", {}, {});
  RunDeviceCompactPartialAcceptanceReplay(
      CreateModel(GetOrtEnv(), std::move(config)), false);
}

TEST(WebGpuFixedStatePoolTest, AdversarialReplayPreservesRequestState) {
  if (!HasWebGpuProvider()) {
    GTEST_SKIP() << "No native or registered WebGPU EP is available.";
  }
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-hybrid");
  ClearProviders(*config);
  SetProviderOption(*config, "webgpu", {}, {});
  RunDeviceAdversarialReplay(CreateModel(GetOrtEnv(), std::move(config)));
}

TEST(WebGpuFixedStatePoolTest, PartialRequestsShareBackingTransfers) {
  if (FindRegisteredEpDevices("WebGpuExecutionProvider").empty()) {
    GTEST_SKIP() << "No WebGPU EP device is registered.";
  }
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-hybrid");
  ClearProviders(*config);
  SetProviderOption(*config, "webgpu", {}, {});
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  ReplayCountingDevice device{*model->p_device_kvcache_};
  ScopedKeyValueCacheDevice guard{*model, device};
  FixedStatePool pool{model, 6};
  std::array<int, 6> ids{};
  for (size_t slot = 0; slot < ids.size(); ++slot) {
    auto reservation = pool.Reserve(One(&ids[slot], 1));
    for (const auto& binding : reservation.Bindings()) {
      auto data = WrapTensor<float>(device, *binding.output);
      std::vector<float> values(data.size(), static_cast<float>(slot + 1));
      data.CopyFromCpu(values);
    }
    reservation.Commit();
  }

  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  const std::array<size_t, 3> order{5, 1, 3};
  const std::array<size_t, 3> kept{1, 2, 3};
  std::array<std::array<std::vector<float>, 4>, 6> expected;
  for (size_t slot = 0; slot < ids.size(); ++slot) {
    for (size_t layer = 0; layer < 4; ++layer) {
      expected[slot][layer].assign(layer < 2 ? 6 : 8, static_cast<float>(slot + 1));
    }
  }
  const std::array<Request, 3> requests{
      Request{&ids[5], 5, 3}, Request{&ids[1], 5, 3}, Request{&ids[3], 5, 3}};
  {
    auto reservation = pool.Reserve(requests);
    for (size_t row = 0; row < order.size(); ++row) {
      ASSERT_GT(reservation.Handles()[row].slot, 0u);
      for (size_t layer = 0; layer < reservation.Bindings().size(); ++layer) {
        const auto& binding = reservation.Bindings()[layer];
        auto output = WrapTensor<float>(device, *binding.output);
        std::fill(output.CpuSpan().begin(), output.CpuSpan().end(), -999.0f);
        output.CopyCpuToDevice();
        OrtValue& update = binding.state_update_value
                               ? *binding.state_update_value
                               : *binding.state_update_capsule;
        auto data = WrapTensor<float>(device, update);
        auto values = data.CopyDeviceToCpu();
        const size_t row_elements = RowElements(update);
        for (size_t index = 0; index < row_elements; ++index) {
          values[row * row_elements + index] = static_cast<float>(row + index + 1) / 8;
        }
        data.CopyCpuToDevice();
        auto source = std::vector<float>(expected[order[row]][layer]);
        auto source_bytes = cpu->WrapMemory<uint8_t>(
            {reinterpret_cast<uint8_t*>(source.data()), source.size() * sizeof(float)});
        auto destination_bytes = cpu->WrapMemory<uint8_t>(
            {reinterpret_cast<uint8_t*>(expected[order[row]][layer].data()), source.size() * sizeof(float)});
        auto update_bytes = cpu->WrapMemory<uint8_t>(
            {reinterpret_cast<uint8_t*>(values.data() + row * row_elements), row_elements * sizeof(float)});
        const bool conv = binding.state_update_value != nullptr;
        StateUpdateReplayDesc reference{
            source_bytes, destination_bytes,
            conv ? update_bytes : DeviceSpan<uint8_t>{},
            conv ? DeviceSpan<uint8_t>{} : update_bytes.subspan(0, 24),
            conv ? DeviceSpan<uint8_t>{} : update_bytes.subspan(24, 24),
            conv ? DeviceSpan<uint8_t>{} : update_bytes.subspan(48, 48),
            2, conv ? 3u : 2u, conv ? 0u : 2u, conv ? 0u : 1u,
            3, static_cast<uint32_t>(kept[row]), sizeof(float),
            conv ? StateUpdateReplayKind::CausalConv : StateUpdateReplayKind::GatedDeltaNet};
        cpu->ReplayStateUpdates(&reference, 1);
      }
      reservation.CommitPrefix(row, 4, kept[row]);
    }
    reservation.Commit();
  }

  std::array<Request, 6> all;
  for (size_t slot = 0; slot < ids.size(); ++slot) {
    all[slot] = Request{&ids[slot], 6};
  }
  auto reservation = pool.Reserve(all);  // Materializes the queued replay before gathering.
  ASSERT_EQ(device.buffers.size(), 6u);  // Two convolution tensors, each with input, updates, and one bank.
  size_t downloads = 0, uploads = 0, staged_bytes = 0;
  for (const auto& buffer : device.buffers) {
    EXPECT_EQ(buffer->downloads, 1u);
    EXPECT_LE(buffer->uploads, 1u);
    downloads += buffer->downloads;
    uploads += buffer->uploads;
    staged_bytes += buffer->allocated_bytes;
  }
  EXPECT_EQ(downloads, 6u);
  EXPECT_EQ(uploads, 2u);
  EXPECT_EQ(staged_bytes, 576u);
  for (size_t layer = 0; layer < reservation.Bindings().size(); ++layer) {
    auto data = WrapTensor<float>(device, *reservation.Bindings()[layer].input);
    const auto actual = data.CopyDeviceToCpu();
    for (size_t slot = 0; slot < ids.size(); ++slot) {
      const auto& values = expected[slot][layer];
      EXPECT_EQ(std::memcmp(actual.data() + slot * values.size(), values.data(),
                            values.size() * sizeof(float)),
                0);
    }
  }
  RecordProperty("replay_downloads", static_cast<int>(downloads));
  RecordProperty("replay_uploads", static_cast<int>(uploads));
  RecordProperty("peak_host_mirror_bytes", static_cast<int>(staged_bytes));
}

TEST(WebGpuFixedStatePoolTest, ConvolutionSharedMirrorsObserveEarlierDescriptorWrites) {
  if (FindRegisteredEpDevices("WebGpuExecutionProvider").empty()) {
    GTEST_SKIP() << "No WebGPU EP device is registered.";
  }
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-hybrid");
  ClearProviders(*config);
  SetProviderOption(*config, "webgpu", {}, {});
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  auto& gpu = *model->p_device_kvcache_;
  ReplayCountingDevice device{gpu};
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  for (size_t repetitions : {1u, 4u}) {
    constexpr size_t slots = 8, guard = 4, conv_row = 6, gdn_row = 8, capsule_row = 24;
    constexpr size_t conv_source = guard, gdn_source = conv_source + slots * conv_row;
    constexpr size_t conv_values = gdn_source + slots * gdn_row;
    constexpr size_t capsules = conv_values + slots * conv_row;
    constexpr size_t conv_bank = guard, gdn_bank = conv_bank + slots * conv_row;
    std::vector<float> inputs(capsules + slots * capsule_row + guard);
    std::vector<float> bank(gdn_bank + slots * gdn_row + guard);
    for (size_t index = 0; index < inputs.size(); ++index) {
      inputs[index] = static_cast<float>(index % 13 + 1) / 16;
    }
    for (size_t index = 0; index < bank.size(); ++index) {
      bank[index] = -static_cast<float>(index + 1);
    }
    const auto upload = [&](std::vector<float>& values) {
      auto buffer = gpu.Allocate<uint8_t>(values.size() * sizeof(float));
      auto host = cpu->WrapMemory<uint8_t>(
          {reinterpret_cast<uint8_t*>(values.data()), values.size() * sizeof(float)});
      buffer.CopyFrom(host);  // Zero-offset upload does not allocate a host mirror.
      return buffer;
    };
    auto input_buffer = upload(inputs);
    auto bank_buffer = upload(bank);
    auto expected = bank;
    auto host_inputs = cpu->WrapMemory<uint8_t>(
        {reinterpret_cast<uint8_t*>(inputs.data()), inputs.size() * sizeof(float)});
    auto host_bank = cpu->WrapMemory<uint8_t>(
        {reinterpret_cast<uint8_t*>(expected.data()), expected.size() * sizeof(float)});
    const auto make_descriptors = [&](DeviceSpan<uint8_t> in, DeviceSpan<uint8_t> out) {
      const auto view = [](DeviceSpan<uint8_t> buffer, size_t offset, size_t elements) {
        return buffer.subspan(offset * sizeof(float), elements * sizeof(float));
      };
      std::vector<StateUpdateReplayDesc> descriptors;
      const std::array<size_t, 3> order{6, 1, 3};
      for (size_t repeat = 0; repeat < repetitions; ++repeat) {
        for (size_t row = 0; row < order.size(); ++row) {
          const size_t slot = order[row];
          descriptors.push_back(StateUpdateReplayDesc{
              view(in, conv_source + slot * conv_row, conv_row),
              view(out, conv_bank + slot * conv_row, conv_row),
              view(in, conv_values + slot * conv_row, conv_row),
              {},
              {},
              {},
              2,
              3,
              0,
              0,
              3,
              static_cast<uint32_t>(row + 1),
              sizeof(float),
              StateUpdateReplayKind::CausalConv});
        }
        // This source is a destination written earlier in this same replay call.
        descriptors.push_back(StateUpdateReplayDesc{
            view(out, conv_bank + 6 * conv_row, conv_row),
            view(out, conv_bank + 5 * conv_row, conv_row),
            view(in, conv_values + 5 * conv_row, conv_row),
            {},
            {},
            {},
            2,
            3,
            0,
            0,
            3,
            2,
            sizeof(float),
            StateUpdateReplayKind::CausalConv});
      }
      return descriptors;
    };
    auto reference = make_descriptors(host_inputs, host_bank);
    cpu->ReplayStateUpdates(reference.data(), reference.size());
    auto descriptors = make_descriptors(input_buffer, bank_buffer);
    device.ReplayStateUpdates(descriptors.data(), descriptors.size());
    ASSERT_EQ(device.buffers.size(), 2u);
    EXPECT_EQ(device.buffers[0]->downloads, 1u);
    EXPECT_EQ(device.buffers[0]->uploads, 0u);
    EXPECT_EQ(device.buffers[1]->downloads, 1u);
    EXPECT_EQ(device.buffers[1]->uploads, 1u);
    EXPECT_EQ(device.buffers[0]->allocated_bytes + device.buffers[1]->allocated_bytes, 1920u);
    const auto actual = bank_buffer.CopyDeviceToCpu();
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(float)), 0);
    const auto unchanged_inputs = input_buffer.CopyDeviceToCpu();
    EXPECT_EQ(std::memcmp(unchanged_inputs.data(), inputs.data(), inputs.size() * sizeof(float)), 0);
  }
}

TEST(WebGpuFixedStatePoolTest, DeferredGpuRunFailurePoisonsPoolBeforeNextReader) {
  GetOrtEnv();
  if (!HasWebGpuProvider()) {
    GTEST_SKIP() << "No native or registered WebGPU EP is available.";
  }
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-hybrid");
  ClearProviders(*config);
  SetProviderOption(*config, "webgpu", {}, {});
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  FixedStateTestDevice device{*model->p_device_kvcache_, DeviceType::WEBGPU};
  ScopedKeyValueCacheDevice guard{*model, device};
  FixedStatePool pool{model, 2};
  const auto fill = [&](OrtValue& tensor, float value) {
    auto data = WrapTensor<float>(device, tensor);
    const std::vector<float> values(data.size(), value);
    data.CopyFromCpu(values);
  };
  {
    auto reservation = pool.Reserve(One(kRequestA, 2));
    for (const auto& binding : reservation.Bindings()) {
      fill(*binding.output, 4.0f);
    }
    reservation.Commit();
  }
  const auto handle = pool.HandleFor(kRequestA);
  const auto generation = pool.StateGeneration(handle);
  {
    const std::array<Request, 2> requests{Request{kRequestA, 6, 3}, Request{kRequestB, 4, 3}};
    auto reservation = pool.Reserve(requests);
    for (const auto& binding : reservation.Bindings()) {
      fill(*binding.output, 99.0f);
      if (binding.state_update_value) fill(*binding.state_update_value, 1.0f);
      if (binding.state_update_capsule) fill(*binding.state_update_capsule, 1.0f);
    }
    reservation.CommitPrefix(0, 4, 2);
    reservation.CommitPrefix(1, 4, 2);
    device.invalidate_second_gdn = true;
    reservation.PrepareCommit();
    EXPECT_EQ(reservation.State(), FixedStateReservationState::Prepared);
    EXPECT_EQ(device.gdn_replay_count, 0u);
    EXPECT_EQ(pool.StateGeneration(handle), generation);
    EXPECT_EQ(pool.CommittedTokens(handle), 2u);
    EXPECT_FALSE(pool.OwnsCommittedSlot(kRequestB));
    reservation.PublishCommit();
    EXPECT_EQ(reservation.State(), FixedStateReservationState::Committed);
    EXPECT_EQ(pool.StateGeneration(handle), generation + 1);
    EXPECT_EQ(pool.CommittedTokens(handle), 4u);
    EXPECT_TRUE(pool.OwnsCommittedSlot(kRequestB));
    EXPECT_EQ(pool.Snapshot().reserved_slots, 0u);
    EXPECT_EQ(pool.Snapshot().committed_slots, 2u);
    EXPECT_TRUE(pool.Snapshot().healthy);
  }
  try {
    pool.Reserve(One(kRequestA, 5));
    FAIL() << "Expected the second ORT recurrent replay run to fail.";
  } catch (const Ort::Exception& error) {
    EXPECT_NE(std::string(error.what()).find("kept_count must be in [1, capacity]"), std::string::npos)
        << error.what();
  }
  EXPECT_EQ(device.gdn_replay_count, 2u);
  EXPECT_EQ(pool.StateGeneration(handle), generation + 1);
  EXPECT_EQ(pool.CommittedTokens(handle), 4u);
  EXPECT_TRUE(pool.OwnsCommittedSlot(kRequestB));
  EXPECT_EQ(pool.Snapshot().reserved_slots, 0u);
  EXPECT_EQ(pool.Snapshot().committed_slots, 2u);
  EXPECT_FALSE(pool.Snapshot().healthy);
  // One published row was really replayed; failure blocks readers of the incomplete state.
  ASSERT_FALSE(device.replayed_destination.empty());
  auto actual = device.replayed_destination.CopyDeviceToCpu();
  std::array<float, 8> replayed{};
  ASSERT_EQ(actual.size(), sizeof(replayed));
  std::memcpy(replayed.data(), actual.data(), actual.size());
  for (const auto value : replayed) EXPECT_FLOAT_EQ(value, 6.0f);
  EXPECT_THROW(pool.Reserve(One(kRequestA, 5)), std::logic_error);
  EXPECT_THROW(pool.CapturePrefixCheckpoint(kRequestA), std::logic_error);
}

TEST_F(FixedStatePoolTest, CapacityOverflowLeavesPoolUntouched) {
  auto pool = MakePool(1);
  const std::array<Request, 2> requests{Request{kRequestA, 1}, Request{kRequestB, 1}};  // 2 > capacity 1.
  EXPECT_THROW(pool->Reserve(requests), std::runtime_error);

  const auto snapshot = pool->Snapshot();
  EXPECT_TRUE(snapshot.healthy);
  EXPECT_EQ(snapshot.free_slots, 1u);
  EXPECT_EQ(snapshot.reserved_slots, 0u);
  EXPECT_EQ(snapshot.committed_slots, 0u);
  EXPECT_EQ(snapshot.active_staging_bytes, 0u);
  EXPECT_THROW(pool->HandleFor(kRequestA), std::runtime_error);
}

TEST_F(FixedStatePoolTest, RejectsEmptyReservation) {
  auto pool = MakePool(1);
  const std::span<const Request> empty;
  EXPECT_THROW(pool->Reserve(empty), std::invalid_argument);
  EXPECT_EQ(pool->AvailableSlots(), 1u);
}

TEST_F(FixedStatePoolTest, NotEnoughFreeSlotsForNewAdmissions) {
  auto pool = MakePool(2);
  MakeResident(*pool, kRequestA, 1.0f);  // Occupies one of two slots.
  // One resident row plus two brand-new rows needs three slots but only two exist.
  const std::array<Request, 3> requests{Request{kRequestA, 1}, Request{kRequestB, 1},
                                        Request{kRequestC, 1}};
  EXPECT_THROW(pool->Reserve(requests), std::runtime_error);
  EXPECT_TRUE(pool->Snapshot().healthy);
  EXPECT_EQ(pool->AvailableSlots(), 1u);
}

TEST_F(FixedStatePoolTest, RejectsDuplicateRequestsAndConcurrentReservations) {
  auto pool = MakePool(2);
  const std::array<Request, 2> duplicate{Request{kRequestA, 1}, Request{kRequestA, 1}};
  EXPECT_THROW(pool->Reserve(duplicate), std::runtime_error);

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  auto other = One(kRequestB);
  EXPECT_THROW(pool->Reserve(other), std::logic_error);  // Only one live reservation permitted.
}

TEST_F(FixedStatePoolTest, RejectsReleaseWhileReservationIsLive) {
  auto pool = MakePool(2);
  const auto handle_a = MakeResident(*pool, kRequestA, 3.0f);
  auto requests = One(kRequestB);
  auto reservation = pool->Reserve(requests);
  EXPECT_THROW(pool->Release(handle_a), std::logic_error);  // Idle contract during a reservation.
}

TEST_F(FixedStatePoolTest, ReservationHoldsTheLockForItsWholeLifetime) {
  auto pool = MakePool(2);
  {
    auto requests_a = One(kRequestA, /*target_tokens=*/2);
    auto reservation_a = pool->Reserve(requests_a);
    FillStagedRows(reservation_a, 0, 1.0f);
    reservation_a.Commit();
    EXPECT_EQ(reservation_a.State(), FixedStateReservationState::Committed);

    // A committed reservation still holds the single-reservation lock and its staging memory until
    // it is destroyed, so admitting the next batch or releasing a slot is refused while it is alive.
    auto requests_b = One(kRequestB, /*target_tokens=*/2);
    EXPECT_THROW(pool->Reserve(requests_b), std::logic_error);
    EXPECT_THROW(pool->Release(pool->HandleFor(kRequestA)), std::logic_error);
  }

  // The reservation is gone: the lock is free again. A second reservation likewise holds the lock
  // for its own lifetime, so it too must be dropped before the committed slot can be released.
  {
    auto requests_b = One(kRequestB, /*target_tokens=*/2);
    auto reservation_b = pool->Reserve(requests_b);
    reservation_b.Discard();
  }
  pool->Release(pool->HandleFor(kRequestA));
  EXPECT_EQ(pool->AvailableSlots(), 2u);
}

TEST_F(FixedStatePoolTest, DiscardRejectsCommittedButNotTerminalReservations) {
  auto pool = MakePool(1);
  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  FillStagedRows(reservation, 0, 1.0f);
  reservation.Commit();
  EXPECT_THROW(reservation.Discard(), std::logic_error);  // A published commit is irreversible.
  EXPECT_EQ(reservation.State(), FixedStateReservationState::Committed);
}

TEST_F(FixedStatePoolTest, PublishCommitAfterPoolDestructionFailsFast) {
  std::optional<FixedStateReservation> reservation;
  {
    auto pool = MakePool(1);
    auto requests = One(kRequestA);
    reservation.emplace(pool->Reserve(requests));
    FillStagedRows(*reservation, 0, 1.0f);
    reservation->PrepareCommit();
    ASSERT_EQ(reservation->State(), FixedStateReservationState::Prepared);
  }
  // The pool is gone. Publication cannot succeed without fixed state while a composite transaction
  // continues publishing its other resources, so this lifecycle violation terminates.
  EXPECT_DEATH_IF_SUPPORTED(reservation->PublishCommit(), "");
  EXPECT_EQ(reservation->State(), FixedStateReservationState::Failed);
  reservation.reset();
}

TEST_F(FixedStatePoolTest, RejectsForeignAndStaleHandles) {
  auto pool = MakePool(1);
  auto other_pool = MakePool(1);
  const auto handle_a = MakeResident(*pool, kRequestA, 2.0f);

  // Foreign handle: right shape, wrong pool.
  FixedStateSlotHandle foreign = handle_a;
  foreign.pool = other_pool.get();
  EXPECT_THROW(pool->Release(foreign), std::runtime_error);
  EXPECT_THROW(pool->StateGeneration(foreign), std::runtime_error);
  EXPECT_THROW(pool->CommittedTokens(foreign), std::runtime_error);

  // Stale handle: slot reused by a new request after release.
  pool->Release(handle_a);
  MakeResident(*pool, kRequestB, 8.0f);  // Reuses the slot with a fresh generation.
  EXPECT_THROW(pool->Release(handle_a), std::runtime_error);
  EXPECT_THROW(pool->StateGeneration(handle_a), std::runtime_error);
  EXPECT_THROW(pool->CommittedTokens(handle_a), std::runtime_error);
  EXPECT_THROW(pool->HandleFor(kRequestC), std::runtime_error);  // Unknown request.
}

TEST_F(FixedStatePoolTest, ReservationAccessorsSafeAfterPoolDestruction) {
  std::optional<FixedStateReservation> reservation;
  std::vector<std::string> input_names;
  {
    auto pool = MakePool(1);
    auto requests = One(kRequestA);
    reservation.emplace(pool->Reserve(requests));
    FillStagedRows(*reservation, 0, 7.0f);
    for (const auto& binding : reservation->Bindings()) {
      input_names.emplace_back(binding.input_name);
    }
  }

  // The pool is gone, but the reservation still owns its handles and binding names.
  EXPECT_EQ(reservation->State(), FixedStateReservationState::Failed);
  ASSERT_EQ(reservation->Handles().size(), 1u);
  EXPECT_EQ(reservation->Handles()[0].request_id, kRequestA);
  ASSERT_EQ(reservation->Bindings().size(), input_names.size());
  for (size_t index = 0; index < input_names.size(); ++index) {
    EXPECT_EQ(input_names[index], reservation->Bindings()[index].input_name);
  }
  EXPECT_FLOAT_EQ(
      reservation->Bindings()[0].output->GetTensorData<float>()[0], 7.0f);
  EXPECT_THROW(reservation->ValidateCommit(), std::logic_error);
  EXPECT_THROW(reservation->PrepareCommit(), std::logic_error);
  reservation.reset();
}

// --- Direct geometry-contract tests (model-independent) ---

TEST(FixedStateGeometryTest, AcceptsDynamicBatchAndDerivesRowElements) {
  const std::array<int64_t, 3> input{-1, 2, 3};
  const std::array<int64_t, 3> output{-1, 2, 3};
  const auto geometry = ValidateFixedStateGeometry(
      "past", kFloat, input, "present", kFloat, output);
  EXPECT_EQ(geometry.row_element_count, 6u);
  EXPECT_EQ(geometry.fixed_batch_size, 0u);
}

TEST(FixedStateGeometryTest, AcceptsFixedBatchAndReportsIt) {
  const std::array<int64_t, 2> input{4, 5};
  const std::array<int64_t, 2> output{4, 5};
  const auto geometry = ValidateFixedStateGeometry(
      "past", kFloat, input, "present", kFloat, output);
  EXPECT_EQ(geometry.row_element_count, 5u);
  EXPECT_EQ(geometry.fixed_batch_size, 4u);
}

TEST(FixedStateGeometryTest, AdoptsFixedOutputBatchWhenInputIsDynamic) {
  const std::array<int64_t, 3> input{-1, 2, 3};
  const std::array<int64_t, 3> output{4, 2, 3};
  const auto geometry = ValidateFixedStateGeometry(
      "past", kFloat, input, "present", kFloat, output);
  EXPECT_EQ(geometry.row_element_count, 6u);
  EXPECT_EQ(geometry.fixed_batch_size, 4u);  // A fixed output batch constrains the whole binding.
}

TEST(FixedStateGeometryTest, AdoptsFixedInputBatchWhenOutputIsDynamic) {
  const std::array<int64_t, 3> input{4, 2, 3};
  const std::array<int64_t, 3> output{-1, 2, 3};
  const auto geometry = ValidateFixedStateGeometry(
      "past", kFloat, input, "present", kFloat, output);
  EXPECT_EQ(geometry.fixed_batch_size, 4u);
}

TEST(FixedStateGeometryTest, RejectsZeroBatch) {
  const std::array<int64_t, 2> input{0, 5};
  const std::array<int64_t, 2> output{0, 5};
  EXPECT_THROW(ValidateFixedStateGeometry("past", kFloat, input, "present", kFloat, output),
               std::runtime_error);
}

TEST(FixedStateGeometryTest, RejectsAsymmetricFixedBatch) {
  const std::array<int64_t, 2> input{4, 5};
  const std::array<int64_t, 2> output{2, 5};
  EXPECT_THROW(ValidateFixedStateGeometry("past", kFloat, input, "present", kFloat, output),
               std::runtime_error);
}

TEST(FixedStateGeometryTest, RejectsMismatchedDtype) {
  const std::array<int64_t, 2> input{-1, 5};
  const std::array<int64_t, 2> output{-1, 5};
  EXPECT_THROW(ValidateFixedStateGeometry("past", kFloat, input, "present", kDouble, output),
               std::runtime_error);
}

TEST(FixedStateGeometryTest, RejectsRankMismatch) {
  const std::array<int64_t, 3> input{-1, 2, 3};
  const std::array<int64_t, 2> output{-1, 6};
  EXPECT_THROW(ValidateFixedStateGeometry("past", kFloat, input, "present", kFloat, output),
               std::runtime_error);
}

TEST(FixedStateGeometryTest, RejectsDynamicNonBatchDimension) {
  const std::array<int64_t, 3> input{-1, -1, 3};
  const std::array<int64_t, 3> output{-1, -1, 3};
  EXPECT_THROW(ValidateFixedStateGeometry("past", kFloat, input, "present", kFloat, output),
               std::runtime_error);
}

TEST(FixedStateGeometryTest, RejectsIncompatibleNonBatchGeometry) {
  const std::array<int64_t, 3> input{-1, 2, 3};
  const std::array<int64_t, 3> output{-1, 2, 4};
  EXPECT_THROW(ValidateFixedStateGeometry("past", kFloat, input, "present", kFloat, output),
               std::runtime_error);
}

TEST(FixedStateGeometryTest, RejectsScalarWithoutBatchAxis) {
  const std::array<int64_t, 0> input{};
  const std::array<int64_t, 0> output{};
  EXPECT_THROW(ValidateFixedStateGeometry("past", kFloat, input, "present", kFloat, output),
               std::runtime_error);
}

}  // namespace
}  // namespace test
}  // namespace Generators
