// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <limits>

#include <gtest/gtest.h>

#include "engine/cache_manager.h"
#include "engine/engine_invariants.h"
#include "engine/paged_key_value_cache.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"
#include "models/io/kv_cache.h"
#include "models/model_state_manifest.h"

namespace Generators {
namespace test {
namespace {

std::unique_ptr<PagedKeyValueCache> MakePagedCache(const std::shared_ptr<Model>& model) {
  return std::make_unique<PagedKeyValueCache>(model);
}

class PagedKeyValueCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_ = LoadDummyDecoderModel();
    Config::Engine::DynamicBatching dynamic_batching;
    dynamic_batching.block_size = 4;
    dynamic_batching.num_blocks = 3;
    dynamic_batching.max_batch_size = 2;
    model_->config_->engine.dynamic_batching = dynamic_batching;
    assign_target_ =
        MakeDoublesEngine(model_, /*capacity=*/2, EosToken(*model_)).engine;
    cache_ = MakePagedCache(model_);
  }

  std::shared_ptr<Request> AddCommittedRequest(
      std::array<int32_t, 4> prompt) {
    auto request =
        CreateRequestWithPrompt(assign_target_, prompt);
    cache_->Add(request);
    cache_->AppendTokens(request);
    return request;
  }

  static RequestStepPlan PlanEntry(
      const std::shared_ptr<Request>& request,
      size_t target_cache_slots,
      bool newly_admitted = false,
      size_t whole_sequence_cache_slots = 0) {
    RequestStepPlan entry;
    entry.request = request;
    entry.request_id = request.get();
    entry.target_cache_slots = target_cache_slots;
    entry.whole_sequence_cache_slots = whole_sequence_cache_slots;
    entry.newly_admitted = newly_admitted;
    return entry;
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<Engine> assign_target_;
  std::unique_ptr<PagedKeyValueCache> cache_;
};

TEST_F(PagedKeyValueCacheTest, DeferredActiveRequestRunsAfterCapacityIsReleased) {
  auto first = AddCommittedRequest({2, 3, 4, 5});
  auto second = AddCommittedRequest({6, 7, 8, 9});

  StepPlan plan;
  plan.requests.push_back(PlanEntry(first, 5));
  plan.requests.push_back(PlanEntry(second, 5));

  const auto result = cache_->PlanStepResources(plan);

  ASSERT_TRUE(result.executable);
  EXPECT_TRUE(result.capacity_deferred);
  EXPECT_EQ(result.unserviceable_request_id, nullptr);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, first);

  const std::array reservation_requests{
      PagedCacheReservationRequest{first.get(), 5, false},
  };
  auto reservation = cache_->Reserve(reservation_requests);
  EXPECT_TRUE(
      ValidateCacheInvariants(cache_->Snapshot(reservation)).empty());

  reservation.Commit();
  const auto snapshot = cache_->Snapshot();
  EXPECT_TRUE(ValidateCacheInvariants(snapshot).empty());
  ASSERT_EQ(snapshot.requests.size(), 2u);
  EXPECT_EQ(snapshot.requests[0].request_id, first.get());
  EXPECT_EQ(snapshot.requests[0].used_slots, 5u);
  EXPECT_EQ(snapshot.requests[0].block_ids.size(), 2u);
  EXPECT_EQ(snapshot.requests[1].request_id, second.get());
  EXPECT_EQ(snapshot.requests[1].used_slots, 4u);
  EXPECT_EQ(snapshot.requests[1].block_ids.size(), 1u);

  cache_->Remove(first);
  StepPlan next_plan;
  next_plan.requests.push_back(PlanEntry(second, 5));

  const auto next_result = cache_->PlanStepResources(next_plan);

  ASSERT_TRUE(next_result.executable);
  EXPECT_FALSE(next_result.capacity_deferred);
  EXPECT_EQ(next_result.unserviceable_request_id, nullptr);
  ASSERT_EQ(next_plan.requests.size(), 1u);
  EXPECT_EQ(next_plan.requests[0].request, second);

  const std::array next_reservation_requests{
      PagedCacheReservationRequest{second.get(), 5, false},
  };
  auto next_reservation = cache_->Reserve(next_reservation_requests);
  next_reservation.Commit();

  const auto next_snapshot = cache_->Snapshot();
  EXPECT_TRUE(ValidateCacheInvariants(next_snapshot).empty());
  ASSERT_EQ(next_snapshot.requests.size(), 1u);
  EXPECT_EQ(next_snapshot.requests[0].request_id, second.get());
  EXPECT_EQ(next_snapshot.requests[0].used_slots, 5u);
  EXPECT_EQ(next_snapshot.requests[0].block_ids.size(), 2u);
}

TEST_F(PagedKeyValueCacheTest, ReportsCommittedBoundaryForResident) {
  auto request = AddCommittedRequest({2, 3, 4, 5});

  EXPECT_EQ(cache_->CommittedSlots(request.get()), 4u);
  EXPECT_THROW(cache_->CommittedSlots(this), StepPlanningConsistencyError);
}

TEST_F(PagedKeyValueCacheTest, DeferredActiveRequestsStillConsumeAdmissionCapacity) {
  auto unserviceable = AddCommittedRequest({2, 3, 4, 5});
  auto fitting = AddCommittedRequest({6, 7, 8, 9});
  auto pending = CreateRequestWithPrompt(
      assign_target_, std::array<int32_t, 1>{10});

  StepPlan plan;
  plan.requests.push_back(PlanEntry(unserviceable, 13));
  plan.requests.push_back(PlanEntry(fitting, 4));
  plan.requests.push_back(PlanEntry(pending, 1, true));

  const auto result = cache_->PlanStepResources(plan);

  ASSERT_TRUE(result.executable);
  EXPECT_TRUE(result.capacity_deferred);
  EXPECT_EQ(result.unserviceable_request_id, unserviceable.get());
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, fitting);
  EXPECT_FALSE(plan.requests[0].newly_admitted);
}

// A chunked prefill asks for one chunk at a time, but admission has to be decided on the whole
// prompt: the pool is three blocks of four slots, so a prompt of thirteen slots can never fit even
// though its first chunk would.
TEST_F(PagedKeyValueCacheTest, PromptTooLargeForThePoolIsUnserviceableEvenWhenItsChunkFits) {
  auto pending = CreateRequestWithPrompt(
      assign_target_, std::array<int32_t, 1>{10});

  StepPlan plan;
  plan.requests.push_back(PlanEntry(pending, /*target_cache_slots=*/1, /*newly_admitted=*/true,
                                    /*whole_sequence_cache_slots=*/13));

  const auto result = cache_->PlanStepResources(plan);

  EXPECT_FALSE(result.executable);
  EXPECT_EQ(result.unserviceable_request_id, pending.get());
  EXPECT_TRUE(plan.requests.empty());
}

// Admission also has to wait for enough free blocks to hold the whole prompt, so a request never
// starts a chunked prefill it cannot finish.
TEST_F(PagedKeyValueCacheTest, AdmissionWaitsUntilTheWholePromptFits) {
  auto committed = AddCommittedRequest({2, 3, 4, 5});
  auto pending = CreateRequestWithPrompt(
      assign_target_, std::array<int32_t, 1>{10});

  StepPlan plan;
  plan.requests.push_back(PlanEntry(committed, /*target_cache_slots=*/4));
  // One block is already taken, leaving two of the three: the chunk needs one, the prompt needs
  // three.
  plan.requests.push_back(PlanEntry(pending, /*target_cache_slots=*/1, /*newly_admitted=*/true,
                                    /*whole_sequence_cache_slots=*/9));

  const auto result = cache_->PlanStepResources(plan);

  ASSERT_TRUE(result.executable);
  EXPECT_TRUE(result.capacity_deferred);
  EXPECT_EQ(result.unserviceable_request_id, nullptr);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, committed);
}

TEST_F(PagedKeyValueCacheTest, OmittedResidentKeepsItsCommittedBlockTable) {
  auto omitted = AddCommittedRequest({2, 3, 4, 5});
  auto scheduled = AddCommittedRequest({6, 7, 8, 9});
  const auto before = cache_->Snapshot();

  StepPlan plan;
  plan.requests.push_back(PlanEntry(scheduled, 5));

  const auto result = cache_->PlanStepResources(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  const std::array reservation_requests{
      PagedCacheReservationRequest{scheduled.get(), 5, false},
  };
  auto reservation = cache_->Reserve(reservation_requests);
  reservation.Commit();

  const auto after = cache_->Snapshot();
  ASSERT_EQ(before.requests.size(), 2u);
  ASSERT_EQ(after.requests.size(), 2u);
  EXPECT_EQ(after.requests[0].request_id, omitted.get());
  EXPECT_EQ(after.requests[0].block_ids, before.requests[0].block_ids);
  EXPECT_EQ(after.requests[0].used_slots, before.requests[0].used_slots);
  EXPECT_EQ(after.requests[1].request_id, scheduled.get());
  EXPECT_EQ(after.requests[1].used_slots, 5u);
}

TEST_F(PagedKeyValueCacheTest, OmittedResidentsStillLimitNewAdmissions) {
  auto first = AddCommittedRequest({2, 3, 4, 5});
  auto second = AddCommittedRequest({6, 7, 8, 9});
  auto pending = CreateRequestWithPrompt(
      assign_target_, std::array<int32_t, 1>{10});

  StepPlan plan;
  plan.requests.push_back(PlanEntry(pending, 1, true, 1));

  const auto result = cache_->PlanStepResources(plan);

  EXPECT_FALSE(result.executable);
  EXPECT_TRUE(result.capacity_deferred);
  EXPECT_EQ(result.outcome.kind, StepOutcomeKind::CapacityDeferred);
  EXPECT_TRUE(plan.requests.empty());
}

TEST_F(PagedKeyValueCacheTest, BlockedPrefillDoesNotPreventLaterAdmission) {
  auto resident = AddCommittedRequest({2, 3, 4, 5});
  auto blocked = CreateRequestWithPrompt(
      assign_target_, std::array<int32_t, 1>{10});
  auto fitting = CreateRequestWithPrompt(
      assign_target_, std::array<int32_t, 1>{11});

  StepPlan plan;
  plan.requests.push_back(PlanEntry(resident, 4));
  plan.requests.push_back(PlanEntry(blocked, 1, true, 9));
  plan.requests.push_back(PlanEntry(fitting, 1, true, 4));

  const auto result = cache_->PlanStepResources(plan);

  ASSERT_TRUE(result.executable);
  EXPECT_TRUE(result.capacity_deferred);
  ASSERT_EQ(plan.requests.size(), 2u);
  EXPECT_EQ(plan.requests[0].request, resident);
  EXPECT_EQ(plan.requests[1].request, fitting);
}

TEST_F(PagedKeyValueCacheTest, InterleavedAdmissionAndResidentSubsetAreSelectedByIdentity) {
  model_->config_->engine.dynamic_batching->max_batch_size = 3;
  cache_ = MakePagedCache(model_);

  auto omitted = AddCommittedRequest({2, 3, 4, 5});
  auto resident = AddCommittedRequest({6, 7, 8, 9});
  auto pending = CreateRequestWithPrompt(
      assign_target_, std::array<int32_t, 1>{10});

  StepPlan plan;
  plan.requests.push_back(PlanEntry(pending, 1, true, 1));
  plan.requests.push_back(PlanEntry(resident, 4));

  const auto result = cache_->PlanStepResources(plan);

  ASSERT_TRUE(result.executable);
  EXPECT_FALSE(result.capacity_deferred);
  ASSERT_EQ(plan.requests.size(), 2u);
  EXPECT_EQ(plan.requests[0].request, pending);
  EXPECT_TRUE(plan.requests[0].newly_admitted);
  EXPECT_EQ(plan.requests[1].request, resident);
  EXPECT_FALSE(plan.requests[1].newly_admitted);

  const auto snapshot = cache_->Snapshot();
  ASSERT_EQ(snapshot.requests.size(), 2u);
  EXPECT_EQ(snapshot.requests[0].request_id, omitted.get());
  EXPECT_EQ(snapshot.requests[0].used_slots, 4u);
}

TEST_F(PagedKeyValueCacheTest, GlobalOnlyPrefillChunkReservesWholePrompt) {
  auto pending = CreateRequestWithPrompt(
      assign_target_, std::array<int32_t, 9>{2, 3, 4, 5, 6, 7, 8, 9, 10});

  StepPlan plan;
  plan.requests.push_back(
      PlanEntry(pending, /*target_cache_slots=*/3,
                /*newly_admitted=*/true,
                /*whole_sequence_cache_slots=*/9));

  const auto result = cache_->PlanStepResources(plan);

  ASSERT_TRUE(result.executable);
  const std::array reservation_requests{
      PagedCacheReservationRequest{pending.get(), 3, true, 9},
  };
  auto reservation = cache_->Reserve(reservation_requests);
  EXPECT_EQ(reservation.ReservedBlockCount(), 3u);
}

TEST(PagedKeyValueCacheManifestTest, AllocatesOnlySparseLogicalLayersUsingDecoderBindings) {
  auto model = LoadSyntheticPagedModel();
  ASSERT_TRUE(model->config_->model.decoder.state_groups.has_value());
  ASSERT_EQ(model->config_->model.decoder.state_groups->size(), 1u);
  EXPECT_EQ(model->config_->model.decoder.state_groups->front().layer_ids,
            std::vector<int>({1, 4}));

  auto cache = MakePagedCache(model);

  const auto values = cache->Cache();
  const auto input_names = cache->Names();
  const auto output_names = cache->OutputNames();

  ASSERT_EQ(values.size(), 2u);
  ASSERT_EQ(input_names.size(), 2u);
  ASSERT_EQ(output_names.size(), 2u);
  EXPECT_STREQ(input_names[0].first, "past_key_values.1.key");
  EXPECT_STREQ(input_names[0].second, "past_key_values.1.value");
  EXPECT_STREQ(output_names[0].first, "present.1.key");
  EXPECT_STREQ(output_names[0].second, "present.1.value");
  EXPECT_STREQ(input_names[1].first, "past_key_values.4.key");
  EXPECT_STREQ(input_names[1].second, "past_key_values.4.value");
  EXPECT_STREQ(output_names[1].first, "present.4.key");
  EXPECT_STREQ(output_names[1].second, "present.4.value");
  EXPECT_EQ(
      values[0].first->GetTensorTypeAndShapeInfo()->GetElementType(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
}

TEST(PagedKeyValueCacheManifestTest, DenseLegacyManifestRetainsSequentialBindings) {
  auto model = LoadDummyDecoderModel();
  model->config_->engine.dynamic_batching = Config::Engine::DynamicBatching{};
  model->config_->engine.dynamic_batching->block_size = 4;
  model->config_->engine.dynamic_batching->num_blocks = 3;
  ASSERT_FALSE(model->config_->model.decoder.state_groups.has_value());

  auto cache = MakePagedCache(model);
  const auto input_names = cache->Names();
  const auto output_names = cache->OutputNames();

  ASSERT_EQ(input_names.size(), 1u);
  ASSERT_EQ(output_names.size(), 1u);
  EXPECT_STREQ(input_names[0].first, "past_key_values.0.key");
  EXPECT_STREQ(input_names[0].second, "past_key_values.0.value");
  EXPECT_STREQ(output_names[0].first, "present.0.key");
  EXPECT_STREQ(output_names[0].second, "present.0.value");
  EXPECT_EQ(
      cache->Cache()[0].first->GetTensorTypeAndShapeInfo()->GetElementType(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
}

TEST(PagedKeyValueCacheManifestTest, CacheManagerConstructsFromSparseManifest) {
  auto manager = CacheManager::Create(LoadSyntheticPagedModel());

  EXPECT_TRUE(manager->SupportsDynamicBatching());
  EXPECT_EQ(manager->Snapshot().total_blocks, 128u);
}

TEST(PagedKeyValueCacheManifestTest, AllocationUsesPhysicalHeadWidth) {
  auto model = LoadSyntheticPagedModel();
  model->config_->model.decoder.head_size = 2;
  auto cache = MakePagedCache(model);
  EXPECT_EQ(PagedKeyValueCacheBytesPerBlock(model), 64u);
  for (const auto& values : cache->Cache()) {
    EXPECT_EQ(values.first->GetTensorTypeAndShapeInfo()->GetShape().back(), 1);
    EXPECT_EQ(values.second->GetTensorTypeAndShapeInfo()->GetShape().back(), 1);
  }
}

TEST(PagedKeyValueCacheManifestTest, PackedScaleCapacityUsesExactBytes) {
  constexpr size_t payload = 16 * 2 * 256 * 4 * 128;
  constexpr size_t scales = 16 * 2 * 256 * 4 * 2;
  EXPECT_EQ(payload + scales, 4259840u);
  EXPECT_EQ(ComputePagedBlockCapacityFromBytes(1024 * (payload + scales), 1.0f, 0,
                                               payload + scales),
            921u);
  EXPECT_THROW(ComputePagedBlockCapacityFromBytes(1024, 1.0f, 0, 0), std::invalid_argument);
  EXPECT_THROW(ComputePagedBlockCapacityFromBytes(1024, 1.0f, 0, 1,
                                                  std::numeric_limits<size_t>::max()),
               std::overflow_error);
}

TEST(PagedKeyValueCacheManifestTest, ScaleCachesBindAliasesAndPreserveBlockTableOffset) {
  auto model = CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged-scales");
  auto cache = MakePagedCache(model);
  auto params = CreateGeneratorParams(*model);
  ModelIO state(*params, *model);
  EXPECT_EQ(PagedKeyValueCacheBytesPerBlock(model), 96u);
  cache->UpdateState(state, {});
  ASSERT_EQ(state.inputs_.size(), 9u);
  ASSERT_EQ(state.outputs_.size(), 8u);
  EXPECT_STREQ(state.input_names_[8], "block_table");
  for (size_t index = 4; index < 8; ++index) {
    EXPECT_EQ(state.inputs_[index], state.outputs_[index]);
    EXPECT_EQ(state.inputs_[index]->GetTensorTypeAndShapeInfo()->GetShape(),
              std::vector<int64_t>({128, 4, 1}));
    EXPECT_EQ(state.inputs_[index]->GetTensorTypeAndShapeInfo()->GetElementType(),
              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    const auto* scale_bytes = static_cast<const uint8_t*>(state.inputs_[index]->GetTensorRawData());
    for (size_t offset = 0; offset < 128 * 4 * sizeof(Ort::Float16_t); ++offset) {
      EXPECT_EQ(scale_bytes[offset], 0u);
    }
  }
  EXPECT_STREQ(state.input_names_[4], "past_key_values.1.key_scale");
  EXPECT_STREQ(state.output_names_[7], "present.4.value_scale");
  auto* first_scale = state.inputs_[4];
  cache->UpdateState(state, {});
  EXPECT_EQ(state.inputs_[4], first_scale);
  EXPECT_STREQ(state.input_names_[8], "block_table");
}

// Executable coverage for the aliasing contract: the ONNX graph actually runs with every scale
// buffer bound as both a past input and a present output, across a prefill step and several decode
// steps. The bind-level test above can only show that the pointers match; this one shows ORT
// accepts the aliased binding for a real session run and that the Engine keeps stepping with it.
TEST(PagedKeyValueCacheManifestTest, RunsPrefillAndDecodeStepsWithAliasedScaleBuffers) {
  auto model = CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged-scales");
  auto engine = std::make_shared<Engine>(model);
  auto request = CreateEngineRequest(engine);
  TurnOptions options;
  options.max_generated_tokens = 4;
  options.min_generated_tokens = 4;
  const std::array<int32_t, 3> prompt{2, 3, 4};
  request->BeginTurn(prompt, options);

  std::array<EngineEvent, 8> events;
  std::vector<int32_t> generated;
  for (int step = 0; step < 32 && !request->IsTurnComplete(); ++step) {
    const size_t count = engine->Run(events);
    for (size_t i = 0; i < count; ++i) {
      if (events[i].request.get() == request.get() && (events[i].flags & EngineEventFlagToken)) {
        generated.push_back(events[i].token);
      }
    }
  }
  EXPECT_TRUE(request->IsTurnComplete());
  EXPECT_EQ(generated.size(), 4u);
  EXPECT_TRUE(ValidateRequestInvariants(request->Snapshot()).empty());
}

// The scale buffers are Engine-owned and updated in place, so both sides of the binding must point
// at one allocation and that allocation must survive the step boundary. If a step ever bound a
// fresh buffer for the present side, the scales it produced would be discarded and the next step
// would dequantize the block it had just written against stale data.
TEST(PagedKeyValueCacheManifestTest, ScaleBuffersStayAliasedAndPersistAcrossSteps) {
  auto model = CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged-scales");
  auto cache = MakePagedCache(model);
  auto params = CreateGeneratorParams(*model);
  ModelIO state(*params, *model);

  cache->UpdateState(state, {});
  ASSERT_EQ(state.inputs_.size(), 9u);
  ASSERT_EQ(state.outputs_.size(), 8u);
  std::array<const void*, 4> first_step_data{};
  for (size_t index = 4; index < 8; ++index) {
    ASSERT_EQ(state.inputs_[index], state.outputs_[index]);
    // One buffer, two distinct graph names: the model reads `past` and writes `present`.
    ASSERT_STRNE(state.input_names_[index], state.output_names_[index]);
    first_step_data[index - 4] = state.inputs_[index]->GetTensorRawData();
    ASSERT_NE(first_step_data[index - 4], nullptr);
  }

  // Stands in for the model's in-place scale update: the synthetic graph does not quantize, so the
  // observable contract here is buffer identity and persistence, not the scale values themselves.
  static_cast<uint8_t*>(state.outputs_[4]->GetTensorMutableRawData())[0] = 0xAB;

  cache->UpdateState(state, {});
  ASSERT_EQ(state.inputs_.size(), 9u);
  ASSERT_EQ(state.outputs_.size(), 8u);
  for (size_t index = 4; index < 8; ++index) {
    EXPECT_EQ(state.inputs_[index], state.outputs_[index]);
    EXPECT_EQ(state.inputs_[index]->GetTensorRawData(), first_step_data[index - 4]);
  }
  EXPECT_EQ(static_cast<const uint8_t*>(state.inputs_[4]->GetTensorRawData())[0], 0xABu);
  EXPECT_STREQ(state.input_names_[8], "block_table");
}

TEST(PagedKeyValueCacheManifestTest, RequiresScaleTemplatesAsAPair) {
  for (int field = 0; field < 4; ++field) {
    auto model = CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged-scales");
    auto& inputs = model->config_->model.decoder.inputs;
    auto& outputs = model->config_->model.decoder.outputs;
    switch (field) {
      case 0:
        inputs.past_key_scale_names.clear();
        break;
      case 1:
        inputs.past_value_scale_names.clear();
        break;
      case 2:
        outputs.present_key_scale_names.clear();
        break;
      default:
        outputs.present_value_scale_names.clear();
        break;
    }
    try {
      static_cast<void>(MakePagedCache(model));
      FAIL() << "Expected an unpaired scale template to be rejected, field " << field;
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string{error.what()}.find("both a past and a present"), std::string::npos)
          << "field " << field << ": " << error.what();
    }
  }
  // Clearing both sides of one side-of-cache is the supported way to bind fewer scale caches.
  auto model = CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged-scales");
  model->config_->model.decoder.inputs.past_value_scale_names.clear();
  model->config_->model.decoder.outputs.present_value_scale_names.clear();
  auto cache = MakePagedCache(model);
  auto params = CreateGeneratorParams(*model);
  ModelIO state(*params, *model);
  cache->UpdateState(state, {});
  // Two layers of key/value, one key scale per layer, then the block table.
  EXPECT_EQ(state.inputs_.size(), 7u);
  EXPECT_EQ(state.outputs_.size(), 6u);
  EXPECT_STREQ(state.input_names_[6], "block_table");
  EXPECT_EQ(PagedKeyValueCacheBytesPerBlock(model), 80u);
}

// Byte accounting reads the element type per layer while allocation commits to one type for the
// whole pool, so the two agree only if every binding shares a type. Tie them together directly:
// the bytes the pool actually holds must equal bytes-per-block times the block count that sizing
// produced, for both the plain and the per-token quantized fixture.
TEST(PagedKeyValueCacheManifestTest, AccountingMatchesAllocatedBytesAndTypes) {
  for (const char* path : {MODEL_PATH "engine/synthetic-paged",
                           MODEL_PATH "engine/synthetic-paged-scales"}) {
    auto model = CreateModel(GetOrtEnv(), path);
    auto cache = MakePagedCache(model);
    auto params = CreateGeneratorParams(*model);
    ModelIO state(*params, *model);
    cache->UpdateState(state, {});

    const auto kv = cache->Cache();
    ASSERT_FALSE(kv.empty()) << path;
    const auto kv_type = kv.front().first->GetTensorTypeAndShapeInfo()->GetElementType();
    const int64_t blocks = kv.front().first->GetTensorTypeAndShapeInfo()->GetShape()[0];

    const auto& decoder = model->config_->model.decoder;
    for (const int layer : {1, 4}) {
      EXPECT_EQ(model->session_info_.GetInputDataType(
                    ComposeKeyValueName(decoder.inputs.past_key_names, layer)),
                kv_type)
          << path;
      EXPECT_EQ(model->session_info_.GetInputDataType(
                    ComposeKeyValueName(decoder.inputs.past_value_names, layer)),
                kv_type)
          << path;
      EXPECT_EQ(model->session_info_.GetOutputDataType(
                    ComposeKeyValueName(decoder.outputs.present_key_names, layer)),
                kv_type)
          << path;
      EXPECT_EQ(model->session_info_.GetOutputDataType(
                    ComposeKeyValueName(decoder.outputs.present_value_names, layer)),
                kv_type)
          << path;
    }

    // Every cache and scale tensor is bound on both sides, so outputs_ counts exactly the buffers
    // the pool owns. Neither fixture is windowed, so all of them are billed per full-attention
    // block by PagedKeyValueCacheBytesPerBlock.
    size_t allocated_bytes = 0;
    for (size_t index = 0; index < state.outputs_.size(); ++index) {
      const auto info = state.inputs_[index]->GetTensorTypeAndShapeInfo();
      size_t elements = 1;
      for (const auto dimension : info->GetShape()) {
        elements *= static_cast<size_t>(dimension);
      }
      allocated_bytes += elements * Ort::SizeOf(info->GetElementType());
    }
    EXPECT_EQ(allocated_bytes,
              PagedKeyValueCacheBytesPerBlock(model) * static_cast<size_t>(blocks))
        << path;
  }
}

TEST(PagedKeyValueCacheManifestTest, RejectsScaleBlockCountMismatch) {
  auto model = CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged-scales");
  model->config_->engine.dynamic_batching->num_blocks = 64;
  EXPECT_THROW(MakePagedCache(model), std::runtime_error);
}

TEST(PagedKeyValueCacheManifestTest, RejectsMalformedAndDuplicateScaleTemplates) {
  auto model = CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged-scales");
  model->config_->model.decoder.inputs.past_key_scale_names = "scale.%s";
  EXPECT_THROW(MakePagedCache(model), std::runtime_error);
  model->config_->model.decoder.inputs.past_key_scale_names =
      model->config_->model.decoder.inputs.past_value_scale_names;
  EXPECT_THROW(MakePagedCache(model), std::runtime_error);
}

TEST(PagedKeyValueCacheManifestTest, RejectsRankFourTensorAsScaleCache) {
  auto model = LoadSyntheticPagedModel();
  auto& inputs = model->config_->model.decoder.inputs;
  auto& outputs = model->config_->model.decoder.outputs;
  // ValidateScaleBindings seeds its uniqueness set from the key/value templates and rejects a
  // colliding scale name before ScaleCacheType ever inspects the tensor, and the past/present
  // templates are required as a pair. Aliasing both key bindings onto their value counterparts
  // frees "past_key_values.%d.key" and "present.%d.key" to stand in for a wrongly shaped scale
  // pair, so the rank check is the guard that actually fires.
  inputs.past_key_names = inputs.past_value_names;
  outputs.present_key_names = outputs.present_value_names;
  inputs.past_key_scale_names = "past_key_values.%d.key";
  outputs.present_key_scale_names = "present.%d.key";
  try {
    static_cast<void>(MakePagedCache(model));
    FAIL() << "Expected a rank-four tensor to be rejected as a scale cache";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find("rank three"), std::string::npos) << error.what();
  }
}

TEST(PagedKeyValueCacheManifestTest, ReportsScaleGeometryMismatchPerDimension) {
  {
    auto model = CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged-scales");
    model->config_->engine.dynamic_batching->block_size = 8;
    try {
      static_cast<void>(MakePagedCache(model));
      FAIL() << "Expected a block-size mismatch to be rejected";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string{error.what()}.find("block-size dimension"), std::string::npos)
          << error.what();
    }
  }
  {
    auto model = CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged-scales");
    model->config_->model.decoder.num_key_value_heads = 2;
    try {
      static_cast<void>(MakePagedCache(model));
      FAIL() << "Expected a head-count mismatch to be rejected";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string{error.what()}.find("head dimension"), std::string::npos)
          << error.what();
    }
  }
}

// The scale shape check must use the same wildcard-tolerant comparison the state manifest applies
// to key/value bindings: the model builder emits symbolic num_blocks and block_size dims, so an
// exact vector comparison would reject a present/past pair that differs only in which dims the
// exporter happened to resolve.
TEST(PagedKeyValueCacheManifestTest, ScaleShapeComparisonTreatsSymbolicDimsAsWildcards) {
  EXPECT_TRUE(StateShapesCompatible({-1, 4, 1}, {128, 4, 1}));
  EXPECT_TRUE(StateShapesCompatible({128, -1, 1}, {-1, 4, 1}));
  EXPECT_FALSE(StateShapesCompatible({128, 4, 1}, {128, 4, 2}));
  EXPECT_FALSE(StateShapesCompatible({128, 4, 1}, {128, 4}));
}

TEST(PagedKeyValueCacheManifestTest, BlockCapacityUsesParticipatingLayerCount) {
  // Bytes one block costs across `layers` full-attention layers of the geometry used below:
  // block_size 4 * one KV head * head width 2 * 2-byte elements * two caches (key and value).
  const auto primary_bytes = [](size_t layers) { return size_t{4} * 1 * 2 * 2 * 2 * layers; };
  ASSERT_EQ(primary_bytes(2), 64u);
  ASSERT_EQ(primary_bytes(6), 192u);

  // 10240 bytes at the default 90% utilization leaves a 9216-byte budget.
  EXPECT_EQ(ComputePagedBlockCapacityFromBytes(10240, 1.0f, 0, primary_bytes(2)), 144u);
  EXPECT_EQ(ComputePagedBlockCapacityFromBytes(10240, 1.0f, 0, primary_bytes(6)), 48u);

  // One auxiliary layer with the same geometry adds 32 bytes to the primary model's 64 bytes per
  // block, so both pools together fit 96 blocks in the same 90%-adjusted memory budget.
  EXPECT_EQ(ComputePagedBlockCapacityFromBytes(10240, 1.0f, 0, primary_bytes(2), 32), 96u);

  EXPECT_EQ(ComputePagedBlockCapacityFromBytes(10240, 1.0f, 1024, primary_bytes(2)), 128u);

  EXPECT_THROW(ComputePagedBlockCapacityFromBytes(10240, 1.0f, 9216, primary_bytes(2)),
               std::runtime_error);
  EXPECT_THROW(ComputePagedBlockCapacityFromBytes(10240, 1.0f, 0, primary_bytes(2),
                                                  std::numeric_limits<size_t>::max()),
               std::overflow_error);
}

// The production path: per-block bytes come from the graph, and capacity divides the budget by
// them. The quantized fixture pays 96 bytes per block against the plain fixture's 64 because each
// layer adds two float16 scales per slot, so the same budget holds fewer blocks.
TEST(PagedKeyValueCacheManifestTest, CapacityUsesBytesPerBlockFromTheGraph) {
  const size_t plain_bytes = PagedKeyValueCacheBytesPerBlock(LoadSyntheticPagedModel());
  const size_t scaled_bytes = PagedKeyValueCacheBytesPerBlock(
      CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged-scales"));
  EXPECT_EQ(plain_bytes, 64u);
  EXPECT_EQ(scaled_bytes, 96u);
  EXPECT_EQ(ComputePagedBlockCapacityFromBytes(10240, 1.0f, 0, plain_bytes), 144u);
  EXPECT_EQ(ComputePagedBlockCapacityFromBytes(10240, 1.0f, 0, scaled_bytes), 96u);
}

TEST(PagedKeyValueCacheManifestTest, RejectsPerBlockByteOverflow) {
  auto model = LoadSyntheticPagedModel();
  model->config_->engine.dynamic_batching->block_size = std::numeric_limits<size_t>::max();
  EXPECT_THROW(PagedKeyValueCacheBytesPerBlock(model), std::overflow_error);
}

TEST(PagedKeyValueCacheManifestTest, ExplicitBlockCountCoversBothPools) {
  // Without a head, an explicit num_blocks is used verbatim.
  EXPECT_EQ(
      ResolveConfiguredPagedBlockCount(
          /*configured_num_blocks=*/100,
          /*primary_bytes_per_block=*/64,
          /*auxiliary_bytes_per_block=*/0),
      100u);

  // num_blocks is the combined budget: 100 blocks of 64 bytes is 6400 bytes, which holds 66 blocks
  // of the 96 bytes one target block plus one head block cost together.
  EXPECT_EQ(
      ResolveConfiguredPagedBlockCount(
          /*configured_num_blocks=*/100,
          /*primary_bytes_per_block=*/64,
          /*auxiliary_bytes_per_block=*/32),
      66u);

  // DSpark also reserves fixed spill blocks for each active query. Deduct those bytes from the
  // configured budget before splitting the remainder between target and drafter blocks.
  EXPECT_EQ(
      ResolveConfiguredPagedBlockCount(
          /*configured_num_blocks=*/100,
          /*primary_bytes_per_block=*/64,
          /*auxiliary_bytes_per_block=*/32,
          /*auxiliary_reserved_memory_bytes=*/1024),
      56u);

  EXPECT_THROW(
      ResolveConfiguredPagedBlockCount(
          /*configured_num_blocks=*/16,
          /*primary_bytes_per_block=*/64,
          /*auxiliary_bytes_per_block=*/32,
          /*auxiliary_reserved_memory_bytes=*/1024),
      std::runtime_error);

  // Fixed auxiliary state, including a preallocated windowed drafter pool, also consumes the
  // configured budget when there is no per-block head.
  EXPECT_EQ(
      ResolveConfiguredPagedBlockCount(
          /*configured_num_blocks=*/100,
          /*primary_bytes_per_block=*/64,
          /*auxiliary_bytes_per_block=*/0,
          /*auxiliary_reserved_memory_bytes=*/64),
      99u);

  // A budget too small to leave a block for each pool is rejected rather than silently allocating
  // an extra head pool outside the configured sizing contract.
  EXPECT_THROW(
      ResolveConfiguredPagedBlockCount(
          /*configured_num_blocks=*/1,
          /*primary_bytes_per_block=*/32,
          /*auxiliary_bytes_per_block=*/64),
      std::runtime_error);

  EXPECT_THROW(
      ResolveConfiguredPagedBlockCount(
          /*configured_num_blocks=*/std::numeric_limits<size_t>::max(),
          /*primary_bytes_per_block=*/64,
          /*auxiliary_bytes_per_block=*/32),
      std::runtime_error);

  EXPECT_THROW(
      ResolveConfiguredPagedBlockCount(
          /*configured_num_blocks=*/1,
          /*primary_bytes_per_block=*/64,
          /*auxiliary_bytes_per_block=*/0,
          /*auxiliary_reserved_memory_bytes=*/64),
      std::runtime_error);
}

TEST(PagedKeyValueCacheManifestTest, AllocatesSparseSlidingAndFullLayerCaches) {
  auto model = LoadSyntheticPagedModel();
  auto& decoder = model->config_->model.decoder;
  decoder.sliding_window = Config::Model::Decoder::SlidingWindow{};
  decoder.sliding_window->window_size = 4;
  decoder.sliding_window->layers = {1};
  decoder.inputs.block_table_windowed = decoder.inputs.block_table;
  model->config_->search.chunk_size = 4;

  auto cache = MakePagedCache(model);
  const auto values = cache->Cache();

  ASSERT_EQ(values.size(), 2u);
  EXPECT_EQ(
      values[0].first->GetTensorTypeAndShapeInfo()->GetShape(),
      std::vector<int64_t>({16, 4, 1, 1}));
  EXPECT_EQ(
      values[1].first->GetTensorTypeAndShapeInfo()->GetShape(),
      std::vector<int64_t>({128, 4, 1, 1}));
}

TEST(PagedKeyValueCacheManifestTest, RejectsSlidingWindowLayersOutsidePagedGroup) {
  auto model = LoadSyntheticPagedModel();
  auto& decoder = model->config_->model.decoder;
  decoder.sliding_window = Config::Model::Decoder::SlidingWindow{};
  decoder.sliding_window->window_size = 4;
  decoder.sliding_window->layers = {0};
  decoder.inputs.block_table_windowed = decoder.inputs.block_table;
  model->config_->search.chunk_size = 4;

  EXPECT_THROW(
      {
        try {
          auto cache = MakePagedCache(model);
        } catch (const std::runtime_error& error) {
          EXPECT_NE(
              std::string{error.what()}.find(
                  "Every sliding-window layer must belong to the paged_kv decoder state group"),
              std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST(PagedKeyValueCacheManifestTest, RejectsMultiplePagedGroups) {
  auto model = LoadSyntheticPagedModel();
  auto& groups = *model->config_->model.decoder.state_groups;
  auto second_paged_group = groups.front();
  groups.front().layer_ids = {1};
  second_paged_group.layer_ids = {4};
  groups.push_back(std::move(second_paged_group));

  EXPECT_THROW(
      {
        try {
          auto manager = CacheManager::Create(model);
        } catch (const std::runtime_error& error) {
          EXPECT_NE(
              std::string{error.what()}.find(
                  "requires exactly one paged_kv decoder state group"),
              std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST(PagedKeyValueCacheManifestTest, RejectsFixedStateGroupsAbsentFromSession) {
  // Fixed decoder state groups are now supported (the composite manager owns a FixedStatePool), but
  // their bindings still have to resolve to real session inputs and outputs. The synthetic-paged
  // session has no such tensors, so pool construction rejects the group at session validation.
  auto model = LoadSyntheticPagedModel();
  Config::Model::Decoder::StateGroup fixed_group;
  fixed_group.kind = Config::Model::Decoder::StateGroupKind::FixedConv;
  fixed_group.layer_ids = {0};
  model->config_->model.decoder.state_groups->push_back(std::move(fixed_group));

  EXPECT_THROW(
      {
        try {
          PagedCacheManager manager{model};
        } catch (const std::runtime_error& error) {
          EXPECT_NE(
              std::string{error.what()}.find("was not found"),
              std::string::npos)
              << error.what();
          throw;
        }
      },
      std::runtime_error);
}

TEST(PagedKeyValueCacheManifestTest, PublicEngineAcceptsPackedFixedState) {
  auto model = LoadSyntheticCompositeModel();

  EXPECT_NO_THROW({
    auto engine = std::make_shared<Engine>(model);
  });
}

TEST(PagedKeyValueCacheManifestTest, RejectsFixedStateWithStaticBatching) {
  auto model = LoadSyntheticCompositeModel();
  model->config_->engine.dynamic_batching.reset();
  EXPECT_THROW(
      {
        try {
          auto manager = CacheManager::Create(model);
        } catch (const std::runtime_error& error) {
          EXPECT_NE(
              std::string{error.what()}.find("require engine.dynamic_batching"),
              std::string::npos)
              << error.what();
          throw;
        }
      },
      std::runtime_error);
}

TEST(PagedKeyValueCacheManifestTest, LegacyCompositeEntryPointsRejectBeforeDivergence) {
  auto model = LoadSyntheticCompositeModel();
  PagedCacheManager manager{model};
  auto owner = MakeDoublesEngine(
                   model, /*capacity=*/1, EosToken(*model))
                   .engine;
  const std::array<int32_t, 3> prompt{2, 3, 4};
  auto request = CreateRequestWithPrompt(owner, prompt);

  EXPECT_THROW(manager.Allocate({request}), std::logic_error);
  EXPECT_THROW(manager.Step(), std::logic_error);
  EXPECT_FALSE(manager.IsResident(request));
  EXPECT_EQ(manager.ResidentRequestCount(), 0u);
}

TEST(PagedKeyValueCacheManifestTest, RejectsMalformedPagedGroup) {
  auto model = LoadSyntheticPagedModel();
  model->config_->model.decoder.state_groups->front().layer_ids.clear();

  EXPECT_THROW(
      {
        try {
          auto cache = MakePagedCache(model);
        } catch (const std::runtime_error& error) {
          EXPECT_NE(
              std::string{error.what()}.find("requires a non-empty paged_kv decoder state group"),
              std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST(PagedKeyValueCacheManifestTest, RejectsEmptyLegacyPagedGroup) {
  auto model = LoadDummyDecoderModel();
  model->config_->model.decoder.num_hidden_layers = 0;
  model->config_->engine.dynamic_batching = Config::Engine::DynamicBatching{};
  model->config_->engine.dynamic_batching->block_size = 4;
  model->config_->engine.dynamic_batching->num_blocks = 3;

  EXPECT_THROW(
      {
        try {
          auto cache = MakePagedCache(model);
        } catch (const std::runtime_error& error) {
          EXPECT_NE(
              std::string{error.what()}.find("at least one paged_kv decoder layer"),
              std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

}  // namespace
}  // namespace test
}  // namespace Generators
