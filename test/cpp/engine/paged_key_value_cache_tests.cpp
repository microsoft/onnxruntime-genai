// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <algorithm>
#include <limits>

#include <gtest/gtest.h>

#include "engine/cache_manager.h"
#include "engine/engine_invariants.h"
#include "engine/paged_key_value_cache.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"
#include "models/io/kv_cache.h"
#include "models/model_state_manifest.h"
#include "logging.h"

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

TEST_F(PagedKeyValueCacheTest, ValidatedRemovalReleasesAndReindexesCommittedTables) {
  auto first = AddCommittedRequest({2, 3, 4, 5});
  auto second = AddCommittedRequest({6, 7, 8, 9});

  cache_->ValidateRemove(first.get());
  cache_->RemoveValidated(first.get());

  EXPECT_FALSE(cache_->OwnsRequest(first.get()));
  EXPECT_TRUE(cache_->OwnsRequest(second.get()));
  EXPECT_EQ(cache_->CommittedSlots(second.get()), 4u);
  const auto snapshot = cache_->Snapshot();
  ASSERT_EQ(snapshot.requests.size(), 1u);
  EXPECT_EQ(snapshot.requests.front().request_id, second.get());
  EXPECT_EQ(snapshot.free_blocks, 2u);
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
  model->config_->engine.dynamic_batching->prefix_caching = false;
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

TEST(PagedKeyValueCacheManifestTest, RemovalReusesCommittedSlidingWindowCapacity) {
  auto model = LoadSyntheticPagedModel();
  model->config_->engine.dynamic_batching->prefix_caching = false;
  auto& decoder = model->config_->model.decoder;
  decoder.sliding_window = Config::Model::Decoder::SlidingWindow{};
  decoder.sliding_window->window_size = 4;
  decoder.sliding_window->layers = {1};
  decoder.inputs.block_table_windowed = decoder.inputs.block_table;
  model->config_->search.chunk_size = 4;
  auto assign_target =
      MakeDoublesEngine(model, /*capacity=*/2, EosToken(*model)).engine;
  auto cache = MakePagedCache(model);

  auto first = CreateRequestWithPrompt(
      assign_target, std::array<int32_t, 4>{2, 3, 4, 5});
  cache->Add(first);
  cache->AppendTokens(first);
  const auto before = cache->Snapshot();
  ASSERT_EQ(before.requests.size(), 1u);

  cache->Remove(first);

  EXPECT_FALSE(cache->OwnsRequest(first.get()));
  const auto removed = cache->Snapshot();
  EXPECT_TRUE(removed.requests.empty());
  EXPECT_EQ(removed.free_blocks, removed.total_blocks);

  auto second = CreateRequestWithPrompt(
      assign_target, std::array<int32_t, 4>{6, 7, 8, 9});
  cache->Add(second);
  cache->AppendTokens(second);
  EXPECT_TRUE(cache->OwnsRequest(second.get()));
}

TEST(PagedKeyValueCacheManifestTest, MtpPrefixCachingRequiresExplicitBatchOneOptIn) {
  auto model = LoadSyntheticPagedModel();
  auto& batching = *model->config_->engine.dynamic_batching;
  model->config_->model.mtp.enabled = true;
  model->config_->model.mtp.filename = "mtp.onnx";
  batching.max_batch_size = 1;
  batching.prefix_caching = true;
  batching.prefix_caching_explicitly_set = false;
  EXPECT_FALSE(ResolvePrefixCachingEnabled(model, 1));
  batching.prefix_caching_explicitly_set = true;
  EXPECT_TRUE(ResolvePrefixCachingEnabled(model, 1));
  batching.max_batch_size = 2;
  EXPECT_THROW(ResolvePrefixCachingEnabled(model, 1), std::runtime_error);
  batching.max_batch_size = 1;
  model->config_->model.dflash2.filename = "dflash.onnx";
  EXPECT_THROW(ResolvePrefixCachingEnabled(model, 1), std::runtime_error);
  model->config_->model.dflash2.filename.clear();
  model->config_->model.mtp.filename.clear();
  EXPECT_THROW(ResolvePrefixCachingEnabled(model, 1), std::runtime_error);
}

TEST(PagedKeyValueCacheManifestTest, PrefixCachingRejectsSlidingWindowCache) {
  auto model = LoadSyntheticPagedModel();
  model->config_->engine.dynamic_batching->prefix_caching_explicitly_set = true;
  auto& decoder = model->config_->model.decoder;
  decoder.sliding_window = Config::Model::Decoder::SlidingWindow{};
  decoder.sliding_window->window_size = 4;
  decoder.sliding_window->layers = {1};
  decoder.inputs.block_table_windowed = decoder.inputs.block_table;

  EXPECT_THROW(MakePagedCache(model), std::runtime_error);
}

TEST(PagedKeyValueCacheManifestTest, OmittedPrefixCachingPreservesSlidingWindowCompatibility) {
  auto model = LoadSyntheticPagedModel();
  auto& decoder = model->config_->model.decoder;
  decoder.sliding_window = Config::Model::Decoder::SlidingWindow{};
  decoder.sliding_window->window_size = 4;
  decoder.sliding_window->layers = {1};
  decoder.inputs.block_table_windowed = decoder.inputs.block_table;
  model->config_->search.chunk_size = 4;

  EXPECT_TRUE(model->config_->engine.dynamic_batching->prefix_caching);
  EXPECT_FALSE(ResolvePrefixCachingEnabled(model, /*auxiliary_bytes_per_block=*/0));
  auto cache = MakePagedCache(model);

  EXPECT_FALSE(cache->PrefixCachingEnabled());
}

TEST(PagedKeyValueCacheManifestTest, OmittedPrefixCachingPreservesAuxiliaryCacheCompatibility) {
  auto model = LoadSyntheticPagedModel();

  PagedKeyValueCache cache(
      model, /*auxiliary_bytes_per_block=*/1,
      /*auxiliary_reserved_memory_bytes=*/0);

  EXPECT_FALSE(cache.PrefixCachingEnabled());
}

TEST(PagedKeyValueCacheManifestTest, SingleBlockPoolKeepsPrefixCachingEnabled) {
  auto model = LoadSyntheticPagedModel();
  auto& batching = *model->config_->engine.dynamic_batching;
  batching.num_blocks = 1;

  auto cache = MakePagedCache(model);

  EXPECT_TRUE(cache->PrefixCachingEnabled());
}

TEST(PagedKeyValueCacheManifestTest, PrefixCachingRejectsDsparkAuxiliaryState) {
  auto model = LoadSyntheticPagedModel();
  auto& batching = *model->config_->engine.dynamic_batching;
  batching.prefix_caching_explicitly_set = true;
  model->config_->model.dflash2.filename = "dspark.onnx";
  model->config_->model.dflash2.is_dspark = true;

  EXPECT_THROW(MakePagedCache(model), std::runtime_error);
}

TEST(PagedKeyValueCacheManifestTest, PrefixCachingSupportsFixedAuxiliaryPool) {
  auto model = LoadSyntheticPagedModel();
  model->config_->engine.dynamic_batching->prefix_caching = true;
  model->config_->engine.dynamic_batching->prefix_caching_explicitly_set = true;

  EXPECT_THROW(
      PagedKeyValueCache(
          model, /*auxiliary_bytes_per_block=*/1,
          /*auxiliary_reserved_memory_bytes=*/0),
      std::runtime_error);
  EXPECT_NO_THROW(
      PagedKeyValueCache(
          model, /*auxiliary_bytes_per_block=*/0,
          /*auxiliary_reserved_memory_bytes=*/1));
}

TEST(PagedKeyValueCacheManifestTest, PrefixCachingAllocatesFixedStateCheckpoints) {
  auto model = LoadSyntheticCompositeModel();
  model->config_->engine.dynamic_batching->prefix_caching = true;

  PagedCacheManager manager{model};
  const auto fixed = manager.FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(fixed->checkpoint_capacity,
            model->config_->engine.dynamic_batching->max_batch_size);
  EXPECT_EQ(fixed->checkpoint_count, 0u);
}

TEST(PagedKeyValueCacheManifestTest, OptionalDraftCheckpointDoesNotExhaustHybridTargetPool) {
  auto model = LoadSyntheticCompositeModel();
  auto& batching = *model->config_->engine.dynamic_batching;
  batching.prefix_caching = true;
  batching.max_batch_size = 1;
  batching.num_blocks = 4;
  const size_t block_bytes = PagedKeyValueCacheBytesPerBlock(model);
  bool enabled = true;
  auto manager = CacheManager::Create(model, /*auxiliary_bytes_per_block=*/0,
                                      /*auxiliary_reserved_memory_bytes=*/0,
                                      4 * block_bytes, &enabled);
  EXPECT_FALSE(enabled);
  EXPECT_EQ(manager->Snapshot().total_blocks, 4u);
  manager.reset();

  manager = CacheManager::Create(model, 0, 0, 2 * block_bytes, &enabled);
  EXPECT_TRUE(enabled);
  EXPECT_EQ(manager->Snapshot().total_blocks, 2u);
}

TEST(PagedKeyValueCacheManifestTest, DraftSnapshotCountIsLimitedByConfigBudgetAndTargetCheckpoints) {
  auto model = LoadSyntheticCompositeModel();
  auto& batching = *model->config_->engine.dynamic_batching;
  batching.prefix_caching = true;
  batching.max_batch_size = 4;
  batching.num_blocks = 8;
  const size_t block_bytes = PagedKeyValueCacheBytesPerBlock(model);
  for (const auto [requested, expected] :
       {std::pair<size_t, size_t>{0, 0}, {1, 1}, {3, 3}, {100, 4}}) {
    batching.dflash2_max_snapshots = requested;
    auto manager = CacheManager::Create(model, 0, 0, block_bytes);
    EXPECT_EQ(manager->DraftCheckpointCapacity(), expected);
    EXPECT_EQ(manager->Snapshot().total_blocks, 8u - expected);
  }
  batching.dflash2_max_snapshots = 100;
  auto manager = CacheManager::Create(model, 0, 0, 3 * block_bytes);
  EXPECT_EQ(manager->DraftCheckpointCapacity(), 2u);
  EXPECT_EQ(manager->Snapshot().total_blocks, 2u);
  manager.reset();

  batching.prefix_caching = false;
  manager = CacheManager::Create(model, 0, 0, block_bytes);
  EXPECT_EQ(manager->DraftCheckpointCapacity(), 0u);
  EXPECT_EQ(manager->Snapshot().total_blocks, 8u);
}

TEST(PagedKeyValueCacheManifestTest, SnapshotSizingReportsOneBlockMinimumAndBudgetReduction) {
  auto model = LoadSyntheticCompositeModel();
  auto& batching = *model->config_->engine.dynamic_batching;
  batching.prefix_caching = true;
  batching.max_batch_size = 4;
  batching.num_blocks = 9;
  const size_t block_bytes = PagedKeyValueCacheBytesPerBlock(model);
  const auto previous_logging = g_log.enabled;
  SetLogBool("enabled", true);
  testing::internal::CaptureStderr();
  batching.dflash2_max_snapshots = 1;
  auto smaller = CacheManager::Create(model, 0, 0, 2 * block_bytes);
  batching.dflash2_max_snapshots = 4;
  auto larger = CacheManager::Create(model, 0, 0, 2 * block_bytes);
  auto reduced = CacheManager::Create(model, 0, 0, 3 * block_bytes);
  const auto log = testing::internal::GetCapturedStderr();
  SetLogBool("enabled", previous_logging);

  EXPECT_EQ(smaller->DraftCheckpointCapacity(), 1u);
  EXPECT_EQ(smaller->Snapshot().total_blocks, 7u);
  EXPECT_EQ(larger->DraftCheckpointCapacity(), 4u);
  EXPECT_EQ(larger->Snapshot().total_blocks, 1u);
  EXPECT_EQ(reduced->DraftCheckpointCapacity(), 2u);
  EXPECT_EQ(reduced->Snapshot().total_blocks, 3u);
  EXPECT_NE(log.find("requested=1, effective=1; target cache: blocks=7, block_size=4, bytes=" +
                     std::to_string(7 * block_bytes)),
            std::string::npos);
  EXPECT_NE(log.find("requested=4, effective=4; target cache: blocks=1, block_size=4, bytes=" +
                     std::to_string(block_bytes)),
            std::string::npos);
  EXPECT_NE(log.find("requested=4, effective=2; target cache: blocks=3, block_size=4, bytes=" +
                     std::to_string(3 * block_bytes)),
            std::string::npos);
  EXPECT_NE(log.find("draft snapshot reservation: bytes=" + std::to_string(8 * block_bytes)),
            std::string::npos);
}

TEST(PagedKeyValueCacheManifestTest, SnapshotRuntimeProfileOverridesBaseConfiguration) {
  auto model = LoadSyntheticCompositeModel();
  auto& batching = *model->config_->engine.dynamic_batching;
  batching.prefix_caching = true;
  batching.max_batch_size = 4;
  batching.num_blocks = 8;
  batching.dflash2_max_snapshots = 1;
  OverlayConfig(*model->config_, R"({"runtime_profiles":[{
    "id":"snapshot-default","eligibility":{"minimum_total_device_memory_bytes":0},
    "overlay":{"engine":{"dynamic_batching":{"dflash2_max_snapshots":2}}}
  }]})");
  ApplyRuntimeProfile(*model->config_, 32ull * 1024 * 1024 * 1024);
  const size_t block_bytes = PagedKeyValueCacheBytesPerBlock(model);

  auto cache = CacheManager::Create(model, 0, 0, block_bytes);
  EXPECT_EQ(cache->DraftCheckpointCapacity(), 2u);
  EXPECT_EQ(cache->Snapshot().total_blocks, 6u);
  EXPECT_EQ(model->config_->engine.dynamic_batching->dflash2_max_snapshots, 2u);
}

TEST(PagedKeyValueCacheManifestTest, SnapshotRuntimeProfileHonorsBudgetAndTargetCapacity) {
  auto model = LoadSyntheticCompositeModel();
  auto& batching = *model->config_->engine.dynamic_batching;
  batching.prefix_caching = true;
  batching.max_batch_size = 4;
  batching.num_blocks = 8;
  batching.dflash2_max_snapshots = 0;
  OverlayConfig(*model->config_, R"({"runtime_profiles":[{
    "id":"large-snapshot-pool","eligibility":{"minimum_total_device_memory_bytes":0},
    "overlay":{"engine":{"dynamic_batching":{"dflash2_max_snapshots":100}}}
  }]})");
  ApplyRuntimeProfile(*model->config_, 32ull * 1024 * 1024 * 1024);
  const size_t block_bytes = PagedKeyValueCacheBytesPerBlock(model);
  auto target_limited = CacheManager::Create(model, 0, 0, block_bytes);
  EXPECT_EQ(target_limited->DraftCheckpointCapacity(), 4u);
  EXPECT_EQ(target_limited->Snapshot().total_blocks, 4u);

  auto budget_limited = CacheManager::Create(model, 0, 0, 3 * block_bytes);
  EXPECT_EQ(budget_limited->DraftCheckpointCapacity(), 2u);
  EXPECT_EQ(budget_limited->Snapshot().total_blocks, 2u);
  EXPECT_EQ(model->config_->engine.dynamic_batching->dflash2_max_snapshots, 100u);
}

TEST(PagedKeyValueCacheManifestTest, EngineConstructionAppliesSnapshotConfigurationToCacheAndDrafter) {
  auto model = LoadSyntheticCompositeModel();
  auto& draft = model->config_->model.dflash2;
  draft.filename = (fs::path{".."} / "synthetic-dspark" / "dflash2-fp16-aux.onnx").string();
  draft.num_hidden_layers = 1;
  draft.num_key_value_heads = 1;
  draft.head_size = 1;
  draft.block_size = 4;
  draft.num_draft_tokens = 3;
  draft.selector_top_k = 2;
  draft.mask_token_id = 31;
  draft.sliding_window = 8;
  draft.main_aux_hidden_states = "hidden_states";

  auto defaults = Engine::CreateDependencies(model);
  ASSERT_NE(defaults.dflash2_drafter, nullptr);
  EXPECT_EQ(defaults.cache_manager->DraftCheckpointCapacity(), 1u);
  EXPECT_EQ(defaults.dflash2_drafter->AvailablePrefixCheckpoints(), 1u);

  OverlayConfig(*model->config_, R"({"engine":{"dynamic_batching":{"dflash2_max_snapshots":3}}})");
  auto larger = Engine::CreateDependencies(model);
  ASSERT_NE(larger.dflash2_drafter, nullptr);
  EXPECT_EQ(larger.cache_manager->DraftCheckpointCapacity(), 3u);
  EXPECT_EQ(larger.dflash2_drafter->AvailablePrefixCheckpoints(), 3u);

  OverlayConfig(*model->config_, R"({"engine":{"dynamic_batching":{"dflash2_max_snapshots":0}}})");
  auto disabled = Engine::CreateDependencies(model);
  ASSERT_NE(disabled.dflash2_drafter, nullptr);
  EXPECT_EQ(disabled.cache_manager->DraftCheckpointCapacity(), 0u);
  EXPECT_EQ(disabled.dflash2_drafter->AvailablePrefixCheckpoints(), 0u);

  EXPECT_EQ(larger.cache_manager->DraftCheckpointCapacity(), 3u);
  EXPECT_EQ(larger.dflash2_drafter->AvailablePrefixCheckpoints(), 3u);
  EXPECT_EQ(defaults.cache_manager->DraftCheckpointCapacity(), 1u);
  EXPECT_EQ(defaults.dflash2_drafter->AvailablePrefixCheckpoints(), 1u);
  EXPECT_EQ(model->config_->engine.dynamic_batching->dflash2_max_snapshots, 0u);
}

class Dflash2SnapshotEngineTest : public ::testing::TestWithParam<size_t> {
 protected:
  static std::shared_ptr<Model> LoadModel(bool with_drafter) {
    auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-composite");
    config->search.chunk_size = 4;
    config->model.decoder.filename = "decoder-per-token.onnx";
    auto model = CreateModel(GetOrtEnv(), std::move(config));
    if (with_drafter) {
      auto& draft = model->config_->model.dflash2;
      draft.filename = (fs::path{".."} / "synthetic-dspark" / "dflash2-fp16-aux.onnx").string();
      draft.num_hidden_layers = 1;
      draft.num_key_value_heads = 1;
      draft.head_size = 1;
      draft.block_size = 4;
      draft.num_draft_tokens = 3;
      draft.selector_top_k = 2;
      draft.mask_token_id = 31;
      draft.sliding_window = 8;
      draft.main_aux_hidden_states = "hidden_states";
    } else {
      model->config_->engine.dynamic_batching->prefix_caching = false;
    }
    return model;
  }

  static std::shared_ptr<Request> Submit(
      const std::shared_ptr<Engine>& engine, std::span<const int32_t> prompt) {
    auto request = engine->CreateRequest();
    request->BeginTurn(prompt, /*max_generated_tokens=*/6);
    return request;
  }

  static void Drain(
      Engine& engine, const std::vector<std::shared_ptr<Request>>& requests,
      std::vector<std::vector<int32_t>>& outputs,
      std::vector<GenerationFinishReason>& reasons) {
    outputs.resize(requests.size());
    reasons.resize(requests.size(), GenerationFinishReason::None);
    std::array<EngineEvent, 32> events;
    size_t steps = 0;
    while (engine.HasPendingRequests()) {
      ASSERT_LT(++steps, 1000u) << "Requests stopped making progress.";
      const size_t count = engine.Run(events);
      for (size_t i = 0; i < count; ++i) {
        const auto& event = events[i];
        if (event.flags & EngineEventFlagFailed) {
          ASSERT_NO_THROW(engine.Run(events));
        }
        ASSERT_EQ(event.flags & EngineEventFlagFailed, 0u);
        const auto found = std::find(requests.begin(), requests.end(), event.request);
        ASSERT_NE(found, requests.end());
        const size_t index = static_cast<size_t>(found - requests.begin());
        if (event.flags & EngineEventFlagToken) {
          outputs[index].push_back(event.token);
        }
        if (event.flags & EngineEventFlagTurnFinished) {
          reasons[index] = event.finish_reason;
        }
      }
    }
    for (const auto reason : reasons) {
      EXPECT_NE(reason, GenerationFinishReason::None);
    }
  }
};

TEST_P(Dflash2SnapshotEngineTest, ReusesMultiplePromptsWithFullLeasedPoolAndCancellation) {
  const size_t limit = GetParam();
  const std::array<std::array<int32_t, 5>, 4> prompts{
      std::array<int32_t, 5>{5, 9, 13, 7, 8}, {7, 2, 20, 4, 6}, {3, 8, 2, 15, 6}, {11, 12, 13, 14, 15}};
  auto baseline = std::make_shared<Engine>(LoadModel(false));
  std::array<std::vector<int32_t>, 4> expected;
  for (size_t i = 0; i < prompts.size(); ++i) {
    auto request = Submit(baseline, prompts[i]);
    std::vector<std::vector<int32_t>> outputs;
    std::vector<GenerationFinishReason> reasons;
    Drain(*baseline, {request}, outputs, reasons);
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_FALSE(outputs[0].empty());
    expected[i] = outputs[0];
    request->Close();
  }

  auto model = LoadModel(true);
  OverlayConfig(*model->config_, R"({"engine":{"dynamic_batching":{"dflash2_max_snapshots":)" +
                                     std::to_string(limit) + "}}}");
  auto dependencies = Engine::CreateDependencies(model);
  auto cache = dependencies.cache_manager;
  auto* drafter = dependencies.dflash2_drafter.get();
  ASSERT_NE(drafter, nullptr);
  ASSERT_EQ(cache->DraftCheckpointCapacity(), limit);
  auto engine = std::make_shared<Engine>(model, std::move(dependencies));
  ASSERT_GT(engine->MaxDraftTokensPerStep(), 0u);
  const size_t warm_count = std::max(size_t{1}, limit);
  for (size_t i = 0; i < warm_count; ++i) {
    auto request = Submit(engine, prompts[i]);
    std::vector<std::vector<int32_t>> outputs;
    std::vector<GenerationFinishReason> reasons;
    Drain(*engine, {request}, outputs, reasons);
    ASSERT_FALSE(HasFatalFailure());
    EXPECT_EQ(outputs[0], expected[i]);
    request->Close();
  }
  ASSERT_EQ(drafter->AvailablePrefixCheckpoints(), 0u);

  std::vector<std::shared_ptr<Request>> requests;
  std::vector<std::shared_ptr<const PrefixCacheMatch>> leases;
  for (size_t i = 0; i < warm_count; ++i) {
    requests.push_back(Submit(engine, prompts[i]));
    auto match = cache->MatchPrefix(*requests.back());
    ASSERT_NE(match, nullptr);
    ASSERT_EQ(match->token_count, 4u);
    ASSERT_NE(match->fixed_state_checkpoint, nullptr);
    EXPECT_EQ(match->draft_checkpoint != nullptr, limit != 0);
    leases.push_back(std::move(match));
  }
  requests.push_back(Submit(engine, prompts[2]));
  std::vector<std::vector<int32_t>> outputs;
  std::vector<GenerationFinishReason> reasons;
  Drain(*engine, requests, outputs, reasons);
  ASSERT_FALSE(HasFatalFailure());
  for (size_t i = 0; i < warm_count; ++i) {
    EXPECT_EQ(outputs[i], expected[i]);
    const auto match = cache->MatchPrefix(*requests[i]);
    ASSERT_NE(match, nullptr);
    EXPECT_EQ(match->draft_checkpoint, leases[i]->draft_checkpoint);
    EXPECT_EQ(requests[i]->TurnCachedPromptTokens(), 4u);
  }
  EXPECT_EQ(outputs.back(), expected[2]);
  {
    const auto match = cache->MatchPrefix(*requests.back());
    ASSERT_NE(match, nullptr);
    EXPECT_EQ(match->token_count, 4u);
    EXPECT_EQ(match->draft_checkpoint, nullptr);
  }
  leases.clear();
  for (const auto& request : requests) {
    request->Close();
  }

  auto replacement = Submit(engine, prompts[3]);
  outputs.clear();
  reasons.clear();
  Drain(*engine, {replacement}, outputs, reasons);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(outputs[0], expected[3]);
  {
    const auto match = cache->MatchPrefix(*replacement);
    ASSERT_NE(match, nullptr);
    EXPECT_EQ(match->draft_checkpoint != nullptr, limit != 0);
  }
  EXPECT_EQ(drafter->AvailablePrefixCheckpoints(), 0u);
  replacement->Close();

  const std::array<int32_t, 9> cancel_prompt{21, 22, 23, 24, 25, 26, 27, 28, 29};
  auto cancel_reference = Submit(baseline, cancel_prompt);
  outputs.clear();
  reasons.clear();
  Drain(*baseline, {cancel_reference}, outputs, reasons);
  ASSERT_FALSE(HasFatalFailure());
  const auto expected_after_cancel = outputs[0];
  cancel_reference->Close();
  auto canceled = Submit(engine, cancel_prompt);
  std::array<EngineEvent, 32> events;
  EXPECT_EQ(engine->Run(events), 0u);
  ASSERT_EQ(canceled->ProcessedSequenceLength(), 4);
  auto canceled_match = cache->MatchPrefix(*canceled);
  ASSERT_NE(canceled_match, nullptr);
  EXPECT_EQ(canceled_match->token_count, 4u);
  EXPECT_EQ(canceled_match->draft_checkpoint != nullptr, limit != 0);
  ASSERT_TRUE(canceled->Cancel(canceled->CurrentTurnId()));
  outputs.clear();
  reasons.clear();
  Drain(*engine, {canceled}, outputs, reasons);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(reasons[0], GenerationFinishReason::Canceled);
  EXPECT_TRUE(outputs[0].empty());
  canceled->Close();

  auto canceled_replay = Submit(engine, cancel_prompt);
  outputs.clear();
  reasons.clear();
  Drain(*engine, {canceled_replay}, outputs, reasons);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(outputs[0], expected_after_cancel);
  EXPECT_EQ(canceled_replay->TurnCachedPromptTokens(), 4u);
  canceled_match.reset();
  canceled_replay->Close();

  auto after_cancel = Submit(engine, prompts[3]);
  outputs.clear();
  reasons.clear();
  Drain(*engine, {after_cancel}, outputs, reasons);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(outputs[0], expected[3]);
  EXPECT_EQ(after_cancel->TurnCachedPromptTokens(), 4u);
  after_cancel->Close();
  EXPECT_FALSE(engine->HasPendingRequests());
  EXPECT_EQ(cache->ResidentRequestCount(), 0u);
  EXPECT_TRUE(ValidateCacheInvariants(cache->Snapshot()).empty());
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_GT(stats.draft_forward_passes, 0u);
  EXPECT_GT(stats.draft_tokens_proposed, 0u);
  EXPECT_EQ(stats.dflash2_failures, 0u);
}

INSTANTIATE_TEST_SUITE_P(SnapshotLimits, Dflash2SnapshotEngineTest, ::testing::Values(0u, 1u, 2u));

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
