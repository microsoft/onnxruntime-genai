// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>

#include <gtest/gtest.h>

#include "engine/cache_manager.h"
#include "engine/engine_invariants.h"
#include "engine/paged_key_value_cache.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"
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
        MintAssignedRequest(assign_target_, *model_, prompt);
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

TEST_F(PagedKeyValueCacheTest, DeferredActiveRequestsStillConsumeAdmissionCapacity) {
  auto unserviceable = AddCommittedRequest({2, 3, 4, 5});
  auto fitting = AddCommittedRequest({6, 7, 8, 9});
  auto pending = MintAssignedRequest(
      assign_target_, *model_, std::array<int32_t, 1>{10});

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
  auto pending = MintAssignedRequest(
      assign_target_, *model_, std::array<int32_t, 1>{10});

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
  auto pending = MintAssignedRequest(
      assign_target_, *model_, std::array<int32_t, 1>{10});

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
  auto pending = MintAssignedRequest(
      assign_target_, *model_, std::array<int32_t, 1>{10});

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
  auto blocked = MintAssignedRequest(
      assign_target_, *model_, std::array<int32_t, 1>{10});
  auto fitting = MintAssignedRequest(
      assign_target_, *model_, std::array<int32_t, 1>{11});

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
  auto pending = MintAssignedRequest(
      assign_target_, *model_, std::array<int32_t, 1>{10});

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
  auto pending = MintAssignedRequest(
      assign_target_, *model_,
      std::array<int32_t, 9>{2, 3, 4, 5, 6, 7, 8, 9, 10});

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

TEST(PagedKeyValueCacheManifestTest, AllocatesOnlySparseLogicalLayersUsingTheirExactBindings) {
  auto model = LoadSyntheticPagedModel();
  ASSERT_TRUE(model->config_->model.decoder.state_groups.has_value());
  ASSERT_EQ(model->config_->model.decoder.state_groups->size(), 1u);
  EXPECT_EQ(model->config_->model.decoder.state_groups->front().layer_ids,
            std::vector<int>({1, 4}));
  EXPECT_EQ(model->config_->model.decoder.inputs.past_key_names, "legacy_past.%d.key");

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

TEST(PagedKeyValueCacheManifestTest, BlockCapacityUsesParticipatingLayerCount) {
  EXPECT_EQ(
      ComputePagedBlockCapacity(
          /*available_memory_bytes=*/10240,
          /*gpu_utilization_factor=*/1.0f,
          /*reserved_memory_bytes=*/0,
          /*block_size=*/4,
          /*num_key_value_heads=*/1,
          /*head_size=*/2,
          /*full_layer_count=*/2,
          /*element_size=*/2),
      144u);
  EXPECT_EQ(
      ComputePagedBlockCapacity(
          /*available_memory_bytes=*/10240,
          /*gpu_utilization_factor=*/1.0f,
          /*reserved_memory_bytes=*/0,
          /*block_size=*/4,
          /*num_key_value_heads=*/1,
          /*head_size=*/2,
          /*full_layer_count=*/6,
          /*element_size=*/2),
      48u);

  // One auxiliary layer with the same geometry adds 32 bytes to the primary model's 64 bytes per
  // block, so both pools together fit 96 blocks in the same 90%-adjusted memory budget.
  EXPECT_EQ(
      ComputePagedBlockCapacity(
          /*available_memory_bytes=*/10240,
          /*gpu_utilization_factor=*/1.0f,
          /*reserved_memory_bytes=*/0,
          /*block_size=*/4,
          /*num_key_value_heads=*/1,
          /*head_size=*/2,
          /*full_layer_count=*/2,
          /*element_size=*/2,
          /*auxiliary_bytes_per_block=*/32),
      96u);

  EXPECT_EQ(
      ComputePagedBlockCapacity(
          /*available_memory_bytes=*/10240,
          /*gpu_utilization_factor=*/1.0f,
          /*reserved_memory_bytes=*/1024,
          /*block_size=*/4,
          /*num_key_value_heads=*/1,
          /*head_size=*/2,
          /*full_layer_count=*/2,
          /*element_size=*/2),
      128u);

  EXPECT_THROW(
      ComputePagedBlockCapacity(
          /*available_memory_bytes=*/10240,
          /*gpu_utilization_factor=*/1.0f,
          /*reserved_memory_bytes=*/9216,
          /*block_size=*/4,
          /*num_key_value_heads=*/1,
          /*head_size=*/2,
          /*full_layer_count=*/2,
          /*element_size=*/2),
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
  fixed_group.kind = Config::Model::Decoder::StateGroupKind::Fixed;
  fixed_group.layer_ids = {0};
  fixed_group.state = Config::Model::Decoder::StateBinding{
      "past_fixed.%d", "present_fixed.%d"};
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

TEST(PagedKeyValueCacheManifestTest, RejectsMalformedPagedGroup) {
  auto model = LoadSyntheticPagedModel();
  model->config_->model.decoder.state_groups->front().key.reset();

  EXPECT_THROW(
      {
        try {
          auto cache = MakePagedCache(model);
        } catch (const std::runtime_error& error) {
          EXPECT_NE(
              std::string{error.what()}.find("with key and value bindings"),
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
