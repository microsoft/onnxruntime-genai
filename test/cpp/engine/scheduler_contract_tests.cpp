// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Component-contract tests for the continuous-batching DynamicBatchScheduler. They drive a real
// scheduler against a recording CacheManager double and a tiny CPU fixture model that is used only
// to mint real Request objects: the model never runs, so these tests need no GPU and no real
// inference. They assert the scheduler's admission contract -- which requests it admits, in what
// order, how it honors capacity backpressure, and how it releases completed and removed requests --
// by observing the calls it makes to the cache manager.

#include <array>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "engine/scheduler.h"
#include "engine_test_helpers.h"
#include "engine_test_doubles.h"

namespace Generators {
namespace test {
namespace {

// A distinct in-vocabulary, non-EOS prompt per request; contents are irrelevant because the model
// never runs. Kept well within the fixture model's small vocabulary.
std::vector<int32_t> Prompt(int32_t seed) {
  const int32_t base = 2 + (seed % 8);
  return {base, base + 1, base + 2};
}

struct ReverseExecutionOrderCacheManager : RecordingCacheManager {
  using RecordingCacheManager::RecordingCacheManager;

  void OrderStepForExecution(StepPlan& plan) const override {
    std::reverse(plan.requests.begin(), plan.requests.end());
  }
};

class SchedulerContractTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_ = LoadDummyDecoderModel();
    model_->config_->engine.dynamic_batching =
        Config::Engine::DynamicBatching{};
    // A permissive engine used purely as the assignment target when minting requests.
    assign_target_ = MakeDoublesEngine(model_, /*capacity=*/1024, EosToken(*model_)).engine;
  }

  std::shared_ptr<Request> Assigned(int32_t seed) {
    auto prompt = Prompt(seed);
    return CreateRequestWithPrompt(assign_target_, prompt);
  }

  void MakePrefillResident(
      DynamicBatchScheduler& scheduler,
      CacheManager& cache,
      const std::shared_ptr<Request>& request) {
    scheduler.AddRequest(request);
    cache.Allocate({request});
    request->Schedule();
  }

  void MakeDecodeResident(
      DynamicBatchScheduler& scheduler,
      CacheManager& cache,
      const std::shared_ptr<Request>& request) {
    MakePrefillResident(scheduler, cache, request);
    auto logits = model_->p_device_inputs_->Allocate<float>(
        static_cast<size_t>(model_->config_->model.vocab_size));
    auto cpu_logits = logits.CpuSpan();
    std::fill(cpu_logits.begin(), cpu_logits.end(), 0.0f);
    cpu_logits[5] = 100.0f;
    logits.CopyCpuToDevice();

    const auto before = request->Snapshot();
    RequestStepPlan plan;
    plan.request = request;
    plan.request_id = request.get();
    plan.sequence_length_before = before.current_sequence_length;
    plan.target_cache_slots =
        static_cast<size_t>(before.current_sequence_length);
    PrepareRequestStep(model_, plan);
    request->SaveStateForTransaction();
    const auto result = request->ApplyLogitsForTransaction(logits);
    request->CommitStateForTransaction();
    request->CommitStep(plan, result);
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<Engine> assign_target_;
};

TEST_F(SchedulerContractTest, DynamicPlansSingleAssignedRequestWithoutAdmissionMutation) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);

  auto request = Assigned(10);
  scheduler.AddRequest(request);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, request);
  EXPECT_EQ(plan.requests[0].unprocessed_token_count, 3u);
  EXPECT_EQ(request->status_, RequestStatus::Assigned);
  EXPECT_EQ(cache->AllocatedCount(), 0u);
}

TEST_F(SchedulerContractTest, MissingImageProducesPreparationWithoutLogicalProgress) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, 8);
  DynamicBatchScheduler scheduler(model_, cache);
  auto request = Assigned(10);
  auto image = std::make_shared<MediaItem>();
  image->rows = 2;
  request->ReserveMedia(1);
  const std::array<MediaSpan, 1> media{{{0, image}}};
  request->AppendMedia(media);
  scheduler.AddRequest(request);
  const auto before = request->Snapshot();
  StepPlan plan;
  const auto result = scheduler.PlanStep(plan);
  EXPECT_FALSE(result.executable);
  EXPECT_EQ(result.outcome.kind, StepOutcomeKind::PreparationProgress);
  EXPECT_EQ(result.preparation_request, request);
  EXPECT_EQ(result.preparation_item, image);
  EXPECT_TRUE(plan.Empty());
  EXPECT_EQ(request->Snapshot().current_sequence_length, before.current_sequence_length);
  EXPECT_EQ(request->ProcessedSequenceLength(), 0);
  EXPECT_EQ(cache->AllocatedCount(), 0u);
  EXPECT_TRUE(scheduler.HasPendingRequests());
}

TEST_F(SchedulerContractTest, PendingImageWaitsForRunnableDecodeToDrain) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, 8);
  DynamicBatchScheduler scheduler(model_, cache);
  auto image_request = Assigned(10);
  auto image = std::make_shared<MediaItem>();
  image->rows = 2;
  image_request->ReserveMedia(1);
  const std::array<MediaSpan, 1> media{{{0, image}}};
  image_request->AppendMedia(media);
  scheduler.AddRequest(image_request);
  auto decode = Assigned(20);
  MakeDecodeResident(scheduler, *cache, decode);
  StepPlan plan;
  auto logits = model_->p_device_inputs_->Allocate<float>(static_cast<size_t>(model_->config_->model.vocab_size));
  std::fill(logits.CpuSpan().begin(), logits.CpuSpan().end(), 0.0f);
  logits.CpuSpan()[5] = 100.0f;
  logits.CopyCpuToDevice();
  for (size_t step = 0; step < 32; ++step) {
    const auto result = scheduler.PlanStep(plan);
    ASSERT_TRUE(result.executable);
    ASSERT_EQ(plan.requests.size(), 1u);
    EXPECT_EQ(plan.requests[0].request, decode);
    EXPECT_FALSE(plan.requests[0].is_prefill);
    EXPECT_FALSE(image->features);
    EXPECT_EQ(image_request->ProcessedSequenceLength(), 0);
    PrepareRequestStep(model_, plan.requests[0]);
    decode->SaveStateForTransaction();
    const auto sampled = decode->ApplyLogitsForTransaction(logits);
    decode->CommitStateForTransaction();
    decode->CommitStep(plan.requests[0], sampled);
  }
  scheduler.RemoveRequest(decode);
  const auto preparation = scheduler.PlanStep(plan);
  EXPECT_FALSE(preparation.executable);
  EXPECT_EQ(preparation.outcome.kind, StepOutcomeKind::PreparationProgress);
  EXPECT_EQ(preparation.preparation_request, image_request);
  EXPECT_EQ(preparation.preparation_item, image);
  EXPECT_EQ(image_request->ProcessedSequenceLength(), 0);
}

TEST_F(SchedulerContractTest, PrefillStopsBeforeTheNextMissingImage) {
  for (const auto chunk : {std::optional<size_t>{}, std::optional<size_t>{0}, std::optional<size_t>{64}}) {
    for (const bool prefix_caching : {false, true}) {
      model_->config_->search.chunk_size = chunk;
      model_->config_->engine.dynamic_batching->prefix_caching = prefix_caching;
      auto cache = std::make_shared<RecordingCacheManager>(model_, 8);
      DynamicBatchScheduler scheduler(model_, cache);
      auto request = Assigned(10);
      auto first = std::make_shared<MediaItem>();
      first->rows = 1;
      first->features = std::make_shared<Tensor>(GetDeviceInterface(DeviceType::CPU), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      auto second = std::make_shared<MediaItem>();
      second->rows = 1;
      request->ReserveMedia(2);
      const std::array<MediaSpan, 2> media{{{0, first}, {2, second}}};
      request->AppendMedia(media);
      scheduler.AddRequest(request);
      StepPlan plan;
      const auto result = scheduler.PlanStep(plan);
      ASSERT_TRUE(result.executable);
      ASSERT_EQ(plan.requests.size(), 1u);
      EXPECT_EQ(plan.token_count, 2u);
      EXPECT_EQ(plan.requests[0].unprocessed_token_count, 2u);
      EXPECT_EQ(plan.requests[0].logits_row_index, 1u);
      EXPECT_EQ(request->ProcessedSequenceLength(), 0);
    }
  }
}

TEST_F(SchedulerContractTest, CapacityBlockedPeerDoesNotSuppressResidentImagePreparation) {
  for (const bool occupied_slot : {false, true}) {
    auto cache = std::make_shared<RecordingCacheManager>(model_, occupied_slot ? 1 : 8);
    DynamicBatchScheduler scheduler(model_, cache);
    auto request = Assigned(10);
    auto image = std::make_shared<MediaItem>();
    image->rows = 2;
    request->ReserveMedia(1);
    const std::array<MediaSpan, 1> media{{{1, image}}};
    request->AppendMedia(media);
    request->StagePrefixAdoption(1);
    request->CommitPrefixAdoption();
    MakePrefillResident(scheduler, *cache, request);
    auto peer = Assigned(20);
    scheduler.AddRequest(peer);
    if (!occupied_slot) cache->SetCanAllocate(false);

    StepPlan plan;
    const auto preparation = scheduler.PlanStep(plan);
    ASSERT_FALSE(preparation.executable);
    ASSERT_EQ(preparation.outcome.kind, StepOutcomeKind::PreparationProgress);
    EXPECT_EQ(preparation.preparation_request, request);
    EXPECT_EQ(preparation.preparation_item, image);
    EXPECT_TRUE(plan.Empty());
    EXPECT_EQ(request->ProcessedSequenceLength(), 1);
    EXPECT_EQ(peer->status_, RequestStatus::Assigned);

    image->features = std::make_shared<Tensor>(GetDeviceInterface(DeviceType::CPU), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ASSERT_TRUE(scheduler.PlanStep(plan).executable);
    ASSERT_EQ(plan.requests.size(), 1u);
    EXPECT_EQ(plan.requests[0].request, request);
    scheduler.RemoveRequest(request);
    cache->SetCanAllocate(true);
    ASSERT_TRUE(scheduler.PlanStep(plan).executable);
    ASSERT_EQ(plan.requests.size(), 1u);
    EXPECT_EQ(plan.requests[0].request, peer);
  }
}

TEST_F(SchedulerContractTest, OccupiedPagedSlotDoesNotSuppressResidentImagePreparation) {
  auto& config = *model_->config_->engine.dynamic_batching;
  config.block_size = 4;
  config.num_blocks = 4;
  config.max_batch_size = 1;
  config.max_scheduled_tokens = 1;
  auto cache = std::make_shared<PagedCacheManager>(model_);
  DynamicBatchScheduler scheduler(model_, cache);
  auto request = Assigned(10);
  auto image = std::make_shared<MediaItem>();
  image->rows = 2;
  request->ReserveMedia(1);
  const std::array<MediaSpan, 1> media{{{1, image}}};
  request->AppendMedia(media);
  request->StagePrefixAdoption(1);
  request->CommitPrefixAdoption();
  MakePrefillResident(scheduler, *cache, request);
  auto peer = Assigned(20);
  scheduler.AddRequest(peer);

  StepPlan plan;
  const auto result = scheduler.PlanStep(plan);
  ASSERT_EQ(result.outcome.kind, StepOutcomeKind::PreparationProgress);
  EXPECT_EQ(result.preparation_request, request);
  EXPECT_TRUE(plan.Empty());
  EXPECT_EQ(request->ProcessedSequenceLength(), 1);
  image->features = std::make_shared<Tensor>(GetDeviceInterface(DeviceType::CPU), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  ASSERT_TRUE(scheduler.PlanStep(plan).executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, request);
  scheduler.RemoveRequest(request);
  ASSERT_TRUE(scheduler.PlanStep(plan).executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, peer);
}

TEST_F(SchedulerContractTest, MissingImageDoesNotSuppressUnserviceablePeerFailure) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, 8);
  DynamicBatchScheduler scheduler(model_, cache);
  auto request = Assigned(10);
  auto image = std::make_shared<MediaItem>();
  image->rows = 2;
  request->ReserveMedia(1);
  const std::array<MediaSpan, 1> media{{{0, image}}};
  request->AppendMedia(media);
  scheduler.AddRequest(request);
  auto peer = Assigned(20);
  scheduler.AddRequest(peer);
  cache->SetUnserviceableRequest(peer);
  StepPlan plan;
  const auto result = scheduler.PlanStep(plan);
  EXPECT_FALSE(result.executable);
  EXPECT_EQ(result.outcome.kind, StepOutcomeKind::UnserviceableRequest);
  EXPECT_EQ(result.unserviceable_request_id, peer.get());
  EXPECT_FALSE(result.preparation_item);
}

TEST_F(SchedulerContractTest, OccupiedPagedSlotDefersQueuedImagesWithoutFeatureThrashing) {
  for (const int ready_image : {-1, 0, 1}) {
    auto& config = *model_->config_->engine.dynamic_batching;
    config.block_size = 4;
    config.num_blocks = 8;
    config.max_batch_size = 1;
    config.max_scheduled_tokens = 1;
    config.media_device_bytes = 16;
    auto cache = std::make_shared<PagedCacheManager>(model_);
    DynamicBatchScheduler scheduler(model_, cache);
    auto completed = Assigned(10);
    MakePrefillResident(scheduler, *cache, completed);
    completed->status_ = RequestStatus::TurnComplete;

    std::array<std::shared_ptr<MediaItem>, 2> images;
    for (size_t i = 0; i < images.size(); ++i) {
      auto request = Assigned(static_cast<int32_t>(20 + i));
      images[i] = std::make_shared<MediaItem>();
      images[i]->rows = 2;
      images[i]->feature_bytes = config.media_device_bytes;
      if (static_cast<int>(i) == ready_image)
        images[i]->features = std::make_shared<Tensor>(GetDeviceInterface(DeviceType::CPU), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      request->ReserveMedia(1);
      const std::array<MediaSpan, 1> media{{{1, images[i]}}};
      request->AppendMedia(media);
      request->StagePrefixAdoption(1);
      request->CommitPrefixAdoption();
      scheduler.AddRequest(request);
    }

    StepPlan plan;
    for (int attempt = 0; attempt < 3; ++attempt) {
      const auto blocked = scheduler.PlanStep(plan);
      EXPECT_FALSE(blocked.executable);
      EXPECT_EQ(blocked.outcome.kind, StepOutcomeKind::CapacityDeferred);
      EXPECT_TRUE(blocked.capacity_deferred);
      EXPECT_FALSE(blocked.preparation_item);
      EXPECT_TRUE(plan.Empty());
    }
    scheduler.RemoveRequest(completed);
    const auto resumed = scheduler.PlanStep(plan);
    EXPECT_TRUE(resumed.executable || resumed.outcome.kind == StepOutcomeKind::PreparationProgress);
  }
}

TEST_F(SchedulerContractTest, MissingImageDoesNotSuppressLaterQueuedTextPrefill) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, 8);
  DynamicBatchScheduler scheduler(model_, cache);
  auto image_request = Assigned(10);
  auto image = std::make_shared<MediaItem>();
  image->rows = 2;
  image_request->ReserveMedia(1);
  const std::array<MediaSpan, 1> media{{{0, image}}};
  image_request->AppendMedia(media);
  auto text = Assigned(10);
  scheduler.AddRequest(image_request);
  scheduler.AddRequest(text);
  StepPlan plan;
  const auto result = scheduler.PlanStep(plan);
  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, text);
  EXPECT_EQ(image_request->ProcessedSequenceLength(), 0);
}

TEST_F(SchedulerContractTest, QueuedImagePreparationRespectsPerRequestCapacityFailures) {
  for (const bool unserviceable : {false, true}) {
    auto cache = std::make_shared<RecordingCacheManager>(model_, 8);
    DynamicBatchScheduler scheduler(model_, cache);
    const std::array requests{Assigned(10), Assigned(20)};
    for (const auto& request : requests) {
      auto image = std::make_shared<MediaItem>();
      image->rows = 2;
      request->ReserveMedia(1);
      request->AppendMedia(std::array<MediaSpan, 1>{{{0, image}}});
      scheduler.AddRequest(request);
    }
    if (unserviceable)
      cache->SetUnserviceableRequest(requests[0]);
    else
      cache->SetCapacityDeferredRequest(requests[0]);

    StepPlan plan;
    const auto result = scheduler.PlanStep(plan);
    EXPECT_TRUE(plan.Empty());
    if (unserviceable) {
      EXPECT_EQ(result.outcome.kind, StepOutcomeKind::UnserviceableRequest);
      EXPECT_EQ(result.unserviceable_request_id, requests[0].get());
      EXPECT_FALSE(result.preparation_item);
    } else {
      EXPECT_EQ(result.outcome.kind, StepOutcomeKind::PreparationProgress);
      EXPECT_EQ(result.preparation_request, requests[1]);
    }
  }
}

TEST_F(SchedulerContractTest, QueuedImagePreparationProbesTheProspectivePrefixBoundary) {
  auto& config = *model_->config_->engine.dynamic_batching;
  config.block_size = 4;
  config.num_blocks = 2;
  config.max_batch_size = 1;
  config.max_scheduled_tokens = 1;
  config.prefix_caching = true;
  auto cache = std::make_shared<PagedCacheManager>(model_);
  DynamicBatchScheduler scheduler(model_, cache);
  auto source = CreateRequestWithPrompt(assign_target_, std::array<int32_t, 5>{4, 4, 4, 4, 4});
  source->StagePrefixAdoption(4);
  source->CommitPrefixAdoption();
  MakePrefillResident(scheduler, *cache, source);
  StepPlan publication;
  RequestStepPlan entry;
  entry.request = source;
  entry.request_id = source.get();
  entry.target_cache_slots = 4;
  entry.is_prefill = true;
  publication.requests.push_back(entry);
  auto reservation = cache->ReserveStep(publication);
  reservation->Commit();
  cache->SealCommittedBlocks(publication);
  scheduler.RemoveRequest(source);

  auto request = CreateRequestWithPrompt(assign_target_, std::array<int32_t, 7>{4, 4, 4, 4, 4, 4, 4});
  auto image = std::make_shared<MediaItem>();
  image->rows = 2;
  request->ReserveMedia(1);
  request->AppendMedia(std::array<MediaSpan, 1>{{{4, image}}});
  scheduler.AddRequest(request);
  const auto match = cache->MatchPrefix(*request);
  ASSERT_TRUE(match);
  ASSERT_EQ(match->token_count, 4u);

  StepPlan plan;
  const auto result = scheduler.PlanStep(plan);
  EXPECT_EQ(result.outcome.kind, StepOutcomeKind::PreparationProgress);
  EXPECT_EQ(result.preparation_request, request);
  EXPECT_EQ(result.preparation_item, image);
  EXPECT_EQ(request->ProcessedSequenceLength(), 0);
  EXPECT_TRUE(plan.Empty());
}

TEST_F(SchedulerContractTest, MediaChunkFinishesAnOffsetCacheBlockExactly) {
  for (const auto chunk : {std::optional<size_t>{}, std::optional<size_t>{0}, std::optional<size_t>{64}}) {
    for (const bool prefix_caching : {false, true}) {
      model_->config_->search.chunk_size = chunk;
      model_->config_->engine.dynamic_batching->prefix_caching = prefix_caching;
      model_->config_->engine.dynamic_batching->block_size = 16;
      auto cache = std::make_shared<RecordingCacheManager>(model_, 8);
      DynamicBatchScheduler scheduler(model_, cache);
      const std::vector<int32_t> tokens(30, 4);
      auto request = CreateRequestWithPrompt(assign_target_, tokens);
      auto image = std::make_shared<MediaItem>();
      image->rows = tokens.size();
      image->features = std::make_shared<Tensor>(GetDeviceInterface(DeviceType::CPU), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      request->ReserveMedia(1);
      const std::array<MediaSpan, 1> media{{{0, image}}};
      request->AppendMedia(media);
      request->StagePrefixAdoption(3);
      request->CommitPrefixAdoption();
      scheduler.AddRequest(request);
      StepPlan plan;
      const auto result = scheduler.PlanStep(plan);
      ASSERT_TRUE(result.executable);
      ASSERT_EQ(plan.requests.size(), 1u);
      EXPECT_EQ(plan.token_count, prefix_caching ? 13u : 27u);
      EXPECT_EQ(plan.requests[0].target_cache_slots, prefix_caching ? 16u : 30u);
      EXPECT_EQ(request->ProcessedSequenceLength(), 3);
    }
  }
}

TEST_F(SchedulerContractTest, MediaPrefillUsesTheFurthestBudgetedBlockBoundary) {
  for (const size_t image_rows : {size_t{1}, size_t{96}}) {
    for (const size_t chunk : {size_t{0}, size_t{20}, size_t{64}}) {
      for (const size_t budget : {size_t{12}, size_t{50}, size_t{64}}) {
        model_->config_->search.chunk_size = chunk;
        auto& config = *model_->config_->engine.dynamic_batching;
        config.prefix_caching = true;
        config.block_size = 16;
        config.max_scheduled_tokens = budget;
        auto cache = std::make_shared<RecordingCacheManager>(model_, 8);
        DynamicBatchScheduler scheduler(model_, cache);
        const std::vector<int32_t> tokens(96, 4);
        auto request = CreateRequestWithPrompt(assign_target_, tokens);
        auto image = std::make_shared<MediaItem>();
        image->rows = image_rows;
        image->features = std::make_shared<Tensor>(GetDeviceInterface(DeviceType::CPU), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        request->ReserveMedia(1);
        const std::array<MediaSpan, 1> media{{{0, image}}};
        request->AppendMedia(media);
        request->StagePrefixAdoption(3);
        request->CommitPrefixAdoption();
        scheduler.AddRequest(request);

        const size_t limit = std::min(budget, chunk == 0 ? size_t{93} : chunk);
        const size_t boundary = (3 + limit) / config.block_size * config.block_size;
        const size_t expected = boundary > 3 ? boundary - 3 : limit;
        StepPlan plan;
        ASSERT_TRUE(scheduler.PlanStep(plan).executable);
        ASSERT_EQ(plan.requests.size(), 1u);
        EXPECT_EQ(plan.token_count, expected);
        EXPECT_EQ(plan.requests[0].target_cache_slots, 3 + expected);
        EXPECT_EQ(plan.requests[0].logits_row_index, expected - 1);
      }
    }
  }
}

TEST_F(SchedulerContractTest, DisabledResolvedPrefixCachingDoesNotAlignMediaPrefills) {
  for (const bool windowed : {false, true}) {
    for (const size_t image_rows : {size_t{1}, size_t{7}}) {
      auto model = LoadSyntheticPagedModel();
      model->config_->search.chunk_size = 3;
      auto& config = *model->config_->engine.dynamic_batching;
      config.max_scheduled_tokens = 3;
      ASSERT_TRUE(config.prefix_caching.value_or(true));
      if (windowed) {
        auto& decoder = model->config_->model.decoder;
        decoder.sliding_window.emplace();
        decoder.sliding_window->window_size = 4;
        decoder.sliding_window->layers = {1};
        decoder.inputs.block_table_windowed = decoder.inputs.block_table;
      }
      const size_t auxiliary_bytes = windowed ? 0 : 1;
      ASSERT_FALSE(ResolvePrefixCachingEnabled(model, auxiliary_bytes));
      auto cache = std::make_shared<PagedCacheManager>(model, auxiliary_bytes);
      DynamicBatchScheduler scheduler(model, cache);
      auto request = CreateRequestWithPrompt(assign_target_, std::vector<int32_t>(10, 4));
      auto image = std::make_shared<MediaItem>();
      image->rows = image_rows;
      image->features = std::make_shared<Tensor>(GetDeviceInterface(DeviceType::CPU), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      request->ReserveMedia(1);
      request->AppendMedia(std::array<MediaSpan, 1>{{{0, image}}});
      request->StagePrefixAdoption(3);
      request->CommitPrefixAdoption();
      MakePrefillResident(scheduler, *cache, request);

      StepPlan plan;
      ASSERT_TRUE(scheduler.PlanStep(plan).executable);
      ASSERT_EQ(plan.requests.size(), 1u);
      EXPECT_EQ(plan.token_count, 3u);
      EXPECT_EQ(plan.requests[0].target_cache_slots, 6u);
      EXPECT_EQ(plan.requests[0].logits_row_index, 2u);
    }
  }
}

TEST_F(SchedulerContractTest, DynamicPreservesRequestOrder) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);

  auto first = Assigned(10);
  auto second = Assigned(20);
  auto third = Assigned(30);
  scheduler.AddRequest(first);
  scheduler.AddRequest(second);
  scheduler.AddRequest(third);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 3u);
  EXPECT_EQ(plan.requests[0].request, first);
  EXPECT_EQ(plan.requests[1].request, second);
  EXPECT_EQ(plan.requests[2].request, third);
}

TEST_F(SchedulerContractTest, ExecutionOrderingPreservesAssignedTokenBudgets) {
  auto cache = std::make_shared<ReverseExecutionOrderCacheManager>(
      model_, /*capacity=*/8);
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 3;
  DynamicBatchScheduler scheduler(model_, cache);

  // Unequal remaining prompt lengths under the one Engine-global token budget: the first request
  // takes its single token, leaving the second only two of the three budgeted tokens.
  const std::array<int32_t, 1> short_prompt{5};
  auto first = CreateRequestWithPrompt(assign_target_, short_prompt);
  auto second = CreateRequestWithPrompt(assign_target_, Prompt(20));
  scheduler.AddRequest(first);
  scheduler.AddRequest(second);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 2u);
  EXPECT_EQ(plan.requests[0].request, second);
  EXPECT_EQ(plan.requests[0].scheduling_order, 1u);
  EXPECT_EQ(plan.requests[0].unprocessed_token_count, 2u);
  EXPECT_EQ(plan.requests[0].packed_token_offset, 0u);
  EXPECT_EQ(plan.requests[0].logits_row_index, 1u);
  EXPECT_EQ(plan.requests[1].request, first);
  EXPECT_EQ(plan.requests[1].scheduling_order, 0u);
  EXPECT_EQ(plan.requests[1].unprocessed_token_count, 1u);
  EXPECT_EQ(plan.requests[1].packed_token_offset, 2u);
  EXPECT_EQ(plan.requests[1].logits_row_index, 2u);
  EXPECT_EQ(plan.token_count, 3u);
}

TEST_F(SchedulerContractTest, DynamicHonorsCapacityBackpressure) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/2);
  DynamicBatchScheduler scheduler(model_, cache);

  auto first = Assigned(10);
  auto second = Assigned(20);
  auto third = Assigned(30);
  scheduler.AddRequest(first);
  scheduler.AddRequest(second);
  scheduler.AddRequest(third);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  EXPECT_TRUE(result.capacity_deferred);
  ASSERT_EQ(plan.requests.size(), 2u);
  EXPECT_EQ(plan.requests[0].request, first);
  EXPECT_EQ(plan.requests[1].request, second);
  EXPECT_EQ(first->status_, RequestStatus::Assigned);
  EXPECT_EQ(second->status_, RequestStatus::Assigned);
  EXPECT_EQ(third->status_, RequestStatus::Assigned);
}

TEST_F(SchedulerContractTest, DynamicRetainsTurnCompleteRequestsUntilExplicitRemoval) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);

  auto request = Assigned(10);
  MakePrefillResident(scheduler, *cache, request);
  ASSERT_EQ(cache->AllocatedCount(), 1u);

  request->status_ = RequestStatus::TurnComplete;
  auto second = Assigned(20);
  scheduler.AddRequest(second);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  EXPECT_EQ(cache->deallocate_calls, 0);
  EXPECT_EQ(cache->AllocatedCount(), 1u);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, second);

  scheduler.RemoveRequest(request);

  EXPECT_EQ(cache->deallocate_calls, 1);
  EXPECT_EQ(cache->AllocatedCount(), 0u);
  EXPECT_FALSE(cache->IsResident(request));
}

TEST_F(SchedulerContractTest, DynamicTurnCompleteResidencyAppliesCapacityBackpressure) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/1);
  DynamicBatchScheduler scheduler(model_, cache);

  auto completed = Assigned(10);
  MakePrefillResident(scheduler, *cache, completed);
  completed->status_ = RequestStatus::TurnComplete;

  auto waiting = Assigned(20);
  scheduler.AddRequest(waiting);
  StepPlan plan;

  const auto blocked = scheduler.PlanStep(plan);

  EXPECT_FALSE(blocked.executable);
  EXPECT_TRUE(blocked.capacity_deferred);
  EXPECT_EQ(cache->deallocate_calls, 0);
  EXPECT_EQ(cache->AllocatedCount(), 1u);
  EXPECT_TRUE(plan.requests.empty());

  scheduler.RemoveRequest(completed);
  const auto admitted = scheduler.PlanStep(plan);

  ASSERT_TRUE(admitted.executable);
  EXPECT_EQ(cache->deallocate_calls, 1);
  EXPECT_EQ(cache->AllocatedCount(), 0u);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, waiting);
}

TEST_F(SchedulerContractTest, DynamicResidentQueuedRequestIsNotReadmitted) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);

  auto request = Assigned(10);
  MakeDecodeResident(scheduler, *cache, request);
  request->status_ = RequestStatus::Assigned;
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, request);
  EXPECT_FALSE(plan.requests[0].newly_admitted);
  EXPECT_EQ(cache->AllocatedCount(), 1u);
}

// Engine::BeginTurn enrolls a request again when a turn restarts after a rewind or a cancellation. The request never left
// the pool, so the second enrollment must not queue it twice.
TEST_F(SchedulerContractTest, DynamicEnrollingAnEnrolledRequestAgainPlansItOnce) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);

  auto request = Assigned(10);
  scheduler.AddRequest(request);
  scheduler.AddRequest(request);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, request);
}

// Removing a request from the dynamic scheduler deallocates its cache resources immediately.
TEST_F(SchedulerContractTest, DynamicRemoveReleasesCacheResources) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);

  auto request = Assigned(10);
  MakePrefillResident(scheduler, *cache, request);
  ASSERT_EQ(cache->AllocatedCount(), 1u);

  scheduler.RemoveRequest(request);

  EXPECT_GE(cache->deallocate_calls, 1);
  EXPECT_EQ(cache->AllocatedCount(), 0u);
  EXPECT_FALSE(scheduler.HasPendingRequests());
}

TEST_F(SchedulerContractTest, LegacyDynamicScheduleCannotBypassStepPlan) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);

  EXPECT_THROW(scheduler.Schedule(), std::logic_error);
}

TEST_F(SchedulerContractTest, DynamicPlanningDoesNotMutateAdmissionState) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);
  auto first = Assigned(10);
  auto second = Assigned(20);
  scheduler.AddRequest(first);
  scheduler.AddRequest(second);
  StepPlan plan;
  plan.transaction_id = 17;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 2u);
  EXPECT_EQ(plan.requests[0].request, first);
  EXPECT_EQ(plan.requests[0].packed_token_offset, 0u);
  EXPECT_EQ(plan.requests[0].logits_row_index, 2u);
  EXPECT_EQ(plan.requests[1].request, second);
  EXPECT_EQ(plan.requests[1].packed_token_offset, 3u);
  EXPECT_EQ(plan.requests[1].logits_row_index, 5u);
  EXPECT_EQ(plan.token_count, 6u);
  EXPECT_FALSE(plan.graph_capture_eligible);
  EXPECT_EQ(first->status_, RequestStatus::Assigned);
  EXPECT_EQ(second->status_, RequestStatus::Assigned);
  EXPECT_EQ(cache->AllocatedCount(), 0u);
  EXPECT_EQ(cache->allocate_calls, 0);
}

TEST_F(SchedulerContractTest, DynamicPlanningKeepsActiveWorkWhenNewAdmissionIsDeferred) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);
  auto active = Assigned(10);
  MakePrefillResident(scheduler, *cache, active);
  ASSERT_EQ(active->status_, RequestStatus::Active);

  auto deferred = Assigned(20);
  scheduler.AddRequest(deferred);
  cache->SetCanAllocate(false);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  EXPECT_TRUE(result.capacity_deferred);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, active);
  EXPECT_FALSE(plan.requests[0].newly_admitted);
  EXPECT_EQ(deferred->status_, RequestStatus::Assigned);
  EXPECT_EQ(cache->AllocatedCount(), 1u);
}

TEST_F(SchedulerContractTest, DynamicPlanningReportsNoWorkWithoutMutation) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);
  StepPlan plan;
  plan.transaction_id = 9;

  const auto result = scheduler.PlanStep(plan);

  EXPECT_FALSE(result.executable);
  EXPECT_EQ(result.outcome.kind, StepOutcomeKind::NoWork);
  EXPECT_EQ(result.outcome.transaction_id, 9u);
  EXPECT_TRUE(plan.Empty());
}

TEST_F(SchedulerContractTest, UnserviceableCandidateDoesNotBlockSmallerWork) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);
  auto too_large = Assigned(10);
  auto fitting = Assigned(20);
  scheduler.AddRequest(too_large);
  scheduler.AddRequest(fitting);
  cache->SetUnserviceableRequest(too_large);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  EXPECT_EQ(result.unserviceable_request_id, too_large.get());
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, fitting);
  EXPECT_EQ(too_large->status_, RequestStatus::Assigned);
}

TEST_F(SchedulerContractTest, UnserviceableActiveGrowthIsNotDeferred) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);
  auto active = Assigned(10);
  MakePrefillResident(scheduler, *cache, active);
  cache->SetUnserviceableRequest(active);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  EXPECT_FALSE(result.executable);
  EXPECT_FALSE(result.capacity_deferred);
  EXPECT_EQ(result.outcome.kind,
            StepOutcomeKind::UnserviceableRequest);
  EXPECT_EQ(result.outcome.request_id, active.get());
}

TEST_F(SchedulerContractTest, DecodeFirstOrderingUsesTheExactGlobalBudget) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 5;
  model_->config_->engine.dynamic_batching->max_batch_size = 8;
  DynamicBatchScheduler scheduler(model_, cache);
  auto resident_prefill = Assigned(10);
  auto decode = Assigned(20);
  auto new_prefill = Assigned(30);
  MakePrefillResident(scheduler, *cache, resident_prefill);
  MakeDecodeResident(scheduler, *cache, decode);
  scheduler.AddRequest(new_prefill);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 3u);
  EXPECT_EQ(plan.requests[0].request, decode);
  EXPECT_EQ(plan.requests[0].unprocessed_token_count, 1u);
  EXPECT_EQ(plan.requests[0].packed_token_offset, 0u);
  EXPECT_EQ(plan.requests[1].request, resident_prefill);
  EXPECT_EQ(plan.requests[1].unprocessed_token_count, 3u);
  EXPECT_EQ(plan.requests[1].packed_token_offset, 1u);
  EXPECT_EQ(plan.requests[2].request, new_prefill);
  EXPECT_EQ(plan.requests[2].unprocessed_token_count, 1u);
  EXPECT_EQ(plan.requests[2].packed_token_offset, 4u);
  EXPECT_EQ(plan.token_count, 5u);
  EXPECT_FALSE(plan.graph_capture_eligible);
}

TEST_F(SchedulerContractTest, DecodeDemandCanExhaustTheGlobalBudget) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 2;
  model_->config_->engine.dynamic_batching->max_batch_size = 8;
  DynamicBatchScheduler scheduler(model_, cache);
  auto first = Assigned(10);
  auto second = Assigned(20);
  auto third = Assigned(30);
  auto prefill = Assigned(40);
  MakeDecodeResident(scheduler, *cache, first);
  MakeDecodeResident(scheduler, *cache, second);
  MakeDecodeResident(scheduler, *cache, third);
  scheduler.AddRequest(prefill);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  EXPECT_TRUE(result.capacity_deferred);
  ASSERT_EQ(plan.requests.size(), 2u);
  EXPECT_EQ(plan.requests[0].request, first);
  EXPECT_EQ(plan.requests[1].request, second);
  EXPECT_EQ(plan.token_count, 2u);
  EXPECT_TRUE(plan.graph_capture_eligible);
}

TEST_F(SchedulerContractTest, PrefillRespectsChunkAndGlobalCaps) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 2;
  model_->config_->search.chunk_size = 1;
  DynamicBatchScheduler scheduler(model_, cache);
  auto prompt = Prompt(10);
  auto request = CreateRequestWithPrompt(assign_target_, prompt);
  scheduler.AddRequest(request);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].unprocessed_token_count, 1u);
  EXPECT_EQ(plan.token_count, 1u);
  EXPECT_FALSE(plan.graph_capture_eligible);
}

TEST_F(SchedulerContractTest, CacheQueryCapBoundsOversizedRequestChunk) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  cache->SetMaxQueryTokensPerRequest(2);
  model_->config_->search.chunk_size = 7;
  DynamicBatchScheduler scheduler(model_, cache);
  auto prompt = Prompt(10);
  auto request = CreateRequestWithPrompt(assign_target_, prompt);
  scheduler.AddRequest(request);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].unprocessed_token_count, 2u);
  EXPECT_EQ(plan.token_count, 2u);
}

TEST_F(SchedulerContractTest, CacheQueryCapBoundsUnchunkedRequest) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  cache->SetMaxQueryTokensPerRequest(2);
  model_->config_->search.chunk_size = 0;
  DynamicBatchScheduler scheduler(model_, cache);
  auto prompt = Prompt(10);
  auto request = CreateRequestWithPrompt(assign_target_, prompt);
  scheduler.AddRequest(request);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].unprocessed_token_count, 2u);
  EXPECT_EQ(plan.token_count, 2u);
}

TEST_F(SchedulerContractTest, GlobalCapChunksPromptWithoutSearchChunkSize) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 2;
  DynamicBatchScheduler scheduler(model_, cache);
  auto request = Assigned(10);
  ASSERT_FALSE(request->PrefillChunkSize().has_value());
  scheduler.AddRequest(request);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].unprocessed_token_count, 2u);
  EXPECT_EQ(plan.requests[0].whole_sequence_cache_slots, 3u);
  EXPECT_EQ(plan.requests[0].target_cache_slots, 2u);
  EXPECT_FALSE(plan.graph_capture_eligible);
}

TEST_F(SchedulerContractTest, BlockedPrefillDoesNotHideLaterFittingRequest) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);
  auto blocked = Assigned(10);
  auto fitting = Assigned(20);
  scheduler.AddRequest(blocked);
  scheduler.AddRequest(fitting);
  cache->SetCapacityDeferredRequest(blocked);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  EXPECT_TRUE(result.capacity_deferred);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, fitting);
  EXPECT_EQ(plan.token_count, 3u);
}

TEST_F(SchedulerContractTest, DynamicScheduledRequestsRejectInvalidPlanTokenCount) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);
  auto request = Assigned(10);
  scheduler.AddRequest(request);
  StepPlan plan;
  ASSERT_TRUE(scheduler.PlanStep(plan).executable);
  ASSERT_EQ(plan.requests.size(), 1u);

  plan.requests[0].unprocessed_token_count = 0;
  EXPECT_THROW(scheduler.CreateScheduledRequests(plan), std::runtime_error);

  plan.requests[0].unprocessed_token_count = 4;
  EXPECT_THROW(scheduler.CreateScheduledRequests(plan), std::runtime_error);
}

TEST_F(SchedulerContractTest, ProductionCachePreservesOmittedResident) {
  auto& config = *model_->config_->engine.dynamic_batching;
  config.block_size = 4;
  config.num_blocks = 4;
  config.max_batch_size = 2;
  config.max_scheduled_tokens = 1;
  auto cache = std::make_shared<PagedCacheManager>(model_);
  DynamicBatchScheduler scheduler(model_, cache);

  auto omitted = Assigned(10);
  auto decode = Assigned(20);
  MakePrefillResident(scheduler, *cache, omitted);
  MakeDecodeResident(scheduler, *cache, decode);
  const auto before = cache->Snapshot();
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, decode);
  EXPECT_TRUE(plan.graph_capture_eligible);

  auto reservation = cache->ReserveStep(plan);
  reservation->Commit();
  const auto after = cache->Snapshot();
  ASSERT_EQ(before.requests.size(), 2u);
  ASSERT_EQ(after.requests.size(), 2u);
  EXPECT_EQ(after.requests[0].request_id, omitted.get());
  EXPECT_EQ(after.requests[0].used_slots, before.requests[0].used_slots);
  EXPECT_EQ(after.requests[0].block_ids, before.requests[0].block_ids);
}

TEST_F(SchedulerContractTest, ProductionCacheReservesGloballyChunkedPrompt) {
  auto& config = *model_->config_->engine.dynamic_batching;
  config.block_size = 4;
  config.num_blocks = 3;
  config.max_batch_size = 1;
  config.max_scheduled_tokens = 2;
  auto cache = std::make_shared<PagedCacheManager>(model_);
  DynamicBatchScheduler scheduler(model_, cache);
  auto request = CreateRequestWithPrompt(
      assign_target_, std::array<int32_t, 9>{2, 3, 4, 5, 6, 7, 8, 9, 10});
  scheduler.AddRequest(request);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].unprocessed_token_count, 2u);
  EXPECT_EQ(plan.requests[0].whole_sequence_cache_slots, 9u);
  auto reservation = cache->ReserveStep(plan);
  ASSERT_NE(reservation->PagedReservation(), nullptr);
  EXPECT_EQ(reservation->PagedReservation()->ReservedBlockCount(), 3u);
}

}  // namespace
}  // namespace test
}  // namespace Generators
