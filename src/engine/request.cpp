// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "request.h"

#include "engine.h"
#include "sequence_positions.h"
#include "../search.h"

namespace Generators {

namespace {

DeviceSpan<int32_t> AllocateOnDevice(GeneratorParams& params,
                                     std::span<const int32_t> input_ids) {
  auto device_tokens = params.p_device->Allocate<int32_t>(input_ids.size());
  auto cpu_tokens = device_tokens.CpuSpan();
  std::copy(input_ids.begin(), input_ids.end(), cpu_tokens.begin());
  device_tokens.CopyCpuToDevice();
  return device_tokens;
}

void ValidateAppendLength(const GeneratorParams& params,
                          size_t current_sequence_length,
                          size_t token_count) {
  const size_t max_length = static_cast<size_t>(params.search.max_length);
  if (current_sequence_length >= max_length ||
      token_count >= max_length - current_sequence_length) {
    throw std::runtime_error(
        "Input tokens must leave room for at least one generated token before max_length (" +
        std::to_string(params.search.max_length) + ").");
  }
}

}  // namespace

Request::Request(std::shared_ptr<GeneratorParams> params)
    : params_{params},
      rng_{CreateRandomGenerator(params->search.random_seed)},
      search_{CreateSearch(*params)} {
  // A request is one sequence: the engine batches requests, not rows within a request. Several
  // places here read row 0 only (UnprocessedTokens, CurrentSequenceLength) or take the tail of the
  // next-token span, so a wider search would silently mirror the wrong row's tokens.
  if (params->search.batch_size != 1) {
    throw std::runtime_error("A request must have search.batch_size == 1; batch across requests instead.");
  }
  // Beam search does not implement the deferred completion contract below, so its next tokens would
  // never be copied back from the device.
  if (params->search.num_beams != 1) {
    throw std::runtime_error("A request must have search.num_beams == 1; beam search is not supported by the engine.");
  }

  // The engine drives one independent search per request, so completion is batched: see
  // ScheduledRequests::GenerateNextTokens().
  search_->DeferCompletion(true);
}

void Request::OnFirstExternalReference() noexcept {
  externally_abandoned_.store(false, std::memory_order_release);
}

void Request::OnLastExternalReference() noexcept {
  externally_abandoned_.store(true, std::memory_order_release);
}

bool Request::IsExternallyAbandoned() const noexcept {
  return externally_abandoned_.load(std::memory_order_acquire);
}

void Request::Assign(std::shared_ptr<Engine> engine) {
  if (status_ != RequestStatus::Unassigned) {
    throw std::runtime_error("Cannot add the request to the engine since it is already assigned.");
  }
  if (prefill_input_ids_.empty()) {
    throw std::runtime_error("Cannot add a request with no input tokens to the engine.");
  }
  engine_ = engine;
  status_ = RequestStatus::Assigned;

  auto device_tokens = AllocateOnDevice(*params_, prefill_input_ids_);
  processed_sequence_length_ = 0;
  search_->AppendTokens(device_tokens);
  prompt_sequence_length_ = CurrentSequenceLength();
  seen_sequence_length_ = CurrentSequenceLength();
  tokens_host_.reserve(params_->search.max_length);
  tokens_host_.insert(tokens_host_.end(), prefill_input_ids_.begin(), prefill_input_ids_.end());
  prefill_input_ids_.clear();
}

void Request::PrepareForStep(size_t max_generated_token_indices) {
  if (next_unseen_token_index_ > unseen_token_indices_.size()) {
    throw std::logic_error("The unseen token cursor is outside the generated-token index queue.");
  }

  const size_t unread_token_count =
      unseen_token_indices_.size() - next_unseen_token_index_;
  const bool append_would_grow =
      max_generated_token_indices >
      unseen_token_indices_.capacity() - unseen_token_indices_.size();
  if (next_unseen_token_index_ != 0 &&
      (append_would_grow ||
       next_unseen_token_index_ >= unread_token_count)) {
    const auto unread_begin =
        unseen_token_indices_.begin() +
        static_cast<std::vector<size_t>::difference_type>(
            next_unseen_token_index_);
    unseen_token_indices_.erase(unseen_token_indices_.begin(), unread_begin);
    next_unseen_token_index_ = 0;
  }

  if (max_generated_token_indices >
      unseen_token_indices_.max_size() - unseen_token_indices_.size()) {
    throw std::length_error(
        "The generated-token index queue cannot represent this step.");
  }
  const size_t required_capacity =
      unseen_token_indices_.size() + max_generated_token_indices;
  if (required_capacity <= unseen_token_indices_.capacity()) {
    return;
  }

  // Grow geometrically from the actual unread output needed by this step. Unlike reserving
  // max_length, this keeps a streaming caller's index storage small while avoiding one allocation
  // per token when output is allowed to accumulate.
  const size_t target_capacity =
      required_capacity > unseen_token_indices_.max_size() / 2
          ? unseen_token_indices_.max_size()
          : required_capacity * 2;
  unseen_token_indices_.reserve(target_capacity);
}

void Request::Schedule() {
  if (status_ != RequestStatus::Assigned) {
    throw std::runtime_error("Request cannot be scheduled unless it has been previously added to the engine.");
  }

  if (!search_ || CurrentSequenceLength() == 0) {
    throw std::runtime_error("Cannot schedule a request with no tokens.");
  }

  status_ = RequestStatus::Active;
}

void Request::Remove() {
  if (status_ == RequestStatus::Unassigned) {
    throw std::runtime_error("Cannot close a request that has not been submitted to an engine.");
  }
  if (IsClosed(status_)) {
    return;
  }

  auto engine = engine_.lock();
  if (!engine) {
    CompleteClose();
    return;
  }
  engine->RemoveRequest(shared_from_this());
}

void Request::CompleteClose() {
  engine_.reset();
  status_ = RequestStatus::Closed;
}

void Request::AddTokens(std::span<const int32_t> tokens) {
  if (tokens.empty())
    throw std::runtime_error("Expected at least one token for generation. Received 0.");

  if (status_ != RequestStatus::Unassigned) {
    if (IsTurnComplete()) {
      throw std::runtime_error(
          "AddTokens only accepts initial input; use the continuation API for another turn.");
    }
    if (IsClosed(status_)) {
      throw std::runtime_error("Cannot add tokens to a closed request.");
    }
    throw std::runtime_error("AddTokens only accepts initial input before submission to an engine.");
  }

  ValidateAppendLength(*params_, prefill_input_ids_.size(), tokens.size());
  std::copy(tokens.begin(), tokens.end(), std::back_inserter(prefill_input_ids_));
}

void Request::Continue(std::span<const int32_t> tokens) {
  if (IsClosed(status_)) {
    throw std::runtime_error("Cannot continue a closed request.");
  }
  if (tokens.empty())
    throw std::runtime_error("Expected at least one token for continuation. Received 0.");
  if (!IsTurnComplete()) {
    throw std::runtime_error("Continue is only valid after the current turn is complete.");
  }

  auto engine = engine_.lock();
  if (!engine) {
    throw std::runtime_error("Cannot continue a request after its engine has been destroyed.");
  }
  const DeviceType cache_device = params_->model_->p_device_kvcache_->GetType();
  if (!SupportsContinuousDecoding(cache_device)) {
    throw std::runtime_error(
        "Continuous decoding is not supported on the selected KV-cache device type (" +
        to_string(cache_device) + ").");
  }
  engine->ValidateRequestCanContinue(shared_from_this());
  ValidateAppendLength(*params_, static_cast<size_t>(CurrentSequenceLength()), tokens.size());
  if (tokens_host_.capacity() < tokens_host_.size() + tokens.size()) {
    throw std::logic_error("The request host token mirror does not have reserved continuation capacity.");
  }

  auto device_tokens = AllocateOnDevice(*params_, tokens);
  search_->SaveStateForTransaction();
  try {
    search_->AppendTokens(device_tokens);
    search_->CommitStateForTransaction();
  } catch (...) {
    const auto append_error = std::current_exception();
    try {
      search_->RestoreStateForTransaction();
    } catch (...) {
      engine->HandleContinuationRestoreFailure(
          shared_from_this(), append_error, std::current_exception());
    }
    std::rethrow_exception(append_error);
  }

  tokens_host_.insert(tokens_host_.end(), tokens.begin(), tokens.end());
  prompt_sequence_length_ = CurrentSequenceLength();
  status_ = RequestStatus::Assigned;
}

int64_t Request::CurrentSequenceLength() const {
  return search_->GetSequenceLength();
}

int64_t Request::CommittedSequenceLength() const {
  return CurrentSequenceLength() - static_cast<int64_t>(staged_draft_count_);
}

void Request::SetDraftTokens(std::span<const int32_t> tokens) {
  if (staged_draft_count_ != 0) {
    throw std::runtime_error("Cannot replace draft tokens while a step is in flight.");
  }
  if (IsClosed(status_)) {
    throw std::runtime_error("Cannot propose draft tokens for a closed request.");
  }
  draft_tokens_.clear();
  if (tokens.empty()) {
    return;
  }

  const auto& search = params_->search;
  // Verification compares the target model's argmax against each draft, which only reproduces the
  // request's own token stream when that stream is greedy.
  if (search.do_sample && search.top_k != 1 && search.temperature != 0) {
    throw std::runtime_error("Speculative draft tokens require a greedy request.");
  }
  // The draft rows are read before any logits processor runs, so a processor that would change a
  // row's argmax has to be inactive for every position this step verifies.
  if (search.repetition_penalty != 1.0f || search.no_repeat_ngram_size > 0 ||
      search.min_length > CurrentSequenceLength()) {
    throw std::runtime_error(
        "Speculative draft tokens require repetition_penalty 1, no_repeat_ngram_size 0, and a "
        "sequence already past min_length.");
  }
  auto engine = engine_.lock();
  if (!engine) {
    throw std::runtime_error("Cannot propose draft tokens before the request is added to an engine.");
  }
  const size_t max_drafts = engine->MaxDraftTokensPerStep();
  if (max_drafts == 0) {
    throw std::runtime_error("This engine cannot roll back a rejected draft token.");
  }
  if (tokens.size() > max_drafts) {
    throw std::runtime_error(
        "A step accepts at most " + std::to_string(max_drafts) + " draft tokens.");
  }
  ValidateAppendLength(*params_, static_cast<size_t>(CurrentSequenceLength()), tokens.size());
  if (tokens_host_.capacity() < tokens_host_.size() + tokens.size()) {
    throw std::logic_error("The request host token mirror does not have reserved draft capacity.");
  }
  draft_tokens_.assign(tokens.begin(), tokens.end());
}

std::span<const int32_t> Request::StagedDraftTokens() const {
  return std::span<const int32_t>{draft_tokens_}.subspan(0, staged_draft_count_);
}

bool Request::IsStopToken(int32_t token) const {
  const auto& stop_tokens = params_->config.model.eos_token_id;
  return std::find(stop_tokens.begin(), stop_tokens.end(), token) != stop_tokens.end();
}

void Request::AppendDraftsForTransaction(size_t draft_count) {
  if (draft_count == 0) {
    return;
  }
  if (staged_draft_count_ != 0 || draft_count > draft_tokens_.size()) {
    throw std::logic_error("The step staged more draft tokens than the request proposed.");
  }

  const std::span<const int32_t> drafts{draft_tokens_.data(), draft_count};
  auto device_tokens = AllocateOnDevice(*params_, drafts);
  search_->AppendTokens(device_tokens);
  tokens_host_.insert(tokens_host_.end(), drafts.begin(), drafts.end());
  staged_draft_count_ = draft_count;
  accepted_draft_count_ = 0;
}

void Request::RewindDraftsForTransaction(size_t accepted_count) {
  if (accepted_count > staged_draft_count_) {
    throw std::logic_error("The step accepted more draft tokens than it staged.");
  }
  const size_t rejected_count = staged_draft_count_ - accepted_count;
  accepted_draft_count_ = accepted_count;
  if (rejected_count == 0) {
    return;
  }

  staged_draft_count_ = accepted_count;
  tokens_host_.resize(tokens_host_.size() - rejected_count);
  search_->RewindTo(static_cast<size_t>(CurrentSequenceLength()) - rejected_count);
}

void Request::DiscardStagedDrafts() noexcept {
  if (staged_draft_count_ != 0) {
    tokens_host_.resize(tokens_host_.size() - staged_draft_count_);
    staged_draft_count_ = 0;
  }
  accepted_draft_count_ = 0;
}

RequestStateSnapshot Request::Snapshot() const {
  const int64_t current = CommittedSequenceLength();
  RequestStateSnapshot snapshot;
  snapshot.request_id = this;
  snapshot.status = status_;
  snapshot.current_sequence_length = current;
  snapshot.processed_sequence_length = processed_sequence_length_;
  snapshot.seen_sequence_length = seen_sequence_length_;
  snapshot.is_prefill = IsPrefill();
  return snapshot;
}

int64_t Request::ProcessedSequenceLength() const {
  return processed_sequence_length_;
}

size_t Request::ScheduledTokenCount() const {
  const size_t unprocessed = static_cast<size_t>(CurrentSequenceLength() - processed_sequence_length_);
  return std::min(scheduled_token_count_, unprocessed);
}

void Request::ScheduleTokens() {
  const size_t unprocessed = static_cast<size_t>(CurrentSequenceLength() - processed_sequence_length_);
  scheduled_token_count_ = Generators::ScheduledTokenCount(unprocessed, params_->search.chunk_size);
}

void Request::BindScheduledTokenCount(size_t token_count) {
  // A speculative step also sends the drafts the transaction is about to stage onto the sequence.
  const int64_t remaining = CurrentSequenceLength() - processed_sequence_length_ +
                            static_cast<int64_t>(draft_tokens_.size() - staged_draft_count_);
  if (remaining <= 0 || token_count == 0 ||
      token_count > static_cast<size_t>(remaining)) {
    throw std::runtime_error(
        "The dynamic step token count must be positive and no greater than the remaining tokens.");
  }
  scheduled_token_count_ = token_count;
}

bool Request::IsChunkComplete() const {
  return processed_sequence_length_ + static_cast<int64_t>(ScheduledTokenCount()) >= CurrentSequenceLength();
}

void Request::AdvanceChunk() {
  processed_sequence_length_ += static_cast<int64_t>(ScheduledTokenCount());
}

int32_t Request::UnseenToken() {
  if (next_unseen_token_index_ >= unseen_token_indices_.size())
    throw std::runtime_error("All tokens have been seen.");

  const size_t token_index = unseen_token_indices_[next_unseen_token_index_++];
  if (token_index >= tokens_host_.size())
    throw std::runtime_error("The unseen token index is outside the host token sequence.");
  seen_sequence_length_ = std::max(seen_sequence_length_, static_cast<int64_t>(token_index + 1));
  const int32_t token = tokens_host_[token_index];
  if (next_unseen_token_index_ == unseen_token_indices_.size()) {
    unseen_token_indices_.clear();
    next_unseen_token_index_ = 0;
  }
  return token;
}

bool Request::HasUnseenTokens() const {
  return next_unseen_token_index_ < unseen_token_indices_.size();
}

DeviceSpan<int32_t> Request::UnprocessedTokens() {
  auto sequence = search_->GetSequence(0);
  return sequence.subspan(processed_sequence_length_, ScheduledTokenCount());
}

std::span<const int32_t> Request::UnprocessedTokensCpu() const {
  const size_t begin = static_cast<size_t>(processed_sequence_length_);
  const size_t end = begin + ScheduledTokenCount();
  if (end > tokens_host_.size())
    throw std::runtime_error("The host token mirror is out of sync with the search sequence.");

  return std::span<const int32_t>{tokens_host_}.subspan(begin, end - begin);
}

bool Request::IsTurnComplete() const {
  return status_ == RequestStatus::TurnComplete;
}

bool Request::IsPrefill() const {
  return processed_sequence_length_ < prompt_sequence_length_;
}

void Request::GenerateNextTokens(DeviceSpan<float> logits) {
  PrepareGeneration(logits);

  auto& search_params = search_->params_->search;
  if (!search_params.do_sample || search_params.top_k == 1 || search_params.temperature == 0) {
    search_->SelectTop();
  } else {
    // The user explicitly called TopKTopP on a beam search
    if (search_params.num_beams != 1)
      throw std::runtime_error("TopK and TopP cannot be used with a beam search");

    // Sanity checks
    if (search_params.top_p < 0.0f || search_params.top_p > 1.0f)
      throw std::runtime_error("top_p must be between 0.0 and 1.0");
    if (search_params.top_k < 0)
      throw std::runtime_error("top_k must be 0 or greater");

    if (search_params.top_p > 0.0f && search_params.top_p < 1.0f && search_params.top_k > 1) {
      search_->SampleTopKTopP(search_params.top_k, search_params.top_p, search_params.temperature,
                              rng_);
    } else if (search_params.top_k > 1) {
      search_->SampleTopK(search_params.top_k, search_params.temperature, rng_);
    } else {
      assert(search_params.top_k == 0);
      search_->SampleTopP(search_params.top_p, search_params.temperature, rng_);
    }
  }
}

void Request::ValidateEngineCompatibility() const {
  const auto& search = params_->search;
  if (search.batch_size != 1 || search.num_beams != 1) {
    throw std::runtime_error("Engine requests require batch_size and num_beams to both be 1.");
  }
  if (search.top_p < 0.0f || search.top_p > 1.0f) {
    throw std::runtime_error("top_p must be between 0.0 and 1.0");
  }
  if (search.top_k < 0) {
    throw std::runtime_error("top_k must be 0 or greater");
  }
}

void Request::SaveStateForTransaction() {
  search_->SaveStateForTransaction();
  transaction_rng_ = rng_;
}

void Request::SaveStateForExternalSamplingTransaction() {
  search_->SaveStateForExternalSamplingTransaction();
  transaction_rng_ = rng_;
}

RequestStepResult Request::ApplyLogitsForTransaction(DeviceSpan<float> logits) {
  PrepareGenerationForTransaction(logits);
  SelectNextToken();
  return StageGeneration();
}

void Request::PrepareGenerationForTransaction(DeviceSpan<float> logits) {
  // Accepted drafts are already on the sequence, so this is the boundary past which any further
  // growth is the token the sampler itself produced.
  sequence_length_before_sampling_ = CurrentSequenceLength();
  ApplyLogitsProcessors(logits);
}

RequestStepResult Request::StageGenerationForTransaction() {
  return StageGeneration();
}

void Request::RestoreStateForTransaction() {
  search_->RestoreStateForTransaction();
  rng_ = transaction_rng_;
  DiscardStagedDrafts();
}

void Request::QueueStateRestoreForTransaction() {
  search_->QueueStateRestoreForTransaction();
  rng_ = transaction_rng_;
  DiscardStagedDrafts();
}

void Request::CompleteStateRestoreForTransaction() {
  search_->CompleteStateRestoreForTransaction();
}

void Request::CommitStateForTransaction() {
  search_->CommitStateForTransaction();
}

void Request::CommitStep(const RequestStepPlan& plan,
                         const RequestStepResult& result) noexcept {
  // ScheduledRequests reserved every append below before model execution. tokens_host_ retains its
  // existing max-length reservation, so nothing at this commit boundary allocates.
  const size_t accepted_drafts = accepted_draft_count_;
  const size_t first_accepted_index = tokens_host_.size() - accepted_drafts;
  for (size_t offset = 0; offset < accepted_drafts; ++offset) {
    unseen_token_indices_.push_back(first_accepted_index + offset);
  }
  if (result.token_appended) {
    const size_t token_index = tokens_host_.size();
    tokens_host_.push_back(result.token);
    unseen_token_indices_.push_back(token_index);
  }
  // A verify step reserved cache slots for every draft; the rejected ones were never committed.
  processed_sequence_length_ =
      static_cast<int64_t>(plan.target_cache_slots - (plan.draft_token_count - accepted_drafts));
  status_ = result.done ? RequestStatus::TurnComplete : RequestStatus::Active;
  draft_tokens_.clear();
  staged_draft_count_ = 0;
  accepted_draft_count_ = 0;
}

void Request::ApplyLogitsProcessors(DeviceSpan<float> logits) {
  search_->SetLogits(logits);
  auto& search_params = search_->params_->search;
  search_->ApplyMinLength(search_params.min_length);
  search_->ApplyRepetitionPenalty(search_params.repetition_penalty);
  search_->ApplyNoRepeatNgram(search_params.no_repeat_ngram_size);
}

void Request::SelectNextToken() {
  auto& search_params = search_->params_->search;
  if (!search_params.do_sample || search_params.top_k == 1 || search_params.temperature == 0) {
    search_->SelectTop();
  } else if (search_params.top_p > 0.0f && search_params.top_p < 1.0f &&
             search_params.top_k > 1) {
    search_->SampleTopKTopP(search_params.top_k, search_params.top_p,
                            search_params.temperature, rng_);
  } else if (search_params.top_k > 1) {
    search_->SampleTopK(search_params.top_k, search_params.temperature, rng_);
  } else {
    search_->SampleTopP(search_params.top_p, search_params.temperature, rng_);
  }
}

RequestStepResult Request::StageGeneration() {
  search_->CompleteGeneration();
  const bool done = search_->IsDone();
  const bool token_appended = CurrentSequenceLength() > sequence_length_before_sampling_;
  int32_t token = 0;
  if (token_appended) {
    token = search_->GetNextTokens().CpuSpan().back();
  }
  return RequestStepResult{
      token,
      token_appended,
      done,
  };
}

void Request::PrepareGeneration(DeviceSpan<float> logits) {
  processed_sequence_length_ = search_->GetSequence(0).size();
  ApplyLogitsProcessors(logits);
}

const Config::Search& Request::SearchOptions() const {
  return search_->params_->search;
}

bool Request::BindNextTokensSlot(DeviceSpan<int32_t> slot) {
  return search_->BindNextTokensSlot(slot);
}

bool Request::SupportsBatchedSampling() const {
  return search_->SupportsBatchedSampling();
}

void Request::OnNextTokensSampled() {
  search_->OnNextTokensSampled();
}

BatchedSamplerState& Request::SamplingState(BatchedSampler& sampler) {
  if (!batched_sampler_state_ || !sampler.OwnsState(*batched_sampler_state_))
    batched_sampler_state_ = sampler.CreateState(search_->params_->search.random_seed);
  return *batched_sampler_state_;
}

void Request::CompleteGeneration() {
  search_->CompleteGeneration();

  const size_t sequence_length = static_cast<size_t>(CurrentSequenceLength());
  if (sequence_length > tokens_host_.size()) {
    const size_t new_token_count = sequence_length - tokens_host_.size();
    auto next_tokens = search_->GetNextTokens().CpuSpan();
    if (new_token_count > next_tokens.size())
      throw std::runtime_error("The search produced fewer tokens than it appended to the sequence.");

    const size_t first_new_token = tokens_host_.size();
    tokens_host_.insert(tokens_host_.end(), next_tokens.end() - new_token_count, next_tokens.end());
    for (size_t token_index = first_new_token; token_index < tokens_host_.size(); ++token_index) {
      unseen_token_indices_.push_back(token_index);
    }
  }

  if (search_->IsDone()) {
    status_ = RequestStatus::TurnComplete;
  }
}

std::shared_ptr<GeneratorParams> Request::Params() {
  return params_;
}

void Request::SetOpaqueData(void* data) {
  opaque_data_ = data;
}

void* Request::GetOpaqueData() {
  return opaque_data_;
}

}  // namespace Generators
