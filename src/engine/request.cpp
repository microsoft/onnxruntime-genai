// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "request.h"

#include <cmath>

#include "engine.h"
#include "sequence_positions.h"
#include "../constrained_logits_processor.h"
#include "../search.h"

namespace Generators {

DeviceSpan<int32_t> Request::AllocateOnDevice(
    GeneratorParams& params,
    std::span<const int32_t> input_ids) {
  auto device_tokens = params.p_device->Allocate<int32_t>(input_ids.size());
  auto cpu_tokens = device_tokens.CpuSpan();
  std::copy(input_ids.begin(), input_ids.end(), cpu_tokens.begin());
  device_tokens.CopyCpuToDevice();
  return device_tokens;
}

void Request::ValidateAppendLength(
    size_t max_total_tokens,
    size_t current_sequence_length,
    size_t token_count) {
  if (current_sequence_length >= max_total_tokens ||
      token_count >= max_total_tokens - current_sequence_length) {
    throw std::runtime_error(
        "Appending input_tokens_count (" + std::to_string(token_count) +
        ") to current_sequence_length (" +
        std::to_string(current_sequence_length) +
        ") must leave room for at least one generated token before "
        "max_total_tokens (" +
        std::to_string(max_total_tokens) + ").");
  }
}

void TurnOptions::ValidateOwnerThread() const {
  auto bound_request = request.lock();
  if (!bound_request) {
    throw std::runtime_error(
        "Cannot use Turn options after their Request has been destroyed.");
  }
  bound_request->ValidateOwnerThread();
}

void Request::ValidateOwnerThread() const {
  auto engine = engine_.lock();
  if (!engine) {
    if (IsClosed(status_)) {
      throw std::runtime_error("Cannot use a closed request.");
    }
    throw std::runtime_error(
        "Cannot use a Request after its Engine has been destroyed.");
  }
  engine->ValidateOwnerThread();
}

Request::Request(
    std::shared_ptr<GeneratorParams> params,
    size_t max_total_tokens,
    std::shared_ptr<std::atomic<bool>> abandonment_pending)
    : max_total_tokens_{max_total_tokens},
      params_{params},
      rng_{CreateRandomGenerator(params->search.random_seed)},
      search_{CreateSearch(*params)},
      abandonment_pending_{std::move(abandonment_pending)} {
  draft_tokens_.reserve(kMaxDraftTokensPerStep);
  // A request is one sequence: the engine batches requests, not rows within a request. Several
  // places here read row 0 only (UnprocessedTokens, CurrentSequenceLength) or take the tail of the
  // next-token span, so a wider search would silently mirror the wrong row's tokens.
  if (params->search.batch_size != 1) {
    throw std::runtime_error(
        "Engine requests require search.batch_size == 1; actual value is " +
        std::to_string(params->search.batch_size) +
        ". Batch across requests instead.");
  }
  // Beam search does not implement the deferred completion contract below, so its next tokens would
  // never be copied back from the device.
  if (params->search.num_beams != 1) {
    throw std::runtime_error(
        "Engine requests require search.num_beams == 1; actual value is " +
        std::to_string(params->search.num_beams) +
        ". Beam search is not supported by the Engine.");
  }
  if (params->guidance_ff_tokens_enabled) {
    throw std::runtime_error("Guidance fast-forward tokens are not supported by the engine.");
  }

  guidance_logits_processor_ = CreateGuidanceLogitsProcessor(params);

  // The engine drives one independent search per request, so completion is batched: see
  // ScheduledRequests::GenerateNextTokens().
  search_->DeferCompletion(true);
  tokens_host_.reserve(max_total_tokens_);
}

Request::~Request() = default;

void Request::OnFirstExternalReference() noexcept {
  externally_abandoned_.store(false, std::memory_order_release);
}

void Request::OnLastExternalReference() noexcept {
  externally_abandoned_.store(true, std::memory_order_release);
  abandonment_pending_->store(true, std::memory_order_release);
}

bool Request::IsExternallyAbandoned() const noexcept {
  return externally_abandoned_.load(std::memory_order_acquire);
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

void Request::Close() {
  if (IsClosed(status_)) {
    return;
  }

  auto engine = engine_.lock();
  if (!engine) {
    CompleteClose();
    return;
  }
  engine->CloseRequest(shared_from_this());
}

bool Request::Cancel(uint64_t turn_id) {
  auto engine = engine_.lock();
  if (!engine) {
    throw std::runtime_error(
        "Cannot cancel after the request's engine has been destroyed.");
  }
  return engine->CancelRequest(shared_from_this(), turn_id);
}

void Request::CompleteClose() noexcept {
  engine_.reset();
  status_ = RequestStatus::Closed;
  guidance_transaction_checkpoint_.reset();
  guidance_logits_processor_.reset();
  batched_sampler_state_.reset();
  search_.reset();
  params_.reset();
  std::vector<int32_t>{}.swap(tokens_host_);
}

uint64_t Request::BeginTurn(
    std::span<const int32_t> tokens,
    std::optional<size_t> max_generated_tokens) {
  if (IsClosed(status_)) {
    throw std::runtime_error("Cannot begin a turn for a closed request.");
  }
  auto engine = engine_.lock();
  if (!engine) {
    throw std::runtime_error(
        "Cannot begin a turn after the request's engine has been destroyed.");
  }
  return engine->BeginTurn(
      shared_from_this(), tokens, max_generated_tokens);
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
  if (tokens.empty()) {
    draft_tokens_.clear();
    return;
  }
  if (!IsExecuting(status_) || IsPrefill()) {
    throw std::runtime_error(
        "Speculative draft tokens may only be proposed when the request is ready to decode.");
  }
  if (guidance_logits_processor_) {
    throw std::runtime_error("Speculative draft tokens are not supported with guidance.");
  }

  const auto& search = params_->search;
  // Sampled verification extracts a bounded target distribution for every row. Pure nucleus
  // sampling has no bounded sparse support and stays on the standard path.
  if (search.do_sample && search.top_k != 1 && search.temperature != 0 && search.top_k <= 0) {
    throw std::runtime_error("Sampled speculative draft tokens require top_k greater than 1.");
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
    throw std::runtime_error("This engine does not support speculative draft verification.");
  }
  if (tokens.size() > max_drafts) {
    throw std::runtime_error(
        "A step accepts at most " + std::to_string(max_drafts) + " draft tokens.");
  }
  ValidateAppendLength(
      max_total_tokens_, static_cast<size_t>(CurrentSequenceLength()), tokens.size());
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
  draft_verification_completed_generation_ = false;
}

void Request::CommitAcceptedDraftsForTransaction(size_t accepted_count) {
  if (accepted_count > staged_draft_count_) {
    throw std::logic_error("The step accepted more draft tokens than it staged.");
  }
  const size_t proposed_count = staged_draft_count_;
  const size_t committed_length =
      static_cast<size_t>(CurrentSequenceLength()) - proposed_count;
  tokens_host_.resize(tokens_host_.size() - proposed_count);
  search_->RewindTo(committed_length);
  staged_draft_count_ = 0;
  accepted_draft_count_ = 0;
  draft_verification_completed_generation_ = false;

  for (size_t offset = 0; offset < accepted_count; ++offset) {
    const int32_t token = draft_tokens_[offset];
    const int64_t sequence_length_before = CurrentSequenceLength();
    search_->CommitToken(token);
    const int64_t sequence_length_after = CurrentSequenceLength();
    if (sequence_length_after == sequence_length_before + 1) {
      tokens_host_.push_back(token);
      ++staged_draft_count_;
      ++accepted_draft_count_;
    } else if (sequence_length_after != sequence_length_before ||
               !search_->IsDone()) {
      throw std::logic_error(
          "Committing an accepted draft produced an invalid sequence transition.");
    }
    const bool turn_limit_reached =
        turn_max_generated_tokens_ &&
        turn_generated_tokens_ + accepted_draft_count_ >=
            *turn_max_generated_tokens_;
    if (search_->IsDone() || turn_limit_reached) {
      draft_verification_completed_generation_ = true;
      break;
    }
  }
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

void Request::RecordSampledDraftAcceptance(size_t accepted_count) {
  if (staged_draft_count_ != 0 || accepted_count > draft_tokens_.size()) {
    throw std::logic_error("Sampled verification recorded an invalid accepted draft prefix.");
  }
  tokens_host_.insert(tokens_host_.end(), draft_tokens_.begin(),
                      draft_tokens_.begin() + static_cast<ptrdiff_t>(accepted_count));
  staged_draft_count_ = accepted_count;
  accepted_draft_count_ = accepted_count;
}

void Request::DiscardStagedDrafts() noexcept {
  if (staged_draft_count_ != 0) {
    tokens_host_.resize(tokens_host_.size() - staged_draft_count_);
    staged_draft_count_ = 0;
  }
  accepted_draft_count_ = 0;
  draft_verification_completed_generation_ = false;
}

RequestStateSnapshot Request::Snapshot() const {
  const int64_t current = CommittedSequenceLength();
  RequestStateSnapshot snapshot;
  snapshot.request_id = this;
  snapshot.status = status_;
  snapshot.current_sequence_length = current;
  snapshot.processed_sequence_length = processed_sequence_length_;
  snapshot.is_prefill = IsPrefill();
  snapshot.has_current_turn = has_current_turn_;
  snapshot.current_turn_id = current_turn_id_;
  snapshot.finish_reason = finish_reason_;
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
        "The dynamic step token count (" + std::to_string(token_count) +
        ") must be positive and no greater than the remaining token count (" +
        std::to_string(remaining) + ").");
  }
  scheduled_token_count_ = token_count;
}

bool Request::IsChunkComplete() const {
  return processed_sequence_length_ + static_cast<int64_t>(ScheduledTokenCount()) >= CurrentSequenceLength();
}

void Request::AdvanceChunk() {
  processed_sequence_length_ += static_cast<int64_t>(ScheduledTokenCount());
}

void Request::AppendTokensForAuxiliaryDecoder(std::span<const int32_t> tokens) {
  if (status_ != RequestStatus::Active || tokens.empty() ||
      processed_sequence_length_ != CurrentSequenceLength()) {
    throw std::logic_error(
        "Auxiliary decoder tokens require an active request with no pending rows.");
  }
  ValidateAppendLength(*params_, static_cast<size_t>(CurrentSequenceLength()), tokens.size());
  auto device_tokens = AllocateOnDevice(*params_, tokens);
  search_->AppendTokens(device_tokens);
  tokens_host_.insert(tokens_host_.end(), tokens.begin(), tokens.end());
}

void Request::AppendTokensForAuxiliaryDecoder(DeviceSpan<int32_t> tokens) {
  if (status_ != RequestStatus::Active || tokens.empty() ||
      processed_sequence_length_ != CurrentSequenceLength()) {
    throw std::logic_error(
        "Auxiliary decoder tokens require an active request with no pending rows.");
  }
  ValidateAppendLength(*params_, static_cast<size_t>(CurrentSequenceLength()), tokens.size());
  search_->AppendTokens(tokens);
  const int32_t non_pad = params_->config.model.pad_token_id == 0 ? 1 : 0;
  tokens_host_.insert(tokens_host_.end(), tokens.size(), non_pad);
}

void Request::RewindAuxiliaryDecoderTo(size_t sequence_length) {
  if (status_ != RequestStatus::Active ||
      processed_sequence_length_ != CurrentSequenceLength() ||
      sequence_length > static_cast<size_t>(processed_sequence_length_)) {
    throw std::logic_error(
        "Auxiliary decoder rewind requires an active request at a processed sequence boundary.");
  }
  search_->RewindTo(sequence_length);
  tokens_host_.resize(sequence_length);
  processed_sequence_length_ = static_cast<int64_t>(sequence_length);
  scheduled_token_count_ = 0;
}

void Request::CommitAuxiliaryDecoderStep() noexcept {
  processed_sequence_length_ += static_cast<int64_t>(ScheduledTokenCount());
  scheduled_token_count_ = 0;
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

void Request::GenerateNextTokens(DeviceSpan<float> logits, bool guidance_applied) {
  PrepareGeneration(logits, guidance_applied);

  auto& search_params = search_->params_->search;
  if (!search_params.do_sample || search_params.top_k == 1 || search_params.temperature == 0) {
    search_->SelectTop();
  } else {
    // The user explicitly called TopKTopP on a beam search
    if (search_params.num_beams != 1)
      throw std::runtime_error("TopK and TopP cannot be used with a beam search");

    // Sanity checks
    if (!std::isfinite(search_params.top_p) ||
        search_params.top_p < 0.0f || search_params.top_p > 1.0f)
      throw std::runtime_error(
          "top_p (" + std::to_string(search_params.top_p) +
          ") must be finite and between 0.0 and 1.0.");
    if (search_params.top_k < 0)
      throw std::runtime_error(
          "top_k (" + std::to_string(search_params.top_k) +
          ") must be 0 or greater.");

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
  if (!std::isfinite(search.top_p) ||
      search.top_p < 0.0f || search.top_p > 1.0f) {
    throw std::runtime_error(
        "top_p (" + std::to_string(search.top_p) +
        ") must be finite and between 0.0 and 1.0.");
  }
  if (search.top_k < 0) {
    throw std::runtime_error(
        "top_k (" + std::to_string(search.top_k) +
        ") must be 0 or greater.");
  }
}

void Request::SaveStateForTransaction() {
  auto guidance_checkpoint = guidance_logits_processor_
                                 ? guidance_logits_processor_->Clone()
                                 : nullptr;
  search_->SaveStateForTransaction();
  guidance_transaction_checkpoint_ = std::move(guidance_checkpoint);
  transaction_rng_ = rng_;
  transaction_processed_sequence_length_ = processed_sequence_length_;
  transaction_tokens_host_size_ = tokens_host_.size();
}

void Request::SaveStateForExternalSamplingTransaction() {
  auto guidance_checkpoint = guidance_logits_processor_
                                 ? guidance_logits_processor_->Clone()
                                 : nullptr;
  search_->SaveStateForExternalSamplingTransaction();
  guidance_transaction_checkpoint_ = std::move(guidance_checkpoint);
  transaction_rng_ = rng_;
  transaction_processed_sequence_length_ = processed_sequence_length_;
  transaction_tokens_host_size_ = tokens_host_.size();
}

RequestStepResult Request::ApplyLogitsForTransaction(DeviceSpan<float> logits,
                                                     bool guidance_applied) {
  const auto sequence_length_before = CurrentSequenceLength();
  PrepareGenerationForTransaction(logits, guidance_applied);
  SelectNextToken();
  return StageGeneration(sequence_length_before);
}

void Request::PrepareGenerationForTransaction(DeviceSpan<float> logits,
                                              bool guidance_applied) {
  ApplyLogitsProcessors(logits, guidance_applied);
}

RequestStepResult Request::StageGenerationForTransaction(
    const RequestStepPlan& plan) {
  // Accepted drafts have already extended the sequence, so the baseline has to match the one
  // ApplyLogitsForTransaction reads after they commit. Without the offset a step that ends on an
  // unappended EOS would look like it appended a token.
  return StageGeneration(
      plan.sequence_length_before + static_cast<int64_t>(accepted_draft_count_));
}

RequestStepResult Request::StageDraftCompletionForTransaction() {
  if (!draft_verification_completed_generation_) {
    throw std::logic_error(
        "Draft completion was staged before verification completed generation.");
  }

  GenerationFinishReason finish_reason = GenerationFinishReason::ContextLimit;
  const bool turn_limit_reached =
      turn_max_generated_tokens_ &&
      turn_generated_tokens_ + accepted_draft_count_ >=
          *turn_max_generated_tokens_;
  if (turn_limit_reached) {
    finish_reason = GenerationFinishReason::TurnLimit;
  } else {
    const auto next_tokens = search_->GetNextTokens().CpuSpan();
    if (search_->IsDone() && !next_tokens.empty() &&
        contains(params_->config.model.eos_token_id, next_tokens.back())) {
      finish_reason = GenerationFinishReason::EosToken;
    }
  }
  RequestStepResult result{
      0,
      false,
      true,
      finish_reason,
  };
  StageVisibleTokens(result, accepted_draft_count_, std::nullopt);
  return result;
}

void Request::RestoreStateForTransaction() {
  search_->RestoreStateForTransaction();
  rng_ = transaction_rng_;
  processed_sequence_length_ = transaction_processed_sequence_length_;
  tokens_host_.resize(transaction_tokens_host_size_);
  staged_draft_count_ = 0;
  accepted_draft_count_ = 0;
  scheduled_token_count_ = 0;
  if (guidance_transaction_checkpoint_) {
    guidance_logits_processor_ = std::move(guidance_transaction_checkpoint_);
  }
}

void Request::QueueStateRestoreForTransaction() {
  search_->QueueStateRestoreForTransaction();
  rng_ = transaction_rng_;
  processed_sequence_length_ = transaction_processed_sequence_length_;
  tokens_host_.resize(transaction_tokens_host_size_);
  staged_draft_count_ = 0;
  accepted_draft_count_ = 0;
  scheduled_token_count_ = 0;
}

void Request::CompleteStateRestoreForTransaction() {
  search_->CompleteStateRestoreForTransaction();
  if (guidance_transaction_checkpoint_) {
    guidance_logits_processor_ = std::move(guidance_transaction_checkpoint_);
  }
}

void Request::CommitStateForTransaction() {
  search_->CommitStateForTransaction();
  guidance_transaction_checkpoint_.reset();
}

void Request::CommitStep(const RequestStepPlan& plan,
                         const RequestStepResult& result) noexcept {
  // Draft verification has already retained the accepted prefix in the host mirror and Search.
  // tokens_host_ retains its existing max-length reservation, so the final append cannot allocate
  // at this commit boundary.
  const size_t accepted_drafts = accepted_draft_count_;
  turn_generated_tokens_ += accepted_drafts;
  if (result.token_appended) {
    tokens_host_.push_back(result.token);
    ++turn_generated_tokens_;
  }
  // A verify step reserved cache slots for every draft; the rejected ones were never committed.
  processed_sequence_length_ =
      static_cast<int64_t>(plan.target_cache_slots - (plan.draft_token_count - accepted_drafts));
  status_ = result.done ? RequestStatus::TurnComplete : RequestStatus::Active;
  if (result.done) {
    finish_reason_ = result.finish_reason;
  }
  draft_tokens_.clear();
  staged_draft_count_ = 0;
  accepted_draft_count_ = 0;
  draft_verification_completed_generation_ = false;
}

// Records the tokens this step makes externally visible, in sequence order. Accepted drafts are
// already in tokens_host_; a freshly sampled token is only appended by CommitStep.
void Request::StageVisibleTokens(RequestStepResult& result,
                                 size_t committed_count,
                                 std::optional<int32_t> sampled_token) const {
  if (committed_count + (sampled_token ? 1u : 0u) > result.visible_tokens.size()) {
    throw std::logic_error(
        "A single engine step produced more visible tokens than it can report.");
  }
  if (committed_count > tokens_host_.size()) {
    throw std::logic_error(
        "The host token mirror is missing tokens this step committed.");
  }
  const auto committed = std::span<const int32_t>{tokens_host_}.last(committed_count);
  std::copy(committed.begin(), committed.end(), result.visible_tokens.begin());
  result.visible_token_count = committed.size();
  if (sampled_token) {
    result.visible_tokens[result.visible_token_count++] = *sampled_token;
  }
}

void Request::ApplyLogitsProcessors(DeviceSpan<float> logits,
                                    bool guidance_applied) {
  search_->SetLogits(logits);
  if (guidance_logits_processor_ && !guidance_applied) {
    guidance_logits_processor_->ProcessLogits(logits);
  }
  auto& search_params = search_->params_->search;
  search_->ApplyMinLength(search_params.min_length);
  search_->ApplyRepetitionPenalty(search_params.repetition_penalty);
  search_->ApplyNoRepeatNgram(search_params.no_repeat_ngram_size);
}

void Request::ResetGuidanceForNewTurn() {
  if (guidance_logits_processor_) {
    guidance_logits_processor_->Reset();
  }
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

RequestStepResult Request::StageGeneration(int64_t sequence_length_before) {
  search_->CompleteGeneration();
  const bool search_done = search_->IsDone();
  const bool token_appended = CurrentSequenceLength() > sequence_length_before;
  const auto next_tokens = search_->GetNextTokens().CpuSpan();
  const int32_t token = next_tokens.empty() ? 0 : next_tokens.back();
  const size_t generated_tokens_after_step =
      turn_generated_tokens_ + accepted_draft_count_ +
      static_cast<size_t>(token_appended);
  const bool turn_limit_reached =
      turn_max_generated_tokens_ &&
      generated_tokens_after_step >= *turn_max_generated_tokens_;
  const bool context_limit_reached =
      static_cast<size_t>(CurrentSequenceLength()) >= max_total_tokens_;
  GenerationFinishReason finish_reason = GenerationFinishReason::None;
  if (search_done && !next_tokens.empty() &&
      contains(params_->config.model.eos_token_id, token)) {
    finish_reason = GenerationFinishReason::EosToken;
  } else if (turn_limit_reached) {
    finish_reason = GenerationFinishReason::TurnLimit;
  } else if (context_limit_reached || search_done) {
    finish_reason = GenerationFinishReason::ContextLimit;
  }
  const bool done = finish_reason != GenerationFinishReason::None;
  RequestStepResult result{
      token,
      token_appended,
      done,
      finish_reason,
  };
  StageVisibleTokens(
      result, accepted_draft_count_,
      token_appended ? std::optional<int32_t>{token} : std::nullopt);
  CommitGuidanceToken(result);
  if (done && guidance_logits_processor_) {
    guidance_logits_processor_->Reset();
  }
  return result;
}

void Request::CommitGuidanceToken(const RequestStepResult& result) {
  if (guidance_logits_processor_ && result.token_appended) {
    int32_t token = result.token;
    guidance_logits_processor_->CommitTokens(std::span<int32_t>{&token, 1});
  }
}

void Request::PrepareGeneration(DeviceSpan<float> logits,
                                bool guidance_applied) {
  processed_sequence_length_ = search_->GetSequence(0).size();
  ApplyLogitsProcessors(logits, guidance_applied);
}

std::span<const uint32_t> Request::GetReadyGuidanceMask() {
  return guidance_logits_processor_ ? guidance_logits_processor_->GetReadyMask()
                                    : std::span<const uint32_t>{};
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

RequestStepResult Request::CompleteGeneration() {
  search_->CompleteGeneration();
  const auto next_tokens = search_->GetNextTokens().CpuSpan();

  const size_t sequence_length = static_cast<size_t>(CurrentSequenceLength());
  size_t new_token_count{};
  int32_t token{};
  if (sequence_length > tokens_host_.size()) {
    new_token_count = sequence_length - tokens_host_.size();
    if (new_token_count > next_tokens.size())
      throw std::runtime_error("The search produced fewer tokens than it appended to the sequence.");
    auto new_tokens = next_tokens.last(new_token_count);

    tokens_host_.insert(tokens_host_.end(), new_tokens.begin(), new_tokens.end());
    token = new_tokens.back();
    if (guidance_logits_processor_) {
      guidance_logits_processor_->CommitTokens(new_tokens);
    }
    turn_generated_tokens_ += new_token_count;
  }

  const bool turn_limit_reached =
      turn_max_generated_tokens_ &&
      turn_generated_tokens_ >= *turn_max_generated_tokens_;
  const bool context_limit_reached =
      static_cast<size_t>(CurrentSequenceLength()) >= max_total_tokens_;
  if (search_->IsDone() || turn_limit_reached || context_limit_reached) {
    status_ = RequestStatus::TurnComplete;
    if (search_->IsDone() && !next_tokens.empty() &&
        contains(params_->config.model.eos_token_id, next_tokens.back())) {
      finish_reason_ = GenerationFinishReason::EosToken;
    } else if (turn_limit_reached) {
      finish_reason_ = GenerationFinishReason::TurnLimit;
    } else {
      finish_reason_ = GenerationFinishReason::ContextLimit;
    }
    if (guidance_logits_processor_) {
      guidance_logits_processor_->Reset();
    }
  }
  RequestStepResult result{
      token,
      new_token_count != 0,
      Generators::IsTurnComplete(status_),
      finish_reason_,
  };
  StageVisibleTokens(result, new_token_count, std::nullopt);
  return result;
}

}  // namespace Generators
