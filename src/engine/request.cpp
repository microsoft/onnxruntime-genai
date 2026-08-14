// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "request.h"

#include "engine.h"
#include "sequence_positions.h"
#include "../constrained_logits_processor.h"
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

}  // namespace

Request::Request(std::shared_ptr<GeneratorParams> params)
    : params_{params}, search_{CreateSearch(*params.get())} {
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
  if (params->guidance_ff_tokens_enabled) {
    throw std::runtime_error("Guidance fast-forward tokens are not supported by the engine.");
  }

  const bool guidance_requested = !params->guidance_type.empty() || !params->guidance_data.empty();
  if (guidance_requested && !params->model_) {
    throw std::runtime_error("Engine guidance requires request parameters associated with a model.");
  }
  if (params->model_) {
    guidance_logits_processor_ = CreateGuidanceLogitsProcessor(*params->model_, params);
  }
  if (guidance_requested && !guidance_logits_processor_) {
    throw std::runtime_error("Engine guidance is unavailable. Build with use_guidance=true and provide both guidance type and data.");
  }

  // The engine drives one independent search per request, so completion is batched: see
  // ScheduledRequests::GenerateNextTokens().
  search_->DeferCompletion(true);
}

Request::~Request() = default;

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

void Request::Schedule() {
  if (status_ != RequestStatus::Assigned) {
    throw std::runtime_error("Request cannot be scheduled unless it has been previously added to the engine.");
  }

  if (!search_ || CurrentSequenceLength() == 0) {
    throw std::runtime_error("Cannot schedule a request with no tokens.");
  }

  status_ = RequestStatus::InProgress;
}

void Request::Remove() {
  auto engine = engine_.lock();
  if (engine) {
    engine->RemoveRequest(shared_from_this());
  }
  status_ = RequestStatus::Unassigned;
}

void Request::AddTokens(std::span<const int32_t> tokens) {
  if (tokens.size() == 0)
    throw std::runtime_error("Expected at least one token for generation. Received 0.");

  if (tokens.size() + CurrentSequenceLength() > params_->search.max_length)
    throw std::runtime_error("Input tokens size (" +
                             std::to_string(tokens.size()) +
                             ") exceeds the max length (" +
                             std::to_string(params_->search.max_length) + ")");

  if (status_ == RequestStatus::Unassigned) {
    std::copy(tokens.begin(), tokens.end(), std::back_inserter(prefill_input_ids_));
  } else if (status_ == RequestStatus::InProgress) {
    throw std::runtime_error("Cannot add tokens to a request that is in progress.");
  } else if (status_ == RequestStatus::Completed) {
    if (HasUnseenTokens()) {
      throw std::runtime_error("Consume all generated tokens before continuing a completed request.");
    }
    auto device_tokens = AllocateOnDevice(*params_, tokens);
    search_->AppendTokens(device_tokens);
    prompt_sequence_length_ = CurrentSequenceLength();
    tokens_host_.insert(tokens_host_.end(), tokens.begin(), tokens.end());
    seen_sequence_length_ = CurrentSequenceLength();
    status_ = RequestStatus::InProgress;
  }
}

int64_t Request::CurrentSequenceLength() const {
  return search_->GetSequenceLength();
}

RequestStateSnapshot Request::Snapshot() const {
  const int64_t current = CurrentSequenceLength();
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

bool Request::IsChunkComplete() const {
  return processed_sequence_length_ + static_cast<int64_t>(ScheduledTokenCount()) >= CurrentSequenceLength();
}

void Request::AdvanceChunk() {
  processed_sequence_length_ += static_cast<int64_t>(ScheduledTokenCount());
}

int32_t Request::UnseenToken() {
  if (static_cast<size_t>(seen_sequence_length_) >= tokens_host_.size())
    throw std::runtime_error("All tokens have been seen.");

  return tokens_host_[seen_sequence_length_++];
}

bool Request::HasUnseenTokens() const {
  return seen_sequence_length_ < CurrentSequenceLength();
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

bool Request::IsDone() const {
  return status_ == RequestStatus::Completed;
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
      search_->SampleTopKTopP(search_params.top_k, search_params.top_p, search_params.temperature);
    } else if (search_params.top_k > 1) {
      search_->SampleTopK(search_params.top_k, search_params.temperature);
    } else {
      assert(search_params.top_k == 0);
      search_->SampleTopP(search_params.top_p, search_params.temperature);
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
  if (guidance_logits_processor_) {
    guidance_transaction_checkpoint_ = guidance_logits_processor_->Clone();
  }
  search_->SaveStateForTransaction();
}

void Request::SaveStateForExternalSamplingTransaction() {
  if (guidance_logits_processor_) {
    guidance_transaction_checkpoint_ = guidance_logits_processor_->Clone();
  }
  search_->SaveStateForExternalSamplingTransaction();
}

RequestStepResult Request::ApplyLogitsForTransaction(DeviceSpan<float> logits) {
  const auto sequence_length_before = CurrentSequenceLength();
  PrepareGenerationForTransaction(logits);
  SelectNextToken();
  return StageGeneration(sequence_length_before);
}

void Request::PrepareGenerationForTransaction(DeviceSpan<float> logits) {
  ApplyLogitsProcessors(logits);
}

RequestStepResult Request::StageGenerationForTransaction(
    const RequestStepPlan& plan) {
  return StageGeneration(plan.sequence_length_before);
}

void Request::RestoreStateForTransaction() {
  search_->RestoreStateForTransaction();
  if (guidance_transaction_checkpoint_) {
    guidance_logits_processor_ = std::move(guidance_transaction_checkpoint_);
  }
}

void Request::QueueStateRestoreForTransaction() {
  search_->QueueStateRestoreForTransaction();
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
                         const RequestStepResult& result) {
  if (result.token_appended) {
    tokens_host_.push_back(result.token);
  }
  processed_sequence_length_ = static_cast<int64_t>(plan.target_cache_slots);
  status_ = result.done ? RequestStatus::Completed : RequestStatus::InProgress;
  if (result.done && guidance_logits_processor_) {
    guidance_logits_processor_->Reset();
  }
}

void Request::ApplyLogitsProcessors(DeviceSpan<float> logits) {
  search_->SetLogits(logits);
  if (guidance_logits_processor_) {
    guidance_logits_processor_->ProcessLogits(logits);
  }
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
                            search_params.temperature);
  } else if (search_params.top_k > 1) {
    search_->SampleTopK(search_params.top_k, search_params.temperature);
  } else {
    search_->SampleTopP(search_params.top_p, search_params.temperature);
  }
}

RequestStepResult Request::StageGeneration(int64_t sequence_length_before) {
  search_->CompleteGeneration();
  const bool done = search_->IsDone();
  const bool token_appended = CurrentSequenceLength() > sequence_length_before;
  int32_t token = 0;
  if (token_appended) {
    token = search_->GetNextTokens().CpuSpan().back();
  }
  RequestStepResult result{
      token,
      token_appended,
      done,
  };
  CommitGuidanceToken(result);
  return result;
}

void Request::CommitGuidanceToken(const RequestStepResult& result) {
  if (guidance_logits_processor_ && result.token_appended) {
    int32_t token = result.token;
    guidance_logits_processor_->CommitTokens(std::span<int32_t>{&token, 1});
  }
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
  const size_t sequence_length_before = static_cast<size_t>(CurrentSequenceLength());
  search_->CompleteGeneration();

  const size_t sequence_length = static_cast<size_t>(CurrentSequenceLength());
  if (sequence_length > tokens_host_.size()) {
    const size_t new_token_count = sequence_length - tokens_host_.size();
    auto next_tokens = search_->GetNextTokens().CpuSpan();
    if (new_token_count > next_tokens.size())
      throw std::runtime_error("The search produced fewer tokens than it appended to the sequence.");

    tokens_host_.insert(tokens_host_.end(), next_tokens.end() - new_token_count, next_tokens.end());
  }

  if (sequence_length > sequence_length_before) {
    auto next_tokens = search_->GetNextTokens().CpuSpan();
    auto new_tokens = next_tokens.subspan(next_tokens.size() - (sequence_length - sequence_length_before));
    if (guidance_logits_processor_) {
      guidance_logits_processor_->CommitTokens(new_tokens);
    }
  }

  if (search_->IsDone()) {
    status_ = RequestStatus::Completed;
    if (guidance_logits_processor_) {
      guidance_logits_processor_->Reset();
    }
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
