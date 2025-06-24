// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "request.h"

#include "engine.h"
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
    : params_{params}, search_{CreateSearch(*params.get())} {}

void Request::Assign(std::shared_ptr<Engine> engine) {
  if (status_ != RequestStatus::Unassigned) {
    throw std::runtime_error("Cannot add the request to the engine since it is already assigned.");
  }
  engine_ = engine;
  status_ = RequestStatus::Assigned;

  auto device_tokens = AllocateOnDevice(*params_, prefill_input_ids_);
  processed_sequence_length_ = CurrentSequenceLength();
  search_->AppendTokens(device_tokens);
  seen_sequence_length_ = CurrentSequenceLength();
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
    auto device_tokens = AllocateOnDevice(*params_, tokens);
    search_->AppendTokens(device_tokens);
  }
}

int64_t Request::CurrentSequenceLength() const {
  return search_->GetSequenceLength();
}

int32_t Request::UnseenToken() {
  auto sequence = search_->GetSequence(0).CopyDeviceToCpu();
  if (static_cast<size_t>(seen_sequence_length_) >= sequence.size())
    throw std::runtime_error("All tokens have been seen.");

  return sequence[seen_sequence_length_++];
}

bool Request::HasUnseenTokens() const {
  return seen_sequence_length_ < CurrentSequenceLength();
}

DeviceSpan<int32_t> Request::UnprocessedTokens() {
  auto sequence = search_->GetSequence(0);
  auto unprocessed_tokens = sequence.subspan(processed_sequence_length_, CurrentSequenceLength() - processed_sequence_length_);
  return unprocessed_tokens;
}

bool Request::IsDone() const {
  return status_ == RequestStatus::Completed;
}

bool Request::IsPrefill() const {
  return is_prefill_;
}

void Request::GenerateNextTokens(DeviceSpan<float> logits) {
  processed_sequence_length_ = search_->GetSequence(0).size();
  is_prefill_ = false;

  search_->SetLogits(logits);
  auto& search_params = search_->params_->search;
  search_->ApplyMinLength(search_params.min_length);
  search_->ApplyRepetitionPenalty(search_params.repetition_penalty);

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

  if (search_->IsDone()) {
    status_ = RequestStatus::Completed;
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
