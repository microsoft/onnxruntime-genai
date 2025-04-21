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
  assigned_time_ = std::chrono::system_clock::now();
  status_ = RequestStatus::Assigned;

  auto device_tokens = AllocateOnDevice(*params_, unprocessed_input_ids_);
  search_->AppendTokens(device_tokens);
  unprocessed_input_ids_.clear();
}

void Request::Schedule() {
  if (status_ != RequestStatus::Assigned) {
    throw std::runtime_error("Request cannot be scheduled unless it has been previously added to the engine.");
  }

  if (!search_ || search_->GetSequenceLength() == 0) {
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
  if (status_ == RequestStatus::Unassigned) {
    unprocessed_input_ids_ = std::vector<int32_t>(tokens.begin(), tokens.end());
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

DeviceSpan<int32_t> Request::UnprocessedTokens() {
  auto sequence = search_->GetSequence(0);
  return sequence.subspan(processed_tokens_, sequence.size() - processed_tokens_);
}

void Request::GenerateNextTokens(DeviceSpan<float> logits) {
  search_->SetLogits(logits);
  auto& search_params = search_->params_->search;
  search_->ApplyMinLength(search_params.min_length);
  search_->ApplyRepetitionPenalty(search_params.repetition_penalty);

  if (!search_params.do_sample || search_params.top_k == 1 || search_params.temperature == 0) {
    search_->SelectTop();
    return;
  }

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

}  // namespace Generators
