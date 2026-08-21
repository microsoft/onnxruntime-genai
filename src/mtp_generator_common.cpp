// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mtp_generator_common.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "gemma4_assistant_generator.h"
#include "generators.h"
#include "models/model.h"
#include "models/model_type.h"
#include "models/utils.h"
#include "mtp_generator.h"

namespace Generators {

std::unique_ptr<MtpGeneratorInterface> CreateMtpGenerator(
    const Model& target_model, const Model& draft_model, const GeneratorParams& params) {
  if (ModelType::IsGemma4Assistant(draft_model.config_->model.type))
    return std::make_unique<Gemma4AssistantGenerator>(target_model, draft_model, params);
  return std::make_unique<MtpGenerator>(target_model, draft_model, params);
}

void MtpGeneratorBase::GenerateNextToken() {
  if (!primed_)
    throw std::runtime_error("MTP generation requires AppendTokens before GenerateNextToken");

  if (pending_tokens_.empty() && !done_) {
    const size_t committed = sequence_.size();
    RunRound();
    QueueCommittedTokens(committed);
  }

  if (pending_tokens_.empty()) return;
  emitted_sequence_.push_back(pending_tokens_.front());
  pending_tokens_.pop_front();
  stats_.tokens_emitted++;
}

void MtpGeneratorBase::QueueCommittedTokens(size_t first_index) {
  for (size_t index = first_index; index < sequence_.size(); ++index) {
    pending_tokens_.push_back(sequence_[index]);
    stats_.tokens_queued++;
    if (IsEos(sequence_[index]) || index + 1 >= static_cast<size_t>(max_length_)) {
      done_ = true;
      return;
    }
  }
}

bool MtpGeneratorBase::IsEos(int32_t token) const {
  return contains(eos_token_ids_, token);
}

int CountAcceptedDrafts(const int32_t* drafts, const int32_t* verify_argmax, int count,
                        int32_t target_prediction) {
  if (count <= 0 || drafts[0] != target_prediction) return 0;
  int accepted = 1;
  while (accepted < count && drafts[accepted] == verify_argmax[accepted - 1]) ++accepted;
  return accepted;
}

void ValidateMtpPair(const Model& target_model, const Model& draft_model,
                     const GeneratorParams& params) {
  if (params.search.batch_size != 1 || params.search.num_beams != 1 ||
      params.search.num_return_sequences != 1)
    throw std::runtime_error(
        "MTP generation supports only batch_size=1, num_beams=1, and num_return_sequences=1");
  if (!params.guidance_type.empty())
    throw std::runtime_error("MTP generation does not support guided generation");
  if (target_model.p_device_->GetType() != draft_model.p_device_->GetType())
    throw std::runtime_error("MTP generation requires target and draft models on the same device type");
  if (target_model.config_->model.vocab_size != draft_model.config_->model.vocab_size)
    throw std::runtime_error("MTP generation requires matching target and draft vocabularies");
}

int32_t ArgmaxRow(const float* values, int vocab_size) {
  return static_cast<int32_t>(std::max_element(values, values + vocab_size) - values);
}

template <typename T>
int32_t ArgmaxHalfRow(const uint8_t* values, int vocab_size) {
  const auto* typed_values = reinterpret_cast<const T*>(values);
  int32_t best = 0;
  float best_value = ToFloat32(typed_values[0]);
  for (int index = 1; index < vocab_size; ++index) {
    const float value = ToFloat32(typed_values[index]);
    if (value > best_value) {
      best = index;
      best_value = value;
    }
  }
  return best;
}

int32_t ArgmaxRow(const uint8_t* values, ONNXTensorElementDataType type, int vocab_size) {
  if (type == Ort::TypeToTensorType<float>)
    return ArgmaxRow(reinterpret_cast<const float*>(values), vocab_size);
  if (type == Ort::TypeToTensorType<Ort::Float16_t>)
    return ArgmaxHalfRow<Ort::Float16_t>(values, vocab_size);
  if (type == Ort::TypeToTensorType<Ort::BFloat16_t>)
    return ArgmaxHalfRow<Ort::BFloat16_t>(values, vocab_size);
  throw std::runtime_error("MTP generation encountered an unsupported logits type");
}

void CopyTensorRow(OrtValue& source, int row, Tensor& destination, DeviceInterface& device) {
  if (source.GetTensorTypeAndShapeInfo()->GetElementType() != destination.GetType())
    throw std::runtime_error("MTP generation encountered a tensor type mismatch");
  const size_t row_bytes = destination.GetByteSpan().size();
  const size_t offset = static_cast<size_t>(row) * row_bytes;
  if (source.GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_CPU) {
    auto host = destination.GetByteSpan().CpuSpan();
    std::memcpy(host.data(), static_cast<const uint8_t*>(source.GetTensorRawData()) + offset, row_bytes);
    destination.GetByteSpan().CopyCpuToDevice();
    return;
  }
  auto source_bytes = ByteWrapTensor(device, source);
  destination.GetByteSpan().CopyFrom(source_bytes.subspan(offset, row_bytes));
}

SpeculativeStats FinalizeSpeculativeStats(SpeculativeStats stats, size_t buffered_tokens) {
  stats.tokens_buffered = buffered_tokens;
  if (stats.draft_tokens_evaluated > 0)
    stats.acceptance_rate = static_cast<float>(stats.draft_tokens_accepted) /
                            static_cast<float>(stats.draft_tokens_evaluated);
  if (stats.rounds > 0) {
    stats.avg_draft_tokens_per_round = static_cast<float>(stats.draft_tokens_proposed) /
                                       static_cast<float>(stats.rounds);
    stats.mean_emitted_tokens_per_round = static_cast<float>(stats.tokens_emitted) /
                                          static_cast<float>(stats.rounds);
  }
  return stats;
}

}  // namespace Generators
