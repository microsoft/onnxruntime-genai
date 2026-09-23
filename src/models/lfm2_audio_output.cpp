// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/lfm2_audio_output.h"
#include "models/multi_modal.h"
#include "models/utils.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>

namespace Generators {

namespace {

// Names of LiquidAI's depthformer and audio embedding exports (github.com/Liquid4All/onnx-export).
constexpr const char* kDepthformerInputs[] = {"hidden_states", "depth_slices_in", "step_idx", "prev_token",
                                              "past_keys", "past_values", "seqlens_k", "total_seq_len"};
constexpr const char* kDepthformerOutputs[] = {"logits", "depth_slices", "new_keys", "new_values"};
constexpr const char* kEmbeddingInputs[] = {"audio_codes"};
constexpr const char* kEmbeddingOutputs[] = {"audio_embeds"};

std::vector<int64_t> InputShape(OrtSession& session, const char* name, const char* graph) {
  const auto names = session.GetInputNames();
  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] == name) {
      return session.GetInputTypeInfo(i)->GetTensorTypeAndShapeInfo().GetShape();
    }
  }
  throw std::runtime_error(std::string("Lfm2AudioOutput: the ") + graph + " model has no input named \"" + name + "\".");
}

template <typename T>
std::unique_ptr<OrtValue> MakeTensor(std::span<const T> values, std::vector<int64_t> shape) {
  auto tensor = OrtValue::CreateTensor<T>(Ort::Allocator::GetWithDefaultOptions(), shape);
  std::copy(values.begin(), values.end(), tensor->template GetTensorMutableData<T>());
  return tensor;
}

template <typename T>
std::unique_ptr<OrtValue> MakeScalar(T value) {
  return MakeTensor<T>(std::span<const T>{&value, 1}, {});
}

}  // namespace

Lfm2AudioOutput::Lfm2AudioOutput(const MultiModalLanguageModel& model, const GeneratorParams& params)
    : model_{model},
      config_{model.config_->model.audio_output},
      interleaved_{params.search.audio_interleaved},
      temperature_{params.search.audio_temperature},
      top_k_{params.search.audio_top_k},
      placeholder_token_id_{model.config_->model.audio_token_id},
      rng_{CreateRandomGenerator(params.search.random_seed)},
      modality_left_{config_.interleaved_n_text} {
  if (params.BatchBeamSize() != 1) {
    throw std::runtime_error("Lfm2AudioOutput: speech output needs a batch size of 1 and no beam search.");
  }

  const auto hidden_shape = InputShape(*model_.depthformer_session_, "hidden_states", "depthformer");
  const auto slices_shape = InputShape(*model_.depthformer_session_, "depth_slices_in", "depthformer");
  const auto keys_shape = InputShape(*model_.depthformer_session_, "past_keys", "depthformer");
  if (hidden_shape.size() != 2 || slices_shape.size() != 3 || keys_shape.size() != 5 ||
      config_.num_codebooks <= 0 || slices_shape[1] != config_.num_codebooks) {
    throw std::runtime_error("Lfm2AudioOutput: the depthformer model does not have the expected inputs for " +
                             std::to_string(config_.num_codebooks) + " codebooks.");
  }
  hidden_size_ = hidden_shape[1];
  depth_size_ = slices_shape[2];
  num_layers_ = keys_shape[0];
  num_kv_heads_ = keys_shape[2];
  head_size_ = keys_shape[4];

  // RunDepthformer checks it against the logits the depthformer produces.
  if (config_.codebook_size <= 0) {
    throw std::runtime_error("Lfm2AudioOutput: model.audio_output.codebook_size " +
                             std::to_string(config_.codebook_size) + " must be positive.");
  }

  // The search has to pick the placeholder whatever its sampling settings are.
  const int vocab_size = model_.config_->model.vocab_size;
  if (placeholder_token_id_ <= 0 || placeholder_token_id_ >= vocab_size) {
    throw std::runtime_error("Lfm2AudioOutput: model.audio_token_id " + std::to_string(placeholder_token_id_) +
                             " must be set to a token inside the vocabulary of " + std::to_string(vocab_size) +
                             " tokens.");
  }
  placeholder_logits_ = model_.p_device_->Allocate<float>(static_cast<size_t>(vocab_size));
  auto logits = placeholder_logits_.CpuSpan();
  std::fill(logits.begin(), logits.end(), std::numeric_limits<float>::lowest());
  logits[static_cast<size_t>(placeholder_token_id_)] = 0.0f;
  placeholder_logits_.CopyCpuToDevice();
}

void Lfm2AudioOutput::BeginStep(DeviceSpan<int32_t>& next_tokens, bool is_prompt) {
  if (is_prompt || next_tokens.size() != 1) {
    // A new turn starts in text, whatever the last one ended in.
    modality_ = Modality::Text;
    previous_was_text_ = false;
    text_done_ = false;
    modality_left_ = config_.interleaved_n_text;
    pending_embedding_.clear();
  } else if (previous_was_text_) {
    const int32_t token = next_tokens.CopyDeviceToCpu()[0];
    if (interleaved_) {
      text_done_ = text_done_ || token == config_.text_end_token_id;
      if (modality_left_ == 0 || text_done_) {
        modality_ = Modality::Audio;
        modality_left_ = config_.interleaved_n_audio;
      }
    } else if (token == config_.audio_start_token_id) {
      modality_ = Modality::Audio;
    }
  }

  --modality_left_;
  previous_was_text_ = modality_ == Modality::Text;
}

void Lfm2AudioOutput::WritePendingFrame(OrtValue& inputs_embeds) {
  const auto info = inputs_embeds.GetTensorTypeAndShapeInfo();
  if (info->GetElementCount() != pending_embedding_.size()) {
    throw std::runtime_error("Lfm2AudioOutput: expected the decoder to take one embedding of " +
                             std::to_string(pending_embedding_.size()) + " values after an audio frame, got " +
                             std::to_string(info->GetElementCount()) + ".");
  }

  auto bytes = ByteWrapTensor(DeviceForTensor(inputs_embeds, *model_.p_device_inputs_), inputs_embeds);
  auto cpu = bytes.CpuSpan();
  if (info->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    std::memcpy(cpu.data(), pending_embedding_.data(), pending_embedding_.size() * sizeof(float));
  } else if (info->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    auto* half = reinterpret_cast<uint16_t*>(cpu.data());
    std::transform(pending_embedding_.begin(), pending_embedding_.end(), half, FastFloat32ToFloat16);
  } else {
    throw std::runtime_error("Lfm2AudioOutput: the decoder's inputs_embeds must be float or float16.");
  }
  bytes.CopyCpuToDevice();
  pending_embedding_.clear();
}

DeviceSpan<float> Lfm2AudioOutput::SampleFrame(OrtValue& hidden_states) {
  const auto info = hidden_states.GetTensorTypeAndShapeInfo();
  const auto shape = info->GetShape();
  if (shape.empty() || shape.back() != hidden_size_) {
    throw std::runtime_error("Lfm2AudioOutput: the decoder's hidden states do not match the depthformer's input of " +
                             std::to_string(hidden_size_) + " values.");
  }

  // The last position of [batch, sequence, hidden_size].
  const size_t count = static_cast<size_t>(hidden_size_);
  const size_t first = info->GetElementCount() - count;
  auto bytes = ByteWrapTensor(DeviceForTensor(hidden_states, *model_.p_device_inputs_), hidden_states).CopyDeviceToCpu();
  std::vector<float> hidden(count);
  if (info->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    std::memcpy(hidden.data(), bytes.data() + first * sizeof(float), count * sizeof(float));
  } else if (info->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    const auto* half = reinterpret_cast<const uint16_t*>(bytes.data()) + first;
    std::transform(half, half + count, hidden.begin(), FastFloat16ToFloat32);
  } else {
    throw std::runtime_error("Lfm2AudioOutput: the decoder's hidden states must be float or float16.");
  }

  std::vector<int64_t> frame = RunDepthformer(hidden);

  // As the reference orders it: the count first, then the end-of-audio code, which wins.
  if (interleaved_ && modality_left_ == 0 && !text_done_) {
    modality_ = Modality::Text;
    modality_left_ = config_.interleaved_n_text;
  }
  const int64_t end_of_audio = config_.codebook_size - 1;
  if (frame[0] == end_of_audio) {
    std::fill(frame.begin(), frame.end(), end_of_audio);
    modality_ = Modality::Text;
  } else {
    audio_codes_.insert(audio_codes_.end(), frame.begin(), frame.end());
  }

  EmbedFrame(frame);
  return placeholder_logits_;
}

std::vector<int64_t> Lfm2AudioOutput::RunDepthformer(std::span<const float> hidden) {
  const auto hidden_tensor = MakeTensor<float>(hidden, {1, hidden_size_});
  const std::vector<int64_t> cache_shape{num_layers_, 1, num_kv_heads_, 0, head_size_};
  auto keys = OrtValue::CreateTensor<float>(Ort::Allocator::GetWithDefaultOptions(), cache_shape);
  auto values = OrtValue::CreateTensor<float>(Ort::Allocator::GetWithDefaultOptions(), cache_shape);
  const std::vector<float> no_slices(static_cast<size_t>(config_.num_codebooks * depth_size_), 0.0f);
  auto slices = MakeTensor<float>(no_slices, {1, config_.num_codebooks, depth_size_});

  // One run per codebook: each code is sampled given the ones before it in the frame. The projection
  // of the hidden state is computed on the first run and handed back in on the rest.
  std::vector<int64_t> frame;
  int64_t previous_code = 0;
  for (int32_t step = 0; step < config_.num_codebooks; ++step) {
    const auto step_index = MakeScalar<int64_t>(step);
    const auto previous = MakeTensor<int64_t>(std::span<const int64_t>{&previous_code, 1}, {1});
    const auto past_length = MakeTensor<int32_t>(std::span<const int32_t>{&step, 1}, {1});
    const auto total_length = MakeScalar<int32_t>(step + 1);
    const OrtValue* inputs[] = {hidden_tensor.get(), slices.get(), step_index.get(), previous.get(),
                                keys.get(), values.get(), past_length.get(), total_length.get()};
    auto outputs = model_.depthformer_session_->Run(nullptr, kDepthformerInputs, inputs, std::size(inputs),
                                                    kDepthformerOutputs, std::size(kDepthformerOutputs));

    // The code is sampled from the first codebook_size logits, so there have to be at least that many.
    const auto logits_info = outputs[0]->GetTensorTypeAndShapeInfo();
    const auto logits_shape = logits_info->GetShape();
    if (logits_info->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || logits_shape.empty() ||
        logits_shape.back() < config_.codebook_size) {
      throw std::runtime_error("Lfm2AudioOutput: model.audio_output.codebook_size is " +
                               std::to_string(config_.codebook_size) + ", but the depthformer produces " +
                               std::to_string(logits_shape.empty() ? 0 : logits_shape.back()) + " float logits.");
    }
    const float* logits = outputs[0]->GetTensorData<float>();
    previous_code = SampleCode({logits, static_cast<size_t>(config_.codebook_size)});
    frame.push_back(previous_code);
    if (step == 0) {
      slices = std::move(outputs[1]);
    }
    keys = std::move(outputs[2]);
    values = std::move(outputs[3]);
  }
  return frame;
}

int64_t Lfm2AudioOutput::SampleCode(std::span<const float> logits) {
  if (temperature_ <= 0.0f || top_k_ == 1) {
    return std::max_element(logits.begin(), logits.end()) - logits.begin();
  }

  std::vector<int64_t> order(logits.size());
  std::iota(order.begin(), order.end(), int64_t{0});
  const size_t kept = top_k_ > 0 ? std::min(static_cast<size_t>(top_k_), order.size()) : order.size();
  std::partial_sort(order.begin(), order.begin() + kept, order.end(),
                    [&](int64_t a, int64_t b) { return logits[a] > logits[b]; });

  std::vector<double> weights(kept);
  const float highest = logits[order[0]];
  for (size_t i = 0; i < kept; ++i) {
    weights[i] = std::exp(static_cast<double>(logits[order[i]] - highest) / temperature_);
  }
  std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
  return order[distribution(rng_)];
}

void Lfm2AudioOutput::EmbedFrame(std::span<const int64_t> frame) {
  // Every codebook has its own rows of the one embedding table.
  std::vector<int64_t> rows(frame.size());
  for (size_t codebook = 0; codebook < frame.size(); ++codebook) {
    rows[codebook] = frame[codebook] + static_cast<int64_t>(codebook) * config_.codebook_size;
  }
  const auto codes = MakeTensor<int64_t>(rows, {1, static_cast<int64_t>(rows.size())});
  const OrtValue* inputs[] = {codes.get()};
  auto outputs = model_.audio_embedding_session_->Run(nullptr, kEmbeddingInputs, inputs, std::size(inputs),
                                                      kEmbeddingOutputs, std::size(kEmbeddingOutputs));

  const auto info = outputs[0]->GetTensorTypeAndShapeInfo();
  const auto shape = info->GetShape();  // [1, num_codebooks, hidden_size]
  if (info->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || shape.empty() ||
      info->GetElementCount() != frame.size() * static_cast<size_t>(shape.back())) {
    throw std::runtime_error("Lfm2AudioOutput: the audio embedding model must produce one float embedding per codebook.");
  }
  const size_t width = static_cast<size_t>(shape.back());
  const float* embeddings = outputs[0]->GetTensorData<float>();
  pending_embedding_.assign(width, 0.0f);
  for (size_t codebook = 0; codebook < frame.size(); ++codebook) {
    for (size_t i = 0; i < width; ++i) {
      pending_embedding_[i] += embeddings[codebook * width + i];
    }
  }
}

OrtValue* Lfm2AudioOutput::GetAudioCodes() {
  const int64_t num_frames = static_cast<int64_t>(audio_codes_.size()) / config_.num_codebooks;
  audio_codes_tensor_ = MakeTensor<int64_t>(audio_codes_, {num_frames, config_.num_codebooks});
  return audio_codes_tensor_.get();
}

}  // namespace Generators
