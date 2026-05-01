// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "cohere_model.h"

#include <algorithm>
#include <cctype>
#include <map>

namespace Generators {

// --- CohereEncoderState ---

CohereEncoderState::CohereEncoderState(const WhisperModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {}

void CohereEncoderState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  // Add audio features
  audio_features_ = std::make_unique<AudioFeatures>(*this, model_.config_->model.encoder.inputs.audio_features, extra_inputs);
  audio_features_->Add();

  // Compute num_frames from audio input shape and audio_stride
  auto shape = audio_features_->GetShape();
  int audio_stride = model_.config_->model.encoder.audio_stride;

  if (audio_stride > 0 && shape.size() == 2) {
    // Raw audio input [batch, samples]
    int T_enc = static_cast<int>(shape[1]) / audio_stride + 1;
    num_frames_ = T_enc * 2;  // *2 so GetNumFrames()/2 == T_enc for CrossCache
  } else if (audio_stride > 0 && shape.size() == 3) {
    // Mel input [batch, n_mels, T_mel]
    int T_mel = static_cast<int>(shape[2]);
    int T_enc = (T_mel - 1) / 8 + 1;
    num_frames_ = T_enc * 2;
  } else {
    throw std::runtime_error("Cohere Transcribe requires audio_stride > 0 in config. "
                             "Got audio_stride=" + std::to_string(audio_stride) +
                             " with input rank=" + std::to_string(shape.size()));
  }

  // Add mel_length input if provided
  for (const auto& [name, value] : extra_inputs) {
    if (name == "mel_length") {
      mel_length_ = std::move(reinterpret_cast<Tensor*>(value.get())->ort_tensor_);
      input_names_.push_back("mel_length");
      inputs_.push_back(mel_length_.get());
      break;
    }
  }

  // Add encoder hidden states output if the model has it
  if (model_.session_info_.HasOutput(model_.config_->model.encoder.outputs.hidden_states)) {
    auto hidden_states_shape = std::array<int64_t, 3>{params_->BatchBeamSize(), GetNumFrames() / 2, model_.config_->model.encoder.hidden_size};
    hidden_states_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), hidden_states_shape, audio_features_->GetType());
    outputs_.push_back(hidden_states_.get());
    output_names_.push_back(model_.config_->model.encoder.outputs.hidden_states.c_str());
  }
}

void CohereEncoderState::SetChunkAudioFeatures(std::shared_ptr<Tensor> audio_features_tensor, std::shared_ptr<Tensor> mel_length_tensor) {
  // Replace audio features input
  auto* ort_tensor = audio_features_tensor->ort_tensor_.get();
  auto shape_info = ort_tensor->GetTensorTypeAndShapeInfo();
  auto shape = shape_info->GetShape();
  int audio_stride = model_.config_->model.encoder.audio_stride;

  if (audio_stride > 0 && shape.size() == 3) {
    int T_mel = static_cast<int>(shape[2]);
    int T_enc = (T_mel - 1) / 8 + 1;
    num_frames_ = T_enc * 2;
  }

  // Find and replace the audio features in inputs_
  std::string audio_name = model_.config_->model.encoder.inputs.audio_features;
  for (size_t i = 0; i < input_names_.size(); ++i) {
    if (input_names_[i] == audio_name.c_str() || std::string(input_names_[i]) == audio_name) {
      // We need to keep the AudioFeatures object alive - create a new one
      // But since AudioFeatures takes extra_inputs, we directly replace the pointer
      inputs_[i] = ort_tensor;
      break;
    }
  }

  // Replace mel_length
  if (mel_length_tensor) {
    mel_length_ = std::move(mel_length_tensor->ort_tensor_);
    for (size_t i = 0; i < input_names_.size(); ++i) {
      if (std::string(input_names_[i]) == "mel_length") {
        inputs_[i] = mel_length_.get();
        break;
      }
    }
  }

  // Update hidden states output shape if applicable
  if (model_.session_info_.HasOutput(model_.config_->model.encoder.outputs.hidden_states)) {
    auto hidden_states_shape = std::array<int64_t, 3>{params_->BatchBeamSize(), GetNumFrames() / 2, model_.config_->model.encoder.hidden_size};
    hidden_states_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), hidden_states_shape,
                                            shape_info->GetElementType());
    // Update the output pointer
    for (size_t i = 0; i < output_names_.size(); ++i) {
      if (std::string(output_names_[i]) == model_.config_->model.encoder.outputs.hidden_states) {
        outputs_[i] = hidden_states_.get();
        break;
      }
    }
  }

  // Mark encoder as needing to run again
  first_run_ = true;
}

DeviceSpan<float> CohereEncoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.encoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.encoder.run_options.value());
  }
  State::Run(*model_.session_encoder_);
  return {};
}

// --- CohereState ---

CohereState::CohereState(const WhisperModel& model, const GeneratorParams& params, DeviceSpan<int32_t> sequence_lengths)
    : State{params, model},
      model_{model} {
  encoder_state_ = std::make_unique<CohereEncoderState>(model, params);
  // decoder_state_ and cross_cache_ created in SetExtraInputs after num_frames is known
}

void CohereState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  // Extract and store chunk data before passing to encoder
  std::vector<ExtraInput> encoder_inputs;
  std::map<int, std::shared_ptr<Tensor>> indexed_mels;
  std::map<int, std::shared_ptr<Tensor>> indexed_mel_lengths;

  for (const auto& [name, tensor] : extra_inputs) {
    if (name == "cohere_chunk_count") {
      total_chunks_ = static_cast<int>(tensor->ort_tensor_->GetTensorData<int64_t>()[0]);
      current_chunk_ = 0;
    } else if (name.substr(0, 13) == "cohere_chunk_" && name.find("mel_length") == std::string::npos) {
      // Parse index from cohere_chunk_N
      int idx = std::stoi(name.substr(13));
      indexed_mels[idx] = tensor;
    } else if (name.substr(0, 24) == "cohere_chunk_mel_length_") {
      int idx = std::stoi(name.substr(24));
      indexed_mel_lengths[idx] = tensor;
    } else {
      encoder_inputs.push_back({name, tensor});
    }
  }

  // Store chunks in sorted order
  for (auto& [idx, tensor] : indexed_mels)
    chunk_mels_.push_back(tensor);
  for (auto& [idx, tensor] : indexed_mel_lengths)
    chunk_mel_lengths_.push_back(tensor);

  encoder_state_->SetExtraInputs(encoder_inputs);

  // Now that num_frames is known, create the decoder
  decoder_state_ = std::make_unique<WhisperDecoderState>(model_, *params_, encoder_state_->GetNumFrames());

  // Create cross cache
  if (encoder_state_->HasCrossKVCacheOutputs()) {
    cross_cache_ = std::make_unique<CrossCache>(*this, encoder_state_->GetNumFrames() / 2);
    encoder_state_->AddCrossCache(cross_cache_);
    decoder_state_->AddCrossCache(cross_cache_);
    transpose_k_cache_buffer_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), cross_cache_->GetShape(), cross_cache_->GetType());
  }

  if (!encoder_state_->HasCrossKVCacheOutputs()) {
    decoder_state_->inputs_.push_back(encoder_state_->GetHiddenStates());
    decoder_state_->input_names_.push_back(model_.config_->model.decoder.inputs.encoder_hidden_states.c_str());
  }
}

bool CohereState::AdvanceToNextChunk() {
  if (!HasMoreChunks()) return false;

  size_t chunk_idx = current_chunk_;
  auto& next_mel = chunk_mels_[chunk_idx];
  auto& next_mel_length = chunk_mel_lengths_[chunk_idx];

  encoder_state_->SetChunkAudioFeatures(next_mel, next_mel_length);

  int new_num_frames = encoder_state_->GetNumFrames();

  encoder_state_->outputs_.clear();
  encoder_state_->output_names_.clear();

  decoder_state_ = std::make_unique<WhisperDecoderState>(model_, *params_, new_num_frames);

  if (encoder_state_->HasCrossKVCacheOutputs()) {
    cross_cache_ = std::make_unique<CrossCache>(*this, new_num_frames / 2);
    encoder_state_->AddCrossCache(cross_cache_);
    decoder_state_->AddCrossCache(cross_cache_);
    transpose_k_cache_buffer_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), cross_cache_->GetShape(), cross_cache_->GetType());
  }

  if (!encoder_state_->HasCrossKVCacheOutputs()) {
    decoder_state_->inputs_.push_back(encoder_state_->GetHiddenStates());
    decoder_state_->input_names_.push_back(model_.config_->model.decoder.inputs.encoder_hidden_states.c_str());
  }

  // Reset state for new chunk
  first_run_ = true;
  current_chunk_++;
  return true;
}

void CohereState::SaveChunkTokens(const int32_t* tokens, size_t count) {
  completed_chunk_tokens_.emplace_back(tokens, tokens + count);
}

static std::string StripAsciiWhitespace(const std::string& s) {
  // Mirrors Python str.strip(): strips ASCII whitespace (space, \t, \n, \r, \v, \f).
  static const char* ws = " \t\n\r\v\f";
  const size_t start = s.find_first_not_of(ws);
  if (start == std::string::npos) return {};
  const size_t end = s.find_last_not_of(ws);
  return s.substr(start, end - start + 1);
}

// --- Helpers for overlap-aware chunk text merging --------------------------
//
// The processor emits OVERLAPPING audio chunks; adjacent transcripts share
// the same words at their seam. We dedup by finding the longest run of
// matching normalized words between the tail of `prev` and the head of `next`
// and dropping it from `next`. We also strip stray sentence-end punctuation
// from the prev tail when the continuation starts with a lowercase word, so
// "...UK." + "come from..." becomes "...UK come from..." not "...UK. come...".

static std::string ToLowerAscii(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) out.push_back(static_cast<char>(std::tolower(c)));
  return out;
}

static std::string NormalizeWord(const std::string& w) {
  // Lowercase and strip leading/trailing ASCII punctuation for matching.
  static const char* punct = ".,!?;:\"')(";
  size_t a = w.find_first_not_of(punct);
  if (a == std::string::npos) return {};
  size_t b = w.find_last_not_of(punct);
  return ToLowerAscii(w.substr(a, b - a + 1));
}

static std::vector<std::string> SplitOnWhitespace(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  const size_t n = s.size();
  while (i < n) {
    while (i < n && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    size_t j = i;
    while (j < n && !std::isspace(static_cast<unsigned char>(s[j]))) ++j;
    if (j > i) out.emplace_back(s.substr(i, j - i));
    i = j;
  }
  return out;
}

static bool StartsWithLowerAscii(const std::string& w) {
  // First non-punct alpha char check.
  for (unsigned char c : w) {
    if (std::isalpha(c)) return std::islower(c) != 0;
    if (std::isalnum(c)) return false;
  }
  return false;
}

static std::string RStripSentencePunct(std::string w) {
  while (!w.empty()) {
    char c = w.back();
    if (c == '.' || c == '!' || c == '?') {
      w.pop_back();
    } else {
      break;
    }
  }
  return w;
}

// Append `next` onto `prev` with overlap dedup. Returns merged text.
static std::string MergeWithOverlapDedup(const std::string& prev, const std::string& next,
                                         size_t min_match = 3, size_t max_lookback = 25) {
  if (prev.empty()) return next;
  if (next.empty()) return prev;
  auto pw = SplitOnWhitespace(prev);
  auto nw = SplitOnWhitespace(next);
  if (pw.empty()) return next;
  if (nw.empty()) return prev;

  std::vector<std::string> pn(pw.size()), nn(nw.size());
  for (size_t i = 0; i < pw.size(); ++i) pn[i] = NormalizeWord(pw[i]);
  for (size_t i = 0; i < nw.size(); ++i) nn[i] = NormalizeWord(nw[i]);

  size_t best_k = 0;
  size_t upper = std::min({max_lookback, pw.size(), nw.size()});
  for (size_t k = upper; k >= min_match; --k) {
    bool ok = true;
    for (size_t i = 0; i < k; ++i) {
      if (pn[pn.size() - k + i] != nn[i]) { ok = false; break; }
    }
    if (ok) { best_k = k; break; }
    if (k == 0) break;
  }
  if (best_k == 0 && !pn.back().empty() && pn.back() == nn.front()) {
    best_k = 1;
  }

  // Strip terminal sentence punctuation from prev's last word if the
  // continuation begins mid-sentence (lowercase).
  if (best_k < nw.size() && StartsWithLowerAscii(nw[best_k])) {
    pw.back() = RStripSentencePunct(pw.back());
  }

  std::string out;
  for (size_t i = 0; i < pw.size(); ++i) {
    if (i) out += ' ';
    out += pw[i];
  }
  for (size_t i = best_k; i < nw.size(); ++i) {
    out += ' ';
    out += nw[i];
  }
  return out;
}

std::string CohereState::GetJoinedChunkText(const Tokenizer& tokenizer, const std::string& separator) const {
  // Decode each chunk, strip whitespace, then merge with overlap-aware word
  // dedup at the seams (chunks are emitted with audio overlap by the processor).
  // `separator` is honored only as the inter-chunk fallback when no overlap is
  // detected (e.g. zh/ja/ko/vi want "" — but those are not handled here yet).
  (void)separator;
  std::string merged;
  for (const auto& chunk : completed_chunk_tokens_) {
    if (chunk.empty()) continue;
    auto raw = tokenizer.Decode(std::span<const int32_t>(chunk.data(), chunk.size()));
    auto stripped = StripAsciiWhitespace(raw);
    if (stripped.empty()) continue;
    if (merged.empty()) {
      merged = std::move(stripped);
    } else {
      merged = MergeWithOverlapDedup(merged, stripped);
    }
  }
  return merged;
}

DeviceSpan<float> CohereState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (encoder_state_->IsFirstRun()) {
    encoder_state_->Run(current_length, next_tokens, next_indices);
    decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length, first_run_);
    return decoder_state_->Run(current_length, next_tokens, next_indices);
  }

  decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length, first_run_);
  auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);
  first_run_ = false;
  return logits;
}

OrtValue* CohereState::GetInput(const char* name) {
  for (size_t i = 0; i < encoder_state_->input_names_.size(); i++) {
    if (std::strcmp(encoder_state_->input_names_[i], name) == 0)
      return encoder_state_->inputs_[i];
  }
  for (size_t i = 0; i < decoder_state_->input_names_.size(); i++) {
    if (std::strcmp(decoder_state_->input_names_[i], name) == 0)
      return decoder_state_->inputs_[i];
  }
  return State::GetInput(name);
}

OrtValue* CohereState::GetOutput(const char* name) {
  for (size_t i = 0; i < encoder_state_->output_names_.size(); i++) {
    if (std::strcmp(encoder_state_->output_names_[i], name) == 0)
      return encoder_state_->outputs_[i];
  }
  for (size_t i = 0; i < decoder_state_->output_names_.size(); i++) {
    if (std::strcmp(decoder_state_->output_names_[i], name) == 0)
      return decoder_state_->outputs_[i];
  }
  return State::GetOutput(name);
}

// --- CohereModel ---

std::unique_ptr<State> CohereModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<CohereState>(*this, params, sequence_lengths);
}

}  // namespace Generators
