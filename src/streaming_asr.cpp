// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// StreamingASR implementation — high-level streaming speech recognition.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>

#include "generators.h"
#include "streaming_asr.h"

namespace Generators {

// ─── Vocabulary loading ─────────────────────────────────────────────────────

void StreamingASR::LoadVocab() {
  if (vocab_loaded_) return;

  auto config_path = model_.config_->config_path;

  // Try tokens.txt
  auto tokens_path = config_path / "tokens.txt";
  std::ifstream tokens_file(tokens_path.string());
  if (tokens_file.is_open()) {
    std::string line;
    while (std::getline(tokens_file, line)) {
      // Format: "token_text index" (space-separated) or "token_text\tindex" (tab-separated)
      auto sep_pos = line.rfind(' ');
      if (sep_pos == std::string::npos) sep_pos = line.rfind('\t');
      if (sep_pos != std::string::npos) {
        vocab_.push_back(line.substr(0, sep_pos));
      } else {
        vocab_.push_back(line);
      }
    }
    vocab_loaded_ = true;
    return;
  }

  // Fallback: use tokenizer
  try {
    auto tokenizer = model_.CreateTokenizer();
    vocab_.resize(cache_config_.vocab_size);
    for (int i = 0; i < cache_config_.vocab_size; ++i) {
      try {
        std::vector<int32_t> ids = {static_cast<int32_t>(i)};
        vocab_[i] = tokenizer->Decode(ids);
      } catch (...) {
        vocab_[i] = "";
      }
    }
    vocab_loaded_ = true;
    return;
  } catch (...) {
  }

  // Last resort
  vocab_.resize(cache_config_.vocab_size);
  for (int i = 0; i < cache_config_.vocab_size; ++i) {
    vocab_[i] = "<" + std::to_string(i) + ">";
  }
  vocab_loaded_ = true;
}

// ─── StreamingASR ───────────────────────────────────────────────────────────

StreamingASR::StreamingASR(Model& model)
    : model_{model} {
  // Get the NemotronSpeechModel to access its sessions
  auto* nemotron_model = dynamic_cast<NemotronSpeechModel*>(&model);
  if (!nemotron_model) {
    throw std::runtime_error("StreamingASR requires a nemotron_speech model type. Got: " + model.config_->model.type);
  }

  encoder_session_ = nemotron_model->session_encoder_.get();
  decoder_session_ = nemotron_model->session_decoder_.get();
  joiner_session_ = nemotron_model->session_joiner_.get();
  cache_config_ = nemotron_model->cache_config_;

  // Initialize mel extractor (standalone, no ORT dependencies)
  // mel_extractor_ is already constructed with default MelConfig via member init

  // Initialize mel pre-encode cache (zeros for first chunk)
  mel_pre_encode_cache_.assign(
      static_cast<size_t>(kNumMels) * cache_config_.pre_encode_cache_size, 0.0f);
  is_first_chunk_ = true;

  // Initialize streaming state
  auto& allocator = model_.allocator_cpu_;
  encoder_cache_.Initialize(cache_config_, allocator);
  decoder_state_.Initialize(cache_config_, allocator);
}

StreamingASR::~StreamingASR() = default;

void StreamingASR::Reset() {
  auto& allocator = model_.allocator_cpu_;
  encoder_cache_.Reset(cache_config_, allocator);
  decoder_state_.Reset(cache_config_, allocator);
  full_transcript_.clear();
  mel_extractor_.Reset();
  mel_pre_encode_cache_.assign(
      static_cast<size_t>(kNumMels) * cache_config_.pre_encode_cache_size, 0.0f);
  is_first_chunk_ = true;
  audio_buffer_.clear();
  chunk_index_ = 0;
}

std::string StreamingASR::TranscribeChunk(const float* audio_data, size_t num_samples) {
  LoadVocab();

  // Append incoming audio to accumulation buffer
  audio_buffer_.insert(audio_buffer_.end(), audio_data, audio_data + num_samples);

  std::string result;
  const size_t chunk_sz = static_cast<size_t>(cache_config_.chunk_samples);

  // Process complete chunks of audio (each = chunk_samples = 8960 samples = 56 mel frames)
  while (audio_buffer_.size() >= chunk_sz) {
    // Compute mel for this chunk
    auto [mel_data, num_frames] = mel_extractor_.Process(audio_buffer_.data(), chunk_sz);

    // Prepend pre-encode cache → feed [cache | new_mel] to encoder
    result += ProcessMelChunk(mel_data, num_frames);

    // Advance by full chunk (no overlap — NeMo native: shift == chunk)
    audio_buffer_.erase(audio_buffer_.begin(),
                        audio_buffer_.begin() + static_cast<ptrdiff_t>(chunk_sz));
  }

  return result;
}

std::string StreamingASR::Flush() {
  LoadVocab();

  std::string result;
  const size_t chunk_sz = static_cast<size_t>(cache_config_.chunk_samples);

  // Process any remaining audio (pad to full chunk with silence)
  if (!audio_buffer_.empty()) {
    audio_buffer_.resize(chunk_sz, 0.0f);

    auto [mel_data, num_frames] = mel_extractor_.Process(audio_buffer_.data(), chunk_sz);
    result += ProcessMelChunk(mel_data, num_frames);

    audio_buffer_.clear();
  }

  return result;
}

std::string StreamingASR::ProcessMelChunk(const std::vector<float>& mel_data, int num_frames) {
  auto& allocator = model_.allocator_cpu_;
  int cache_size = cache_config_.pre_encode_cache_size;  // 9

  // Build encoder input: [pre_encode_cache (9 mel) | new_mel (num_frames mel)]
  int total_mel_frames = cache_size + num_frames;  // e.g. 9 + 56 = 65

  // Create processed_signal: [1, num_mels, total_mel_frames]
  auto signal_shape = std::array<int64_t, 3>{1, kNumMels, total_mel_frames};
  auto processed_signal = OrtValue::CreateTensor(allocator, signal_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  float* signal_data = processed_signal->GetTensorMutableData<float>();

  // Fill row by row: mel is [kNumMels, time] in row-major layout
  for (int m = 0; m < kNumMels; ++m) {
    // Pre-encode cache columns for this mel bin
    std::memcpy(signal_data + m * total_mel_frames,
                mel_pre_encode_cache_.data() + m * cache_size,
                cache_size * sizeof(float));
    // New mel columns for this mel bin
    std::memcpy(signal_data + m * total_mel_frames + cache_size,
                mel_data.data() + m * num_frames,
                num_frames * sizeof(float));
  }

  // Update pre-encode cache: save last cache_size mel frames from current chunk
  // (these will be prepended to the next chunk)
  if (num_frames >= cache_size) {
    for (int m = 0; m < kNumMels; ++m) {
      std::memcpy(mel_pre_encode_cache_.data() + m * cache_size,
                  mel_data.data() + m * num_frames + (num_frames - cache_size),
                  cache_size * sizeof(float));
    }
  } else {
    // Short chunk: shift existing cache left, append new frames
    int keep = cache_size - num_frames;
    for (int m = 0; m < kNumMels; ++m) {
      std::memmove(mel_pre_encode_cache_.data() + m * cache_size,
                   mel_pre_encode_cache_.data() + m * cache_size + num_frames,
                   keep * sizeof(float));
      std::memcpy(mel_pre_encode_cache_.data() + m * cache_size + keep,
                  mel_data.data() + m * num_frames,
                  num_frames * sizeof(float));
    }
  }
  is_first_chunk_ = false;

  // Create processed_signal_length: [1]
  auto len_shape = std::array<int64_t, 1>{1};
  auto signal_length = OrtValue::CreateTensor(allocator, len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *signal_length->GetTensorMutableData<int64_t>() = static_cast<int64_t>(total_mel_frames);

  // Encoder inputs
  const char* enc_input_names[] = {
      "audio_signal", "length",
      "cache_last_channel", "cache_last_time", "cache_last_channel_len"};
  OrtValue* enc_inputs[] = {
      processed_signal.get(), signal_length.get(),
      encoder_cache_.cache_last_channel.get(),
      encoder_cache_.cache_last_time.get(),
      encoder_cache_.cache_last_channel_len.get()};

  const char* enc_output_names[] = {
      "outputs", "encoded_lengths",
      "cache_last_channel_next", "cache_last_time_next", "cache_last_channel_next_len"};

  // Run encoder
  auto run_options = OrtRunOptions::Create();
  auto enc_outputs = encoder_session_->Run(
      run_options.get(),
      enc_input_names, enc_inputs, 5,
      enc_output_names, 5);

  // Parse encoder outputs
  auto* encoded = enc_outputs[0].get();
  int64_t encoded_len = *enc_outputs[1]->GetTensorData<int64_t>();

  // Update cache
  encoder_cache_.cache_last_channel = std::move(enc_outputs[2]);
  encoder_cache_.cache_last_time = std::move(enc_outputs[3]);
  encoder_cache_.cache_last_channel_len = std::move(enc_outputs[4]);

  // Run RNNT decoder on ALL encoder output frames (no drop_last needed)
  std::string chunk_text = RunRNNTDecoder(encoded, encoded_len);
  full_transcript_ += chunk_text;
  chunk_index_++;

  return chunk_text;
}

std::string StreamingASR::RunRNNTDecoder(OrtValue* encoder_output, int64_t encoded_len) {
  auto& allocator = model_.allocator_cpu_;
  std::string result;

  auto enc_info = encoder_output->GetTensorTypeAndShapeInfo();
  auto enc_shape = enc_info->GetShape();
  int64_t hidden_dim = enc_shape[1];
  // Decode ALL encoder output frames — pre-encode cache artifacts are already
  // removed by the ONNX graph's baked-in Slice (drop_extra_pre_encoded=2).
  int64_t time_steps = std::min(enc_shape[2], encoded_len);
  const float* enc_data = encoder_output->GetTensorData<float>();

  auto run_options = OrtRunOptions::Create();

  for (int64_t t = 0; t < time_steps; ++t) {
    // Extract single encoder frame: [1, hidden_dim, 1]
    auto frame_shape = std::array<int64_t, 3>{1, hidden_dim, 1};
    auto encoder_frame = OrtValue::CreateTensor(allocator, frame_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    float* frame_data = encoder_frame->GetTensorMutableData<float>();
    for (int64_t d = 0; d < hidden_dim; ++d) {
      frame_data[d] = enc_data[d * enc_shape[2] + t];
    }

    constexpr int kMaxSymbolsPerStep = 10;
    for (int sym = 0; sym < kMaxSymbolsPerStep; ++sym) {
      // ── Step 1: Run decoder (prediction network) ──
      auto targets_shape = std::array<int64_t, 2>{1, 1};
      auto targets = OrtValue::CreateTensor(allocator, targets_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
      *targets->GetTensorMutableData<int32_t>() = decoder_state_.last_token;

      auto tgt_len_shape = std::array<int64_t, 1>{1};
      auto target_length = OrtValue::CreateTensor(allocator, tgt_len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
      *target_length->GetTensorMutableData<int32_t>() = 1;

      const char* dec_input_names[] = {
          "targets", "target_length",
          "states.1", "onnx::Slice_3"};
      OrtValue* dec_inputs[] = {
          targets.get(), target_length.get(),
          decoder_state_.state_1.get(), decoder_state_.state_2.get()};

      const char* dec_output_names[] = {
          "outputs", "prednet_lengths", "states", "162"};

      auto dec_outputs = decoder_session_->Run(
          run_options.get(),
          dec_input_names, dec_inputs, 4,
          dec_output_names, 4);

      // dec_outputs[0] = decoder hidden [1, 640, 1]
      // dec_outputs[1] = prednet_lengths [1]
      // dec_outputs[2] = new states h [2, ?, 640]
      // dec_outputs[3] = new states c [2, ?, 640]

      // ── Step 2: Run joiner (joint network) ──
      const char* join_input_names[] = {
          "encoder_outputs", "decoder_outputs"};
      OrtValue* join_inputs[] = {
          encoder_frame.get(), dec_outputs[0].get()};

      const char* join_output_names[] = {"outputs"};

      auto join_outputs = joiner_session_->Run(
          run_options.get(),
          join_input_names, join_inputs, 2,
          join_output_names, 1);

      // Find argmax
      const float* logits_data = join_outputs[0]->GetTensorData<float>();
      auto logits_shape = join_outputs[0]->GetTensorTypeAndShapeInfo()->GetShape();
      int total_logits = 1;
      for (auto d : logits_shape) total_logits *= static_cast<int>(d);

      int best_token = 0;
      float best_score = logits_data[0];
      for (int i = 1; i < total_logits; ++i) {
        if (logits_data[i] > best_score) {
          best_score = logits_data[i];
          best_token = i;
        }
      }

      // Blank => next time step
      if (best_token == cache_config_.blank_id || best_token >= cache_config_.vocab_size) {
        break;
      }

      // Emit token & update state
      decoder_state_.last_token = best_token;
      decoder_state_.state_1 = std::move(dec_outputs[2]);
      decoder_state_.state_2 = std::move(dec_outputs[3]);

      if (best_token < static_cast<int>(vocab_.size())) {
        std::string token_str = vocab_[best_token];
        // Replace sentencepiece space marker "▁" with space
        size_t pos = 0;
        while ((pos = token_str.find("\xe2\x96\x81", pos)) != std::string::npos) {
          token_str.replace(pos, 3, " ");
          pos += 1;
        }
        result += token_str;
      }
    }
  }

  return result;
}

void StreamingASR::SaveNpy(const std::string& path, const float* data,
                               const std::vector<int64_t>& shape) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return;

    // Build header string: "{'descr': '<f4', 'fortran_order': False, 'shape': (D0, D1, ...), }\n"
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
      shape_str += std::to_string(shape[i]);
      if (i + 1 < shape.size()) shape_str += ", ";
      else if (shape.size() == 1) shape_str += ",";
    }
    shape_str += ")";
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";

    // Pad header to 64-byte alignment (magic=6 + version=2 + header_len=2 + header + '\n')
    size_t prefix_len = 6 + 2 + 2;  // magic + version + header_len
    size_t total = prefix_len + header.size() + 1;  // +1 for trailing '\n'
    size_t pad = (64 - (total % 64)) % 64;
    header.append(pad, ' ');
    header += '\n';

    uint16_t header_len = static_cast<uint16_t>(header.size());

    // Write magic
    f.write("\x93NUMPY", 6);
    // Write version 1.0
    char ver[] = {1, 0};
    f.write(ver, 2);
    // Write header length (little-endian)
    f.write(reinterpret_cast<const char*>(&header_len), 2);
    // Write header
    f.write(header.data(), header.size());
    // Write data
    size_t num_elements = 1;
    for (auto d : shape) num_elements *= static_cast<size_t>(d);
    f.write(reinterpret_cast<const char*>(data), num_elements * sizeof(float));
  }

  std::unique_ptr<StreamingASR> CreateStreamingASR(Model& model) {
  return std::make_unique<StreamingASR>(model);
}

}  // namespace Generators
