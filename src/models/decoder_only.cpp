#include "../generators.h"
#include "decoder_only.h"
#include "utils.h"

#include <cstring>
#include <limits>

namespace Generators {
DecoderOnly_Model::DecoderOnly_Model(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_decoder_ = CreateSession(ort_env, config_->model.decoder.filename, session_options_.get());
  session_info_.Add(*session_decoder_);
}

std::unique_ptr<State> DecoderOnly_Model::CreateState(DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params) const {
  return std::make_unique<DecoderOnly_State>(*this, sequence_lengths_unk, params);
}

DecoderOnly_State::DecoderOnly_State(const DecoderOnly_Model& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      kv_cache_(CreateKeyValueCache(*this)),
      recurrent_state_(CreateRecurrentState(*this)),
      position_inputs_{CreatePositionInputs(*this, sequence_lengths_unk, model_.config_->model.decoder.inputs.attention_mask)} {
  input_ids_.Add();
  position_inputs_->Add();
  AddAttentionBias();
  logits_.Add();
  if (kv_cache_)
    kv_cache_->Add();
  if (recurrent_state_)
    recurrent_state_->Add();
}

void DecoderOnly_State::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  extra_inputs_.Add(extra_inputs, model_.session_decoder_->GetInputNames());
}

DeviceSpan<float> DecoderOnly_State::Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  size_t num_tokens = next_tokens.size();
  const auto& chunk_size_opt = model_.config_->search.chunk_size;

  if (chunk_size_opt.has_value() && chunk_size_opt.value() > 0 && num_tokens > chunk_size_opt.value()) {
    return RunWithChunking(total_length, next_tokens, next_indices, chunk_size_opt.value());
  } else {
    UpdateInputsOutputs(next_tokens, next_indices, total_length);
    if (model_.config_->model.decoder.run_options.has_value()) {
      State::SetRunOptions(model_.config_->model.decoder.run_options.value());
    }

    // Graph capture enabled for token generation case, allowing it to repeat the same graph for each token.
    bool graph_capture_this_run = params_->use_graph_capture && input_ids_.GetShape()[1] == 1;
    State::Run(*model_.session_decoder_, graph_capture_this_run);

    return logits_.Get();
  }
}

DeviceSpan<float> DecoderOnly_State::RunWithChunking(int total_length, DeviceSpan<int32_t>& next_tokens,
                                                     DeviceSpan<int32_t> next_indices, size_t chunk_size) {
  // Chunking logic for context phase - process in chunks based on configured chunk_size
  size_t num_tokens = next_tokens.size();
  size_t processed_tokens = 0;
  int length = total_length - static_cast<int>(num_tokens);

  if (model_.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.decoder.run_options.value());
  }
  while (processed_tokens < num_tokens) {
    size_t current_chunk_size = std::min(chunk_size, num_tokens - processed_tokens);

    // Create subspans for current chunk
    auto chunk_tokens = next_tokens.subspan(processed_tokens, current_chunk_size);
    length = length + static_cast<int>(current_chunk_size);

    // Process this chunk - fills KV cache progressively
    UpdateInputsOutputs(chunk_tokens, next_indices, length);

    // Graph capture is typically disabled during context phase chunking
    bool graph_capture_this_run = false;  // Disable graph capture during chunking
    State::Run(*model_.session_decoder_, graph_capture_this_run);

    processed_tokens += current_chunk_size;
  }

  // Return logits from the last chunk for potential sampling
  return logits_.Get();
}

DeviceSpan<float> DecoderOnly_State::RunTree(
    int stable_length, DeviceSpan<int32_t>& tree_tokens,
    std::span<const int64_t> position_ids,
    std::span<const uint8_t> tree_mask) {
  if (stable_length < 0)
    throw std::runtime_error("Tree decoding received a negative stable cache length.");
  const int tree_width = static_cast<int>(tree_tokens.size());
  if (tree_width == 0)
    throw std::runtime_error("Tree decoding requires at least one token.");
  if (position_ids.size() != tree_tokens.size() ||
      tree_mask.size() != tree_tokens.size() * tree_tokens.size())
    throw std::runtime_error("Tree decoding tensors have inconsistent dimensions.");

  UpdateInputsOutputs(tree_tokens, {}, stable_length + tree_width);
  position_inputs_->SetPositionIDs(position_ids);
  SetTreeAttentionBias(static_cast<size_t>(stable_length), tree_mask);
  if (model_.config_->model.decoder.run_options.has_value())
    State::SetRunOptions(model_.config_->model.decoder.run_options.value());
  State::Run(*model_.session_decoder_, false);
  return logits_.GetAll();
}

void DecoderOnly_State::RewindTo(size_t index) {
  position_inputs_->RewindTo(index);
  if (kv_cache_)
    kv_cache_->RewindTo(index);
  if (recurrent_state_)
    recurrent_state_->RewindTo(index);
}

void DecoderOnly_State::CompactTreeCache(
    size_t stable_length, std::span<const size_t> tree_indices) {
  if (kv_cache_)
    kv_cache_->CompactTree(stable_length, tree_indices);
  position_inputs_->RewindTo(stable_length + tree_indices.size());
}

void DecoderOnly_State::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> beam_indices, int total_length) {
  input_ids_.Update(next_tokens);
  size_t new_length = static_cast<size_t>(input_ids_.GetShape()[1]);

  // Determine effective lengths for position_ids and KV cache based on sliding window config
  int position_length = total_length;
  int kv_cache_length = total_length;

  if (model_.config_->model.decoder.sliding_window.has_value() &&
      model_.config_->model.decoder.sliding_window->window_size > 0) {
    const int window_size = model_.config_->model.decoder.sliding_window->window_size;

    // Position IDs are clamped when slide_inputs is true
    if (model_.config_->model.decoder.sliding_window->slide_inputs) {
      position_length = std::min(total_length, window_size);
    }

    // KV cache is clamped when slide_key_value_cache is true
    if (model_.config_->model.decoder.sliding_window->slide_key_value_cache) {
      kv_cache_length = std::min(total_length, window_size);
    }
  }

  position_inputs_->Update(next_tokens, position_length, static_cast<int>(new_length));
  UpdateAttentionBias(position_length, static_cast<int>(new_length));
  if (kv_cache_)
    kv_cache_->Update(beam_indices, kv_cache_length);
  if (recurrent_state_)
    recurrent_state_->Update();
  logits_.Update(next_tokens, new_length);
}

void DecoderOnly_State::AddAttentionBias() {
  const std::string& name = model_.config_->model.decoder.inputs.attention_bias;
  if (name.empty())
    return;
  if (!model_.session_info_.HasInput(name))
    throw std::runtime_error("Configured attention_bias input '" + name +
                             "' was not found in the decoder graph.");

  attention_bias_type_ = model_.session_info_.GetInputDataType(name);
  if (attention_bias_type_ != Ort::TypeToTensorType<float> &&
      attention_bias_type_ != Ort::TypeToTensorType<Ort::BFloat16_t>)
    throw std::runtime_error("attention_bias only supports float32 or bfloat16 tensors.");

  attention_bias_ =
      std::make_unique<Tensor>(model_.p_device_inputs_, attention_bias_type_);
  attention_bias_shape_ = {params_->BatchBeamSize(), 1, 1, 1};
  attention_bias_->CreateTensor(attention_bias_shape_);
  attention_bias_->GetByteSpan().Zero();
  attention_bias_input_index_ = inputs_.size();
  inputs_.push_back(attention_bias_->GetOrtTensor());
  input_names_.push_back(name.c_str());
}

void DecoderOnly_State::UpdateAttentionBias(int total_length, int new_length) {
  if (!attention_bias_)
    return;
  attention_bias_shape_ = {
      params_->BatchBeamSize(), 1, static_cast<int64_t>(new_length),
      static_cast<int64_t>(total_length)};
  attention_bias_->CreateTensor(attention_bias_shape_);
  attention_bias_->GetByteSpan().Zero();
  inputs_[attention_bias_input_index_] = attention_bias_->GetOrtTensor();
}

void DecoderOnly_State::SetTreeAttentionBias(
    size_t stable_length, std::span<const uint8_t> tree_mask) {
  if (!attention_bias_)
    throw std::runtime_error("Tree decoding requires a configured attention_bias input.");
  const size_t query_length = static_cast<size_t>(attention_bias_shape_[2]);
  const size_t total_length = static_cast<size_t>(attention_bias_shape_[3]);
  if (attention_bias_shape_[0] != 1 ||
      total_length != stable_length + query_length ||
      tree_mask.size() != query_length * query_length)
    throw std::runtime_error("Tree attention bias has an invalid shape.");

  auto bytes = attention_bias_->GetByteSpan();
  auto cpu_bytes = bytes.CpuSpan();
  if (attention_bias_type_ == Ort::TypeToTensorType<float>) {
    auto* values = reinterpret_cast<float*>(cpu_bytes.data());
    std::fill_n(values, total_length * query_length, 0.0f);
    for (size_t query = 0; query < query_length; ++query) {
      for (size_t key = 0; key < query_length; ++key) {
        if (!tree_mask[query * query_length + key])
          values[query * total_length + stable_length + key] =
              std::numeric_limits<float>::lowest();
      }
    }
  } else {
    auto* values = reinterpret_cast<uint16_t*>(cpu_bytes.data());
    std::fill_n(values, total_length * query_length, uint16_t{});
    const uint16_t blocked =
        Float32ToBFloat16(std::numeric_limits<float>::lowest());
    for (size_t query = 0; query < query_length; ++query) {
      for (size_t key = 0; key < query_length; ++key) {
        if (!tree_mask[query * query_length + key])
          values[query * total_length + stable_length + key] = blocked;
      }
    }
  }
  bytes.CopyCpuToDevice();
}

}  // namespace Generators
