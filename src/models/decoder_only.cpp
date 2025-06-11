#include "../generators.h"
#include "decoder_only.h"

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
      position_inputs_{model, *this, sequence_lengths_unk, model_.config_->model.decoder.inputs.attention_mask} {
  input_ids_.Add();
  position_inputs_.Add();
  logits_.Add();
  kv_cache_->Add();
}

void DecoderOnly_State::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  extra_inputs_.Add(extra_inputs, model_.session_decoder_->GetInputNames());
}

DeviceSpan<float> DecoderOnly_State::Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  size_t num_tokens = next_tokens.size();
  const size_t chunk_size = 1024; // Experimental value
  
  if (num_tokens > chunk_size) {
    // Chunking logic for context phase - process in chunks of 512 tokens
    size_t processed_tokens = 0;
    int length = total_length - static_cast<int>(num_tokens);
    while (processed_tokens < num_tokens) {
      size_t current_chunk_size = std::min(chunk_size, num_tokens - processed_tokens);
      
      // Create subspans for current chunk
      auto chunk_tokens = next_tokens.subspan(processed_tokens, current_chunk_size);
      //auto chunk_indices = next_indices.subspan(processed_tokens, current_chunk_size);
      length = length + static_cast<int>(current_chunk_size);
      // Process this chunk - fills KV cache progressively
      UpdateInputsOutputs(chunk_tokens, next_indices, length);
      
      // Graph capture is typically disabled during context phase chunking
      bool graph_capture_this_run = false; // Disable graph capture during chunking
      State::Run(*model_.session_decoder_, graph_capture_this_run);
      
      processed_tokens += current_chunk_size;
    }
    
    // Return logits from the last chunk for potential sampling
    return logits_.Get();
  } else {
    // Original logic for tokens <= 512 (generation phase or small context)
    UpdateInputsOutputs(next_tokens, next_indices, total_length);

    // Graph capture enabled for token generation case, allowing it to repeat the same graph for each token.
    bool graph_capture_this_run = params_->use_graph_capture && input_ids_.GetShape()[1] == 1;
    State::Run(*model_.session_decoder_, graph_capture_this_run);

    return logits_.Get();
  }
}

void DecoderOnly_State::RewindTo(size_t index) {
  position_inputs_.RewindTo(index);
  kv_cache_->RewindTo(index);
}

void DecoderOnly_State::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> beam_indices, int total_length) {
  input_ids_.Update(next_tokens);
  size_t new_length = static_cast<size_t>(input_ids_.GetShape()[1]);
  position_inputs_.Update(next_tokens, total_length, static_cast<int>(new_length));
  kv_cache_->Update(beam_indices, total_length);
  logits_.Update(next_tokens, new_length);
}

}  // namespace Generators
