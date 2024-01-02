#include "../generators.h"
#include "../search.h"
#include "whisper_cpu.h"
#include "debugging.h"

namespace Generators {

Whisper_Model::Whisper_Model(OrtEnv& ort_env, Config& config, OrtSessionOptions& session_options)
    : config_{config} {
  session_decoder_ = OrtSession::Create(ort_env, (config.config_path / config.model_decoder).c_str(), &session_options);
  session_encoder_ = OrtSession::Create(ort_env, (config.config_path / config.model_encoder_decoder_init).c_str(), &session_options);

  InitModelParams();
}

void Whisper_Model::InitModelParams() {
  // We could use this to determine the vocabulary size and if the logits has a width of 1
  auto logits_type_info = session_decoder_->GetOutputTypeInfo(0);
  auto& logits_tensor_info = logits_type_info->GetTensorTypeAndShapeInfo();
  auto logits_shape = logits_tensor_info.GetShape();
  assert(logits_shape.size() == 3);
  logits_uses_seq_len_ = logits_shape[1] == -1;
  vocab_size_ = static_cast<int>(logits_shape[2]);
  layer_count_ = (static_cast<int>(session_decoder_->GetOutputCount()) - 1) / 2;
  score_type_ = logits_tensor_info.GetElementType();

  auto past_shape = session_decoder_->GetInputTypeInfo(3)->GetTensorTypeAndShapeInfo().GetShape();
  head_count_ = static_cast<int>(past_shape[1]);
  hidden_size_ = static_cast<int>(past_shape[3]);

  assert(config_.vocab_size == vocab_size_);
  assert(config_.num_hidden_layers == layer_count_);
  assert(config_.num_attention_heads == head_count_);
  assert(config_.hidden_size == hidden_size_);
}

Whisper_State::Whisper_State(Whisper_Model& model, RoamingArray<int32_t> sequence_lengths_unk, const SearchParams& search_params)
    : model_{&model},
      search_params_{search_params},
      kv_cache_{search_params, model.config_, allocator_cpu_, model.cuda_stream_, model.score_type_, model.past_names_, model.present_names_, model.past_cross_names_, model.present_cross_names_} {
  // CPU only supports float
  assert(model.score_type_ == Ort::TypeToTensorType<float>::type);

  std::unique_ptr<OrtValue> encoder_input_ids, expanded_encoder_input_ids;

  // Use original input_ids. This requires the input_ids for subgraph is also int32.
  // Current shape is (batch_size, sequence_length)
  // Note that we will expand it to (batch_size * num_beams, sequence_length) later.
  decoder_input_ids_shape_ = {search_params_.batch_size, search_params_.sequence_length};
  decoder_input_ids_ = OrtValue::CreateTensor<int32_t>(allocator_cpu_, decoder_input_ids_shape_);
  auto* p_data = decoder_input_ids_->GetTensorMutableData<int32_t>();
  for (auto v : search_params_.input_ids)
    *p_data++ = v;

  auto& inputs = const_cast<SearchParams::Whisper&>(std::get<SearchParams::Whisper>(search_params.inputs));

  expanded_encoder_input_ids = ExpandInputs(inputs.input_features, search_params_.num_beams, allocator_cpu_, DeviceType::CPU, nullptr);
  expanded_decoder_input_ids_ = ExpandInputs(decoder_input_ids_, search_params_.num_beams, allocator_cpu_, DeviceType::CPU, nullptr);
  decoder_input_ids_shape_[0] *= search_params_.num_beams;

  cpu_span<int32_t> sequence_lengths = sequence_lengths_unk;
  for (int i = 0; i < decoder_input_ids_shape_[0]; i++) {
    sequence_lengths[i] = static_cast<int32_t>(search_params_.sequence_length);
  }

  input_names_.push_back("encoder_input_ids");
  inputs_.push_back(expanded_encoder_input_ids.get());
  input_names_.push_back("decoder_input_ids");
  inputs_.push_back(expanded_decoder_input_ids_.get());

  // Allocate space for logits
  logits_shape_ = {decoder_input_ids_shape_[0], decoder_input_ids_shape_[1], model_->vocab_size_};
  logits_ = OrtValue::CreateTensor(allocator_cpu_, logits_shape_, model_->score_type_);
  output_names_.push_back("logits");
  outputs_.push_back(logits_.get());

  encoder_hidden_states_ = OrtValue::CreateTensor<float>(allocator_cpu_, std::array<int64_t, 3>{decoder_input_ids_shape_[0], 1500, 384});

  output_names_.push_back("encoder_hidden_states");
  outputs_.push_back(encoder_hidden_states_.get());

  for (int i = 0; i < model_->layer_count_ * 2; ++i) {
    outputs_.push_back(kv_cache_.presents_[i].get());
    output_names_.push_back(kv_cache_.output_name_strings_[i].c_str());
  }

  for (int i = 0; i < model_->layer_count_ * 2; ++i) {
    outputs_.push_back(kv_cache_.crosses_[i].get());
    output_names_.push_back(kv_cache_.output_cross_name_strings_[i].c_str());
  }

  model_->session_encoder_->Run(nullptr, input_names_.data(), inputs_.data(), input_names_.size(), output_names_.data(), outputs_.data(), output_names_.size());

  input_names_.clear();
  output_names_.clear();
  inputs_.clear();
  outputs_.clear();

  input_names_.push_back("input_ids");
  inputs_.push_back(nullptr);  // Placeholder, will be filled in by UpdateInputs

  output_names_.push_back("logits");
  outputs_.push_back(nullptr);  // Placeholder, will be filled in by UpdateInputs

  for (int i = 0; i < model_->layer_count_ * 2; ++i) {
    inputs_.push_back(nullptr);  // Placeholder, will be filled in by UpdateInputs
    input_names_.push_back(kv_cache_.input_name_strings_[i].c_str());
    outputs_.push_back(kv_cache_.presents_[i].get());
    output_names_.push_back(kv_cache_.output_name_strings_[i].c_str());
  }

  for (int i = 0; i < model_->layer_count_ * 2; ++i) {
    inputs_.push_back(kv_cache_.crosses_[i].get());
    input_names_.push_back(kv_cache_.input_cross_name_strings_[i].c_str());
  }
}

RoamingArray<float> Whisper_State::Run(int current_length, RoamingArray<int32_t> next_tokens_unk, RoamingArray<int32_t> next_indices_unk) {
  cpu_span<int32_t> next_tokens = next_tokens_unk;
  cpu_span<int32_t> next_indices = next_indices_unk;

  if (first_run_)
    first_run_ = false;
  else {
    UpdateInputs(next_tokens, next_indices, current_length);

#if 0
    printf("**Inputs:\r\n");
    DumpTensors(inputs_.data(), input_names_.data(), input_names_.size(), true);
    printf("**Outputs:\r\n");
    DumpTensors(outputs_.data(), output_names_.data(), output_names_.size(), false);
#endif

    model_->session_decoder_->Run(nullptr, input_names_.data(), inputs_.data(), input_names_.size(), output_names_.data(), outputs_.data(), output_names_.size());
  }
  auto type_shape = logits_->GetTensorTypeAndShapeInfo();
  auto shape = type_shape->GetShape();
  assert(type_shape->GetShape().size() == 3);

  return cpu_span<float>{logits_->GetTensorMutableData<ScoreType>(), type_shape->GetElementCount()};
}

void Whisper_State::UpdateInputs(std::span<const int32_t> next_tokens, std::span<const int32_t> beam_indices, int current_length) {
  // Resize input_ids shape once if it doesn't match the decoder shape
  if (decoder_input_ids_shape_[1] != 1) {
    decoder_input_ids_shape_[1] = 1;
    expanded_decoder_input_ids_ = OrtValue::CreateTensor<int32_t>(allocator_cpu_, decoder_input_ids_shape_);
    inputs_[0] = expanded_decoder_input_ids_.get();
  }

  // Update input_ids with next tokens
  int32_t* input_ids_data = expanded_decoder_input_ids_->GetTensorMutableData<int32_t>();
  for (int i = 0; i < decoder_input_ids_shape_[0]; i++)
    input_ids_data[i] = next_tokens[i];

  // Resize the logits shape once if it doesn't match the decoder shape
  if (logits_shape_[1] != 1) {
    logits_shape_[1] = 1;
    logits_ = OrtValue::CreateTensor(allocator_cpu_, logits_shape_, model_->score_type_);
    outputs_[0] = logits_.get();
  }

  kv_cache_.Update(beam_indices, current_length);
  for (size_t i = 0; i < model_->layer_count_ * 2; i++) {
    inputs_[i + 1] = kv_cache_.pasts_[i].get();
    outputs_[i + 1] = kv_cache_.presents_[i].get();
  }
}

}  // namespace Generators
