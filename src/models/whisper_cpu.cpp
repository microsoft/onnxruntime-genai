#include "../generators.h"
#include "../search.h"
#include "whisper_cpu.h"
#include "debugging.h"

namespace Generators {

template <typename T>
void ExpandInputs(const OrtValue& input, int num_beams, OrtAllocator& allocator, std::unique_ptr<OrtValue>& expanded);

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
  assert(model.score_type_ == Ort::TypeToTensorType<float>::type);
  int64_t input_ids_shape[] = {search_params_.batch_size, search_params_.sequence_length};
  cpu_span<int32_t> sequence_lengths = sequence_lengths_unk;

  // Allocate position_ids and attention_mask based on shape of input_ids
  auto element_type = Ort::TypeToTensorType<int32_t>::type;

  std::unique_ptr<OrtValue> encoder_input_ids, expanded_encoder_input_ids;

  // Use original input_ids. This requires the input_ids for subgraph is also int32.
  // Current shape is (batch_size, sequence_length)
  // Note that we will expand it to (batch_size * num_beams, sequence_length) later.
  decoder_input_ids_ = OrtValue::CreateTensor<int32_t>(allocator_cpu_, input_ids_shape);
  auto* p_data = decoder_input_ids_->GetTensorMutableData<int32_t>();
  for (auto v : search_params_.input_ids)
    *p_data++ = v;
  //  for (size_t i=search_params_.input_ids.size();i<1500;i++)
  //    *p_data++ = model.config_.pad_token_id;

  for (int i = 0; i < search_params_.num_beams * search_params_.batch_size; i++) {
    sequence_lengths[i] = static_cast<int32_t>(search_params_.sequence_length);
  }

  auto& inputs = const_cast<SearchParams::Whisper&>(std::get<SearchParams::Whisper>(search_params.inputs));

  if (search_params_.num_beams == 1) {
    expanded_encoder_input_ids = std::move(inputs.input_features);
    expanded_decoder_input_ids_ = std::move(decoder_input_ids_);
  } else {
    ExpandInputs<float>(*inputs.input_features, search_params_.num_beams, allocator_cpu_, expanded_encoder_input_ids);
    ExpandInputs<int32_t>(*decoder_input_ids_, search_params_.num_beams, allocator_cpu_, expanded_decoder_input_ids_);
  }

  input_names_.push_back("encoder_input_ids");
  inputs_.push_back(expanded_encoder_input_ids.get());
  input_names_.push_back("decoder_input_ids");
  inputs_.push_back(expanded_decoder_input_ids_.get());

  // Allocate space for logits
  {
    int64_t logits_shape[] = {search_params_.batch_size * search_params_.num_beams, model_->logits_uses_seq_len_ ? input_ids_shape[1] : 1, model_->vocab_size_};
    logits_ = OrtValue::CreateTensor(allocator_cpu_, logits_shape, model_->score_type_);
    output_names_.push_back("logits");
    outputs_.push_back(logits_.get());
  }

  encoder_hidden_states_ = OrtValue::CreateTensor<float>(allocator_cpu_, std::array<int64_t, 3>{search_params_.batch_size * search_params_.num_beams, 1500, 384});

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
  // The following updates inputs for subgraph

  // Update input_ids with next tokens.
  int batch_beam_size = static_cast<int>(next_tokens.size());
  std::unique_ptr<OrtValue> input_ids = OrtValue::CreateTensor<int32_t>(allocator_cpu_, std::array<int64_t, 2>{batch_beam_size, 1});
  int32_t* input_ids_data = input_ids->GetTensorMutableData<int32_t>();
  for (int i = 0; i < batch_beam_size; i++) {
    input_ids_data[i] = next_tokens[i];
  }
  expanded_decoder_input_ids_ = std::move(input_ids);
  inputs_[0] = expanded_decoder_input_ids_.get();

  // Update logits
  if (model_->logits_uses_seq_len_) {
    int64_t logits_shape[] = {search_params_.batch_size * search_params_.num_beams, 1, model_->vocab_size_};
    logits_ = OrtValue::CreateTensor(allocator_cpu_, logits_shape, Ort::TypeToTensorType<ScoreType>::type);
    outputs_[0] = logits_.get();
  }

  kv_cache_.Update(beam_indices, current_length);
  for (size_t i = 0; i < model_->layer_count_ * 2; i++) {
    inputs_[i + 1] = kv_cache_.pasts_[i].get();
    outputs_[i + 1] = kv_cache_.presents_[i].get();
    inputs_[i + model_->layer_count_ * 2] = kv_cache_.crosses_[i].get();
  }
}

}  // namespace Generators
