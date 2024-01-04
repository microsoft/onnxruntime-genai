#include "../generators.h"
#include "../search.h"
#include "llama_cpu.h"
#include "debugging.h"

namespace Generators {

Llama_State::Llama_State(Llama_Model& model, RoamingArray<int32_t> sequence_lengths_unk, const SearchParams& search_params)
    : model_{&model}, 
      search_params_{search_params},
      kv_cache_{search_params, model.config_, model.model_.allocator_cpu_, model.cuda_stream_, model.score_type_, model.past_names_, model.present_names_},
      position_ids_{model.model_, search_params, model.model_.allocator_cpu_, sequence_lengths_unk} {

  assert(model.score_type_ == Ort::TypeToTensorType<float>::type);
  int64_t input_ids_shape[] = {search_params_.batch_size, search_params_.sequence_length};
  input_ids_ = OrtValue::CreateTensor<int64_t>(model.model_.allocator_cpu_, input_ids_shape);
  auto* p_data = input_ids_->GetTensorMutableData<int64_t>();
  for (auto v : search_params_.input_ids)
    *p_data++ = v;

  input_ids_ = ExpandInputs(input_ids_, search_params_.num_beams, model.model_.allocator_cpu_, model.model_.device_type_, model.model_.cuda_stream_);

  for (auto* input : {input_ids_.get(), position_ids_.position_ids_.get(), position_ids_.attention_mask_.get()})
    inputs_.push_back(input);
  for (auto* name : {"input_ids", "position_ids", "attention_mask"})
    input_names_.push_back(name);

  output_names_.push_back("logits");

  // Allocate space for logits (only works if we know the shape)
  {
    int64_t logits_shape[] = {search_params_.batch_size * search_params_.num_beams, model_->logits_uses_seq_len_ ? input_ids_shape[1] : 1, model_->vocab_size_};
    logits_ = OrtValue::CreateTensor(model.model_.allocator_cpu_, logits_shape, model_->score_type_);
    outputs_.push_back(logits_.get());
  }

  for (int i = 0; i < model_->layer_count_ * 2; ++i) {
    inputs_.push_back(kv_cache_.empty_past_.get());
    input_names_.push_back(kv_cache_.input_name_strings_[i].c_str());
    outputs_.push_back(kv_cache_.presents_[i].get());
    output_names_.push_back(kv_cache_.output_name_strings_[i].c_str());
  }
}

RoamingArray<float> Llama_State::Run(int current_length, RoamingArray<int32_t> next_tokens_unk, RoamingArray<int32_t> next_indices_unk) {
  cpu_span<int32_t> next_tokens = next_tokens_unk;
  cpu_span<int32_t> next_indices = next_indices_unk;

  if (first_run_)
    first_run_ = false;
  else
    UpdateInputs(next_tokens, next_indices, current_length);

#if 0
    printf("**Inputs:\r\n");
    DumpTensors(inputs_.data(), input_names_.data(), input_names_.size(), true);
    printf("**Outputs:\r\n");
    DumpTensors(outputs_.data(), output_names_.data(), output_names_.size(), false);
#endif

  model_->session_decoder_->Run(nullptr, input_names_.data(), inputs_.data(), input_names_.size(), output_names_.data(), outputs_.data(), output_names_.size());

  auto type_shape = logits_->GetTensorTypeAndShapeInfo();
  auto shape = type_shape->GetShape();
  assert(type_shape->GetShape().size() == 3);

  return cpu_span<float>{logits_->GetTensorMutableData<float>(), type_shape->GetElementCount()};
}

void Llama_State::UpdateInputs(std::span<const int32_t> next_tokens, std::span<const int32_t> beam_indices, int current_length) {

  // The following updates inputs for subgraph

  // Update input_ids with next tokens.
  int batch_beam_size = static_cast<int>(next_tokens.size());
  int64_t dims[] = {batch_beam_size, 1};
  std::unique_ptr<OrtValue> input_ids = OrtValue::CreateTensor<int64_t>(model_->model_.allocator_cpu_, dims);
  int64_t* input_ids_data = input_ids->GetTensorMutableData<int64_t>();
  for (int i = 0; i < batch_beam_size; i++) {
    input_ids_data[i] = next_tokens[i];
  }
  input_ids_ = std::move(input_ids);
  inputs_[0] = input_ids_.get();

  position_ids_.Update(current_length);
  inputs_[1] = position_ids_.position_ids_.get();
  inputs_[2] = position_ids_.attention_mask_.get();

  // Update logits
  if (model_->logits_uses_seq_len_) {
    int64_t logits_shape[] = {search_params_.batch_size * search_params_.num_beams, 1, model_->vocab_size_};
    logits_ = OrtValue::CreateTensor(model_->model_.allocator_cpu_, logits_shape, model_->score_type_);
    outputs_[0] = logits_.get();
  }

  kv_cache_.Update(beam_indices, current_length);
  for (size_t i = 0; i < model_->layer_count_ * 2; i++) {
    inputs_[i + 3] = kv_cache_.pasts_[i].get();
    outputs_[i + 1] = kv_cache_.presents_[i].get();
  }
}

}  // namespace Generators