#include "../generators.h"
#include "../search.h"
#include "gpt_cpu.h"
#include "debugging.h"

namespace Generators {

Gpt_State::Gpt_State(Gpt_Model& model, RoamingArray<int32_t> sequence_lengths_unk, const SearchParams& search_params)
    : search_params_{search_params},
      model_{&model},
      kv_cache_{search_params, model.model_.config_, model.model_.allocator_cpu_, model.model_.cuda_stream_, model.score_type_},
      position_ids_{model.model_, search_params, model.model_.allocator_cpu_, sequence_lengths_unk} {
  assert(model.score_type_ == Ort::TypeToTensorType<float>::type);
  int64_t input_ids_shape[] = {search_params_.batch_size, search_params_.sequence_length};
  cpu_span<int32_t> sequence_lengths = sequence_lengths_unk;

  const OrtMemoryInfo& location_cpu = model.model_.allocator_cpu_.GetInfo();

  // Use original input_ids. This requires the input_ids for subgraph is also int32.
  // Current shape is (batch_size, sequence_length)
  // Note that we will expand it to (batch_size * num_beams, sequence_length) later.
  // To avoid cloning input_ids, we use const_cast here since this function does not change its content.
  input_ids_ = OrtValue::CreateTensor<int32_t>(location_cpu, std::span<int32_t>(const_cast<int32_t*>(search_params_.input_ids.data()), input_ids_shape[0] * input_ids_shape[1]), input_ids_shape);

  // Expand (batch_size, sequence_length) to (batch_size * num_beams, sequence_length)
  input_ids_ = ExpandInputs(input_ids_, search_params_.num_beams, model.model_.allocator_cpu_, model_->model_.device_type_, model_->model_.cuda_stream_);

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

  for (int i = 0; i < model_->layer_count_; i++) {
    inputs_.push_back(kv_cache_.empty_past_.get());
    input_names_.push_back(kv_cache_.input_name_strings_[i].c_str());
    outputs_.push_back(kv_cache_.presents_[i].get());
    output_names_.push_back(kv_cache_.output_name_strings_[i].c_str());
  }
}

RoamingArray<float> Gpt_State::Run(int current_length, RoamingArray<int32_t> next_tokens_unk, RoamingArray<int32_t> next_indices_unk) {
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

void Gpt_State::UpdateInputs(cpu_span<const int32_t> next_tokens, cpu_span<const int32_t> beam_indices, int current_length) {
  assert(search_params_.num_beams == 1 || !beam_indices.empty());  // We require beam_indices if we're a beam search

  {
    // Update input_ids with next tokens.
    int batch_beam_size = static_cast<int>(next_tokens.size());
    int64_t dims[] = {batch_beam_size, 1};
    std::unique_ptr<OrtValue> input_ids = OrtValue::CreateTensor<int32_t>(model_->model_.allocator_cpu_, dims);
    auto* input_ids_data = input_ids->GetTensorMutableData<int32_t>();
    memcpy(input_ids_data, next_tokens.data(), batch_beam_size * sizeof(int32_t));
    input_ids_ = std::move(input_ids);
    inputs_[0] = input_ids_.get();
  }

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
  for (size_t i = 0; i < model_->layer_count_; i++) {
    inputs_[i + 3] = kv_cache_.pasts_[i].get();
    outputs_[i + 1] = kv_cache_.presents_[i].get();
  }
}

}  // namespace Generators