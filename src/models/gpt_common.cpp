#include "../generators.h"
#include "../search.h"
#include "model.h"
#include "gpt_common.h"
#include "debugging.h"
#include <iostream>

namespace Generators {

Gpt_Model::Gpt_Model(Model& model, OrtEnv& ort_env, OrtSessionOptions& session_options)
    : model_{model} {
  session_decoder_ = OrtSession::Create(ort_env, (model_.config_.config_path / model_.config_.model_decoder).c_str(), &session_options);
  InitModelParams();
}

void Gpt_Model::InitModelParams() {
  // We could use this to determine the vocabulary size and if the logits has a width of 1
  auto logits_type_info = session_decoder_->GetOutputTypeInfo(0);
  auto& logits_tensor_info = logits_type_info->GetTensorTypeAndShapeInfo();
  auto logits_shape = logits_tensor_info.GetShape();
  assert(logits_shape.size() == 3);
  logits_uses_seq_len_ = logits_shape[1] == -1;
  vocab_size_ = static_cast<int>(logits_shape[2]);
  layer_count_ = static_cast<int>(session_decoder_->GetOutputCount()) - 1;
  score_type_ = logits_tensor_info.GetElementType();

  auto past_shape = session_decoder_->GetInputTypeInfo(3)->GetTensorTypeAndShapeInfo().GetShape();
  head_count_ = static_cast<int>(past_shape[2]);
  hidden_size_ = static_cast<int>(past_shape[4]);

  assert(model_.config_.vocab_size==vocab_size_);
  assert(model_.config_.num_hidden_layers == layer_count_);
  assert(model_.config_.num_attention_heads == head_count_);
  assert(model_.config_.hidden_size == hidden_size_);
}

}  // namespace Generators
