#include "../generators.h"
#include "../search.h"
#include "onnxruntime_cxx_api_2.h"
#include "llama_common.h"
#include "debugging.h"
#include <iostream>

namespace Generators {

Llama_Model::Llama_Model(OrtEnv& ort_env, const ORTCHAR_T* decoder_path) : device_type_{DeviceType::CPU} {
  auto session_options = OrtSessionOptions::Create();
  session_decoder_ = OrtSession::Create(ort_env, decoder_path, session_options.get());
  InitModelParams();
}

#ifdef USE_CUDA
Llama_Model::Llama_Model(OrtEnv& ort_env, const ORTCHAR_T* decoder_path, cudaStream_t cuda_stream) : cuda_stream_{cuda_stream}, device_type_{DeviceType::CUDA} {
  auto session_options = OrtSessionOptions::Create();
  OrtCUDAProviderOptions cuda_options;
  cuda_options.has_user_compute_stream = true;
  cuda_options.user_compute_stream = cuda_stream;
  session_options->AppendExecutionProvider_CUDA(cuda_options);

  session_decoder_ = OrtSession::Create(ort_env, decoder_path, session_options.get());
  InitModelParams();
}
#endif

void Llama_Model::InitModelParams() {
  // We could use this to determine the vocabulary size and if the logits has a width of 1
  auto& logits_type_info = session_decoder_->GetOutputTypeInfo(0)->GetTensorTypeAndShapeInfo();
  auto logits_shape = logits_type_info.GetShape();
  assert(logits_shape.size() == 3);
  logits_uses_seq_len_ = logits_shape[1] == -1;
  vocab_size_ = static_cast<int>(logits_shape[2]);
  layer_count_ = (static_cast<int>(session_decoder_->GetOutputCount()) - 1) / 2;
  score_type_ = logits_type_info.GetElementType();

  auto past_shape = session_decoder_->GetInputTypeInfo(3)->GetTensorTypeAndShapeInfo().GetShape();
  head_count_ = static_cast<int>(past_shape[1]);
  hidden_size_ = static_cast<int>(past_shape[3]);
}

}  // namespace Generators
