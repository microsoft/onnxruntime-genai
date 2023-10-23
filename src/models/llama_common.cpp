#include "../generators.h"
#include "../search.h"
#include "onnxruntime_cxx_api_2.h"
#include "llama_common.h"
#include "debugging.h"
#include <iostream>

namespace Generators {

void GetModelParams(LlamaModelParams& model_params, OrtSession& session)
{
  // We could use this to determine the vocabulary size and if the logits has a width of 1
  auto logits_shape = session.GetOutputTypeInfo(0) -> GetTensorTypeAndShapeInfo().GetShape();
  assert(logits_shape.size() == 3);
  model_params.logits_uses_seq_len = logits_shape[1] == -1;
  model_params.vocab_size = static_cast<int>(logits_shape[2]);
  model_params.layer_count = (static_cast<int>(session.GetOutputCount()) - 1) / 2;

  auto past_shape = session.GetInputTypeInfo(3) -> GetTensorTypeAndShapeInfo().GetShape();
  model_params.head_count = static_cast<int>(past_shape[1]);
  model_params.hidden_size = static_cast<int>(past_shape[3]);
}

}
