// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "decoder.h"
#include "../../models/decoder_only.h"

namespace Generators {

/**
 * @class VarlenDecoderIO
 * @brief Prepares and manages the inputs and outputs for a variable-length decoder model.
 *
 * A variable-length decoder model is one that can handle input sequences of varying lengths
 * within the same batch. This class handles the preparation of the following input and output tensors:
 * Inputs:
 * - Input IDs - int64[total_num_tokens]
 * - Cumulative Sequence Lengths - int32[batch_size + 1]
 * - Past Sequence Lengths - int32[batch_size]
 * Outputs:
 * - Logits - float16/float32[total_num_tokens, vocab_size]
 *
 * The inputs prepared by this class are compatible with models that use the
 * PagedAttention operator.
 */
struct VarlenDecoderIO : DecoderIO {
  VarlenDecoderIO(std::shared_ptr<DecoderOnly_Model> model,
                  ScheduledRequests& scheduled_requests,
                  std::shared_ptr<CacheManager> cache_manager);

  std::vector<DeviceSpan<float>> ProcessLogits() override;

 private:
  void PrepareInputIds(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests);
  void PrepareLogits(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests);

  std::vector<std::unique_ptr<Tensor>> owned_inputs_;
  std::unique_ptr<Tensor> logits_;
  std::unique_ptr<Tensor> logits_fp32_;
};

}  // namespace Generators
