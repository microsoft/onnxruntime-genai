// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>
#include "generation_device_helper.h"

namespace onnxruntime {
class SessionState;
}

class Subgraph {
 public:
  Subgraph(
      OrtAllocator* allocator,
      const Node& node_in,
      const std::string& attribute_name,
      const OrtSession& subgraph_in);
  virtual ~Subgraph() {}

  OrtAllocator* allocator_;
  const Node& node;              // Node that contains the subgraph
  const std::string& attribute;   // Attribute of th node that contains the subgraph. Not used yet.
  const OrtSession& subgraph;    // The subgraph

  int num_implicit_inputs;

  int num_subgraph_inputs;   // Same as subgraph_input_names.size(), keep it for convenience.
  int num_subgraph_outputs;  // Same as subgraph_output_names.size()

  std::vector<std::string> subgraph_input_names;
  std::vector<std::string> subgraph_output_names;

  // Parameters deduced from the subgraph
  int num_heads;
  int head_size;
  int vocab_size;
  int num_layers;
  bool past_present_share_buffer_;
  bool has_decoder_masked_attention_;

  // Setup execution
  void Setup(const SessionState& session_state,
               const SessionState& subgraph_session_state);

#if 0
  FeedsFetchesManager* GetFeedsFetchesManager() {
    return (feeds_fetches_manager_.has_value()) ? &*feeds_fetches_manager_ : nullptr;
  }

  const IExecutionProvider* GetProvider() const;
#endif

  bool IsOutputFloat16() const { return is_output_float16_; }

  virtual void Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                          const std::vector<const NodeArg*>& subgraph_outputs) = 0;

 protected:
  void GetParameters(const TensorShapeProto* past_shape,
                     const TensorShapeProto* logits_shape,
                     bool merged_past);

  void AppendPastSequenceLength(std::vector<std::unique_ptr<OrtValue>>& feeds,
                                OrtAllocator* cpu_allocator,
                                const int32_t init_value);

  void AppendBeamWidthAndCacheIndir(std::vector<std::unique_ptr<OrtValue>>& feeds,
                                    OrtAllocator* cpu_allocator,
                                    OrtAllocator* default_allocator,
                                    const int64_t batch_size,
                                    const int64_t num_beams,
                                    const int64_t max_seq_len);

  const SessionState* session_state_;
  const SessionState* subgraph_session_state_;
  std::optional<FeedsFetchesManager> feeds_fetches_manager_;
  bool is_output_float16_;
};
