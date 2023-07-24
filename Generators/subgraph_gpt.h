// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "subgraph_base.h"

// A class for GPT-2 subgraph inputs and outputs preparation.
class GptSubgraph : public Subgraph {
 public:
  GptSubgraph(
      OrtAllocator* allocator,
      const Node& node_in,
      const std::string& attribute_name,
      const OrtSession& subgraph_in) : Subgraph(allocator, node_in, attribute_name, subgraph_in) {
    first_past_input_index_ = 3;
    first_present_output_index_ = 1;
  }

  // Create inputs for first inference of subgraph.
  void CreateInitialFeeds(
      const OrtValue& input_ids,
      const std::vector<const OrtValue*>& implicit_inputs,
      int num_beams,
      int pad_token_id,
      gsl::span<int32_t>& sequence_lengths,
      std::unique_ptr<OrtValue>& expanded_input_ids,
      const OrtValue* attn_mask_value,
      std::vector<std::unique_ptr<OrtValue>>& feeds,
      const GenerationDeviceHelper::CreateGptInputsFunc& create_gpt_inputs_func,
      const GenerationDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
      IAllocatorUniquePtr<char>& buffer,
      Stream* ort_stream,
      int past_present_share_buffer_max_seq_len = -1,
      bool need_cache_indir = false);

  void Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                const std::vector<const NodeArg*>& subgraph_outputs) override;

  int GetFirstPastInputIndex() const {
    return first_past_input_index_;
  }

  int GetFirstPresentOutputIndex() const {
    return first_present_output_index_;
  }

 private:
  int first_past_input_index_;
  int first_present_output_index_;
};

