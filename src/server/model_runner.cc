#include "model_runner.h"
#include "../tensor.h"
#include <cstdint>
#include <vector>
#include "engine_utils.h"

namespace Generators {

ModelRunner::ModelRunner(const Generators::Model& model) : model_(model) {}

void PadVector(std::vector<std::vector<int32_t>> tensor, int32_t pad_token_id) {
  bool pad_right_{true};

  size_t max_length = 0;
  for (auto& sequence : tensor) max_length = std::max(max_length, sequence.size());

  // Copy and pad the tensor with pad_token_id
  for (size_t i = 0; i < tensor.size(); i++) {
    auto& sequence = tensor[i];

    auto pad_count = max_length - sequence.size();
    sequence.insert(sequence.end(), pad_count, pad_token_id);
  }
}

std::vector<CompletionSequenceGroupOutput> ModelRunner::ExecuteModel(ExecuteModelRequest request) {
  // TODO(yingxiong): copy blocks before inference

  auto params = Generators::CreateGeneratorParams(model_);

  std::vector<int> input_tokens;
  std::vector<std::vector<int>> block_tables;
  std::vector<int> slot_mapping;
  std::vector<int> context_lens;

  int batch_size = 0;
  for (const auto& seq_group_metadata : request.seq_group_metadata_list) {
    for (auto it = seq_group_metadata.seq_data.begin(); it != seq_group_metadata.seq_data.end(); ++it) {
      int seq_id = it->first;
      auto& seq_data = it->second;
      int context_len;
      std::vector<int> block_table;
      if (seq_data.is_prompt) {
        std::vector<int> full_tokens;
        full_tokens.insert(full_tokens.end(), seq_data.prompt_tokens_ids.begin(), seq_data.prompt_tokens_ids.end());
        full_tokens.insert(full_tokens.end(), seq_data.output_token_ids.begin(), seq_data.output_token_ids.end());
        input_tokens.insert(input_tokens.end(), full_tokens.begin(), full_tokens.end());
        context_len = seq_data.num_computed_tokens;
      } else {
        input_tokens.push_back(seq_data.output_token_ids.back());
        context_len = seq_data.GetLen() - 1;
        block_table = seq_group_metadata.block_tables[seq_id];
      }

      int seq_len = std::min(seq_data.GetLen(), context_len + seq_group_metadata.token_chunk_size);
      std::vector<int> slot_mapping;
      for (int i = context_len; i < seq_len; i++) {
        int block_num = block_table[i / cache_config_.block_size];
        int block_offset = i % cache_config_.block_size;
        int slot = block_num * cache_config_.block_size + block_offset;
        slot_mapping.push_back(slot);
      }

      batch_size++;
      context_lens.push_back(seq_len);
      block_tables.push_back(block_table);
    }
  }

  params.input_tokens = {input_tokens.data(), static_cast<size_t>(input_tokens.size())};
  params.batch_size = 1;
  auto named_tensors = std::make_unique<NamedTensors>();
  named_tensors->emplace(
      "context_lens", std::make_shared<Tensor>(OrtValue::CreateTensor<int32_t>(
                          model_.allocator_cpu_.GetInfo(), std::span<int32_t>(context_lens.data(), context_lens.size()),
                          context_lens.size())));
  PadVector(block_tables, 0);
  named_tensors->emplace("block_tables",
                         std::make_shared<Tensor>(OrtValue::CreateTensor<int32_t>(
                             model_.allocator_cpu_.GetInfo(),
                             std::span<int32_t>(block_tables.data(), block_tables.size() * block_tables[0].size()),
                             {block_tables.size(), block_tables[0].size()})));
  named_tensors->emplace(
      "slot_mapping", std::make_shared<Tensor>(OrtValue::CreateTensor<int32_t>(
                          model_.allocator_cpu_.GetInfo(), std::span<int32_t>(slot_mapping.data(), slot_mapping.size()),
                          slot_mapping.size())));

  params.SetInputs(*named_tensors);
  auto output_tokens = RunGenerator(params);
  std::vector<CompletionSequenceGroupOutput> outputs;

  for (int i = 0; i < request.seq_group_metadata_list.size(); ++i) {
    const auto& seq_group_metadata = request.seq_group_metadata_list[i];
    CompletionSequenceGroupOutput group_output;
    for (int j = 0; j < seq_group_metadata.seq_data.size(); ++j) {
      group_output.samples.push_back(SequenceOutput{output_tokens[i + j]});
    }
    outputs.push_back(group_output);
  }

  return outputs;
}

RoamingArray<int32_t> ModelRunner::RunGenerator(const Generators::GeneratorParams& params) {
  auto generator = Generators::CreateGenerator(model_, params);

  generator->ComputeLogits();
  generator->GenerateNextToken();

  return generator->search_->GetNextTokens();
}

};  // namespace Generators