#include "model_runner.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "engine_utils.h"

namespace Generators {

ModelRunner::ModelRunner(std::shared_ptr<Generators::Model> model,
                         const CacheOptions& cache_config)
    : model_(model), cache_config_(cache_config) {}

void PadVector(std::vector<std::vector<int32_t>>& tensor,
               int32_t pad_token_id) {
  size_t max_length = 0;
  for (auto const& sequence : tensor)
    max_length = std::max(max_length, sequence.size());

  // Copy and pad the tensor with pad_token_id
  for (auto& sequence : tensor) {
    auto pad_count = max_length - sequence.size();
    sequence.insert(sequence.end(), pad_count, pad_token_id);
  }
}

std::vector<CompletionSequenceGroupOutput> ModelRunner::ExecuteModel(
    const ExecuteModelRequest& request) {
  // TODO(yingxiong): copy blocks before inference

  auto params = Generators::CreateGeneratorParams(*model_);

  std::vector<int> input_tokens;
  std::vector<std::vector<int>> block_tables;
  std::vector<int> slot_mapping;
  std::vector<int> context_lens;
  std::vector<std::vector<int>> group_seq_ids;

  int batch_size = 0;
  for (const auto& seq_group_metadata : request.seq_group_metadata_list) {
    std::vector<int> seq_ids;
    for (const auto& [seq_id, seq_data] : seq_group_metadata.seq_data) {
      int context_len;
      std::vector<int> block_table;

      seq_ids.push_back(seq_id);
      if (seq_data.stage == SequenceStage::kPrefill) {
        std::vector<int> full_tokens;
        full_tokens.insert(full_tokens.end(),
                           seq_data.prompt_tokens_ids.begin(),
                           seq_data.prompt_tokens_ids.end());
        if (!seq_data.output_token_ids.empty()) {
          full_tokens.insert(full_tokens.end(),
                             seq_data.output_token_ids.begin(),
                             seq_data.output_token_ids.end());
        }
        input_tokens.insert(input_tokens.end(), full_tokens.begin(),
                            full_tokens.end());
        context_len = seq_data.num_computed_tokens;
      } else {
        input_tokens.push_back(seq_data.output_token_ids.back());
        context_len = seq_data.GetLen() - 1;
        block_table = seq_group_metadata.block_tables.at(seq_id);
      }
      int seq_len = std::min(seq_data.GetLen(),
                             context_len + seq_group_metadata.token_chunk_size);
      for (int i = context_len; i < seq_len; i++) {
        auto& btable = seq_group_metadata.block_tables.at(seq_id);
        int block_num = btable[i / cache_config_.block_size_];
        int block_offset = i % cache_config_.block_size_;
        int slot = block_num * cache_config_.block_size_ + block_offset;
        slot_mapping.push_back(slot);
      }

      batch_size++;
      context_lens.push_back(seq_len);
      block_tables.push_back(block_table);
    }
    group_seq_ids.push_back(seq_ids);
  }

  printf("finish batching!\n");

  params->input_ids = input_tokens;
  params->batch_size = 1;
  params->sequence_length = input_tokens.size();
  named_tensors_ = std::make_unique<NamedTensors>();
  std::vector<int64_t> context_lens_shape = {context_lens.size()};
  context_lens_ = std::make_shared<Tensor>(OrtValue::CreateTensor<int32_t>(
      model_->allocator_cpu_.GetInfo(), std::span<int32_t>(context_lens),
      std::span<const int64_t>(context_lens_shape.data(),
                               context_lens_shape.size())));
  named_tensors_->emplace("context_lens", context_lens_);

  PadVector(block_tables, 0);
  std::vector<int64_t> block_tables_shape = {block_tables.size(),
                                             block_tables[0].size()};
  std::vector<int32_t> flat_block_tables;
  for (const auto& block_table : block_tables) {
    flat_block_tables.insert(flat_block_tables.end(), block_table.begin(),
                             block_table.end());
  }
  block_tables_ = std::make_shared<Tensor>(OrtValue::CreateTensor<int32_t>(
      model_->allocator_cpu_.GetInfo(), std::span<int32_t>(flat_block_tables),
      std::span<const int64_t>(block_tables_shape)));
  named_tensors_->emplace("block_tables", block_tables_);

  std::vector<int64_t> slot_mapping_shape = {slot_mapping.size()};
  slot_mapping_ = std::make_shared<Tensor>(OrtValue::CreateTensor<int32_t>(
      model_->allocator_cpu_.GetInfo(), std::span<int32_t>(slot_mapping),
      std::span<const int64_t>(slot_mapping_shape.data(),
                               slot_mapping_shape.size())));
  named_tensors_->emplace("slot_mapping", slot_mapping_);

  std::vector<int32_t> is_span_data{1};
  std::vector<int64_t> is_span_shape{1};

  // is_prompt_ = std::make_shared<Tensor>(OrtValue::CreateTensor<int32_t>(
  //     model_->allocator_cpu_.GetInfo(), std::span<int32_t>(is_span_data),
  //     std::span<int64_t>(is_span_shape)));
  is_prompt_ = std::make_shared<Tensor>(
      OrtValue::CreateTensor<int32_t>(model_->allocator_cpu_, is_span_shape));
  std::copy(is_span_data.begin(), is_span_data.end(), is_prompt_->ort_tensor_->GetTensorMutableData<int32_t>());
  printf("is_prompt_ address: %p\n", is_span_data.data());
  printf("is_prompt_ address: %p\n",
         is_prompt_->ort_tensor_->GetTensorMutableRawData());
  printf("is_prompt_: %d\n",
         is_prompt_->ort_tensor_->GetTensorMutableData<int32_t>()[0]);
  named_tensors_->emplace("is_prompt", is_prompt_);

  params->SetInputs(*named_tensors_);
  auto output_tokens = RunGenerator(*params);
  std::vector<CompletionSequenceGroupOutput> outputs;

  int offset = 0;
  for (int i = 0; i < request.seq_group_metadata_list.size(); ++i) {
    const auto& seq_group_metadata = request.seq_group_metadata_list[i];
    CompletionSequenceGroupOutput group_output;
    for (int j = 0; j < seq_group_metadata.seq_data.size(); ++j) {
      group_output.samples.push_back(
          SequenceOutput{output_tokens[offset], group_seq_ids[i][j]});
      offset++;
    }
    outputs.push_back(group_output);
  }

  return outputs;
}

std::vector<int32_t> ModelRunner::RunGenerator(const GeneratorParams& params) {
  printf("before create generator\n");
  auto generator = Generators::CreateGenerator(*model_, params);
  printf("generator created\n");

  generator->ComputeLogits();
  printf("logits finish\n");
  generator->GenerateNextToken();
  printf("next token finish\n");

  auto result = generator->search_->GetNextTokens().GetCPU();
  return std::vector<int32_t>(result.begin(), result.end());
}

};  // namespace Generators