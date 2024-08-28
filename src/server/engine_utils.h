#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>

namespace Generators {
enum class SequenceStage { kPrefill, kDecode };
enum class SequenceStatus {
  kWaiting,
  kRunning,
  kSwapped,
  kFinishedStopped,
  kFinishedLengthCapped,
  kFinishedAborted,
  kFinishedIgnored

};

enum class AllocateStatus { kOK, kLater, kNever };

struct SamplingParams {
  int n = 1;
  int best_of = 0;
  float presence_penalty = 0.f;
  float repetition_penalty = 1.f;
  float temperature = 1.f;
  float top_p = 1.f;
  int top_k = -1;
  float min_p = 0.f;
  int seed = 0;
  bool use_beam_search = false;
  float length_penalty = 1.f;
  bool early_stopping = false;
  std::string stop = "";
  std::vector<int> stop_token_ids;
  bool include_stop_str_in_output = false;
  bool ignore_eos = false;
  int max_tokens = 16;
  int min_tokens = 0;
  int logprobs = 0;
  int prompt_logprobs = 0;
  bool detokenize = true;
  bool skip_special_tokens = true;
  bool spaces_between_special_tokens = true;
};

struct SequenceData {
  std::vector<int> prompt_tokens_ids;
  std::vector<int> output_token_ids;
  float cumulative_logprob = 0;
  int num_computed_tokens = 0;
  SequenceStage stage = SequenceStage::kPrefill;

  SequenceData(std::vector<int> const& prompt_tokens_ids)
      : prompt_tokens_ids(prompt_tokens_ids) {}

  int GetLen() const;
  void ResetStateForRecompute();
  int GetNumUncomputedTokens() const;
};

struct LLMInputs {
  std::vector<int32_t> prompt_tokens_ids;
  std::string prompt;
};

struct LogicalTokenBlock {
  int block_number;
  int block_size;
  std::vector<int> token_ids;
  int num_tokens;
  LogicalTokenBlock(int block_number, int block_size)
      : block_number(block_number), block_size(block_size) {
    token_ids.reserve(block_size);
    for (int i = 0; i < block_size; i++) {
      token_ids.push_back(-1);
    }
    num_tokens = 0;
  }
};

struct Sequence {
  int seq_id;
  LLMInputs inputs;
  int block_size;
  int eos_token_id;
  SequenceData data;
  SequenceStatus status;
  std::vector<LogicalTokenBlock> logical_token_blocks = {};
  std::string output_text = "";

  Sequence(int seq_id, const LLMInputs& inputs, int block_size,
           int eos_token_id);

  int GetNumNewTokens() const;
  int GetLen() const;
  bool IsPrefill() const;
  bool IsFinished() const;
  void SetStatus(SequenceStatus status);
  void ResetStateForRecompute();
  void AppendTokenId(int token_id);
  void AppendTokensToBlocks(std::vector<int> const& tokens);
};

struct RequestMetrics {
  float arrival_time;
  float last_token_time;
  float first_scheduled_time = -1;
  float first_token_time = -1;
  float time_in_queue = -1;
  float finished_time = -1;
};

struct SequenceGroup {
  std::string request_id;
  std::unordered_map<int, Sequence> seqs_dict;
  float arrival_time;
  SamplingParams sampling_params;
  std::vector<float> embeddings;
  std::shared_ptr<Sequence> encoder_seq;
  RequestMetrics metrics;

  SequenceGroup(std::string& request_id, std::vector<Sequence>& seqs,
                float arrival_time, const SamplingParams& sampling_params,
                std::vector<float>& embeddings,
                std::unique_ptr<Sequence> encoder_seq);

  int GetMaxNumRunningSeqs();
  std::vector<Sequence> GetSeqs() const;
  std::vector<Sequence> GetSeqs(SequenceStatus status) const;

  void Add(Sequence& seq);
  void MaybeSetFirstScheduledTime(float time);
  void MaybeSetFirstTokenTime(float time);
  void UpdateNumComputedTokens(int num_new_computed_tokens);

  bool IsPrefill() const;
  bool IsFinished() const;
};

struct SequenceGroupMetadata {
  std::string request_id;
  bool is_prompt;
  std::unordered_map<int, SequenceData> seq_data;
  SamplingParams sampling_params;
  std::unordered_map<int, std::vector<int>> block_tables;
  bool do_sample = true;
  int token_chunk_size;
  std::vector<int> computed_block_nums;
};

struct ExecuteModelRequest {
  std::vector<SequenceGroupMetadata> seq_group_metadata_list;
  std::vector<std::tuple<int, int>> blocks_to_swap_in;
  std::vector<std::tuple<int, int>> blocks_to_swap_out;
  std::vector<std::tuple<int, int>> blocks_to_swap_copy;
  int num_lookahead_slots;
  int running_queue_size;
};

struct SequenceOutput {
  int output_token;
  int parent_seq_id;
};

struct CompletionSequenceGroupOutput {
  std::vector<SequenceOutput> samples;
};
}  // namespace Generators