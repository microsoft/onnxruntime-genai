#include <unordered_map>
#include <vector>
#include <string>

namespace engine {
enum SequenceStage { kPrefill = 0, kDecode };
enum SequenceStatus {
  kWaiting = 0,
  kRunning,
  kSwapped,
  kFinishedStopped,
  kFinishedLengthCapped,
  kFinishedAborted,
  kFinishedIgnored
};

enum AllocateStatus { kOK = 0, kLater, kNever };

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

  SequenceData(std::vector<int> prompt_tokens_ids)
      : prompt_tokens_ids(prompt_tokens_ids) {}

  int GetLen() const;
  void ResetStateForRecompute();
  int GetNumUncomputedTokens() const;
};

struct LLMInputs {
  std::vector<int> prompt_tokens_ids;
  std::string prompt;
};

struct Sequence {
  int seq_id;
  LLMInputs inputs;
  int block_size;
  int eos_token_id;
  SequenceData data;
  SequenceStatus status;

  Sequence(int seq_id, LLMInputs inputs, int block_size, int eos_token_id);

  int GetNumNewTokens() const;
  int GetLen() const;
  bool IsPrefill() const;
  bool IsFinished() const;
  void SetStatus(SequenceStatus status);
  void ResetStateForRecompute();
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
  Sequence encoder_seq;
  bool is_prefill;
  RequestMetrics metrics;

  SequenceGroup(std::string request_id, std::vector<Sequence> seqs, float arrival_time,
                SamplingParams sampling_params, std::vector<float> embeddings,
                Sequence encoder_seq);

  int GetMaxNumRunningSeqs();
  std::vector<Sequence> GetSeqs();
  std::vector<Sequence> GetSeqs(SequenceStatus status);

  void MaybeSetFirstScheduledTime(float time);
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
}  // namespace engine