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

  Sequence(int seq_id, LLMInputs inputs, int block_size, int eos_token_id);

  int GetNumNewTokens() const;
  int GetLen() const;
  void SetStatus(SequenceStatus status);
  void ResetStateForRecompute();
};

struct SequenceGroup {
  std::string request_id;
  std::vector<Sequence> seqs;
  float arrival_time;
  SamplingParams sampling_params;
  std::vector<float> embeddings;
  Sequence encoder_seq;
  bool is_prefill;

  int GetMaxNumRunningSeqs() const;
  std::vector<Sequence>& GetSeqs();
  std::vector<Sequence>& GetSeqs(SequenceStatus status) const;
};
}  // namespace engine