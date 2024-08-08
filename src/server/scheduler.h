#pragma once

#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>
#include <deque>

#include "engine_utils.h"

using RequestIDType = int64_t;

namespace Generators {

enum PreemptionMode {
  kSwap = 0,
  kRecompute,
};
struct ScheduledSequenceGroup {
  SequenceGroup seq_group;
  int token_chunk_size;
};

class SchedulerBudget {
 public:
  SchedulerBudget(int token_budget, int max_num_seqs)
      : token_budget_(token_budget), max_num_seqs_(max_num_seqs) {}
  int GetTokenBudget() const { return token_budget_; }
  int GetMaxNumSeqs() const { return max_num_seqs_; }
  int GetNumBatchedTokens() const { return num_batched_tokens_; };
  int GetNumCurrSeqs() const { return num_curr_seqs_; };
  int RemainingTokenBudget() const;
  void AddNumBatchedTokens(std::string req_id, int num_batched_tokens);
  void AddNumSeqs(std::string req_id, int num_curr_seqs);
  void SubtractNumBatchedTokens(std::string req_id, int num_batched_tokens);
  void SubtractNumSeqs(std::string req_id, int num_curr_seqs);
  bool CanSchedule(int num_new_tokens, int num_new_seqs);

 private:
  int token_budget_;
  int max_num_seqs_;
  std::unordered_set<std::string> request_ids_num_batched_tokens_;
  std::unordered_set<std::string> request_ids_num_curr_seqs_;
  int num_batched_tokens_ = 0;
  int num_curr_seqs_ = 0;
};

struct ScheduledPrefillOutputs {
  std::vector<ScheduledSequenceGroup> seq_groups;
  std::vector<SequenceGroup> ignored_seq_groups;
  int num_lookahead_slots;
};

struct ScheduledRunningOutputs {
  std::vector<ScheduledSequenceGroup> decode_seq_groups;
  std::vector<ScheduledSequenceGroup> prefill_seq_groups;
  std::vector<SequenceGroup> preempted;
  std::vector<SequenceGroup> swapped_out;
  std::vector<std::tuple<int, int>> blocks_to_swap_out;
  std::vector<std::tuple<int, int>> blocks_to_copy;
  int num_lookahead_slots;
};

struct ScheduledSwappedInOutputs {
  std::vector<ScheduledSequenceGroup> decode_seq_groups;
  std::vector<ScheduledSequenceGroup> prefill_seq_groups;
  std::vector<std::tuple<int, int>> blocks_to_swap_in;
  std::vector<std::tuple<int, int>> blocks_to_copy;
  int num_lookahead_slots;
  std::vector<SequenceGroup> infeasible_seq_groups;
};

struct SchedulerConfig {
  int max_num_bathced_tokens;
  int max_num_seqs;
  int max_model_len;
  bool use_v2_block_manager = false;
  int num_lookahead_slots = 0;
  float delay_factor = 0.0;
  bool enable_chunked_prefill = false;
  bool embedding_mode = false;
  std::string preemption_mode = "";
};

struct SchedulerOutputs {
  std::vector<ScheduledSequenceGroup> scheduled_seq_groups;
  int num_prefill_groups;
  int num_batched_tokens;
  std::vector<std::tuple<int, int>> blocks_to_swap_in;
  std::vector<std::tuple<int, int>> blocks_to_swap_out;
  std::vector<std::tuple<int, int>> blocks_to_copy;
  std::vector<SequenceGroup> ignored_seq_groups;
  int num_lookahead_slots;
  int running_queue_size;
  int preempted;
};

struct ScheduleResult {
  std::vector<SequenceGroupMetadata> seq_group_metadatas;
  SchedulerOutputs scheduler_outputs;
};

class Scheduler {
 public:
  Scheduler(SchedulerConfig& scheduler_config, CacheConfig& cache_config);
  void AddSeqGroup(SequenceGroup seq_group);
  ScheduledPrefillOutputs SchedulePrefill(SchedulerBudget budget,
                                          bool enable_chunking = false);
  ScheduledRunningOutputs ScheduleRunning(SchedulerBudget budget,
                                          bool enable_chunking = false);
  ScheduledSwappedInOutputs ScheduleSwapped(SchedulerBudget budget,
                                            bool enable_chunking = false);
  ScheduleResult Schedule();
  void FreeFinishedRequests();

  PreemptionMode Preempt(SequenceGroup& seq_group,
                         std::vector<std::tuple<int, int>>& blocks_to_swap_out);

 private:
  SchedulerConfig scheduler_config_;
  CacheConfig cache_config_;
  std::deque<SequenceGroup> waiting_;
  std::deque<SequenceGroup> running_;
  std::deque<SequenceGroup> swapped_;

  BlockManager block_manager_;
  int num_cumulative_preemption_ = 0;
};
}  // namespace engine