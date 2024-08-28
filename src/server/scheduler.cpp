#include "scheduler.h"
#include <alloca.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "engine_utils.h"

namespace Generators {

int GetNumNewTokens(const SequenceGroup& seq_group, SequenceStatus status,
                    bool enable_chunking, const SchedulerBudget& budget) {
  int num_new_tokens = 0;
  auto seqs = seq_group.GetSeqs(status);
  for (const auto& seq : seqs) {
    num_new_tokens += seq.GetNumNewTokens();
  }

  if (enable_chunking && seqs.size() == 1) {
    num_new_tokens = std::min(num_new_tokens, budget.RemainingTokenBudget());
  }

  return num_new_tokens;
}

template <typename T>
T MergeSeqGroups(std::initializer_list<T> args) {
  T seq_groups;
  for (const auto& arg : args) {
    seq_groups.reserve(seq_groups.size() + arg.size());
  }
  for (const auto& arg : args) {
    seq_groups.insert(seq_groups.end(), arg.begin(), arg.end());
  }
  return seq_groups;
}

Scheduler::Scheduler(const SchedulerConfig& scheduler_config,
                     std::unique_ptr<CacheManager> block_manager)
    : scheduler_config_(scheduler_config),
      block_manager_(std::move(block_manager)) {}

PreemptionMode Scheduler::Preempt(
    SequenceGroup& seq_group,
    std::vector<std::tuple<int, int>>& blocks_to_swap_out) {
  PreemptionMode preemption_mode = kRecompute;
  // TODO: support swap
  // if (seq_group.GetMaxNumRunningSeqs() == 1) {
  //   preemption_mode = kRecompute;
  // } else {
  //   preemption_mode = kSwap;
  // }

  if (num_cumulative_preemption_ % 50 == 0) {
    // log warning
  }
  num_cumulative_preemption_++;

  // if (preemption_mode == kRecompute) {
  for (auto const& seq : seq_group.GetSeqs(SequenceStatus::kRunning)) {
    auto& raw_seq = seq_group.seqs_dict.at(seq.seq_id);
    raw_seq.status = SequenceStatus::kWaiting;
    block_manager_->Free(raw_seq);
    raw_seq.ResetStateForRecompute();
  }
  // } else {
  //   if (!block_manager_->CanSwapOut(seq_group)) {
  //     throw std::runtime_error(
  //         "Aborted due to the lack of CPU swap space. Please increase the
  //         swap " "space " "to avoid this error.");
  //   }

  //   std::vector<std::tuple<int, int>> mapping =
  //       block_manager_->SwapOut(seq_group);
  //   blocks_to_swap_out.insert(blocks_to_swap_out.end(), mapping.begin(),
  //                             mapping.end());
  //   for (auto& seq : seq_group.GetSeqs(kRunning)) {
  //     seq.SetStatus(SequenceStatus::kSwapped);
  //   }
  // }
  return preemption_mode;
}

void Scheduler::AddSeqGroup(SequenceGroup seq_group) {
  // Add the sequence group to the waiting queue
  waiting_.push_back(seq_group);
}  // namespace engine

ScheduleResult Scheduler::Schedule() {
  auto budget = SchedulerBudget(scheduler_config_.max_num_bathced_tokens,
                                scheduler_config_.max_num_seqs);

  ScheduledPrefillOutputs scheduled_prefill;
  ScheduledRunningOutputs scheduled_running;
  ScheduledSwappedInOutputs scheduled_swapped_in;
  for (auto& seq_group : running_) {
    budget.AddNumSeqs(seq_group.request_id, seq_group.GetMaxNumRunningSeqs());
  }

  if (swapped_.empty()) {
    scheduled_prefill = SchedulePrefill(budget);
  }

  if (scheduled_prefill.seq_groups.empty()) {
    scheduled_running = ScheduleRunning(budget);
    // if ((scheduled_running.preempted.size() +
    //      scheduled_running.swapped_out.size()) == 0) {
    //   scheduled_swapped_in = ScheduleSwapped(budget);
    // }
  }

  if (budget.GetNumBatchedTokens() > scheduler_config_.max_num_bathced_tokens ||
      budget.GetNumCurrSeqs() > scheduler_config_.max_num_seqs) {
    throw std::runtime_error(
        "The number of batched tokens or the number of sequences exceeds the "
        "maximum "
        "limit");
  }

  waiting_.insert(waiting_.begin(), scheduled_running.preempted.begin(),
                  scheduled_running.preempted.end());
  for (auto& scheduled_seq_group : scheduled_prefill.seq_groups) {
    running_.push_back(scheduled_seq_group.seq_group);
  }
  for (auto& scheduled_seq_group : scheduled_running.decode_seq_groups) {
    running_.push_back(scheduled_seq_group.seq_group);
  }
  for (auto& scheduled_seq_group : scheduled_swapped_in.decode_seq_groups) {
    running_.push_back(scheduled_seq_group.seq_group);
  }
  swapped_.insert(swapped_.end(), scheduled_running.swapped_out.begin(),
                  scheduled_running.swapped_out.end());

  std::vector<ScheduledSequenceGroup> scheduled_seq_groups = MergeSeqGroups(
      {scheduled_prefill.seq_groups, scheduled_running.decode_seq_groups,
       scheduled_swapped_in.decode_seq_groups});
  std::vector<std::tuple<int, int>> blocks_to_copy = MergeSeqGroups(
      {scheduled_running.blocks_to_copy, scheduled_swapped_in.blocks_to_copy});

  std::vector<SequenceGroup> ignored_seq_groups =
      MergeSeqGroups({scheduled_prefill.ignored_seq_groups,
                      scheduled_swapped_in.infeasible_seq_groups});

  auto scheduler_output =
      SchedulerOutputs{scheduled_seq_groups,
                       static_cast<int>(scheduled_prefill.seq_groups.size()),
                       budget.GetNumBatchedTokens(),
                       scheduled_swapped_in.blocks_to_swap_in,
                       scheduled_running.blocks_to_swap_out,
                       blocks_to_copy,
                       ignored_seq_groups,
                       scheduler_config_.num_lookahead_slots,
                       static_cast<int>(running_.size()),
                       static_cast<int>(scheduled_running.preempted.size())};

  float now = std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
  std::vector<SequenceGroupMetadata> seq_group_metadata_list;
  seq_group_metadata_list.reserve(scheduled_seq_groups.size());
  for (auto& scheduled_seq_group : scheduled_seq_groups) {
    SequenceGroup seq_group = scheduled_seq_group.seq_group;
    int token_chunk_size = scheduled_seq_group.token_chunk_size;
    seq_group.MaybeSetFirstScheduledTime(now);

    std::unordered_map<int, SequenceData> seq_datas;
    std::unordered_map<int, std::vector<int>> block_tables;
    for (auto& seq : seq_group.GetSeqs(SequenceStatus::kRunning)) {
      seq_datas.try_emplace(seq.seq_id, seq.data);
      block_tables[seq.seq_id] = block_manager_->GetBlockTable(seq);
    }

    // used in prefix caching, empty for now
    std::vector<int> common_computed_block_nums;

    bool do_sample = true;
    if (seq_group.IsPrefill()) {
      auto seqs = seq_group.GetSeqs();
      assert(seqs.size() == 1);

      if (token_chunk_size + seqs[0].data.num_computed_tokens <
          seqs[0].data.GetLen()) {
        do_sample = false;
      }
    }

    bool is_prompt = seq_group.IsPrefill();
    seq_group_metadata_list.emplace_back(SequenceGroupMetadata{
        seq_group.request_id, is_prompt, seq_datas, seq_group.sampling_params,
        block_tables, do_sample, token_chunk_size, common_computed_block_nums});
  }

  return ScheduleResult{seq_group_metadata_list, scheduler_output};
}

ScheduledPrefillOutputs Scheduler::SchedulePrefill(SchedulerBudget& budget,
                                                   bool enable_chunking) {
  std::vector<SequenceGroup> ignored_seq_groups;
  std::vector<ScheduledSequenceGroup> seq_groups;

  while (!waiting_.empty()) {
    SequenceGroup& seq_group = waiting_.front();
    auto waiting_seqs = seq_group.GetSeqs();
    std::cout << "waiting_seqs.size(): " << waiting_seqs.size() << std::endl;
    if (waiting_seqs.size() > 1) {
      throw std::runtime_error(
          "Waiting sequence should have only one prompt sequence");
    }

    int num_new_tokens = GetNumNewTokens(seq_group, SequenceStatus::kWaiting,
                                         enable_chunking, budget);

    std::cout << "num_new_tokens: " << num_new_tokens << std::endl;

    if (!enable_chunking && waiting_seqs[0].GetLen() != num_new_tokens) {
      throw std::runtime_error(
          "The number of new tokens should be equal to the length of the "
          "prompt "
          "sequence");
    }

    int prompt_limit = scheduler_config_.enable_chunked_prefill
                           ? scheduler_config_.max_model_len
                           : std::min(scheduler_config_.max_model_len,
                                      scheduler_config_.max_num_bathced_tokens);

    if (num_new_tokens > prompt_limit) {
      // log warning seq too long
      for (auto& seq : waiting_seqs) {
        seq_group.seqs_dict.at(seq.seq_id)
            .SetStatus(SequenceStatus::kFinishedIgnored);
      }
      ignored_seq_groups.push_back(seq_group);
      waiting_.pop_front();
      continue;
    }

    AllocateStatus can_allocate = block_manager_->CanAllocate(seq_group);
    std::cout << "can_allocate: " << (can_allocate == AllocateStatus::kOK)
              << std::endl;
    if (can_allocate == AllocateStatus::kLater) {
      break;
    } else if (can_allocate == AllocateStatus::kNever) {
      // log warning
      for (auto& seq : waiting_seqs) {
        seq.SetStatus(SequenceStatus::kFinishedIgnored);
      }
      ignored_seq_groups.push_back(seq_group);
      waiting_.pop_front();
      continue;
    }

    int num_new_seqs = seq_group.GetMaxNumRunningSeqs();

    if (num_new_tokens == 0 ||
        !budget.CanSchedule(num_new_tokens, num_new_seqs)) {
      break;
    }

    block_manager_->Allocate(seq_group);
    for (auto const& seq : seq_group.GetSeqs(SequenceStatus::kWaiting)) {
      auto& raw_seq = seq_group.seqs_dict.at(seq.seq_id);
      raw_seq.status = SequenceStatus::kRunning;
    }

    seq_groups.emplace_back(seq_group, num_new_tokens);
    budget.AddNumBatchedTokens(seq_group.request_id, num_new_tokens);
    budget.AddNumSeqs(seq_group.request_id, num_new_seqs);
    waiting_.pop_front();
  }

  return ScheduledPrefillOutputs{seq_groups, ignored_seq_groups, 0};
}

ScheduledRunningOutputs Scheduler::ScheduleRunning(SchedulerBudget& budget,
                                                   bool enable_chunking) {
  std::vector<ScheduledSequenceGroup> decode_seq_groups;
  std::vector<ScheduledSequenceGroup> prefill_seq_groups;
  std::vector<SequenceGroup> preempted;
  std::vector<SequenceGroup> swapped_out;
  std::vector<std::tuple<int, int>> blocks_to_swap_out;
  std::vector<std::tuple<int, int>> blocks_to_copy;

  std::sort(running_.begin(), running_.end(),
            [](const SequenceGroup& a, const SequenceGroup& b) {
              return a.arrival_time < b.arrival_time;
            });

  while (running_.size() > 0) {
    auto seq_group = running_.front();
    int num_running_tokens = GetNumNewTokens(
        seq_group, SequenceStatus::kRunning, enable_chunking, budget);
    if (num_running_tokens == 0) {
      break;
    }

    running_.pop_front();
    while (!block_manager_->CanAppendSlots(
        seq_group, scheduler_config_.num_lookahead_slots)) {
      budget.SubtractNumBatchedTokens(seq_group.request_id, num_running_tokens);
      int num_running_seqs = seq_group.GetMaxNumRunningSeqs();
      budget.SubtractNumSeqs(seq_group.request_id, num_running_seqs);

      if (running_.size() > 0) {
        SequenceGroup victim = running_.back();
        running_.pop_back();
        PreemptionMode preemption_mode = Preempt(victim, blocks_to_swap_out);
        // TODO: support swap
        // if (preemption_mode == kRecompute) {
        preempted.push_back(victim);
        // } else {
        //   swapped_out.push_back(victim);
        // }
      } else {
        PreemptionMode preemption_mode = Preempt(seq_group, blocks_to_swap_out);
        // TODO: support swap
        // if (preemption_mode == kRecompute) {
        preempted.push_back(seq_group);
        // } else {
        //   swapped_out.push_back(seq_group);
        // }
        break;
      }
    }

    if (block_manager_->CanAppendSlots(seq_group,
                                       scheduler_config_.num_lookahead_slots)) {
      for (auto& seq : seq_group.GetSeqs(SequenceStatus::kRunning)) {
        std::vector<std::tuple<int, int>> cows = block_manager_->AppendSlots(
            seq, scheduler_config_.num_lookahead_slots);
        blocks_to_copy.insert(blocks_to_copy.end(), cows.begin(), cows.end());
      }
      if (seq_group.IsPrefill()) {
        prefill_seq_groups.emplace_back(
            ScheduledSequenceGroup{seq_group, num_running_tokens});
      } else {
        decode_seq_groups.emplace_back(ScheduledSequenceGroup{seq_group, 1});
      }
      budget.AddNumBatchedTokens(seq_group.request_id, num_running_tokens);
      if (enable_chunking) {
        budget.AddNumSeqs(seq_group.request_id,
                          seq_group.GetMaxNumRunningSeqs());
      }
    }
  }

  return ScheduledRunningOutputs{decode_seq_groups,
                                 prefill_seq_groups,
                                 preempted,
                                 swapped_out,
                                 blocks_to_swap_out,
                                 blocks_to_copy,
                                 scheduler_config_.num_lookahead_slots};
}

// ScheduledSwappedInOutputs Scheduler::ScheduleSwapped(SchedulerBudget& budget,
//                                                      bool enable_chunking) {
//   std::vector<ScheduledSequenceGroup> decode_seq_groups;
//   std::vector<ScheduledSequenceGroup> prefill_seq_groups;
//   std::vector<std::tuple<int, int>> blocks_to_swap_in;
//   std::vector<std::tuple<int, int>> blocks_to_copy;
//   std::vector<SequenceGroup> infeasible_seq_groups;

//   std::sort(swapped_.begin(), swapped_.end(),
//             [](const SequenceGroup& a, const SequenceGroup& b) {
//               return a.arrival_time < b.arrival_time;
//             });

//   while (swapped_.size() > 0) {
//     SequenceGroup seq_group = swapped_.front();
//     bool is_prefill = seq_group.IsPrefill();
//     AllocateStatus alloc_status = block_manager_->CanSwapIn(
//         seq_group, scheduler_config_.num_lookahead_slots);
//     if (alloc_status == kLater) {
//       break;
//     } else if (alloc_status == kNever) {
//       // log warning
//       for (auto& seq : seq_group.GetSeqs()) {
//         seq.SetStatus(SequenceStatus::kFinishedIgnored);
//       }
//       infeasible_seq_groups.push_back(seq_group);
//       swapped_.pop_front();
//       continue;
//     }

//     int num_new_tokens =
//         GetNumNewTokens(seq_group, SequenceStatus::kSwapped, enable_chunking,
//         budget);
//     int num_new_seqs = seq_group.GetMaxNumRunningSeqs();

//     if (num_new_tokens == 0 ||
//         !budget.CanSchedule(num_new_tokens, num_new_seqs)) {
//       break;
//     }

//     swapped_.pop_front();
//     std::vector<std::tuple<int, int>> mapping =
//         block_manager_->SwapIn(seq_group);
//     blocks_to_swap_in.insert(blocks_to_swap_in.end(), mapping.begin(),
//                              mapping.end());
//     for (auto& seq : seq_group.GetSeqs(SequenceStatus::kSwapped)) {
//       seq.SetStatus(kRunniSequenceStatus::kRunningng);
//     }

//     for (auto& seq : seq_group.GetSeqs(SequenceStatus::kRunning)) {
//       std::vector<std::tuple<int, int>> cows = block_manager_->AppendSlots(
//           seq, scheduler_config_.num_lookahead_slots);
//       blocks_to_copy.insert(blocks_to_copy.end(), cows.begin(), cows.end());
//     }

//     if (is_prefill) {
//       prefill_seq_groups.emplace_back(
//           ScheduledSequenceGroup{seq_group, num_new_tokens});
//     } else {
//       decode_seq_groups.emplace_back(ScheduledSequenceGroup{seq_group, 1});
//     }
//     budget.AddNumBatchedTokens(seq_group.request_id, num_new_tokens);
//     budget.AddNumSeqs(seq_group.request_id, num_new_seqs);
//   }

//   return ScheduledSwappedInOutputs{decode_seq_groups,
//                                    prefill_seq_groups,
//                                    blocks_to_swap_in,
//                                    blocks_to_copy,
//                                    scheduler_config_.num_lookahead_slots,
//                                    infeasible_seq_groups};
// }

void Scheduler::ForkSeq(const Sequence& parent_seq, const Sequence& child_seq) {
  block_manager_->ForkSeq(parent_seq, child_seq);
}

void Scheduler::FreeSeq(const Sequence& seq) { block_manager_->Free(seq); }

void Scheduler::FreeFinishedSeqGroups() {
  std::erase_if(running_, [](const SequenceGroup& seq_group) {
    return seq_group.IsFinished();
  });
}

int SchedulerBudget::RemainingTokenBudget() const {
  return token_budget_ - num_batched_tokens_;
}

void SchedulerBudget::AddNumBatchedTokens(std::string req_id,
                                          int num_batched_tokens) {
  if (request_ids_num_batched_tokens_.find(req_id) !=
      request_ids_num_batched_tokens_.end()) {
    return;
  }
  num_batched_tokens_ += num_batched_tokens;
  request_ids_num_batched_tokens_.insert(req_id);
}

void SchedulerBudget::AddNumSeqs(std::string req_id, int num_curr_seqs) {
  if (request_ids_num_curr_seqs_.find(req_id) !=
      request_ids_num_curr_seqs_.end()) {
    return;
  }
  num_curr_seqs_ += num_curr_seqs;
  request_ids_num_curr_seqs_.insert(req_id);
}

void SchedulerBudget::SubtractNumBatchedTokens(std::string req_id,
                                               int num_batched_tokens) {
  if (request_ids_num_batched_tokens_.find(req_id) ==
      request_ids_num_batched_tokens_.end()) {
    return;
  }
  num_batched_tokens_ -= num_batched_tokens;
  request_ids_num_batched_tokens_.erase(req_id);
}

void SchedulerBudget::SubtractNumSeqs(std::string req_id, int num_curr_seqs) {
  if (request_ids_num_curr_seqs_.find(req_id) ==
      request_ids_num_curr_seqs_.end()) {
    return;
  }
  num_curr_seqs_ -= num_curr_seqs;
  request_ids_num_curr_seqs_.erase(req_id);
}

bool SchedulerBudget::CanSchedule(int num_new_tokens, int num_new_seqs) {
  assert(num_new_tokens >= 0 && num_new_seqs >= 0);
  return num_batched_tokens_ + num_new_tokens <= RemainingTokenBudget() &&
         num_new_seqs + num_new_seqs <= max_num_seqs_;
}

}  // namespace Generators