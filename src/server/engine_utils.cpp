
#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

#include "engine_utils.h"

namespace Generators {

bool IsFinished(SequenceStatus status) {
  switch (status) {
    case SequenceStatus::kFinishedIgnored:
      return true;
    case SequenceStatus::kFinishedAborted:
      return true;
    case SequenceStatus::kFinishedLengthCapped:
      return true;
    case SequenceStatus::kFinishedStopped:
      return true;
    default:
      return false;
  }
}
int SequenceData::GetLen() const {
  return output_token_ids.size() + prompt_tokens_ids.size();
}

void SequenceData::ResetStateForRecompute() {
  num_computed_tokens = 0;
  stage = SequenceStage::kPrefill;
}

int SequenceData::GetNumUncomputedTokens() const {
  return GetLen() - num_computed_tokens;
}

int Sequence::GetNumNewTokens() const {
  if (data.stage == SequenceStage::kDecode) {
    return 1;
  }
  return data.GetNumUncomputedTokens();
}

Sequence::Sequence(int seq_id, const LLMInputs& inputs, int block_size,
                   int eos_token_id)
    : seq_id(seq_id),
      inputs(inputs),
      block_size(block_size),
      eos_token_id(eos_token_id),
      data(inputs.prompt_tokens_ids) {
  status = SequenceStatus::kWaiting;
  AppendTokensToBlocks(inputs.prompt_tokens_ids);
}

int Sequence::GetLen() const { return data.GetLen(); }

bool Sequence::IsPrefill() const {
  return data.stage == SequenceStage::kPrefill;
}

bool Sequence::IsFinished() const { return Generators::IsFinished(status); }

void Sequence::SetStatus(SequenceStatus new_status) { status = new_status; }

void Sequence::ResetStateForRecompute() { data.ResetStateForRecompute(); }

void Sequence::AppendTokenId(int token_id) {
  AppendTokensToBlocks({token_id});
  data.output_token_ids.push_back(token_id);
}

void Sequence::AppendTokensToBlocks(std::vector<int> const& tokens) {
  int cursur = 0;
  while (cursur < tokens.size()) {
    if (logical_token_blocks.empty()) {
      logical_token_blocks.emplace_back(
          LogicalTokenBlock{logical_token_blocks.size(), block_size});
    }
    auto& last_block = logical_token_blocks.back();
    if (last_block.num_tokens == block_size) {
      logical_token_blocks.emplace_back(
          LogicalTokenBlock{logical_token_blocks.size(), block_size});
      last_block = logical_token_blocks.back();
    }
    int empty_slots = block_size - last_block.num_tokens;
    for (auto it = tokens.begin() + cursur; it != tokens.end(); ++it) {
      last_block.token_ids[last_block.num_tokens] = *it;
      last_block.num_tokens++;
      if (last_block.num_tokens == block_size) {
        break;
      }
    }
    cursur += empty_slots;
  }
}

SequenceGroup::SequenceGroup(std::string& request_id,
                             std::vector<Sequence>& seqs, float arrival_time,
                             const SamplingParams& sampling_params,
                             std::vector<float>& embeddings,
                             std::unique_ptr<Sequence> encoder_seq)
    : request_id(std::move(request_id)),
      arrival_time(arrival_time),
      sampling_params(sampling_params),
      embeddings(std::move(embeddings)),
      encoder_seq(std::move(encoder_seq)) {
  for (auto& seq : seqs) {
    seqs_dict.try_emplace(seq.seq_id, std::move(seq));
  }
  metrics = RequestMetrics{arrival_time, arrival_time};
}

int SequenceGroup::GetMaxNumRunningSeqs() {
  if (sampling_params.use_beam_search) {
    return sampling_params.best_of;
  }

  if (sampling_params.best_of > GetSeqs().size()) {
    return sampling_params.best_of;
  }

  int num_unfinished = 0;
  for (auto const& seq : GetSeqs()) {
    if (!seq.IsFinished()) {
      num_unfinished++;
    }
  }
  return num_unfinished;
}

std::vector<Sequence> SequenceGroup::GetSeqs() const {
  std::vector<Sequence> seqs;
  seqs.reserve(seqs_dict.size());
  for (auto const& [key, value] : seqs_dict) {
    seqs.push_back(value);
  }
  return seqs;
}

std::vector<Sequence> SequenceGroup::GetSeqs(SequenceStatus status) const {
  std::vector<Sequence> seqs;
  for (const auto& [key, value] : seqs_dict) {
    if (value.status == status) {
      seqs.push_back(value);
    }
  }
  return seqs;
}

bool SequenceGroup::IsPrefill() const { return seqs_dict.at(GetSeqs()[0].seq_id).IsPrefill(); }

bool SequenceGroup::IsFinished() const {
  return std::all_of(seqs_dict.begin(), seqs_dict.end(),
                     [](auto& kv) { return kv.second.IsFinished(); });
}

void SequenceGroup::Add(Sequence& seq) {
  if (seqs_dict.contains(seq.seq_id)) {
    throw std::runtime_error("Sequence already exists in the group");
  }
  seqs_dict.try_emplace(seq.seq_id, seq);
}

void SequenceGroup::MaybeSetFirstScheduledTime(float time) {
  if (metrics.first_scheduled_time == -1) {
    metrics.first_scheduled_time = time;
    metrics.time_in_queue = time - arrival_time;
  }
}
void SequenceGroup::MaybeSetFirstTokenTime(float time) {
  if (metrics.first_token_time == -1 &&
      GetSeqs()[0].data.output_token_ids.size() == 1) {
    metrics.first_token_time = time;
  }
}

void SequenceGroup::UpdateNumComputedTokens(int num_new_computed_tokens) {
  for (auto& [seq_id, seq] : seqs_dict) {
    if (seq.status != SequenceStatus::kFinishedStopped ||
        seq.status != SequenceStatus::kFinishedLengthCapped ||
        seq.status != SequenceStatus::kFinishedAborted ||
        seq.status != SequenceStatus::kFinishedIgnored) {
      seq.data.num_computed_tokens += num_new_computed_tokens;

      if (seq.data.GetNumUncomputedTokens() == 0) {
        seq.data.stage = SequenceStage::kDecode;
      }
    }
  }
}

}  // namespace Generators