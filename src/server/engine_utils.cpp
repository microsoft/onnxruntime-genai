
#include "engine_utils.h"
#include <utility>
#include <vector>

namespace Generators {

bool IsFinished(SequenceStatus status) {
  switch (status) {
    case kFinishedIgnored:
      return true;
    case kFinishedAborted:
      return true;
    case kFinishedLengthCapped:
      return true;
    case kFinishedStopped:
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
  if (data.stage == kDecode) {
    return 1;
  }
  return data.GetNumUncomputedTokens();
}

int Sequence::GetLen() const { return data.GetLen(); }

bool Sequence::IsPrefill() const { return data.stage == kPrefill; }

bool Sequence::IsFinished() const { return engine::IsFinished(status); }

void Sequence::SetStatus(SequenceStatus new_status) { status = new_status; }

void Sequence::ResetStateForRecompute() { data.ResetStateForRecompute(); }

SequenceGroup::SequenceGroup(std::string request_id, std::vector<Sequence> seqs,
                             float arrival_time, SamplingParams sampling_params,
                             std::vector<float> embeddings,
                             std::unique_ptr<Sequence> encoder_seq)
    : request_id(request_id),
      arrival_time(arrival_time),
      sampling_params(sampling_params),
      embeddings(embeddings),
      encoder_seq(std::move(encoder_seq)) {
  for (auto& seq : seqs) {
    seqs_dict[seq.seq_id] = std::move(seq);
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
  for (auto& seq : GetSeqs()) {
    if (!seq.IsFinished()) {
      num_unfinished++;
    }
  }
  return num_unfinished;
}

std::vector<Sequence> SequenceGroup::GetSeqs() {
  std::vector<Sequence> seqs;
  seqs.reserve(seqs_dict.size());
  for (auto& kv : seqs_dict) {
    seqs.push_back(kv.second);
  }
  return seqs;
}

std::vector<Sequence> SequenceGroup::GetSeqs(SequenceStatus status) {
  std::vector<Sequence> seqs;
  for (auto& kv : seqs_dict) {
    if (kv.second.status == status) {
      seqs.push_back(kv.second);
    }
  }
  return seqs;
}

bool SequenceGroup::IsPrefill() { return GetSeqs()[0].IsPrefill(); }

void SequenceGroup::MaybeSetFirstScheduledTime(float time) {
  if (metrics.first_scheduled_time == -1) {
    metrics.first_scheduled_time = time;
    metrics.time_in_queue = time - arrival_time;
  }
}
}  // namespace engine