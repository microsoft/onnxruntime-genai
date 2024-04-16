#pragma once

#include <random>
#include "sequences.h"
#include "search.h"
#include "models/dml_readback_heap.h"

namespace Generators {

struct BeamSearchScorer;

struct Search_Dml : Search {
  Search_Dml(const GeneratorParams& params, DmlExecutionContext* dml_execution_context, DmlReadbackHeap* dml_readback_heap);

  int GetSequenceLength() const override;
  RoamingArray<int32_t> GetSequenceLengths() override { return sequence_lengths_; }
  RoamingArray<int32_t> GetSequence(int index) override { return sequences_.GetSequence(index); }

  bool IsDone() const override { return done_; }
  void SetLogits(RoamingArray<float> logits) override;

  void ApplyMinLength(int min_length) override;
  void ApplyRepetitionPenalty(float penalty) override;

  std::span<float> GetScores(int batch_beam_index) const;
  Sequences& GetSequences() { return sequences_; }

  cpu_span<int32_t> sequence_lengths_;  // shape (beam_size*batch_size)
  std::unique_ptr<int32_t[]> sequence_lengths_buffer_;

  cpu_span<int32_t> next_tokens_;  // shape (beam_size*batch_size)

  std::span<float> next_token_scores_;  // shape (beam_size*batch_size, vocab_size)

  Sequences sequences_;
  bool done_{};

  DmlExecutionContext* dml_execution_context_;
  DmlReadbackHeap* dml_readback_heap_;
};

struct GreedySearch_Dml : Search_Dml {
  GreedySearch_Dml(const GeneratorParams& params, DmlExecutionContext* dml_execution_context, DmlReadbackHeap* dml_readback_heap);

  RoamingArray<int32_t> GetNextTokens() override;
  RoamingArray<int32_t> GetNextIndices() override { return cpu_span<int32_t>{}; }

  void SelectTop() override;
  void SampleTopK(int k, float temperature) override;
  void SampleTopP(float p, float temperature) override;
  void SampleTopKTopP(int /*k*/, float /*p*/, float /*temperature*/) override;

 private:
  bool PadIfAlreadyEOS(size_t batch_id);
  void SetNextToken(size_t batch_id, int32_t token);
  void AppendNextTokensToSequences();

  std::unique_ptr<int32_t[]> next_tokens_buffer_;
  std::unique_ptr<int32_t[]> temp_topk_buffer_;

  std::span<bool> eos_seen_;  // shape (batch_size)
  std::unique_ptr<bool[]> eos_seen_buffer_;
  int not_done_count_{params_->batch_size};  // When zero, every batch entry is done (starts at batch_size_)

  std::random_device rd_;
  std::mt19937 gen_;
};

struct BeamSearch_Dml : Search_Dml {
  BeamSearch_Dml(const GeneratorParams& params, DmlExecutionContext* dml_execution_context, DmlReadbackHeap* dml_readback_heap);
  ~BeamSearch_Dml();

  RoamingArray<int32_t> GetNextTokens() override;
  RoamingArray<int32_t> GetNextIndices() override;

  void SelectTop() override;

  void Finalize(size_t num_return_sequences, RoamingArray<int32_t> output, RoamingArray<float> sequence_scores) override;

 private:
  void AppendNextTokensToSequences();

  std::unique_ptr<BeamSearchScorer> beam_scorer_;
};

}  // namespace Generators