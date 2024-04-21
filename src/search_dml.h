#pragma once
#include "sequences_cuda.h"

namespace Generators {

struct Search_Dml : Search {
  Search_Dml(const GeneratorParams& params, ID3D12Device* d3d12_device);

  int GetSequenceLength() const override;
  RoamingArray<int32_t> GetSequenceLengths() override { return sequence_lengths_; }
  RoamingArray<int32_t> GetSequence(int index) override { return sequences_.GetSequence(index); }

  void SetLogits(RoamingArray<float> logits);

  void ApplyMinLength(int min_length) override;
  void ApplyRepetitionPenalty(float penalty) override;

  Sequences_Cuda& GetSequences() { return sequences_; }

  cpu_span<int32_t> sequence_lengths_;  // shape (beam_size*batch_size)
  std::unique_ptr<int32_t[]> sequence_lengths_buffer_;

  gpu_span<bool> eos_meet_;  // shape (beam_size*batch_size)
  ComPtr<ID3D12Resource> eos_meet_buffer_;
  ComPtr<ID3D12Resource> done_cpu_;
  ComPtr<ID3D12Device> d3d12_device_;

  gpu_span<int32_t> next_tokens_;  // shape (beam_size*batch_size)

  ComPtr<ID3D12Resource> next_token_scores_buffer_;  // shape (beam_size*batch_size, vocab_size)

  Sequences_Cuda sequences_;
};

struct GreedySearch_Dml : Search_Dml {
  GreedySearch_Dml(const GeneratorParams& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, DmlExecutionContext* execution_context, const OrtDmlApi* ort_dml_api);

  RoamingArray<int32_t> GetNextTokens() override;
  RoamingArray<int32_t> GetNextIndices() override { return RoamingArray<int32_t>(); }

  void SelectTop() override;
  void SampleTopK(int k, float t) override;
  void SampleTopP(float p, float t) override;
  void SampleTopKTopP(int k, float p, float t) override;

 private:
  void CheckForEOS();
  void AppendNextTokensToSequences();

  ComPtr<ID3D12Resource> next_tokens_buffer_;
  ComPtr<IDMLDevice> dml_device_;
  DmlExecutionContext* execution_context_;
  const OrtDmlApi* ort_dml_api_;
};

namespace Processors_Dml {
void MinLength(Search_Dml& search, int min_length);
void RepetitionPenalty(Search_Dml& search, float penalty);
}  // namespace Processors_Dml

}  // namespace Generators