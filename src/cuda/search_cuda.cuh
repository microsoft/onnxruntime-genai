// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Generators {

namespace cuda {

struct ArgMaxData {
  virtual ~ArgMaxData() = default;
};

void Launch_CheckForEOSAndPad(int32_t* next_tokens, int next_tokens_count, bool* eos_meet, int eos_token_id, int pad_token_id, bool* done_cpu, cudaStream_t stream);
void Launch_ExpandInputSequences(const std::span<int32_t> input_sequences, std::span<int32_t> sequences, int batch_size, int beam_size, int max_length, cudaStream_t stream);
void Launch_AppendNextTokensToSequences(std::span<const int32_t> next_tokens, std::span<int32_t> sequences, int batch_beam_size, int past_length, int max_length, cudaStream_t stream);
void Launch_GetLastTokens(int32_t* next_tokens, const int32_t* sequences, int batch_beam_size, int sequence_length, int max_length, cudaStream_t stream);

void LaunchAddProbsKernel(float* log_probs, float* cum_log_probs, const int batch_size, const int num_beams, const int vocab_size, cudaStream_t stream);
void LaunchSetScoreProcessor(float* next_token_scores, int batch_beam_size, int vocab_size, int token, float score, cudaStream_t stream);
void LaunchRepetitionPenaltyProcessor(const int32_t* sequences, float* next_token_scores, int batch_size, int num_beams, int vocab_size, int max_sequence_length, int current_sequence_length, float repetition_penalty, cudaStream_t stream);

void TopPSampling(int32_t* next_token, float* scores, int size, float p, float temperature);
}  // namespace cuda

}  // namespace Generators