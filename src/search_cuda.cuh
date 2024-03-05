namespace Generators {

namespace cuda {

struct ArgMaxData {
  virtual ~ArgMaxData() = default;
};

void Launch_CheckForEOS(int32_t* next_tokens, int next_tokens_count, bool* eos_meet, int eos_token_id, int pad_token_id, bool* done_cpu, cudaStream_t stream);
void LaunchAddProbsKernel(float* log_probs, float* cum_log_probs, const int batch_size, const int num_beams, const int vocab_size, cudaStream_t stream);
void LaunchSetScoreProcessor(float* next_token_scores, int batch_beam_size, int vocab_size, int token, float score, cudaStream_t stream);
void LaunchRepetitionPenaltyProcessor(const int32_t* sequences, float* next_token_scores, int batch_size, int num_beams, int vocab_size, int max_sequence_length, int current_sequence_length, float repetition_penalty, cudaStream_t stream);

void TopPSampling(int32_t* next_token, float* scores, int size, float p, float temperature);
}  // namespace cuda

}  // namespace Generators