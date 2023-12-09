namespace Generators {

namespace cuda {

struct ArgMaxData {
  virtual ~ArgMaxData()=default;
};

void Launch_ArgMax(std::unique_ptr<ArgMaxData>& data, int32_t* next_tokens, const ScoreType* next_token_scores, int batch_size, int vocab_size, cudaStream_t stream);
void Launch_CheckForEOS(int32_t* next_tokens, int next_tokens_count, bool* eos_meet, int eos_token_id, int pad_token_id, bool* done_cpu, cudaStream_t stream);
void LaunchAddProbsKernel(ScoreType* log_probs, ScoreType* cum_log_probs, const int batch_size, const int num_beams, const int vocab_size, cudaStream_t stream);
void LaunchRepetitionPenaltyProcessor(const int32_t* sequences, ScoreType* next_token_scores, int batch_size, int num_beams, int vocab_size, int max_sequence_length, int current_sequence_length, ScoreType repetition_penalty, cudaStream_t stream);
void Launch_log_softmax(ScoreType* values, int count, cudaStream_t stream);

void TopPSampling(int32_t* next_token, ScoreType* scores, int size, float p, float temperature);
}  // namespace cuda

} // namespace Generators