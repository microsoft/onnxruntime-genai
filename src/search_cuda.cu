#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include "generators.h"
#include "search_cuda.cuh"

namespace Generators {
namespace cuda {

#if 0
__global__ void SetInputSequence(int32_t* sequences, const int32_t* input_sequences, int batch_size, int num_beams) {
  // The original inputs are not expanded, this expands them in place into the sequences
std::span<int32_t> sequences_0 = sequences_space_;
for (size_t batch = 0; batch < params_.batch_size; batch++) {
  for (size_t beam = 0; beam < params_.num_beams; beam++) {
    for (int j = 0; j < params_.sequence_length; j++) {
      sequences_0[(batch * params_.num_beams + beam) * params_.max_length + j] =
          static_cast<int32_t>(params_.input_ids[batch * params_.sequence_length + j]);
    }
  }
}

void LaunchSetInputSequence(std::span<int32_t> sequences) {

}
#endif

__global__ void ArgMax(cub::KeyValuePair<int, float>* argmaxen, int32_t* next_tokens, int batch_size) {
  int batch_index = threadIdx.x;
  next_tokens[batch_index] = argmaxen[batch_index].key;
}

struct ArgMaxDataImpl : ArgMaxData {
  cuda_unique_ptr<uint8_t> temp_storage_;
  size_t temp_storage_element_size_{};  // Size per batch, temp_storage_ is this size * batch_size

  gpu_span<cub::KeyValuePair<int, float>> argmaxen_;
  cuda_unique_ptr<cub::KeyValuePair<int, float>> argmaxen_owner_;
};

void Launch_ArgMax(std::unique_ptr<ArgMaxData>& p_data, int32_t* next_tokens, const float* next_token_scores, int batch_size, int vocab_size, cudaStream_t stream) {
  if (!p_data)
    p_data = std::make_unique<ArgMaxDataImpl>();
  auto& data = static_cast<ArgMaxDataImpl&>(*p_data);

  if (!data.temp_storage_) {
    data.argmaxen_owner_ = CudaMallocArray<cub::KeyValuePair<int, float>>(batch_size, &data.argmaxen_);
    CudaCheck() == cub::DeviceReduce::ArgMax(data.temp_storage_.get(), data.temp_storage_element_size_, next_token_scores, &data.argmaxen_[0], vocab_size, stream);
    data.temp_storage_ = CudaMallocArray<uint8_t>(data.temp_storage_element_size_ * batch_size);
  }

  for (int batch_index = 0; batch_index < batch_size; batch_index++)
    CudaCheck() == cub::DeviceReduce::ArgMax(data.temp_storage_.get() + data.temp_storage_element_size_ * batch_index, data.temp_storage_element_size_, next_token_scores + batch_index * vocab_size, &data.argmaxen_[batch_index], vocab_size, stream);
  ArgMax<<<1, batch_size, 0, stream>>>(data.argmaxen_.data(), next_tokens, batch_size);
}

__global__ void log_softmax(float* values, int count) {
  float max = *std::max_element(values, values + count);
  //  std::vector<float> scaled(values.begin(), values.end());
  float sum = 0.0f;
  for (int i = 0; i < count; i++)
    sum += std::exp(values[i] - max);

  float log_max = std::log(sum);
  // std::transform(values, values+count, values, [max, log_max](float v) { return v - max - log_max; });
}

void Launch_log_softmax(float* values, int count, cudaStream_t stream) {
  log_softmax<<<1, 1, 0, stream>>>(values, count);
}

__global__ void CheckForEOS(int32_t* next_tokens, int next_tokens_count, bool* eos_meet, int eos_token_id, int pad_token_id, bool* done_cpu) {
  // Look for EOS tokens, if seen set EOS flag and replace with pad token
  for (size_t batch_id = 0; batch_id < next_tokens_count; ++batch_id) {
    if (next_tokens[batch_id] == eos_token_id || eos_meet[batch_id] == true) {
      eos_meet[batch_id] = true;
      next_tokens[batch_id] = pad_token_id;
    }
  }

  // When all batches are finished, stop earlier to avoid wasting computation.
  // TODO: Merge this with the above so we don't have to double scan. Just keep track of 'batches left'
  {
    size_t batch_id = 0;
    while (batch_id < next_tokens_count) {
      if (eos_meet[batch_id] == false) {
        break;
      }
      ++batch_id;
    }
    if (batch_id == next_tokens_count) {
      *done_cpu = true;
      return;
    }
  }
}

void Launch_CheckForEOS(int32_t* next_tokens, int next_tokens_count, bool* eos_meet, int eos_token_id, int pad_token_id, bool* done_cpu, cudaStream_t stream) {
  CheckForEOS<<<1, 1, 0, stream>>>(next_tokens, next_tokens_count, eos_meet, eos_token_id, pad_token_id, done_cpu);
}

__global__ void AddProbsKernel(float* log_probs,
                               float* cum_log_probs,
                               const int vocab_size,
                               const int total_elements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_beam_index = index / vocab_size;

  if (index < total_elements)
    log_probs[index] += cum_log_probs[batch_beam_index];
}

void LaunchAddProbsKernel(float* log_probs,
                          float* cum_log_probs,
                          const int batch_size,
                          const int num_beams,
                          const int vocab_size,
                          cudaStream_t stream) {
  int total_elements = batch_size * num_beams * vocab_size;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;
  AddProbsKernel<<<gridSize, blockSize, 0, stream>>>(log_probs, cum_log_probs, vocab_size, total_elements);
}

__global__ void RepetitionPenaltyProcessor(const int32_t* sequences, float* next_token_scores, int max_sequence_length, int vocab_size, int total_elements, int current_sequence_length, float repetition_penalty) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total_elements)
    return;

  int batch_beam_index = index / vocab_size;
  int word_id = index % vocab_size;

  const int32_t* current_sequence = sequences + batch_beam_index * max_sequence_length;
  bool found = false;
  for (int i = 0; i < current_sequence_length; i++) {
    if (current_sequence[i] == word_id) {
      found = true;
      break;
    }
  }
  if (found) {
    float score = next_token_scores[index];
    next_token_scores[index] = score < 0 ? score * repetition_penalty : score / repetition_penalty;
  }
}

void LaunchRepetitionPenaltyProcessor(const int32_t* sequences, float* next_token_scores, int batch_size, int num_beams, int vocab_size, int max_sequence_length, int current_sequence_length, float repetition_penalty, cudaStream_t stream) {
  int total_elements = batch_size * num_beams * vocab_size;
  constexpr int blockSize = 256;
  const int gridSize = (total_elements + blockSize - 1) / blockSize;

  RepetitionPenaltyProcessor<<<gridSize, blockSize, 0, stream>>>(sequences, next_token_scores, max_sequence_length, vocab_size, total_elements, current_sequence_length, repetition_penalty);
}

}  // namespace cuda
}  // namespace Generators
