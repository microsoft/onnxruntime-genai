#include "Generators.h"


Search::Search(SearchParams params)
 : params_{params} {

  auto allocator = &Ort::Allocator::GetWithDefaultOptions();
  auto cpu_allocator = allocator;

  int64_t sequences_dims[] = {params_.batch_size, params_.max_length};
  output_sequences = OrtValue::CreateTensor<float>(*allocator, sequences_dims, std::size(sequences_dims));

  // below buffers are on cpu
  search_state_.sequences_space = AllocateBuffer<int32_t>(cpu_allocator,
                                            sequences_space_buffer_,
                                            SafeInt<size_t>(2) * params_.batch_size * params_.max_length);
  memset(search_state_.sequences_space.data(), 0, search_state_.sequences_space.size_bytes());
  sequences_.Init(search_state_.sequences_space, static_cast<int>(params_.batch_size), params_.sequence_length, params_.max_length);

  search_state_.sequence_lengths = AllocateBuffer<int32_t>(cpu_allocator, sequence_lengths_buffer_, params_.batch_size);
  search_state_.eos_meet = AllocateBuffer<bool>(cpu_allocator, eos_meet_buffer_, params_.batch_size);
  memset(search_state_.eos_meet.data(), 0, search_state_.eos_meet.size_bytes());

  search_state_.next_tokens = AllocateBuffer<int32_t>(cpu_allocator, next_tokens_buffer_, SafeInt<size_t>(params_.batch_size));

  // below buffers are on cpu or cuda
  size_t next_token_size = SafeInt<size_t>(params_.batch_size) * params_.vocab_size;
  search_state_.next_token_scores = AllocateBuffer<float>(allocator, next_token_scores_buffer_, next_token_size);
  search_state_.next_positions = AllocateBuffer<int32_t>(allocator, next_positions_buffer_, params_.batch_size);
  
}
