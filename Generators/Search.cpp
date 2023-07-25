#include "Generators.h"

Search::Search(Gpt& model, SearchParams params)
    : model_{model}, params_{params} {
  auto allocator = &Ort::Allocator::GetWithDefaultOptions();
  auto cpu_allocator = allocator;

  int64_t sequences_dims[] = {params_.batch_size, params_.max_length};
  output_sequences_ = OrtValue::CreateTensor<float>(*allocator, sequences_dims, std::size(sequences_dims));

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

  model.CreateInputs(search_state_.sequence_lengths);

  {
    auto shape = model.expanded_input_ids_->GetTensorTypeAndShapeInfo()->GetShape();
    size_t shape_elements = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());

    gsl::span<const int32_t> input_ids{model.expanded_input_ids_->GetTensorMutableData<int32_t>(), shape_elements};
    SetSequence(input_ids);
  }
}

void Search::SetSequence(gsl::span<const int32_t> input_ids_in_cpu) {
  auto batch_beam_size = params_.BatchBeamSize();
  gsl::span<int32_t> sequences_0 = search_state_.sequences_space;
  for (size_t i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < params_.sequence_length; j++) {
      sequences_0[SafeInt<gsl::index>(i) * params_.max_length + j] =
          static_cast<int32_t>(input_ids_in_cpu[SafeInt<gsl::index>(i) * params_.sequence_length + j]);
    }
  }
}

void Search::Run() {
  model_.Run();

  // Generate Next Token
  ProcessLogits();

  gsl::span<int32_t> next_tokens = search_state_.next_tokens;
  gsl::span<bool> eos_meet = search_state_.eos_meet;
  for (size_t batch_id = 0; batch_id < next_tokens.size(); ++batch_id) {
    if (next_tokens[batch_id] == params_.eos_token_id || eos_meet[batch_id] == true) {
      eos_meet[batch_id] = true;
      next_tokens[batch_id] = params_.pad_token_id;
    }
  }

  sequences_.AppendNextTokenToSequences(next_tokens);

  // When all batches are finished, stop earlier to avoid wasting computation.
  {
    gsl::span<bool> eos_meet = search_state_.eos_meet;
    size_t batch_id = 0;
    while (batch_id < eos_meet.size()) {
      if (eos_meet[batch_id] == false) {
        break;
      }
      ++batch_id;
    }
    if (batch_id == eos_meet.size()) {
      done_=true;
      return;
    }
  }

  if (sequences_.GetSequenceLength()==params_.max_length)
    done_=true;
}

void Search::ProcessLogits() {

}

void Search::PrepareNextStep() {

}


void Search::Finalize() {

  auto shape=output_sequences_->GetTensorTypeAndShapeInfo()->GetShape();
  size_t shape_count = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());

  // Copy the sequences to output
  gsl::span<int32_t> output{ output_sequences_->GetTensorMutableData<int32_t>(), shape_count};
  for (int batch_id = 0; batch_id < params_.batch_size; ++batch_id) {
    auto batch_output = output.subspan(
        static_cast<size_t>(batch_id) * params_.max_length,
        params_.max_length);
    gsl::span<const int32_t> sequence_source = sequences_.GetSequence(batch_id);
    gsl::copy(sequence_source, batch_output);
  }
}