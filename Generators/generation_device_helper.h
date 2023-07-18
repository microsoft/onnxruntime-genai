// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if 0
#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/framework/allocator.h"
#endif

#include <vector>
#include "core/common/gsl.h"
#include "contrib_ops/cpu/transformers/logits_processor.h"
#include "contrib_ops/cpu/transformers/generation_shared.h"
#endif

enum DeviceCopyDirection {
  hostToHost = 0,
  hostToDevice = 1,
  deviceToHost = 2,
  deviceToDevice = 3
};

namespace GenerationDeviceHelper {

#ifdef USE_CUDA
using ReorderPastStateFunc = std::function<void(
    const void* cuda_device_prop,  // cudaDeviceProp
    Tensor& past_state,
    Tensor& past_state_staging,
    Stream* stream)>;  // cublasHandle_t

using InitCacheIndirFunc = std::function<void(
    Tensor& cache_indir,
    Stream* stream)>;
#endif

using TopkFunc = std::function<void(
    const OrtValue* input, const int axis, const unsigned k, bool largest, bool sorted,
    OrtAllocator* allocator,
    Stream* stream,  // cudaStream_t
    OrtValue& output_values,
    OrtValue& output_indices)>;

// Create subgraph inputs: input_ids, position_ids and attention_mask (for GPT-2).
using CreateGptInputsFunc = std::function<void(
    const Tensor* original_input_ids,
    const OrtValue* attn_mask_value,
    int num_beams,
    int pad_token_id,
    gsl::span<int32_t>& sequence_lengths,
    OrtAllocator* allocator,
    OrtValue& expanded_input_ids,
    OrtValue& expanded_position_ids,
    OrtValue& expanded_attention_mask)>;

using AddToFeedsFunc = std::function<void(
    Stream* ort_stream,
    std::initializer_list<OrtValue> inputs,
    std::vector<OrtValue>& feeds,
    IAllocatorUniquePtr<char>& buffer,
    OrtAllocator* device_allocator,
    OrtAllocator* host_allocator,
    const OrtMemoryInfo& location)>;

template <typename T>
using InitBeamStateFunc = std::function<void(
    IBeamSearchState<T>* beam_state,
    gsl::span<int32_t>& sequence_lengths,
    int batch_size,
    int num_beams,
    Stream* stream)>;

using CreateBeamScorer = std::function<std::unique_ptr<IBeamScorer>(
    const IGenerationParameters& parameters,
    OrtAllocator*& allocator,
    OrtAllocator*& allocator_cpu,
    Stream* stream)>;

template <typename T>
using InitGreedyStateFunc = std::function<void(
    IGreedySearchState<T>* greedy_state,
    gsl::span<int32_t>& sequence_lengths,
    Stream* stream)>;

template <typename T>
using ProcessLogitsFunc = std::function<void(
    const OrtValue& logits,                                 // logits output of subgraph
    IBeamSearchState<T>* beam_state,          // state
    ISequences* sequences,                    // sequences
    OrtAllocator*& allocator,                                // default allocator
    ILogitsProcessorList* logits_processors,  // logits processors
    IBeamScorer* beam_scorer,                 // beam scorer
    const IGenerationParameters* parameters,  // parameters
    int step,                                               // iteration counter
    Stream* stream,                                         // cuda stream (for CUDA only)
    const IConsoleDumper* dumper)>;           // tensor dumper

template <typename T>
using GreedySearchProcessLogitsFunc = std::function<void(
    const OrtValue& logits,                                 // logits output of subgraph
    IGreedySearchState<T>* greedy_state,      // state
    ISamplingState<T>* sampling_state,        // sampling buffers
    ISequences* sequences,                    // sequences
    OrtAllocator*& allocator,                                // default allocator
    ILogitsProcessorList* logits_processors,  // logits processors
    const IGenerationParameters* parameters,  // parameters
    bool do_sampling,                                       // whether to do sampling
    int step,                                               // iteration counter
    Stream* ort_stream,                                     // cuda stream (for CUDA only)
    const IConsoleDumper* dumper)>;           // tensor dumper

template <typename T>
using DeviceCopyFunc = std::function<void(
    gsl::span<T> target,
    gsl::span<const T> source,
    Stream* stream,
    int copyDirection)>;

// Update subgraph inputs given outputs of last iteration (for GPT-2).
template <typename T>
using UpdateGptFeedsFunc = std::function<void(
    OrtAllocator* allocator,
    Stream* stream,
    const std::vector<OrtValue*>& last_outputs,
    std::vector<OrtValue*>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    bool increase_position,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices_cpu,
    gsl::span<const int32_t> beam_indices_gpu,
    int num_beams,
    int gpt_subgraph_first_past_input_idx,
    int gpt_subgraph_first_present_output_idx,
    bool past_present_share_buffer,
    int past_sequence_len,
    int input_sequence_len,
    bool need_cache_indir)>;

// Create encoder inputs (for encoder-decoder model like T5).
using CreateEncoderInputsFunc = std::function<void(
    const OrtValue* original_encoder_input_ids,
    const OrtValue* attn_mask_value,
    int pad_token_id,
    int start_token_id,
    OrtAllocator* allocator,
    OrtValue& encoder_input_ids,
    OrtValue& encoder_attention_mask,
    OrtValue& decoder_input_ids)>;

// Update decoder inputs given decoder outputs of last iteration (for encoder-decoder model like T5).
template <typename T>
using UpdateDecoderFeedsFunc = std::function<void(
    OrtAllocator* allocator,
    Stream* stream,
    const std::vector<OrtValue*>& last_outputs,
    std::vector<OrtValue*>& next_inputs,
    int num_present_tensors,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    gsl::span<const int32_t> beam_indices_gpu,
    int num_beams,
    int t5_decoder_first_past_input_idx,
    int t5_decoder_first_present_output_idx,
    bool use_sequence_as_input_ids,
    int current_length,
    int input_sequence_len,
    bool past_present_share_buffer,
    bool need_cache_indir,
    Sequences& sequences,
    const IConsoleDumper* dumper)>;

//------------------------------------------------
//  Modified functions for Whisper Model
//------------------------------------------------
using CreateWhisperEncoderInputsFunc = std::function<void(
    const OrtValue* original_encoder_input_features,
    const OrtValue* original_decoder_input_ids_value,
    int start_token_id,
    OrtAllocator* allocator,
    OrtValue& encoder_input_ids,
    OrtValue& decoder_input_ids)>;

template <typename T>
using ExpandBufferFunc = std::function<void(
    Stream* stream,
    const OrtValue& input,
    int num_beams,
    OrtAllocator* allocator,
    OrtValue& expanded,
    bool only_copy_shape,
    int max_sequence_length)>;
}  // namespace GenerationDeviceHelper

// These are CPU specific device helper implementations
namespace GenerationCpuDeviceHelper {
void TopK(
    const OrtValue* input, const int axis, const unsigned k, bool largest, bool sorted,
    OrtAllocator* allocator,
    Stream* stream,
    OrtValue& output_values,
    OrtValue& output_indices);

void AddToFeeds(
    Stream* ort_stream,
    std::initializer_list<OrtValue> inputs,
    std::vector<OrtValue>& feeds,
    IAllocatorUniquePtr<char>& buffer,
    OrtAllocator* device_allocator,
    OrtAllocator* host_allocator,
    const OrtMemoryInfo& location);

template <typename T>
void InitBeamState(IBeamSearchState<T>* beam_state,
                   gsl::span<int32_t>& sequence_lengths,
                   int batch_size,
                   int num_beams,
                   Stream* stream);

template <typename T>
void InitGreedyState(IGreedySearchState<T>* greedy_state,
                     gsl::span<int32_t>& sequence_lengths,
                     Stream* ort_stream);

template <typename T>
void ProcessLogits(const OrtValue& logits,                                 // logits output of subgraph
                     IBeamSearchState<T>* beam_state,          // state
                     ISequences* sequences,                    // sequences
                     OrtAllocator*& allocator,                                // default allocator
                     ILogitsProcessorList* logits_processors,  // logits processors
                     IBeamScorer* beam_scorer,                 // beam scorer
                     const IGenerationParameters* parameters,  // parameters
                     int step,                                               // iteration counter
                     Stream* stream,                                         // cuda stream (for CUDA only)
                     const IConsoleDumper* dumper);            // tensor dumper

template <typename T>
void GreedySearchProcessLogits(const OrtValue& logits,                                 // logits output of subgraph
                                 IGreedySearchState<T>* greedy_state,      // state
                                 ISamplingState<T>* sampling_state,        // sampling buffers
                                 ISequences* sequences,                    // sequences
                                 OrtAllocator*& allocator,                                // default allocator
                                 ILogitsProcessorList* logits_processors,  // logits processors
                                 const IGenerationParameters* parameters,  // parameters
                                 bool do_sampling,                                       // whether to do sampling
                                 int step,                                               // iteration counter
                                 Stream* stream,                                         // cuda stream (for CUDA only)
                                 const IConsoleDumper* dumper);            // tensor dumper

template <typename T>
void DeviceCopy(gsl::span<T> target,
                  gsl::span<const T> source,
                  Stream* stream,
                  int copyDirectionn);

// ---------------------------------------------------------------
// Functions for GPT model only
// ---------------------------------------------------------------

void CreateGptInputs(
    const OrtValue* original_input_ids,
    const OrtValue* attn_mask_value,
    int num_beams,
    int pad_token_id,
    gsl::span<int32_t>& sequence_lengths,
    OrtAllocator* allocator,
    OrtValue& expanded_input_ids,
    OrtValue& expanded_position_ids,
    OrtValue& expanded_attention_mask);

template <typename T>
void UpdateGptFeeds(
    OrtAllocator* allocator,
    Stream* stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    bool increase_position,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices_cpu,
    gsl::span<const int32_t> beam_indices_gpu,
    int num_beams,
    int gpt_subgraph_first_past_input_idx,
    int gpt_subgraph_first_present_output_idx,
    bool past_present_share_buffer,
    int past_sequence_len,
    int input_sequence_len,
    bool need_cache_indir);

// ---------------------------------------------------------------
// Functions for encoder-decoder model like T5
// ---------------------------------------------------------------
void CreateEncoderInputs(
    const Tensor* original_encoder_input_ids,
    const OrtValue* attn_mask_value,
    int pad_token_id,
    int start_token_id,
    OrtAllocator* allocator,
    OrtValue& encoder_input_ids,
    OrtValue& encoder_attention_mask,
    OrtValue& decoder_input_ids);

// Update decoder inputs given decoder outputs of last iteration.
template <typename T>
void UpdateDecoderFeeds(
    OrtAllocator* allocator,
    Stream* stream,
    const std::vector<OrtValue*>& last_outputs,
    std::vector<OrtValue*>& next_inputs,
    int num_present_tensors,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    gsl::span<const int32_t> beam_indices_gpu,
    int num_beams,
    int t5_decoder_first_past_input_idx,
    int t5_decoder_first_present_output_idx,
    bool use_sequence_as_input_ids,
    int current_length,
    int input_sequence_len,
    bool past_present_share_buffer,
    bool need_cache_indir,
    Sequences& sequences,
    const IConsoleDumper* dumper);

// ---------------------------------------------------------------
// Functions for encoder-decoder model with float input like Whisper
// ---------------------------------------------------------------
template <typename T>
void CreateWhisperEncoderInputs(
    const OrtValue* original_encoder_input_features,
    const OrtValue* original_decoder_input_ids_value,
    int start_token_id,
    OrtAllocator* allocator,
    OrtValue& encoder_input_ids,
    OrtValue& decoder_input_ids);

// ---------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------
template <typename T>
void ExpandInputs(const OrtValue& input, int num_beams, OrtAllocator* allocator, OrtValue& expanded);

template <typename T>
void ExpandBuffer(
    Stream* stream,
    const OrtValue& input,
    int num_beams,
    OrtAllocator* allocator,
    OrtValue& expanded,
    bool only_copy_shape,
    int max_sequence_length);

}  // namespace GenerationCpuDeviceHelper
