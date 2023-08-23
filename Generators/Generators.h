// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <assert.h>
#include <functional>
#include <gsl/gsl>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <unordered_set>
#include <vector>

#include "span_utils.h"
#include "SafeInt.hpp"
#include "onnxruntime_cxx_api_2.h"
#include "TensorShape.h"
#include "debugging.h"

using ScoreType = float;

#if USE_CUDA
#include <cuda_runtime.h>

struct CudaDeleter {
  void operator()(void* p) {
    cudaFree(p);
  }
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, CudaDeleter>;

template <typename T>
cuda_unique_ptr<T> CudaMallocArray(size_t count) {
  T* p;
  cudaMallocManaged(&p, sizeof(T) * count);
      //  cudaMalloc(&p, sizeof(T) * count);
  return cuda_unique_ptr<T>{p};
}

struct CudaHostDeleter {
  void operator()(void* p) {
    cudaFreeHost(p);
  }
};

template <typename T>
using cuda_host_unique_ptr = std::unique_ptr<T, CudaHostDeleter>;

template <typename T>
cuda_host_unique_ptr<T> CudaMallocHostArray(size_t count) {
  T* p;
  cudaMallocHost(&p, sizeof(T) * count);
  return cuda_host_unique_ptr<T>{p};
}

struct cuda_event_holder {
  cuda_event_holder() {
    cudaEventCreate(&v_);
  }

  cuda_event_holder(unsigned flags) {
    cudaEventCreateWithFlags(&v_, flags);
  }

  ~cuda_event_holder() {
    if (v_)
      (void)cudaEventDestroy(v_);
  }

  operator cudaEvent_t() { return v_; }

private:
  cudaEvent_t v_{};
};

#endif


using gsl::narrow;

struct Tensor;
struct Stream;
struct IConsoleDumper;

struct OpKernelContextInternal {};
struct SessionState {};
struct NodeArg {};
struct Node {};
struct TensorShapeProto {};

// Macros to disable the copy and/or move ctor and assignment methods
// These are usually placed in the private: declarations for a class.

#define ORT_DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete

#define ORT_DISALLOW_ASSIGNMENT(TypeName) TypeName& operator=(const TypeName&) = delete

#define ORT_DISALLOW_COPY_AND_ASSIGNMENT(TypeName) \
  ORT_DISALLOW_COPY(TypeName);                     \
  ORT_DISALLOW_ASSIGNMENT(TypeName)

#define ORT_DISALLOW_MOVE(TypeName) \
  TypeName(TypeName&&) = delete;    \
  TypeName& operator=(TypeName&&) = delete

#define ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TypeName) \
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(TypeName);           \
  ORT_DISALLOW_MOVE(TypeName)

#define ORT_ENFORCE(condition, ...) assert(condition)

// TODO: Do we need this class or is IAllocator::MakeUniquePtr sufficient/better
struct  BufferDeleter {
  BufferDeleter() = default;
  explicit BufferDeleter(OrtAllocator* alloc)
      : alloc_(alloc) {}

  void operator()(void* p) const {
    if (alloc_)
      alloc_->Free(alloc_, p);
  }

 private:
  // TODO: we may need consider the lifetime of alloc carefully
  // The alloc_ here is the allocator that used to allocate the buffer
  // And need go with the unique_ptr together. If it is using our internal
  // allocator, it is ok as our allocators are global managed. But if it
  // is provide by user, user need to be very careful about it.
  // A weak_ptr may be a choice to reduce the impact, but that require to
  // change our current allocator mgr to use shared_ptr. Will revisit it
  // later.
  OrtAllocator* alloc_{};
};

using BufferUniquePtr = std::unique_ptr<void, BufferDeleter>;
using BufferNakedPtr = void*;

template <typename T>
gsl::span<T> AllocateBuffer(OrtAllocator* allocator,
                            BufferUniquePtr& buffer,
                            size_t elements,
                            bool fill = false,
                            T fill_value = T{}) {
  size_t bytes = SafeInt<size_t>(sizeof(T)) * elements;
  void* data = allocator->Alloc(allocator, bytes);
  BufferUniquePtr temp_buffer(data, BufferDeleter(allocator));
  buffer = std::move(temp_buffer);
  T* first = reinterpret_cast<T*>(buffer.get());
  auto span = gsl::make_span(first, elements);

  if (fill) {
    std::fill_n(first, elements, fill_value);
  }

  return span;
}

template <typename T>
using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

/** Allocate a unique_ptr using allocator_, and return a span to the allocated memory so usage is safe
@param allocator IAllocator to use for the allocation.
@param size Allocation size. Number of elements of type TAlloc, or total size if TAlloc is 'void'.
@param unique_ptr unique_ptr that will control the lifetime of the allocated memory.
@param fill If true, fill the allocated memory with fill_value.
@param fill_value Value to use if 'fill' is true.
@returns A span to provide bounds checked access to the allocated memory.
*/
template <typename TAlloc>
gsl::span<TAlloc> Allocate(OrtAllocator& allocator,
                           size_t size,
                           IAllocatorUniquePtr<TAlloc>& unique_ptr,
                           bool fill = false, TAlloc fill_value = TAlloc{}) {
  unique_ptr = IAllocatorUniquePtr<TAlloc>(reinterpret_cast<TAlloc*>(allocator.Alloc(&allocator, size * sizeof(TAlloc))), [&allocator](void *p) { allocator.Free(&allocator, p);} );
  auto span = gsl::make_span(unique_ptr.get(), size);

  if (fill) {
    // Do't use span.begin() it will cause performance issue and stop compiler to optimize the code
    std::fill_n(unique_ptr.get(), size, fill_value);
  }

  return span;
}

template <typename T>
struct IBeamSearchState {
  gsl::span<T> next_token_logits;      // shape (batch_size * num_beams, vocab_size)
  gsl::span<float> next_token_scores;  // shape (batch_size, num_beams * vocab_size)
  gsl::span<int32_t> next_tokens;      // shape (batch_size, 2 * num_beams)
  gsl::span<int32_t> next_indices;     // shape (batch_size, 2 * num_beams)
  gsl::span<float> next_scores;        // shape (batch_size, 2 * num_beams)
  gsl::span<int32_t> next_positions;   // shape (batch_size, num_beams), empty for T5. Next position for position_ids.
  gsl::span<float> beam_scores;        // shape (batch_size, num_beams)
  gsl::span<float> scores;             // shape (max_length - sequence_length + 1, batch_size, num_beams * vocab_size)
  gsl::span<float> remaining_scores;   // portion of scores that is available for appending next token scores.
  gsl::span<float> topk_buffer;        // temp buffer for topk computation, including:
                                       // 1st stage needs:
                                       //   temp score: (batch_size * num_beams * parts_vocab, 2 * num_beams)
                                       //   temp token: (batch_size * num_beams * parts_vocab, 2 * num_beams)
                                       // 2nd stage needs:
                                       //   temp score: (batch_size * num_beams, 2 * num_beams)
                                       //   temp token: (batch_size * num_beams, 2 * num_beams)
                                       // in total, it will be:
                                       // 2 * (batch_size * num_beams * (parts_vocab + 1), 2 * num_beams)

  gsl::span<int32_t> sequences_device;  // shape (2 * batch_size * max_length)

  Tensor staging_for_past_state_reorder;  // Tensor of shape (batch_size * num_beams, num_heads, max_length, head_size)
};

struct IBeamSearchCpuState {
  gsl::span<int32_t> sequences_space;   // shape (2, batch_size, num_beams, max_seq_length)
  gsl::span<int32_t> sequence_lengths;  // shape (batch_size, num_beams), initial sequence length

  // The following are used only by CUDA operator for data copied from device.
  gsl::span<float> topk_scores;        // shape (batch_size, 2*num_beams), scores of topk candidates (K=2*num_beams).
  gsl::span<int32_t> topk_tokens;      // shape (batch_size, 2*num_beams), tokens of topk candidates.
  gsl::span<int32_t> topk_indices;     // shape (batch_size, 2*num_beams), beam indices of topk candidates.
  gsl::span<float> final_beam_scores;  // shape (batch_size, num_beams)
};

struct IGreedySearchState {
  gsl::span<int32_t> sequences_space;          // shape (2, batch_size, max_length)
  gsl::span<int32_t> sequence_lengths;         // shape (batch_size)
  gsl::span<int32_t> next_positions;           // shape (batch_size, num_beams). Next position value for position_ids.
  gsl::span<bool> eos_meet;                    // shape (batch_size)
  gsl::span<ScoreType> next_token_scores;      // shape (batch_size, vocab_size)
  gsl::span<int32_t> next_tokens;              // shape (batch_size)
//  std::unique_ptr<OrtValue> staging_for_past_state_reorder;       // Tensor of shape (batch_size * num_beams(1), num_heads, max_length, head_size)
};

template <typename T>
struct ISamplingState {
  gsl::span<int> d_index_in;
  gsl::span<int> d_index_out;
  gsl::span<int> d_offset;
  gsl::span<T> d_sorted_score;
  gsl::span<float> d_sorted_softmaxed_score;
  gsl::span<float> d_softmaxed_score;
  gsl::span<float> h_softmaxed_score;
  gsl::span<float> d_sampled;
  gsl::span<float> h_sampled_all;
  gsl::span<int32_t> d_indices;
  gsl::span<int> d_presence_mask;

  BufferUniquePtr storage_buffer;
  size_t temp_storage_bytes;
  std::default_random_engine generator;

  gsl::span<T> sorted_scores;
  gsl::span<T> cumulative_probs;
};

struct ISequences {
  virtual ~ISequences() {}
  virtual gsl::span<const int32_t> GetSequence(int beam_index) const = 0;
  virtual gsl::span<const int32_t> GetCurrentDeviceSequences() const = 0;  // Get all current beam_index sequences in one continuous block (to pass to CUDA)
  virtual gsl::span<int32_t> GetNextDeviceSequences() = 0;                 // Get all next beam_index sequences in one continuous block (to pass to CUDA)
  virtual int GetSequenceLength() const = 0;
};

struct ILogitsProcessorList {
  virtual ~ILogitsProcessorList() {}
  virtual void Process(const ISequences* sequences, gsl::span<float>& next_token_scores, int step) = 0;
};

// Interface for all scorers for beam search or beam sample.
struct IBeamScorer {
  virtual ~IBeamScorer() {}

  virtual void Process(ISequences& sequences,
                       gsl::span<const float> next_scores,
                       gsl::span<const int32_t> next_tokens,
                       gsl::span<const int32_t> next_indices) = 0;

  virtual void Finalize(ISequences& sequences,
                        size_t num_return_sequences,
                        gsl::span<int32_t> output_sequences,
                        gsl::span<float> output_sequence_scores) = 0;

  virtual bool IsDone() const = 0;                    // GPU version will return false here, as it asynchronously queues up the event
  virtual bool IsDoneLater() const { return false; }  // GPU version waits for the asynchous result to complete here

  virtual gsl::span<float> GetNextScores() = 0;
  virtual gsl::span<int32_t> GetNextTokens() = 0;
  virtual gsl::span<int32_t> GetNextIndicesCPU() = 0;
  virtual gsl::span<int32_t> GetNextIndicesGPU() { return {}; }  // If this is non CPU, returns the device buffer of the indices
};

struct IGenerationParameters {
  static constexpr int kModelTypeGpt = 0;
  static constexpr int kModelTypeT5 = 1;
  static constexpr int kModelTypeWhisper = 2;

  static constexpr int kLogitsProcessorTypeWhisper = 1;

  // Parameters from node attributes
  int model_type;  // 0 for GPT-2; 1 for encoder-decoder like T5; 2 for float inputs like Whisper
  int eos_token_id;
  int pad_token_id;
  int decoder_start_token_id;
  int no_repeat_ngram_size;
  bool early_stopping;

  // Parameters from inputs
  int min_length;
  int max_length;
  int num_beams;
  int num_return_sequences;
  float length_penalty;
  float repetition_penalty;
  int batch_size;       // deduce from first dimension of input_ids
  int sequence_length;  // deduce from second dimension of input_ids of GPT-2 or decoder_input_ids of T5
  int logits_processor;

  gsl::span<const int32_t> vocab_mask;
  gsl::span<const int32_t> prefix_vocab_mask;
  gsl::span<const int32_t> presence_mask;

  // Parameters from outputs.
  bool output_scores;  // whether scores existed in output

  // Parameters from subgraph.
  int vocab_size;
  int num_heads;
  int head_size;
  int num_layers;

  // Parameters for TopK/TopP sampling.
  float presence_penalty;
  float filter_value;
  float temperature = 1.0f;
  float top_p = 0.0f;
  int seed = 0;
  int min_tokens_to_keep = 1;
  bool custom_sampling = false;
};

struct BeamSearchParameters : public IGenerationParameters {

  int BatchBeamSize() const { return batch_size * num_beams; }

#if 0
  Status Validate() const;

  int BatchBeamSize() const { return batch_size * num_beams; }

  void ParseFromAttributes(const OpKernelInfo& info);

  void ParseFromInputs(OpKernelContext* context);

  void SetSubgraphParameters(int vocab_size, int num_heads, int head_size, int num_layers);
#endif
};

struct GreedySearchParameters : public BeamSearchParameters {
  int BatchBeamSize() const { return batch_size; }

#if 0
  void ParseFromAttributes(const OpKernelInfo& info);

  void ParseFromInputs(OpKernelContext* context);
#endif
};

struct SamplingParameters : public GreedySearchParameters {
#if 0
  void ParseFromAttributes(const OpKernelInfo& info);

  void ParseFromInputs(OpKernelContext* context);
#endif
};

// Delete this
struct OrtDevice {
  using DeviceType = int8_t;
  using MemoryType = int8_t;
  using DeviceId = int16_t;

  // Pre-defined device types.
  static const DeviceType CPU = 0;
  static const DeviceType GPU = 1;  // Nvidia or AMD
  static const DeviceType FPGA = 2;
  static const DeviceType NPU = 3;  // Ascend

  struct MemType {
    // Pre-defined memory types.
    static const MemoryType DEFAULT = 0;
    static const MemoryType CUDA_PINNED = 1;
    static const MemoryType HIP_PINNED = 2;
    static const MemoryType CANN_PINNED = 3;
  };

  constexpr OrtDevice(DeviceType device_type_, MemoryType memory_type_, DeviceId device_id_)
      : device_type(device_type_),
        memory_type(memory_type_),
        device_id(device_id_) {}

  constexpr OrtDevice() : OrtDevice(CPU, MemType::DEFAULT, 0) {}

  DeviceType Type() const {
    return device_type;
  }

  MemoryType MemType() const {
    return memory_type;
  }

  DeviceId Id() const {
    return device_id;
  }

#if 0
  std::string ToString() const {
    std::ostringstream ostr;
    ostr << "Device:["
         << "DeviceType:" << static_cast<int>(device_type)
         << " MemoryType:" << static_cast<int>(memory_type)
         << " DeviceId:" << device_id
         << "]";
    return ostr.str();
  }
#endif

#if 0
  // This is to make OrtDevice a valid key in hash tables
  size_t Hash() const {
    auto h = std::hash<int>()(device_type);
    onnxruntime::HashCombine(memory_type, h);
    onnxruntime::HashCombine(device_id, h);
    return h;
  }
#endif

  // To make OrtDevice become a valid key in std map
  bool operator<(const OrtDevice& other) const {
    if (device_type != other.device_type)
      return device_type < other.device_type;
    if (memory_type != other.memory_type)
      return memory_type < other.memory_type;

    return device_id < other.device_id;
  }

 private:
  // Device type.
  int32_t device_type : 8;

  // Memory type.
  int32_t memory_type : 8;

  // Device index.
  int32_t device_id : 16;
};

#include "Sequences.h"
#include "feeds_fetches_manager.h"
#include "Search.h"
#include "gpt.h"
