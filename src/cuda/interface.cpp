// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "ort_genai_c.h"  // For OGA_EXPORT
#include "interface.h"
#include "../search.h"
#include "search_cuda.h"
#include "kernels.h"
#include "cuda_topk.h"
#include <charconv>
#include <cstdarg>
#include <cstring>
#include <random>
#include <system_error>

#if defined(_WIN32) || defined(_WIN64)
#define strcasecmp _stricmp
#endif

namespace Generators {

GenaiInterface* gp_genai{};
Ort::Allocator* ort_allocator_{};
const char* device_label = "cuda";

cuda_stream_holder g_stream;
cudaStream_t GetStream() { return g_stream.get(); }

struct GpuMemory final : DeviceBuffer {
  GpuMemory(size_t size) : owned_{true} {
    size_in_bytes_ = size;
    p_device_ = static_cast<uint8_t*>(ort_allocator_->Alloc(size));
  }

  GpuMemory(void* p, size_t size) : owned_{false} {
    size_in_bytes_ = size;
    p_device_ = static_cast<uint8_t*>(p);
  }

  ~GpuMemory() override {
    if (owned_)
      ort_allocator_->Free(p_device_);
    if (p_cpu_)
      ::cudaFreeHost(p_cpu_);
  }

  const char* GetType() const override { return device_label; }

  void AllocateCpu() override {
    if (!p_cpu_)
      CUDA_CHECK(::cudaHostAlloc(&p_cpu_, size_in_bytes_, 0));
  }

  void CopyDeviceToCpu() override {
    AllocateCpu();
    CUDA_CHECK(::cudaMemcpyAsync(p_cpu_, p_device_, size_in_bytes_, ::cudaMemcpyDeviceToHost, GetStream()));
    CUDA_CHECK(::cudaStreamSynchronize(GetStream()));
  }

  void CopyCpuToDevice() override {
    assert(p_cpu_);
    CUDA_CHECK(::cudaMemcpyAsync(p_device_, p_cpu_, size_in_bytes_, ::cudaMemcpyHostToDevice, GetStream()));
  }

  void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) override {
    if (source.GetType() == device_label)
      CUDA_CHECK(::cudaMemcpyAsync(p_device_ + begin_dest, source.p_device_ + begin_source, size_in_bytes,
                                   ::cudaMemcpyDeviceToDevice, GetStream()));
    else
      gp_genai->CopyThroughCpu(*this, begin_dest, source, begin_source, size_in_bytes);
  }

  void Zero() override {
    CUDA_CHECK(::cudaMemsetAsync(p_device_, 0, size_in_bytes_, GetStream()));
  }

  bool owned_;  // If we own the memory, we delete it on destruction
};

template <typename T>
DeviceSpan<T> AllocateCudaSpan(size_t count) {
  return DeviceSpan<T>{std::make_shared<GpuMemory>(count * sizeof(T))};
}

struct CudaSamplerStatePool {
  explicit CudaSamplerStatePool(int initial_capacity) {
    if (initial_capacity > 0) {
      states_ = AllocateCudaSpan<curandState>(initial_capacity);
      capacity_ = initial_capacity;
    }
  }

  int Acquire(int random_seed) {
    int index;
    if (free_indices_.empty()) {
      index = size_++;
      EnsureCapacity(size_);
    } else {
      index = free_indices_.back();
      free_indices_.pop_back();
    }

    const unsigned long long seed = random_seed == -1
                                        ? static_cast<unsigned long long>(std::random_device{}())
                                        : static_cast<unsigned long long>(random_seed);
    cuda::LaunchInitCurandState(seed, states_.Span().data() + index, GetStream());
    return index;
  }

  void Release(int index) {
    free_indices_.push_back(index);
  }

  curandState* Data() { return states_.Span().data(); }

 private:
  void EnsureCapacity(int required_capacity) {
    if (required_capacity <= capacity_)
      return;

    const int new_capacity = std::max(required_capacity, std::max(4, capacity_ * 2));
    auto new_states = AllocateCudaSpan<curandState>(new_capacity);
    if (size_ > 1) {
      CUDA_CHECK(cudaMemcpyAsync(new_states.Span().data(), states_.Span().data(),
                                 static_cast<size_t>(size_ - 1) * sizeof(curandState),
                                 cudaMemcpyDeviceToDevice, GetStream()));
      CUDA_CHECK(cudaStreamSynchronize(GetStream()));
    }
    states_ = std::move(new_states);
    capacity_ = new_capacity;
  }

  DeviceSpan<curandState> states_;
  std::vector<int> free_indices_;
  int size_{};
  int capacity_{};
};

struct CudaBatchedSamplerState final : BatchedSamplerState {
  CudaBatchedSamplerState(std::shared_ptr<CudaSamplerStatePool> pool, int index)
      : pool_{std::move(pool)}, index_{index} {}

  ~CudaBatchedSamplerState() override { pool_->Release(index_); }

  std::shared_ptr<CudaSamplerStatePool> pool_;
  int index_{};
};

struct CudaBatchedSampler final : BatchedSampler {
  CudaBatchedSampler(int max_batch_size, int /*vocab_size*/)
      : state_pool_{std::make_shared<CudaSamplerStatePool>(max_batch_size)},
        max_batch_size_{std::max(max_batch_size, 1)} {
    // The GPU workspace scales with max_batch_size * vocab_size, so it is grown on demand from the
    // first observed batch rather than reserved at model load. Only the cheap host-side planning
    // vectors are sized up front.
    row_order_.reserve(max_batch_size_);
    bucket_offsets_.reserve(static_cast<size_t>(max_batch_size_) + 1);
    bucket_params_.reserve(max_batch_size_);
  }

  std::unique_ptr<BatchedSamplerState> CreateState(int random_seed) override {
    return std::make_unique<CudaBatchedSamplerState>(state_pool_, state_pool_->Acquire(random_seed));
  }

  bool OwnsState(const BatchedSamplerState& state) const override {
    const auto* cuda_state = dynamic_cast<const CudaBatchedSamplerState*>(&state);
    return cuda_state && cuda_state->pool_.get() == state_pool_.get();
  }

  DeviceSpan<int32_t> Sample(std::span<DeviceSpan<float>> scores,
                             std::span<const BatchedSamplingParams> params,
                             std::span<BatchedSamplerState* const> states,
                             int vocab_size) override {
    const int batch_size = static_cast<int>(scores.size());
    if (batch_size == 0 || params.size() != scores.size() || states.size() != scores.size())
      throw std::runtime_error("BatchedSampler requires one parameter set and RNG state per score row.");

    EnsureCapacity(batch_size, vocab_size);
    row_order_.clear();
    bucket_offsets_.clear();
    bucket_params_.clear();
    for (int row = 0; row < batch_size; ++row) {
      if (scores[row].size() != static_cast<size_t>(vocab_size))
        throw std::runtime_error("BatchedSampler score row has the wrong vocabulary size.");

      auto* state = dynamic_cast<CudaBatchedSamplerState*>(states[row]);
      if (!state || state->pool_.get() != state_pool_.get())
        throw std::runtime_error("BatchedSampler received an RNG state from a different sampler.");
      row_order_.push_back(row);
    }

    const auto params_less = [&](int lhs, int rhs) {
      if (params[lhs].k != params[rhs].k)
        return params[lhs].k < params[rhs].k;
      if (params[lhs].p != params[rhs].p)
        return params[lhs].p < params[rhs].p;
      return params[lhs].temperature < params[rhs].temperature;
    };
    // Stable so that rows with identical parameters keep their original relative order, which
    // keeps each request bound to its own RNG state regardless of batch composition.
    std::stable_sort(row_order_.begin(), row_order_.end(), params_less);

    for (int packed_row = 0; packed_row < batch_size; ++packed_row) {
      const int row = row_order_[packed_row];
      if (packed_row == 0 || params[row].k != bucket_params_.back().k ||
          params[row].p != bucket_params_.back().p ||
          params[row].temperature != bucket_params_.back().temperature) {
        bucket_offsets_.push_back(packed_row);
        bucket_params_.push_back(params[row]);
      }
    }
    bucket_offsets_.push_back(batch_size);

    auto score_ptrs_cpu = score_ptrs_.CpuSpan();
    auto output_indices_cpu = output_indices_.CpuSpan();
    auto state_indices_cpu = state_indices_.CpuSpan();
    for (int packed_row = 0; packed_row < batch_size; ++packed_row) {
      const int row = row_order_[packed_row];
      score_ptrs_cpu[packed_row] = scores[row].Span().data();
      output_indices_cpu[packed_row] = row;
      state_indices_cpu[packed_row] = static_cast<CudaBatchedSamplerState*>(states[row])->index_;
    }
    score_ptrs_.CopyCpuToDevice();
    output_indices_.CopyCpuToDevice();
    state_indices_.CopyCpuToDevice();

    // The fast path samples scores[0] in place, so the packed RNG state indices must line up with
    // the original rows. The stable sort guarantees that for a single bucket; check it explicitly so
    // a future ordering change falls back to gather/scatter instead of silently swapping RNG streams.
    bool rows_are_contiguous = bucket_params_.size() == 1;
    for (int row = 0; row < batch_size && rows_are_contiguous; ++row) {
      rows_are_contiguous = row_order_[row] == row &&
                            scores[row].SameBufferAs(scores[0]) &&
                            scores[row].Span().data() == scores[0].Span().data() +
                                                             static_cast<size_t>(row) * vocab_size;
    }

    if (rows_are_contiguous) {
      const auto& sample_params = bucket_params_.front();
      cuda::GetSample(sampling_data_.get(), GetStream(), next_tokens_.Span().data(),
                      scores[0].Span().data(), vocab_size, batch_size,
                      sample_params.k, sample_params.p, sample_params.temperature,
                      state_pool_->Data(), state_indices_.Span().data());
      return next_tokens_.subspan(0, batch_size);
    }

    cuda::LaunchGatherSamplingRows(score_ptrs_.Span().data(), packed_scores_.Span().data(),
                                   batch_size, vocab_size, GetStream());
    for (size_t bucket = 0; bucket < bucket_params_.size(); ++bucket) {
      const int bucket_offset = bucket_offsets_[bucket];
      const int bucket_size = bucket_offsets_[bucket + 1] - bucket_offset;
      const auto& sample_params = bucket_params_[bucket];
      cuda::GetSample(sampling_data_.get(), GetStream(),
                      packed_tokens_.Span().data() + bucket_offset,
                      packed_scores_.Span().data() + static_cast<size_t>(bucket_offset) * vocab_size,
                      vocab_size, bucket_size, sample_params.k, sample_params.p, sample_params.temperature,
                      state_pool_->Data(), state_indices_.Span().data() + bucket_offset);
    }
    cuda::LaunchScatterSamplingTokens(packed_tokens_.Span().data(), output_indices_.Span().data(),
                                      next_tokens_.Span().data(), batch_size, GetStream());
    return next_tokens_.subspan(0, batch_size);
  }

 private:
  void EnsureCapacity(int batch_size, int vocab_size) {
    if (batch_size <= batch_capacity_ && vocab_size == vocab_capacity_)
      return;

    batch_capacity_ = std::max(batch_size, std::min(max_batch_size_, std::max(4, batch_capacity_ * 2)));
    vocab_capacity_ = vocab_size;
    score_ptrs_ = AllocateCudaSpan<const float*>(batch_capacity_);
    output_indices_ = AllocateCudaSpan<int>(batch_capacity_);
    state_indices_ = AllocateCudaSpan<int>(batch_capacity_);
    packed_scores_ = AllocateCudaSpan<float>(static_cast<size_t>(batch_capacity_) * vocab_size);
    packed_tokens_ = AllocateCudaSpan<int32_t>(batch_capacity_);
    next_tokens_ = AllocateCudaSpan<int32_t>(batch_capacity_);
    score_ptrs_.CpuSpan();
    output_indices_.CpuSpan();
    state_indices_.CpuSpan();
    next_tokens_.CpuSpan();

    const size_t buffer_size = cuda::SamplingData::CalculateTotalSize(batch_capacity_, vocab_size, GetStream());
    sampling_buffer_ = AllocateCudaSpan<uint8_t>(buffer_size);
    sampling_data_ = std::make_unique<cuda::SamplingData>(std::random_device{}(), batch_capacity_, vocab_size,
                                                          GetStream(), sampling_buffer_.Span().data(), buffer_size);
  }

  std::shared_ptr<CudaSamplerStatePool> state_pool_;
  DeviceSpan<const float*> score_ptrs_;
  DeviceSpan<int> output_indices_;
  DeviceSpan<int> state_indices_;
  DeviceSpan<float> packed_scores_;
  DeviceSpan<int32_t> packed_tokens_;
  DeviceSpan<int32_t> next_tokens_;
  DeviceSpan<uint8_t> sampling_buffer_;
  std::unique_ptr<cuda::SamplingData> sampling_data_;
  std::vector<int> row_order_;
  std::vector<int> bucket_offsets_;
  std::vector<BatchedSamplingParams> bucket_params_;
  int max_batch_size_{};
  int batch_capacity_{};
  int vocab_capacity_{};
};

struct CudaInterfaceImplBase : DeviceInterface {
  CudaInterfaceImplBase() {
    g_stream.Create();
  }

  ~CudaInterfaceImplBase() {
  }

  void InitOrt(const OrtApi& api, Ort::Allocator& allocator) override {
    assert(!ort_allocator_);
    ort_allocator_ = &allocator;
  }

  Ort::Allocator& GetAllocator() override {
    return *ort_allocator_;
  }

  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override {
    return OrtMemoryInfo::Create("Cuda",
                                 OrtAllocatorType::OrtDeviceAllocator,
                                 0,
                                 OrtMemType::OrtMemTypeDefault);
  }

  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return std::make_shared<GpuMemory>(size);
  }

  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override {
    return std::make_shared<GpuMemory>(p, size);
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override {
    return std::make_unique<GreedySearch_Cuda>(params);
  }

  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override {
    return std::make_unique<BeamSearch_Cuda>(params);
  }

  std::unique_ptr<BatchedSampler> CreateBatchedSampler(size_t max_batch_size, int vocab_size) override {
    return std::make_unique<CudaBatchedSampler>(static_cast<int>(max_batch_size), vocab_size);
  }

  void Synchronize() override {
    CUDA_CHECK(::cudaStreamSynchronize(GetStream()));
  }

  void* GetCudaStream() override {
    return GetStream();
  }

  bool Cast(void* input_data, void* output_data, ONNXTensorElementDataType input_type, ONNXTensorElementDataType output_type, size_t element_count) override {
    if (input_type == output_type)
      throw std::runtime_error("Cast - input and output types are the same");

    if (input_type == Ort::TypeToTensorType<float> && output_type == Ort::TypeToTensorType<Ort::Float16_t>) {
      cuda::LaunchFp32ToFp16(reinterpret_cast<const float*>(input_data), reinterpret_cast<uint16_t*>(output_data), static_cast<int>(element_count), GetStream());
    } else if (input_type == Ort::TypeToTensorType<Ort::Float16_t> && output_type == Ort::TypeToTensorType<float>) {
      cuda::LaunchFp16ToFp32(reinterpret_cast<const uint16_t*>(input_data), reinterpret_cast<float*>(output_data), static_cast<int>(element_count), GetStream());
    } else if (input_type == Ort::TypeToTensorType<Ort::BFloat16_t> && output_type == Ort::TypeToTensorType<float>) {
      cuda::LaunchBf16ToFp32(reinterpret_cast<const uint16_t*>(input_data), reinterpret_cast<float*>(output_data), static_cast<int>(element_count), GetStream());
    } else if (input_type == Ort::TypeToTensorType<int32_t> && output_type == Ort::TypeToTensorType<int64_t>) {
      cuda::LaunchInt32ToInt64(reinterpret_cast<const int32_t*>(input_data), reinterpret_cast<int64_t*>(output_data), static_cast<int>(element_count), GetStream());
    } else
      return false;
    return true;
  }

  bool RunArgMax(const void* logits, ONNXTensorElementDataType logits_type, int num_rows, int vocab_size) {
    if (num_rows <= 0 || vocab_size <= 0)
      return false;

    cudaStream_t stream = GetStream();

    // The Top-K kernel consumes fp32 scores. Cast fp16 input into a cached fp32 scratch buffer;
    // fp32 input is used directly (no copy).
    const float* scores = nullptr;
    const size_t element_count = static_cast<size_t>(num_rows) * vocab_size;
    if (logits_type == Ort::TypeToTensorType<float>) {
      scores = reinterpret_cast<const float*>(logits);
    } else if (logits_type == Ort::TypeToTensorType<Ort::Float16_t>) {
      if (argmax_fp32_count_ < element_count) {
        argmax_fp32_ = CudaMallocArray<float>(element_count);
        argmax_fp32_count_ = element_count;
      }
      cuda::LaunchFp16ToFp32(reinterpret_cast<const uint16_t*>(logits), argmax_fp32_.get(), static_cast<int>(element_count), stream);
      scores = argmax_fp32_.get();
    } else {
      return false;  // Unsupported logits dtype -> caller falls back to host argmax.
    }

    // (Re)allocate the Top-K working set if the problem size grew.
    if (!topk_data_ || topk_batch_ < num_rows || topk_vocab_ != vocab_size) {
      topk_data_ = std::make_unique<cuda::TopkData>(num_rows, vocab_size, stream);
      topk_batch_ = num_rows;
      topk_vocab_ = vocab_size;
    }

    // k=1 dispatches to distributed_select_sort, the fastest path for argmax over a large vocab.
    cuda::RunTopK(topk_data_.get(), stream, scores, vocab_size, num_rows, /*k=*/1);
    return true;
  }

  bool ArgMax(const void* logits, ONNXTensorElementDataType logits_type, int num_rows, int vocab_size, int32_t* out_tokens) override {
    if (!RunArgMax(logits, logits_type, num_rows, vocab_size))
      return false;

    cudaStream_t stream = GetStream();

    // Copy only the small per-row top-1 indices back to the host (strided -> contiguous), then sync.
    if (!argmax_host_ || argmax_host_count_ < static_cast<size_t>(num_rows)) {
      argmax_host_ = CudaMallocHostArray<int32_t>(num_rows);
      argmax_host_count_ = num_rows;
    }
    CUDA_CHECK(cudaMemcpy2DAsync(argmax_host_.get(), sizeof(int32_t),
                                 topk_data_->topk_indices, static_cast<size_t>(topk_data_->topk_stride) * sizeof(int32_t),
                                 sizeof(int32_t), num_rows, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::memcpy(out_tokens, argmax_host_.get(), static_cast<size_t>(num_rows) * sizeof(int32_t));
    return true;
  }

  bool ArgMaxDevice(const void* logits, ONNXTensorElementDataType logits_type, int num_rows, int vocab_size,
                    DeviceSpan<int32_t> out_tokens) override {
    if (out_tokens.size() < static_cast<size_t>(num_rows) ||
        !RunArgMax(logits, logits_type, num_rows, vocab_size))
      return false;

    CUDA_CHECK(cudaMemcpy2DAsync(out_tokens.Span().data(), sizeof(int32_t),
                                 topk_data_->topk_indices,
                                 static_cast<size_t>(topk_data_->topk_stride) * sizeof(int32_t),
                                 sizeof(int32_t), num_rows, cudaMemcpyDeviceToDevice, GetStream()));
    return true;
  }

  bool CopyStateSlots(const void* descs_device, int count, int src_slot, int dst_slot) override {
    cuda::LaunchCopyStateSlots(descs_device, count, src_slot, dst_slot, GetStream());
    return true;
  }

  bool Top2(const void* logits, ONNXTensorElementDataType logits_type, int num_rows, int vocab_size,
            int32_t* out_tokens, float* out_scores) override {
    if (num_rows <= 0 || vocab_size <= 1) return false;

    cudaStream_t stream = GetStream();
    const float* scores = nullptr;
    const size_t element_count = static_cast<size_t>(num_rows) * vocab_size;
    if (logits_type == Ort::TypeToTensorType<float>) {
      scores = reinterpret_cast<const float*>(logits);
    } else if (logits_type == Ort::TypeToTensorType<Ort::Float16_t>) {
      if (argmax_fp32_count_ < element_count) {
        argmax_fp32_ = CudaMallocArray<float>(element_count);
        argmax_fp32_count_ = element_count;
      }
      cuda::LaunchFp16ToFp32(reinterpret_cast<const uint16_t*>(logits), argmax_fp32_.get(), static_cast<int>(element_count), stream);
      scores = argmax_fp32_.get();
    } else {
      return false;
    }

    if (!topk_data_ || topk_batch_ < num_rows || topk_vocab_ != vocab_size) {
      topk_data_ = std::make_unique<cuda::TopkData>(num_rows, vocab_size, stream);
      topk_batch_ = num_rows;
      topk_vocab_ = vocab_size;
    }
    cuda::select_sort::RunTopK(topk_data_.get(), stream, scores, vocab_size, num_rows, /*k=*/2);

    const size_t result_count = static_cast<size_t>(num_rows) * 2;
    if (!top2_indices_host_ || top2_host_count_ < result_count) {
      top2_indices_host_ = CudaMallocHostArray<int32_t>(result_count);
      top2_scores_host_ = CudaMallocHostArray<float>(result_count);
      top2_host_count_ = result_count;
    }
    const size_t source_pitch = static_cast<size_t>(topk_data_->topk_stride);
    CUDA_CHECK(cudaMemcpy2DAsync(top2_indices_host_.get(), 2 * sizeof(int32_t),
                                 topk_data_->topk_indices, source_pitch * sizeof(int32_t),
                                 2 * sizeof(int32_t), num_rows, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpy2DAsync(top2_scores_host_.get(), 2 * sizeof(float),
                                 topk_data_->topk_scores, source_pitch * sizeof(float),
                                 2 * sizeof(float), num_rows, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::memcpy(out_tokens, top2_indices_host_.get(), result_count * sizeof(int32_t));
    std::memcpy(out_scores, top2_scores_host_.get(), result_count * sizeof(float));
    return true;
  }

  bool TopKScores(const void* logits, ONNXTensorElementDataType logits_type, int num_rows, int vocab_size,
                  int k, int32_t* out_tokens, float* out_scores) override {
    if (num_rows <= 0 || vocab_size <= 1 || k <= 0) return false;
    k = std::min(k, vocab_size);

    cudaStream_t stream = GetStream();
    const float* scores = nullptr;
    const size_t element_count = static_cast<size_t>(num_rows) * vocab_size;
    if (logits_type == Ort::TypeToTensorType<float>) {
      scores = reinterpret_cast<const float*>(logits);
    } else if (logits_type == Ort::TypeToTensorType<Ort::Float16_t>) {
      if (argmax_fp32_count_ < element_count) {
        argmax_fp32_ = CudaMallocArray<float>(element_count);
        argmax_fp32_count_ = element_count;
      }
      cuda::LaunchFp16ToFp32(reinterpret_cast<const uint16_t*>(logits), argmax_fp32_.get(), static_cast<int>(element_count), stream);
      scores = argmax_fp32_.get();
    } else {
      return false;
    }

    if (!topk_data_ || topk_batch_ < num_rows || topk_vocab_ != vocab_size) {
      topk_data_ = std::make_unique<cuda::TopkData>(num_rows, vocab_size, stream);
      topk_batch_ = num_rows;
      topk_vocab_ = vocab_size;
    }
    // Dispatch to the fastest available Top-K algorithm for this (batch, vocab, k). The dispatcher
    // benchmarks once per shape and caches the choice; select_sort (the previous hardcoded call) is
    // only efficient for very small k and is ~20x slower than the tuned algorithms at k~20, which
    // dominated speculative-sampling decode time. All algorithms return k sorted-descending
    // (index, score) pairs per row; results and their stride are read from topk_data_ below.
    cuda::RunTopK(topk_data_.get(), stream, scores, vocab_size, num_rows, k);

    const size_t result_count = static_cast<size_t>(num_rows) * k;
    if (!topk_indices_host_ || topk_host_count_ < result_count) {
      topk_indices_host_ = CudaMallocHostArray<int32_t>(result_count);
      topk_scores_host_ = CudaMallocHostArray<float>(result_count);
      topk_host_count_ = result_count;
    }
    const size_t source_pitch = static_cast<size_t>(topk_data_->topk_stride);
    CUDA_CHECK(cudaMemcpy2DAsync(topk_indices_host_.get(), static_cast<size_t>(k) * sizeof(int32_t),
                                 topk_data_->topk_indices, source_pitch * sizeof(int32_t),
                                 static_cast<size_t>(k) * sizeof(int32_t), num_rows, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpy2DAsync(topk_scores_host_.get(), static_cast<size_t>(k) * sizeof(float),
                                 topk_data_->topk_scores, source_pitch * sizeof(float),
                                 static_cast<size_t>(k) * sizeof(float), num_rows, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::memcpy(out_tokens, topk_indices_host_.get(), result_count * sizeof(int32_t));
    std::memcpy(out_scores, topk_scores_host_.get(), result_count * sizeof(float));
    return true;
  }

  bool UpdatePositionIds(void* position_ids, int batch_beam_size, int total_length, int new_kv_length, ONNXTensorElementDataType type) override {
    if (type == Ort::TypeToTensorType<int32_t>)
      cuda::Launch_UpdatePositionIds(static_cast<int32_t*>(position_ids), batch_beam_size, total_length, new_kv_length, GetStream());
    else
      cuda::Launch_UpdatePositionIds(static_cast<int64_t*>(position_ids), batch_beam_size, total_length, new_kv_length, GetStream());
    return true;
  }

  bool UpdateAttentionMask(void* next_mask_data, void* mask_data, int batch_beam_size, int new_kv_length, int total_length, int max_length, bool update_only, ONNXTensorElementDataType type) override {
    if (type == Ort::TypeToTensorType<int32_t>)
      cuda::Launch_UpdateAttentionMask(static_cast<int32_t*>(next_mask_data), static_cast<int32_t*>(mask_data), batch_beam_size, new_kv_length, total_length, max_length, update_only, GetStream());
    else
      cuda::Launch_UpdateAttentionMask(static_cast<int64_t*>(next_mask_data), static_cast<int64_t*>(mask_data), batch_beam_size, new_kv_length, total_length, max_length, update_only, GetStream());
    return true;
  }

  void UpdateCacheIndirection(int32_t* tgt_indir_cache, const int32_t* src_indir_cache, const int32_t* beam_ids, int batch_size, int beam_width, int input_seq_length, int max_seq_length, int current_length) override {
    cuda::UpdateCacheIndirectionKernelLauncher(tgt_indir_cache, src_indir_cache, beam_ids, batch_size, beam_width, input_seq_length, max_seq_length, current_length, GetStream());
  }

  void ReorderPastStates(void* out_buffer, const void* in_buffer, int batch_size, int num_heads, int max_length, int head_size, int chunk_size) override {
    cuda::ReorderPastStatesKernelLauncher(out_buffer, in_buffer, batch_size, num_heads, max_length, head_size, chunk_size, GetStream());
  }

  void CopyCrossQK(float* cross_qk_buffer_data, void** qk_layer_pointers, int token_index, int batch_beam_size, int num_layers, int num_heads, int num_alignment_heads, const int* alignment_heads, int frames, int max_length, int sequence_length) override {
    cuda::LaunchCopyCrossQKSingleDecodeStep(GetStream(), cross_qk_buffer_data, qk_layer_pointers, token_index, batch_beam_size, num_layers, num_heads, num_alignment_heads, alignment_heads, frames, max_length, sequence_length);
  }

  void CopyCrossQK(Ort::Float16_t* cross_qk_buffer_data, void** qk_layer_pointers, int token_index, int batch_beam_size, int num_layers, int num_heads, int num_alignment_heads, const int* alignment_heads, int frames, int max_length, int sequence_length) override {
    cuda::LaunchCopyCrossQKSingleDecodeStep(GetStream(), reinterpret_cast<uint16_t*>(cross_qk_buffer_data), qk_layer_pointers, token_index, batch_beam_size, num_layers, num_heads, num_alignment_heads, alignment_heads, frames, max_length, sequence_length);
  }

  void FinalizeCrossQK(int iteration_number, int context_decoding_len, int batch_size, int num_beams, int max_length, int num_alignment_heads, int frames_of_k, const float* cross_qk_buffer_data, float* cross_qk_output, int num_return_sequences, const int* cache_indir_data) override {
    cuda::LaunchFinalizeCrossQK(GetStream(), iteration_number, context_decoding_len, batch_size, num_beams, max_length, num_alignment_heads, frames_of_k, cross_qk_buffer_data, cross_qk_output, num_return_sequences, cache_indir_data);
  }

  void FinalizeCrossQK(int iteration_number, int context_decoding_len, int batch_size, int num_beams, int max_length, int num_alignment_heads, int frames_of_k, const Ort::Float16_t* cross_qk_buffer_data, Ort::Float16_t* cross_qk_output, int num_return_sequences, const int* cache_indir_data) override {
    cuda::LaunchFinalizeCrossQK(GetStream(), iteration_number, context_decoding_len, batch_size, num_beams, max_length, num_alignment_heads, frames_of_k, reinterpret_cast<const uint16_t*>(cross_qk_buffer_data), reinterpret_cast<uint16_t*>(cross_qk_output), num_return_sequences, cache_indir_data);
  }

  void LaunchAddLogitsMask(float* batch_logits, int batch_beam_size, int vocab_size, const uint32_t* logits_mask) override {
    cuda::LaunchAddLogitsMask(batch_logits, batch_beam_size, vocab_size, logits_mask, GetStream());
  }

  void GetAvailableMemory(size_t& free_bytes, size_t& total_bytes) override {
    cudaMemGetInfo(&free_bytes, &total_bytes);
  }

  // Cached working set for the on-device ArgMax (Top-K, k=1) path.
  std::unique_ptr<cuda::TopkData> topk_data_;
  int topk_batch_{0};
  int topk_vocab_{0};
  cuda_unique_ptr<float> argmax_fp32_;          // fp16 -> fp32 scratch (device)
  size_t argmax_fp32_count_{0};
  cuda_host_unique_ptr<int32_t> argmax_host_;   // pinned host buffer for the small index copy
  size_t argmax_host_count_{0};
  cuda_host_unique_ptr<int32_t> top2_indices_host_;
  cuda_host_unique_ptr<float> top2_scores_host_;
  size_t top2_host_count_{0};
  cuda_host_unique_ptr<int32_t> topk_indices_host_;  // pinned host buffer for the top-k index copy
  cuda_host_unique_ptr<float> topk_scores_host_;     // pinned host buffer for the top-k score copy
  size_t topk_host_count_{0};
};

struct CudaInterfaceImpl final : CudaInterfaceImplBase {
  DeviceType GetType() const override { return DeviceType::CUDA; }
};

struct NvTensorRtRtxInterfaceImpl final : CudaInterfaceImplBase {
  DeviceType GetType() const override { return DeviceType::NvTensorRtRtx; }

  bool SupportsPhi3RopeRewind(const Config& config) const override {
    for (const auto& provider_options : config.model.decoder.session_options.provider_options) {
      if (provider_options.name != "NvTensorRtRtx")
        continue;

      for (const auto& [name, value] : provider_options.options) {
        if (name != "multi_rotary_cache_concat_offset")
          continue;

        int offset{};
        const auto* const value_begin = value.data();
        const auto* const value_end = value_begin + value.size();
        const auto [parse_end, error_code] = std::from_chars(value_begin, value_end, offset);
        return error_code == std::errc{} && parse_end == value_end && offset > 0 &&
               offset <= config.model.context_length;
      }
    }

    return false;
  }
};

std::unique_ptr<DeviceInterface> g_cuda_device;

DeviceInterface& GetCudaDeviceInterface() { return *g_cuda_device; }

LogItems& GetLogItems() { return gp_genai->GetLogItems(); }
std::ostream& operator<<(std::ostream& stream, SGR sgr_code) { return gp_genai->operator_leftshift(stream, sgr_code); }
std::ostream& Log(std::string_view label, std::string_view text) { return gp_genai->Log(label, text); }

// Duplicate of logging.cpp function
std::ostream& Log(std::string_view label, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  va_list args_copy;
  va_copy(args_copy, args);
  size_t len = vsnprintf(0, 0, fmt, args_copy);
  if (len <= 0) {
    throw std::runtime_error("Invalid format");
  }
  std::unique_ptr<char[]> buf(new char[len + 1]);
  vsnprintf(buf.get(), len + 1, fmt, args);
  va_end(args);
  return Log(label, std::string(buf.get(), buf.get() + len));
}

template <>
void DumpSpan<float>(std::ostream& stream, std::span<const float> values) { return gp_genai->DumpSpan(stream, values); }
template <>
void DumpSpan<int>(std::ostream& stream, std::span<const int> values) { return gp_genai->DumpSpan(stream, values); }

void Sequences::AfterAppendNextTokens(DeviceSpan<int32_t>& next_tokens, size_t batch_beam_size) { return gp_genai->Sequences_AfterAppendNextTokens(this, next_tokens, batch_beam_size); }
void Sequences::RewindTo(size_t new_length) { return gp_genai->Sequences_RewindTo(this, new_length); }
}  // namespace Generators

#ifdef _WIN32
// Override default new/delete so that we match the host's allocator
// Previous implementation calls Generators::gp_genai->HeapAllocate(n) or HeapFree(p).
// But memory allocation might be called before gp_genai created so gp_genai might be nullptr, which causes crash.
// Here we just copy the implementation of HeapAllocate and HeapFree to avoid initialization order issue.
_Ret_notnull_ _Post_writable_byte_size_(n) void* operator new(size_t n) {
  return std::malloc(n);
}
void operator delete(void* p) noexcept {
  return std::free(p);
}

void operator delete(void* p, size_t /*size*/) noexcept {
  return std::free(p);
}
#endif

extern "C" {
Generators::DeviceInterface* GetInterface(GenaiInterface* p_genai, const char* deviceType, const OrtApi* ort_api) {
  Generators::gp_genai = p_genai;
  // Ensure Ort::api is initialized in this shared library (onnxruntime-genai-cuda add-on) immediately. Delaying the
  // initialization to CudaInterfaceImplBase::InitOrt would be inadequate, as GetMemoryInfo runs before InitOrt.
  Ort::api = ort_api;
  if (strcasecmp(deviceType, "NvTensorRtRtx") == 0) {
    Generators::g_cuda_device = std::make_unique<Generators::NvTensorRtRtxInterfaceImpl>();
  } else {
    Generators::g_cuda_device = std::make_unique<Generators::CudaInterfaceImpl>();
  }
  return Generators::g_cuda_device.get();
}
}
