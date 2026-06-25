// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>

#include "../generators.h"
#include "moonshine_streaming_processor.h"

namespace Generators {

MoonshineStreamingProcessor::MoonshineStreamingProcessor(Model& model)
    : model_{model} {
  moonshine_model_ = dynamic_cast<MoonshineStreamingModel*>(&model);
  if (!moonshine_model_) {
    throw std::runtime_error(
        "MoonshineStreamingProcessor requires a streaming_enc_dec_asr model type. Got: " +
        model.config_->model.type);
  }
  config_ = moonshine_model_->moonshine_config_;
  ResetState();
  InitVadFromConfig(model);
}

MoonshineStreamingProcessor::~MoonshineStreamingProcessor() = default;

void MoonshineStreamingProcessor::ResetState() {
  auto& alloc = model_.allocator_cpu_;

  // ---- Frontend tensors (zeroed) ----------------------------------------
  {
    auto shape = std::array<int64_t, 2>{1, config_.sample_buffer_size};
    sample_buffer_ = OrtValue::CreateTensor(alloc, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memset(sample_buffer_->GetTensorMutableData<float>(), 0,
                static_cast<size_t>(config_.sample_buffer_size) * sizeof(float));
  }
  {
    auto shape = std::array<int64_t, 1>{1};
    sample_len_ = OrtValue::CreateTensor(alloc, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    *sample_len_->GetTensorMutableData<int64_t>() = 0;
  }
  {
    auto shape = std::array<int64_t, 3>{1, config_.conv1_channels, config_.conv1_buffer_size};
    conv1_buffer_ = OrtValue::CreateTensor(alloc, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memset(conv1_buffer_->GetTensorMutableData<float>(), 0,
                static_cast<size_t>(config_.conv1_channels) *
                    static_cast<size_t>(config_.conv1_buffer_size) * sizeof(float));
  }
  {
    auto shape = std::array<int64_t, 3>{1, config_.conv2_channels, config_.conv2_buffer_size};
    conv2_buffer_ = OrtValue::CreateTensor(alloc, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memset(conv2_buffer_->GetTensorMutableData<float>(), 0,
                static_cast<size_t>(config_.conv2_channels) *
                    static_cast<size_t>(config_.conv2_buffer_size) * sizeof(float));
  }
  {
    auto shape = std::array<int64_t, 1>{1};
    frame_count_ = OrtValue::CreateTensor(alloc, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    *frame_count_->GetTensorMutableData<int64_t>() = 0;
  }

  // ---- Encoder / adapter / memory state ---------------------------------
  accumulated_features_.clear();
  total_features_ = 0;
  encoder_frames_emitted_ = 0;
  adapter_pos_offset_ = 0;
  accumulated_memory_.clear();
  memory_len_ = 0;
  cached_k_cross_.reset();
  cached_v_cross_.reset();
  memory_in_cross_kv_ = 0;
  cross_kv_valid_ = false;
  audio_buffer_.clear();
}

std::unique_ptr<NamedTensors> MoonshineStreamingProcessor::Process(const float* audio_data,
                                                                   size_t num_samples) {
  if (ShouldDropChunk(audio_data, num_samples)) return nullptr;

  audio_buffer_.insert(audio_buffer_.end(), audio_data, audio_data + num_samples);

  const size_t chunk_size = static_cast<size_t>(config_.chunk_samples);
  if (audio_buffer_.size() < chunk_size) {
    return nullptr;  // Not enough audio yet for a streaming chunk.
  }

  std::vector<float> chunk(audio_buffer_.begin(), audio_buffer_.begin() + chunk_size);
  audio_buffer_.erase(audio_buffer_.begin(), audio_buffer_.begin() + chunk_size);

  RunFrontendAndAccumulate(chunk.data(), chunk.size(), /*is_final=*/false);
  if (memory_len_ == 0) return nullptr;  // No stable frames yet.
  RefreshCrossKv();
  return EmitCrossKv(/*is_final=*/false);
}

std::unique_ptr<NamedTensors> MoonshineStreamingProcessor::Flush() {
  std::vector<float> tail = std::move(audio_buffer_);
  audio_buffer_.clear();
  RunFrontendAndAccumulate(tail.data(), tail.size(), /*is_final=*/true);

  std::unique_ptr<NamedTensors> result;
  if (memory_len_ > 0) {
    RefreshCrossKv();
    result = EmitCrossKv(/*is_final=*/true);
  }
  // End-of-utterance: drop all per-stream state so the next utterance starts clean.
  ResetState();
  return result;
}

void MoonshineStreamingProcessor::RunFrontendAndAccumulate(const float* audio, size_t num,
                                                           bool is_final) {
  auto& alloc = model_.allocator_cpu_;
  const int encoder_dim = config_.encoder_dim;
  const int decoder_dim = config_.decoder_dim;

  // ----- frontend.onnx ---------------------------------------------------
  // Always run, even with num==0 on Flush, to allow it to drain any
  // sub-frame samples it may have buffered. Note: the frontend's audio input
  // dim must be > 0, so we skip the call if there's nothing to feed.
  if (num > 0) {
    auto audio_shape = std::array<int64_t, 2>{1, static_cast<int64_t>(num)};
    auto audio_tensor =
        OrtValue::CreateTensor(alloc, audio_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memcpy(audio_tensor->GetTensorMutableData<float>(), audio, num * sizeof(float));

    const char* fe_in_names[]  = {"audio_chunk", "sample_buffer", "sample_len",
                                  "conv1_buffer", "conv2_buffer", "frame_count"};
    OrtValue*   fe_in_values[] = {audio_tensor.get(), sample_buffer_.get(), sample_len_.get(),
                                  conv1_buffer_.get(), conv2_buffer_.get(), frame_count_.get()};
    const char* fe_out_names[] = {"features", "sample_buffer_out", "sample_len_out",
                                  "conv1_buffer_out", "conv2_buffer_out", "frame_count_out"};
    OrtValue*   fe_out_values[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

    moonshine_model_->session_frontend_->Run(
        nullptr,
        fe_in_names, fe_in_values, 6,
        fe_out_names, fe_out_values, 6);

    auto features  = std::unique_ptr<OrtValue>(fe_out_values[0]);
    sample_buffer_ = std::unique_ptr<OrtValue>(fe_out_values[1]);
    sample_len_    = std::unique_ptr<OrtValue>(fe_out_values[2]);
    conv1_buffer_  = std::unique_ptr<OrtValue>(fe_out_values[3]);
    conv2_buffer_  = std::unique_ptr<OrtValue>(fe_out_values[4]);
    frame_count_   = std::unique_ptr<OrtValue>(fe_out_values[5]);

    auto fshape = features->GetTensorTypeAndShapeInfo()->GetShape();
    if (fshape.size() >= 3 && fshape[1] > 0) {
      const int new_features = static_cast<int>(fshape[1]);
      const float* fdata = features->GetTensorData<float>();
      accumulated_features_.insert(accumulated_features_.end(), fdata,
                                   fdata + static_cast<size_t>(new_features) * encoder_dim);
      total_features_ += new_features;
    }
  }

  // ----- encoder / adapter step -----------------------------------------
  // Hold back the lookahead window unless this is the final chunk.
  const int lookahead = config_.total_lookahead;
  const int stable_count = is_final ? total_features_
                                    : std::max(0, total_features_ - lookahead);
  const int new_frames = stable_count - encoder_frames_emitted_;
  if (new_frames <= 0) return;

  // Encoder runs on [window_start : total_features_] for left context.
  const int left_context = config_.left_context_frames;
  const int window_start = std::max(0, encoder_frames_emitted_ - left_context);
  const int window_size = total_features_ - window_start;
  const int start_idx = encoder_frames_emitted_ - window_start;  // where new frames begin in encoded.

  // Build encoder input from a slice of accumulated_features_.
  auto enc_in_shape =
      std::array<int64_t, 3>{1, window_size, encoder_dim};
  auto enc_in_tensor =
      OrtValue::CreateTensor(alloc, enc_in_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memcpy(enc_in_tensor->GetTensorMutableData<float>(),
              accumulated_features_.data() + static_cast<size_t>(window_start) * encoder_dim,
              static_cast<size_t>(window_size) * encoder_dim * sizeof(float));

  const char* enc_in_names[]  = {"features"};
  OrtValue*   enc_in_values[] = {enc_in_tensor.get()};
  const char* enc_out_names[] = {"encoded"};
  OrtValue*   enc_out_values[1] = {nullptr};
  moonshine_model_->session_encoder_->Run(
      nullptr,
      enc_in_names, enc_in_values, 1,
      enc_out_names, enc_out_values, 1);
  auto encoded = std::unique_ptr<OrtValue>(enc_out_values[0]);

  // Slice [:, start_idx : start_idx + new_frames] from encoded[1, window_size, encoder_dim].
  auto new_enc_shape = std::array<int64_t, 3>{1, new_frames, encoder_dim};
  auto new_enc_tensor =
      OrtValue::CreateTensor(alloc, new_enc_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memcpy(new_enc_tensor->GetTensorMutableData<float>(),
              encoded->GetTensorData<float>() +
                  static_cast<size_t>(start_idx) * encoder_dim,
              static_cast<size_t>(new_frames) * encoder_dim * sizeof(float));

  // ----- adapter.onnx ----------------------------------------------------
  auto pos_shape = std::array<int64_t, 1>{1};
  auto pos_tensor =
      OrtValue::CreateTensor(alloc, pos_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *pos_tensor->GetTensorMutableData<int64_t>() = adapter_pos_offset_;

  const char* ad_in_names[]  = {"encoded", "pos_offset"};
  OrtValue*   ad_in_values[] = {new_enc_tensor.get(), pos_tensor.get()};
  const char* ad_out_names[] = {"memory"};
  OrtValue*   ad_out_values[1] = {nullptr};
  moonshine_model_->session_adapter_->Run(
      nullptr,
      ad_in_names, ad_in_values, 2,
      ad_out_names, ad_out_values, 1);
  auto memory_tensor = std::unique_ptr<OrtValue>(ad_out_values[0]);

  // Append adapter output to accumulated memory.
  auto mshape = memory_tensor->GetTensorTypeAndShapeInfo()->GetShape();
  const int produced_frames = (mshape.size() >= 3) ? static_cast<int>(mshape[1]) : 0;
  if (produced_frames > 0) {
    const float* mdata = memory_tensor->GetTensorData<float>();
    accumulated_memory_.insert(accumulated_memory_.end(), mdata,
                               mdata + static_cast<size_t>(produced_frames) * decoder_dim);
    memory_len_ += produced_frames;
  }

  encoder_frames_emitted_ = stable_count;
  adapter_pos_offset_ += new_frames;
  cross_kv_valid_ = false;  // memory grew; cached cross-KV needs to catch up.
}

void MoonshineStreamingProcessor::RefreshCrossKv() {
  if (cross_kv_valid_ || memory_len_ == 0) return;

  auto& alloc = model_.allocator_cpu_;
  const int decoder_dim = config_.decoder_dim;
  const int new_frames = memory_len_ - memory_in_cross_kv_;
  if (new_frames <= 0) {
    cross_kv_valid_ = true;
    return;
  }

  // Run cross_kv.onnx on JUST the new memory frames. cross_kv is a pure
  // per-frame projection (Cast/MatMulInteger/Reshape/Transpose/Concat over
  // layers, no softmax / positional / cross-frame ops), so the projection
  // for the new rows is identical regardless of whether we re-project the
  // full memory each time.
  auto mem_shape = std::array<int64_t, 3>{1, new_frames, decoder_dim};
  auto mem_tensor =
      OrtValue::CreateTensor(alloc, mem_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memcpy(mem_tensor->GetTensorMutableData<float>(),
              accumulated_memory_.data() +
                  static_cast<size_t>(memory_in_cross_kv_) * decoder_dim,
              static_cast<size_t>(new_frames) * decoder_dim * sizeof(float));

  const char* ck_in_names[]  = {"memory"};
  OrtValue*   ck_in_values[] = {mem_tensor.get()};
  const char* ck_out_names[] = {"k_cross", "v_cross"};
  OrtValue*   ck_out_values[2] = {nullptr, nullptr};
  moonshine_model_->session_cross_kv_->Run(
      nullptr,
      ck_in_names, ck_in_values, 1,
      ck_out_names, ck_out_values, 2);

  std::unique_ptr<OrtValue> k_new(ck_out_values[0]);
  std::unique_ptr<OrtValue> v_new(ck_out_values[1]);

  if (memory_in_cross_kv_ == 0) {
    // First chunk of the stream: nothing to concat with.
    cached_k_cross_ = std::make_shared<Tensor>(std::move(k_new));
    cached_v_cross_ = std::make_shared<Tensor>(std::move(v_new));
  } else {
    // Concat cached [L,1,H,M_old,D] + new [L,1,H,new_frames,D] along dim 3.
    const int L = config_.num_decoder_layers;
    const int H = config_.num_decoder_heads;
    const int D = config_.decoder_head_size;
    const int M_old = memory_in_cross_kv_;
    const int M_total = memory_len_;
    auto concat_shape = std::array<int64_t, 5>{L, 1, H, M_total, D};

    auto concat_one = [&](const float* old_data, const float* new_data) {
      auto out =
          OrtValue::CreateTensor(alloc, concat_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      float* dst = out->GetTensorMutableData<float>();
      const size_t row_old = static_cast<size_t>(M_old) * D;
      const size_t row_new = static_cast<size_t>(new_frames) * D;
      const size_t row_total = static_cast<size_t>(M_total) * D;
      for (int l = 0; l < L; ++l) {
        for (int h = 0; h < H; ++h) {
          const size_t lh = static_cast<size_t>(l) * H + h;
          std::memcpy(dst + lh * row_total,
                      old_data + lh * row_old,
                      row_old * sizeof(float));
          std::memcpy(dst + lh * row_total + row_old,
                      new_data + lh * row_new,
                      row_new * sizeof(float));
        }
      }
      return out;
    };

    auto k_full = concat_one(cached_k_cross_->ort_tensor_->GetTensorData<float>(),
                             k_new->GetTensorData<float>());
    auto v_full = concat_one(cached_v_cross_->ort_tensor_->GetTensorData<float>(),
                             v_new->GetTensorData<float>());
    cached_k_cross_ = std::make_shared<Tensor>(std::move(k_full));
    cached_v_cross_ = std::make_shared<Tensor>(std::move(v_full));
  }

  memory_in_cross_kv_ = memory_len_;
  cross_kv_valid_ = true;
}

std::unique_ptr<NamedTensors> MoonshineStreamingProcessor::EmitCrossKv(bool is_final) {
  if (!cached_k_cross_ || !cached_v_cross_) return nullptr;
  auto result = std::make_unique<NamedTensors>();
  result->emplace("k_cross", cached_k_cross_);
  result->emplace("v_cross", cached_v_cross_);

  // Tell the State whether this is the final chunk of the utterance, so it
  // can decide whether to commit only the longest-common-prefix vs the
  // previous pass, or every token of the current pass.
  auto& alloc = model_.allocator_cpu_;
  auto if_shape = std::array<int64_t, 1>{1};
  auto if_tensor =
      OrtValue::CreateTensor(alloc, if_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *if_tensor->GetTensorMutableData<int64_t>() = is_final ? 1 : 0;
  result->emplace("is_final", std::make_shared<Tensor>(std::move(if_tensor)));
  return result;
}

}  // namespace Generators
