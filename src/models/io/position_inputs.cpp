#include "position_inputs.h"

#include <algorithm>

#include "default_position_inputs.h"
#include "qwen_vl_position_inputs.h"
#include "windowed_position_inputs.h"
#include "models/model.h"
#include "models/model_type.h"

namespace Generators {

void PositionIdsBuffer::Create(Tensor& tensor, std::span<const int64_t> shape) const {
  tensor.CreateTensor(shape);
}

void PositionIdsBuffer::Assign(Tensor& tensor, const Model& model, std::unique_ptr<OrtValue>& values, int num_beams) const {
  if (num_beams == 1) {
    Create(tensor, values->GetTensorTypeAndShapeInfo()->GetShape());
    tensor.GetByteSpan().CopyFrom(ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *values));
  } else {
    auto expanded = model.ExpandInputs(values, num_beams);
    Create(tensor, expanded->GetTensorTypeAndShapeInfo()->GetShape());
    tensor.GetByteSpan().CopyFrom(ByteWrapTensor(*model.p_device_inputs_, *expanded));
  }
}

namespace {

struct CapturedPositionIdsBuffer : PositionIdsBuffer {
  explicit CapturedPositionIdsBuffer(int max_capture_length) : max_capture_length_{max_capture_length} {}

  void Create(Tensor& tensor, std::span<const int64_t> shape) const override {
    const bool capture = shape.back() >= 1 && shape.back() <= max_capture_length_;
    // Pre-size for every captured length, including a short initial prompt.
    size_t capacity = Ort::SizeOf(tensor.GetType()) * max_capture_length_;
    for (size_t i = 0; i + 1 < shape.size(); ++i)
      capacity *= shape[i];
    tensor.CreateTensor(shape, capture, capture ? capacity : 0);
  }

 private:
  int max_capture_length_;
};

// Both mask policies use the same device operation and CPU fallback.
void UpdateMaskOnDevice(Tensor& mask, Tensor* next, const std::array<int64_t, 2>& shape,
                        int total_length, int new_length, int max_length) {
  auto& device = *mask.p_device_;
  if (!device.UpdateAttentionMask(next ? next->GetMutableRawData() : nullptr, mask.GetMutableRawData(),
                                  static_cast<int>(shape[0]), new_length, total_length, max_length,
                                  next == nullptr, mask.GetType())) {
    auto mask_span = mask.GetByteSpan();
    auto next_span = next ? next->GetByteSpan() : DeviceSpan<uint8_t>{};
    device.GetCpuFallbackDevice().UpdateAttentionMask(
        next ? next_span.CopyDeviceToCpu().data() : nullptr, mask_span.CopyDeviceToCpu().data(),
        static_cast<int>(shape[0]), new_length, total_length, max_length, next == nullptr, mask.GetType());
    if (next)
      next_span.CopyCpuToDevice();
    else
      mask_span.CopyCpuToDevice();
  }
}

struct DynamicAttentionMaskBuffer : AttentionMaskBuffer {
  DynamicAttentionMaskBuffer(const Model& model, DynamicMaskUpdate update) : model_{model}, update_{update} {}

  void Initialize(Tensor& tensor, std::unique_ptr<OrtValue>& values, int num_beams,
                  std::array<int64_t, 2>& shape) const override {
    tensor.ort_tensor_ = model_.ExpandInputs(values, num_beams);
    shape[0] *= num_beams;
  }

  void Update(Tensor& tensor, std::array<int64_t, 2>& shape, int total_length, int new_length) const override {
    shape[1] = total_length;
    Tensor next{tensor.p_device_, tensor.GetType()};
    next.CreateTensor(shape);
    if (update_ == DynamicMaskUpdate::FillOnes) {
      auto span = next.GetByteSpan();
      auto bytes = span.CpuSpan();
      if (tensor.GetType() == Ort::TypeToTensorType<int32_t>)
        std::fill_n(reinterpret_cast<int32_t*>(bytes.data()), shape[0] * shape[1], int32_t{1});
      else
        std::fill_n(reinterpret_cast<int64_t*>(bytes.data()), shape[0] * shape[1], int64_t{1});
      span.CopyCpuToDevice();
    } else {
      UpdateMaskOnDevice(tensor, &next, shape, total_length, new_length, 0);
    }
    tensor.ort_tensor_ = std::move(next.ort_tensor_);
  }

  void Rewind(Tensor&, std::array<int64_t, 2>& shape, size_t index) const override {
    shape[1] = static_cast<int64_t>(index);
  }

 private:
  const Model& model_;
  DynamicMaskUpdate update_;
};

struct StaticAttentionMaskBuffer : AttentionMaskBuffer {
  explicit StaticAttentionMaskBuffer(int max_length) : max_length_{max_length} {}

  void Initialize(Tensor& tensor, std::unique_ptr<OrtValue>& values, int num_beams,
                  std::array<int64_t, 2>& shape) const override {
    if (shape[1] > max_length_)
      throw std::runtime_error("Attention mask prompt exceeds max_length");
    const auto input_shape = shape;
    shape[0] *= num_beams;
    // Shared KV providers derive the present capacity from the mask width.
    shape[1] = max_length_;
    tensor.CreateTensor(shape, true);
    auto output = tensor.GetByteSpan();
    output.Zero();
    auto input = ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *values);
    const size_t row_bytes = input_shape[1] * Ort::SizeOf(tensor.GetType());
    const size_t stride = max_length_ * Ort::SizeOf(tensor.GetType());
    for (int64_t b = 0; b < input_shape[0]; ++b)
      for (int beam = 0; beam < num_beams; ++beam)
        output.subspan((b * num_beams + beam) * stride, row_bytes).CopyFrom(input.subspan(b * row_bytes, row_bytes));
  }

  void Update(Tensor& tensor, std::array<int64_t, 2>& shape, int total_length, int new_length) const override {
    if (new_length <= 0 || total_length < new_length || total_length > max_length_)
      throw std::runtime_error("Attention mask update exceeds allocated capacity");
    UpdateMaskOnDevice(tensor, nullptr, shape, total_length, new_length, max_length_);
  }

  void Rewind(Tensor& tensor, std::array<int64_t, 2>& shape, size_t index) const override {
    if (index > static_cast<size_t>(max_length_))
      throw std::runtime_error("Attention mask rewind exceeds max_length");
    auto span = tensor.GetByteSpan();
    auto data = span.CopyDeviceToCpu();
    const size_t row_bytes = max_length_ * Ort::SizeOf(tensor.GetType());
    for (int64_t b = 0; b < shape[0]; ++b)
      std::fill(data.begin() + b * row_bytes + index * Ort::SizeOf(tensor.GetType()),
                data.begin() + (b + 1) * row_bytes, uint8_t{0});
    span.CopyCpuToDevice();
  }

 private:
  int max_length_;
};

}  // namespace

std::unique_ptr<PositionIdsBuffer> CreatePositionIdsBuffer(const GeneratorParams& params) {
  if (params.use_graph_capture)
    return std::make_unique<CapturedPositionIdsBuffer>(params.max_graph_capture_length);
  return std::make_unique<PositionIdsBuffer>();
}

std::unique_ptr<AttentionMaskBuffer> CreateAttentionMaskBuffer(
    const Model& model, const GeneratorParams& params, DynamicMaskUpdate update) {
  if (params.use_graph_capture ||
      (params.IsPastPresentShareBufferEnabled(model.config_->model.type) &&
       model.p_device_inputs_->ShouldUseStaticPositionInputsForSharedBuffers(model.config_->model)))
    return std::make_unique<StaticAttentionMaskBuffer>(params.search.max_length);
  return std::make_unique<DynamicAttentionMaskBuffer>(model, update);
}

std::unique_ptr<PositionInputs> CreateStandardPositionInputs(State& state, DeviceSpan<int32_t> sequence_lengths, const std::string& attention_mask_name) {
  // Check for Qwen-VL family models which require 3D mRoPE position IDs
  if (ModelType::IsQwenVLFamily(state.model_.config_->model.type)) {
    return std::make_unique<Qwen2VLPositionInputs>(state.model_, state, sequence_lengths);
  }
  if (state.model_.config_->model.decoder.sliding_window.has_value() && state.model_.config_->model.decoder.sliding_window->slide_inputs) {
    return std::make_unique<WindowedPositionInputs>(state);
  } else {
    return std::make_unique<DefaultPositionInputs>(state.model_, state, sequence_lengths, attention_mask_name);
  }
}

}  // namespace Generators
