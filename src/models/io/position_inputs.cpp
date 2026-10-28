#include "position_inputs.h"

#include <algorithm>

#include "default_position_inputs.h"
#include "qwen_vl_position_inputs.h"
#include "windowed_position_inputs.h"
#include "models/model.h"
#include "models/model_type.h"

namespace Generators {

namespace {

// The batch axis precedes the sequence axis in both 2D IDs and 3D mRoPE IDs.
// Model::ExpandInputs expands axis 0, which is the rotary axis for mRoPE.
void ExpandPositionIds(const Model& model, std::unique_ptr<OrtValue>& values, int num_beams) {
  if (num_beams == 1)
    return;
  auto shape = values->GetTensorTypeAndShapeInfo()->GetShape();
  const auto type = values->GetTensorTypeAndShapeInfo()->GetElementType();
  const size_t row_bytes = shape.back() * Ort::SizeOf(type);
  const size_t rows = values->GetTensorTypeAndShapeInfo()->GetElementCount() / shape.back();
  shape[shape.size() - 2] *= num_beams;
  auto expanded = OrtValue::CreateTensor(model.allocator_cpu_, shape, type);
  auto& cpu = *GetDeviceInterface(DeviceType::CPU);
  auto input = ByteWrapTensor(cpu, *values);
  auto output = ByteWrapTensor(cpu, *expanded);
  for (size_t row = 0; row < rows; ++row)
    for (int beam = 0; beam < num_beams; ++beam)
      output.subspan((row * num_beams + beam) * row_bytes, row_bytes).CopyFrom(input.subspan(row * row_bytes, row_bytes));
  values = std::move(expanded);
}

}  // namespace

void PositionIdsBuffer::Create(Tensor& tensor, std::span<const int64_t> shape) const {
  tensor.CreateTensor(shape);
}

void PositionIdsBuffer::Assign(Tensor& tensor, const Model& model, std::unique_ptr<OrtValue>& values, int num_beams) const {
  ExpandPositionIds(model, values, num_beams);
  tensor.ort_tensor_ = model.ExpandInputs(values, 1);
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

  void Assign(Tensor& tensor, const Model& model, std::unique_ptr<OrtValue>& values, int num_beams) const override {
    auto shape = values->GetTensorTypeAndShapeInfo()->GetShape();
    if (shape.back() < 1 || shape.back() > max_capture_length_) {
      // Tensor retains its static allocation across longer, uncaptured prefills after rewind.
      PositionIdsBuffer::Assign(tensor, model, values, num_beams);
      return;
    }
    ExpandPositionIds(model, values, num_beams);
    shape = values->GetTensorTypeAndShapeInfo()->GetShape();
    Create(tensor, shape);
    tensor.GetByteSpan().CopyFrom(ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *values));
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
                  std::array<int64_t, 2>& shape) override {
    tensor.ort_tensor_ = model_.ExpandInputs(values, num_beams);
    shape[0] *= num_beams;
  }

  void Update(Tensor& tensor, std::array<int64_t, 2>& shape, int total_length, int new_length) override {
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

  void Rewind(Tensor&, std::array<int64_t, 2>& shape, size_t index) override {
    shape[1] = static_cast<int64_t>(index);
  }

 private:
  const Model& model_;
  DynamicMaskUpdate update_;
};

struct StaticAttentionMaskBuffer : AttentionMaskBuffer {
  explicit StaticAttentionMaskBuffer(int max_length) : max_length_{max_length} {}

  void Initialize(Tensor& tensor, std::unique_ptr<OrtValue>& values, int num_beams,
                  std::array<int64_t, 2>& shape) override {
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

  void Update(Tensor& tensor, std::array<int64_t, 2>& shape, int total_length, int new_length) override {
    if (new_length <= 0 || total_length < new_length || total_length > max_length_)
      throw std::runtime_error("Attention mask update exceeds allocated capacity");
    UpdateMaskOnDevice(tensor, nullptr, shape, total_length, new_length, max_length_);
  }

  void Rewind(Tensor& tensor, std::array<int64_t, 2>& shape, size_t index) override {
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

// WebGPU's device mask update assumes an unpadded prefix. Keep a CPU copy so
// prompt padding and rewound entries survive uploads without device readback.
struct CpuStagedAttentionMaskBuffer : StaticAttentionMaskBuffer {
  using StaticAttentionMaskBuffer::StaticAttentionMaskBuffer;

  void Initialize(Tensor& tensor, std::unique_ptr<OrtValue>& values, int num_beams,
                  std::array<int64_t, 2>& shape) override {
    if (!cpu_mask_)
      cpu_mask_ = std::make_unique<Tensor>(GetDeviceInterface(DeviceType::CPU), tensor.GetType());
    StaticAttentionMaskBuffer::Initialize(*cpu_mask_, values, num_beams, shape);
    tensor.CreateTensor(shape, true);
    tensor.GetByteSpan().CopyFrom(cpu_mask_->GetByteSpan());
  }

  void Update(Tensor& tensor, std::array<int64_t, 2>& shape, int total_length, int new_length) override {
    StaticAttentionMaskBuffer::Update(*cpu_mask_, shape, total_length, new_length);
    tensor.GetByteSpan().CopyFrom(cpu_mask_->GetByteSpan());
  }

  void Rewind(Tensor& tensor, std::array<int64_t, 2>& shape, size_t index) override {
    StaticAttentionMaskBuffer::Rewind(*cpu_mask_, shape, index);
    tensor.GetByteSpan().CopyFrom(cpu_mask_->GetByteSpan());
  }

 private:
  std::unique_ptr<Tensor> cpu_mask_;
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
       model.p_device_inputs_->ShouldUseStaticPositionInputsForSharedBuffers(model.config_->model))) {
    if (model.p_device_inputs_->GetType() == DeviceType::WEBGPU)
      return std::make_unique<CpuStagedAttentionMaskBuffer>(params.search.max_length);
    return std::make_unique<StaticAttentionMaskBuffer>(params.search.max_length);
  }
  return std::make_unique<DynamicAttentionMaskBuffer>(model, update);
}

std::unique_ptr<PositionInputs> CreateStandardPositionInputs(State& state, DeviceSpan<int32_t> sequence_lengths, const std::string& attention_mask_name) {
  // Get size of position ids in model
  const auto& model = state.model_;
  const auto& position_ids_name = model.config_->model.decoder.inputs.position_ids;
  const bool has_position_ids = model.session_info_.HasInput(position_ids_name);
  const bool has_3d_position_ids = has_position_ids && model.session_info_.GetInputShape(position_ids_name).size() == 3;

  // Check for Qwen-VL family models which require 3D mRoPE position IDs
  if (ModelType::IsQwenVLFamily(state.model_.config_->model.type) && has_3d_position_ids) {
    return std::make_unique<Qwen2VLPositionInputs>(state.model_, state, sequence_lengths);
  }
  if (state.model_.config_->model.decoder.sliding_window.has_value() && state.model_.config_->model.decoder.sliding_window->slide_inputs) {
    return std::make_unique<WindowedPositionInputs>(state);
  } else {
    return std::make_unique<DefaultPositionInputs>(state.model_, state, sequence_lengths, attention_mask_name);
  }
}

}  // namespace Generators
