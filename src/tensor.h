// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
namespace Generators {

struct Tensor : std::enable_shared_from_this<Tensor>, LeakChecked<Tensor>, ExternalRefCounted<Tensor> {
  Tensor(DeviceInterface* device, ONNXTensorElementDataType type);
  // This constructor assumes CPU device and is mostly useful for external use via API
  Tensor(std::unique_ptr<OrtValue> ort_tensor);
  ~Tensor();

  // A static tensor is allocated once on a buffer which is reused
  // A non-static tensor is allocated as a new OrtValue every time CreateTensor is called
  void CreateTensor(std::span<const int64_t> shape, bool make_static = false);

  void MakeStatic();  // Make the tensor static, if it is not already

  OrtValue* GetOrtTensor();

  template <typename T>
  DeviceSpan<T> GetDeviceSpan() {
    if (ort_tensor_ == nullptr)
      throw std::runtime_error("Tensor: GetDeviceSpan called before CreateTensor");
    return p_device_->WrapMemory(std::span<T>{ort_tensor_->GetTensorMutableData<T>(), GetElementCount()});
  }

  DeviceSpan<uint8_t> GetByteSpan();

  template <typename T>
  T* GetMutableData() {
    if (ort_tensor_ == nullptr)
      throw std::runtime_error("Tensor: GetMutableData called before CreateTensor");
    return ort_tensor_->GetTensorMutableData<T>();
  }

  template <typename T>
  const T* GetData() const {
    if (ort_tensor_ == nullptr)
      throw std::runtime_error("Tensor: GetData called before CreateTensor");
    return ort_tensor_->GetTensorData<T>();
  }

  void* GetMutableRawData();
  const void* GetRawData() const;

  std::vector<int64_t> GetShape() const;

  ONNXTensorElementDataType GetType() const;

  size_t GetElementCount() const;

  std::unique_ptr<OrtValue> ort_tensor_;
  mutable DeviceInterface* p_device_{};
  ONNXTensorElementDataType type_;
  // For static tensors, allocated once
  void* buffer_{};
  size_t bytes_{};
  bool is_static_{};
};

using NamedTensors = std::unordered_map<std::string, std::shared_ptr<Tensor>>;

}  // namespace Generators