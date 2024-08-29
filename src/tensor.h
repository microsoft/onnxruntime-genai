// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
namespace Generators {

struct Tensor : std::enable_shared_from_this<Tensor>, LeakChecked<Tensor> {
  Tensor() = default;
  Tensor(std::unique_ptr<OrtValue> ort_tensor) : ort_tensor_{std::move(ort_tensor)} {}

  std::unique_ptr<OrtValue> ort_tensor_;
  std::shared_ptr<Tensor> external_owner_;  // Set to 'this' when created by the C API to preserve lifetime
};

using NamedTensors = std::unordered_map<std::string, std::shared_ptr<Tensor>>;

}  // namespace Generators