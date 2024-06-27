// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "static_buffer.h"
#include "../generators.h"
#include "../span.h"
#include "../tensor.h"

namespace Generators {

namespace details {

/// <summary>
/// A named Lora Parameters pair
/// </summary>
struct LoraParam : public GeneratorParams::Input {
  LoraParam() = default;
  LoraParam(std::string p_name, std::shared_ptr<Tensor> p) {
    name = std::move(p_name);
    tensor = std::move(p);
  }
};

std::string LoraCacheKey(std::string_view adapter_name, std::string param_name);

static std::shared_ptr<OrtValue> CreateEmptyInput(ONNXTensorElementDataType type);

/// <summary>
/// This class represents a collection of named pairs of
/// Lora Parameters A and B represented by tensors. They are created by the user to be fed
/// into matching (by name) ONNX model inputs.
/// </summary>
class LoraAdapter {
 public:
  /// <summary>
  /// Construct a named adapter
  /// </summary>
  /// <param name="name">a name. LoraAdapterManagement class would make sure it is unique.</param>
  LoraAdapter() = default;

  /// <summary>
  /// Returns adapter name
  /// </summary>
  /// <returns></returns>
  const std::string& GetName() const noexcept { return name_; }

  /// <summary>
  /// Assigns a name to the adapter
  /// </summary>
  /// <param name="name"></param>
  void SetName(std::string name) { name_ = std::move(name); }

  // Returns if the adapter is active
  bool IsActive() const noexcept { return active_; }

  /// <summary>
  /// Activates the adapter. Once activated, no more parameters can be added.
  /// </summary>
  void SetActive() {
    if (active_) {
      throw std::runtime_error("Adapter: " + name_ + " has already been activated");
    }
    // Make sure data is copied to devices as needed
  }

  void Deactivate() {
    active_ = false;
  }

  /// <summary>
  /// Add Lora Parameter to the adapter
  /// </summary>
  /// <param name="parameter"></param>
  void AddParameter(std::string param_name, std::shared_ptr<Tensor> tensor) {
    LoraParam param{param_name, std::move(tensor)};
    auto p = parameters_.emplace(std::move(param_name), param);
    if (!p.second) {
      throw std::runtime_error("Adapter: " + name_ + " already has a parameter named: " + param_name);
    }
  }

 private:

  std::string name_;
  std::unordered_map<std::string, LoraParam> parameters_;
  bool active_{false};
};

}  // namespace details

/// <summary>
/// This class manages the collection of Lora Adapaters
/// </summary>
class LoraAdapaterManagement {
 public:
  LoraAdapaterManagement() = default;
  ~LoraAdapaterManagement() = default;
  LoraAdapaterManagement(const LoraAdapaterManagement&) = delete;
  LoraAdapaterManagement& operator=(const LoraAdapaterManagement&) = delete;

  /// <summary>
  /// Creates a adapter object to which one can add Lora parameters
  /// </summary>
  /// <param name="adapter_name"></param>
  /// <param name="max_beam_batch_size"></param>
  void CreateAdapter(const std::string& adapter_name) {
    auto result = adapters_.emplace(adapter_name, details::LoraAdapter{});
    if (!result.second) {
      throw std::runtime_error("Adapter: " + adapter_name + " already exist");
    }
    result.first->second.SetName(adapter_name);
  }

  /// <summary>
  /// Add named Lora Parameter to the specified adapter
  /// </summary>
  /// <param name="adapter_name"></param>
  /// <param name="param_name"></param>
  /// <param name="p"></param>
  void AddParameter(const std::string& adapter_name, std::string param_name, std::shared_ptr<Tensor> p);

  /// <summary>
  /// Remove Specific Lora adapter and purge device cache if appropriate.
  /// </summary>
  /// <param name="adapter_name"></param>
  void RemoveAdapter(const std::string& adapter_name);

  /// <summary>
  /// Activate specific adapter. More than one adapter can be active at the same time.
  /// Except for the base scenario.
  /// </summary>
  /// <param name="adapter_name"></param>
  void ActivateAdapter(const std::string& adapter_name);

  void DeactiveAdapter(const std::string& adapter_name);

 private:

  using AdapterMap = std::unordered_map<std::string, details::LoraAdapter>;
  AdapterMap adapters_;
};

}  // namespace Generators
