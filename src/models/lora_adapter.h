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
#include "../span.h"
#include "../tensor.h"

namespace Generators {

namespace details {

/// <summary>
/// A named Lora Parameters pair
/// </summary>
struct LoraParam {
  std::string name_;
  std::shared_ptr<Tensor> p_;
  LoraParam(std::string name, std::shared_ptr<Tensor> p) : name_{std::move(name)}, p_{std::move(p)} {}
};

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
  void AddParameter(LoraParam parameter) {
    if (active_) {
      throw std::runtime_error("Adapter: " + name_ + " is active can not add parameters");
    }

    auto p = parameters_.emplace(parameter.name_, std::move(parameter));
    if (!p.second) {
      throw std::runtime_error("Adapter: " + name_ + " already has a parameter named: " + parameter.name_);
    }
  }


 private:
  std::string name_;
  std::unordered_map<std::string_view, LoraParam> parameters_;
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

  /// <summary>
  /// Add named Lora Parameter to the specified adapter
  /// </summary>
  /// <param name="adapter_name"></param>
  /// <param name="param_name"></param>
  /// <param name="p"></param>
  void AddParameter(const std::string& adapter_name, std::string param_name, std::shared_ptr<Tensor> p) {

    auto& adapter = adapters_[adapter_name];
    if (adapter.GetName().empty()) {
      adapter.SetName(adapter_name);
    }

    details::LoraParam param(std::move(param_name), std::move(p));
    adapter.AddParameter(std::move(param));
  }

  void RemoveAdapter(const std::string& adapter_name) {
    auto hit = adapters_.find(adapter_name);
    if (hit == adapters_.end()) {
      throw std::runtime_error("Adapter: " + adapter_name + " does not exist");
    }

    if (hit->second.IsActive()) {
      throw std::runtime_error("Adapter: " + adapter_name + " is active and can not be deleted");
    }

   // Make sure all cache entries are invalidated
   // when cache is present

    adapters_.erase(hit);
  }

  void ActivateAdapter(const std::string& adapter_name) {
    auto hit = adapters_.find(adapter_name);
    if (hit == adapters_.end()) {
      throw std::runtime_error("Adapter: " + adapter_name + " does not exist");
    }

    if (hit->second.IsActive()) {
      throw std::runtime_error("Adapter: " + adapter_name + " is already active");
    }

    hit->second.SetActive();
    active_adapters_.push_back(adapter_name);
  }

  std::span<std::string> GetActiveAdapters() const {
    return active_adapters_;
  }

 private:
  using AdapterMap = std::unordered_map<std::string, details::LoraAdapter>;
  AdapterMap adapters_;
  std::vector<std::string> active_adapters_;

};

}  // namespace Generators
