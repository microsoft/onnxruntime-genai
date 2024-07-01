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

// std::string LoraCacheKey(std::string_view adapter_name, std::string param_name);

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
    if (parameters_.empty()) {
      throw std::runtime_error("Adapter: " + name_ + " has no parameters");
    }
    active_ = true;
  }

  /// <summary>
  /// Deactivates the adapter.
  /// </summary>
  void Deactivate() { active_ = false; }

  /// <summary>
  /// Add Lora Parameter to the adapter
  /// </summary>
  /// <param name="parameter"></param>
  void AddParameter(std::string param_name, std::shared_ptr<Tensor> tensor) {
    LoraParam param{param_name, std::move(tensor)};
    auto p = parameters_.emplace(std::move(param_name), std::move(param));
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
class LoraAdapterManagement {
 public:
  LoraAdapterManagement();
  ~LoraAdapterManagement() = default;
  LoraAdapterManagement(const LoraAdapterManagement&) = delete;
  LoraAdapterManagement& operator=(const LoraAdapterManagement&) = delete;

  /// <summary>
  /// Creates a adapter object to which one can add Lora parameters
  /// </summary>
  /// <param name="adapter_name"></param>
  /// <param name="max_beam_batch_size"></param>
  void CreateAdapter(const std::string& adapter_name);

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
  /// Activate one or more adapter(s). If any of the adapters are already active, an exception is thrown
  /// and no adapters are activated.
  /// </summary>
  /// <param name="adapter_names">adapters to activate</param>
  void ActivateAdapters(std::span<const std::string> adapter_names);

  /// <summary>
  /// Deactivate one or more adapters that are active.
  /// If any of the adapters are not active, the error is not reported,
  /// and it is a no-op for inactive adapters.
  /// </summary>
  /// <param name="adapter_names">adapter names to deactivate</param>
  void DeactiveAdapters(std::span<const std::string> adapter_names);

  /// <summary>
  /// Deactivates one or more adapter(s) that active.
  /// No error is reported.
  /// </summary>
  /// <param name="adapter_names">adapters to deactivate</param>
  void DeactiveAllAdapters();

  /// <summary>
  /// Retrieves names of all active adapters
  /// </summary>
  /// <returns>a vector of string views</returns>
  std::vector<std::string_view> GetActiveAdapterNames() const;

  /// <summary>
  /// Outputs pointers to names and its corresponding OrtValue params
  /// for all active adapters.
  /// </summary>
  /// <typeparam name="NamesOutputIter"></typeparam>
  /// <typeparam name="TensorOutputIter"></typeparam>
  /// <param name="names_out">Output Iterator for param C strings</param>
  /// <param name="ort_values_out">Output Iterator for Lora Param Tensor</param>
  template <class NamesOutputIter, class TensorOutputIter>
  void OutputAdaptersParameters(NamesOutputIter names_out, TensorOutputIter params_out) const {
    for (const auto& [_, adapter] : adapters_) {
      /// XXX: We need to generate empty inputs for inactive adapters,
      if (adapter.IsActive()) {
        for (const auto& [name, param] : adapter.parameters_) {
          *names_out = name.c_str();
          ++names_out;
          *params_out = param.tensor->ort_tensor_;
          ++params_out;
        }
      } else {
        for (const auto& [name, param] : adapter.parameters_) {
          *names_out = name.c_str();
          ++names_out;
          *params_out = CreateEmptyInput(*param.tensor)->ort_tensor_;
          ++params_out;
        }
      }
    }
  }

  /// <summary>
  /// Creates an empty input tensor for a given Lora parameter.
  /// It takes the customer supplied Lora parameter, inherits its memory info and
  /// data type. It the modifies the original shape to denote empty input in the following
  /// way:
  ///  The adapter dimensions (without batch) are
  ///   [hidden_dim, lora_r] and [lora_r, hidden_dim ].
  ///   The empty input shape we would pass would have lora_r set to 0.
  ///   To detect lora_r dim we simply zero out the dim of smaller value.
  ///   The resulting shape would be either [hidden_dim, 0] or [0, hidden_dim].
  /// </summary>
  /// <param name="tensor">Tensor supplied by the user for a Lora parameter</param>
  /// <returns>A Tensor that holds an OrtValue created over a dummy buffer.</returns>
  static std::shared_ptr<Tensor> CreateEmptyInput(const Tensor& tensor);

 private:
  using AdapterMap = std::unordered_map<std::string, details::LoraAdapter>;
  AdapterMap adapters_;
};

}  // namespace Generators
