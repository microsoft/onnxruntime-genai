// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_api.h"

#include "../span.h"
#include "../tensor.h"

#include <memory>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Generators {

struct Model;

namespace details {

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
/// <returns>A OrtValue over a dummy buffer.</returns>
std::shared_ptr<OrtValue> CreateEmptyInput(const OrtValue& tensor);


/// <summary>
/// A named Lora Parameters pair
/// </summary>
struct LoraParam {
  LoraParam() = default;
  LoraParam(std::string name, const std::shared_ptr<Tensor>& parameter);
  std::string name_;
  // We need this as a shared_ptr so we can share while inference is running.
  // This can also be a subject of weak caching
  // This is created over the same user supplied buffer as the originally
  // passed in tensor.
  std::shared_ptr<OrtValue> ort_user_supplied_value_;
  // Copy on device if needed.
  // XXX: Insert caching logic and convert to weak_ptr
  std::shared_ptr<OrtValue> ort_device_value_;
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
  LoraAdapter() = default;
  LoraAdapter(const LoraAdapter&) = delete;
  LoraAdapter& operator=(const LoraAdapter&) = delete;

  /// <summary>
  /// Set name after construction
  /// </summary>
  /// <param name="name"></param>
  void SetName(const std::string& name) {
    name_ = name;
  }

  // Returns if the adapter is active
  bool IsActive() const noexcept {
    std::shared_lock lock(mutex_);
    return active_;
  }

  /// <summary>
  /// Activates the adapter. Once activated, no more parameters can be added.
  /// </summary>
  void SetActive(const Model* model);

  /// <summary>
  /// Deactivates the adapter.
  /// </summary>
  void Deactivate() noexcept {
    std::unique_lock lock(mutex_);
    active_ = false;
  }

  /// <summary>
  /// Add Lora Parameter to the adapter
  /// </summary>
  /// <param name="parameter"></param>
  void AddParameter(std::string param_name, const std::shared_ptr<Tensor>& tensor) {
    std::unique_lock lock(mutex_);

    auto hit =
        std::find_if(parameters_.begin(), parameters_.end(), [&](const auto& p) { return p.name_ == param_name; });

    if (hit != parameters_.end()) {
      throw std::runtime_error("Adapter: " + name_ + " already has a parameter named: " + param_name);
    }

    parameters_.emplace_back(std::move(param_name), tensor);
  }

  /// <summary>
  /// Outputs parameters in the order they were added.
  /// </summary>
  /// <typeparam name="OutNameIter"></typeparam>
  /// <typeparam name="OutParamIter"></typeparam>
  /// <param name="out_name"></param>
  /// <param name="out_param"></param>
  template <typename OutNameIter, typename OutParamIter>
  void GetParameters(OutNameIter& out_name, OutParamIter& out_param) const noexcept {
    std::shared_lock lock(mutex_);
    // Must return the parameters in the same order they were added
    for (const auto& p : parameters_) {
      *out_name = p.name_;
      ++out_name;
      const auto& result_val = (p.ort_device_value_) ? p.ort_device_value_ : p.ort_user_supplied_value_;
      if (active_) {
        *out_param = result_val;
      } else {
        *out_param = CreateEmptyInput(*result_val);
      }
      ++out_param;
    }
  }

 private:
  std::string name_;
  mutable std::shared_mutex mutex_;
  std::vector<LoraParam> parameters_;
  bool active_{false};
};

}  // namespace details

/// <summary>
/// This class manages the collection of Lora Adapters
/// </summary>
class LoraAdapterManagement {
 public:
  LoraAdapterManagement(const Model* model);
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
  void AddParameter(const std::string& adapter_name, std::string param_name, const std::shared_ptr<Tensor>& p);

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
  void DeactivateAdapters(std::span<const std::string> adapter_names);

  /// <summary>
  /// Deactivates one or more adapter(s) that active.
  /// No error is reported.
  /// </summary>
  /// <param name="adapter_names">adapters to deactivate</param>
  void DeactiveAllAdapters();

  /// <summary>
  /// Retrieves names of all active adapters
  /// </summary>
  /// <returns>a vector of C strings</returns>
  std::vector<const char*> GetActiveAdapterNames() const;

  /// <summary>
  /// Outputs pointers to names and its corresponding OrtValue params
  /// for all active adapters.
  /// </summary>
  /// <typeparam name="NamesOutputIter"></typeparam>
  /// <typeparam name="TensorOutputIter"></typeparam>
  /// <param name="names_out">Output Iterator for std::string</param>
  /// <param name="ort_values_out">Output Iterator for std::shared_ptr<OrtValue></param>
  template <class NamesOutputIter, class TensorOutputIter>
  void OutputAdaptersParameters(NamesOutputIter names_out, TensorOutputIter params_out) const {
    std::shared_lock lock(mutex_);
    for (const auto& [_, adapter] : adapters_) {
        adapter.GetParameters(names_out, params_out);
    }
  }

 private:
  const Model* model_; // optional
  mutable std::shared_mutex mutex_;
  using AdapterMap = std::unordered_map<std::string, details::LoraAdapter>;
  AdapterMap adapters_;
};

}  // namespace Generators
