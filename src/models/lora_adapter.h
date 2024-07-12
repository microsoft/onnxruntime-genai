// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_api.h"

#include "../tensor.h"
#include <mutex>
#include <memory>
#include <set>
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
  void SetName(const std::string& name) { name_ = name; }

  /// <summary>
  /// Returns number of parameters for buffer estimates
  /// </summary>
  size_t GetParamNum() const noexcept {
    std::shared_lock lock(mutex_);
    return parameters_.size();
  }

  /// <summary>
  /// Add Lora Parameter to the adapter
  /// </summary>
  /// <param name="parameter"></param>
  void AddParameter(const Model* model, std::string param_name, const std::shared_ptr<Tensor>& tensor) {
    std::unique_lock lock(mutex_);

    auto hit =
        std::find_if(parameters_.begin(), parameters_.end(), [&](const auto& p) { return p.name_ == param_name; });

    if (hit != parameters_.end()) {
      throw std::runtime_error("Adapter: " + name_ + " already has a parameter named: " + param_name);
    }

    auto& param = parameters_.emplace_back(std::move(param_name), tensor);
    if (model != nullptr) {
      MakeDeviceCopyIfNeeded(*model, param);
    }
  }

  /// <summary>
  /// Outputs OrtValues for all of the parameters.
  /// If the model is not on CPU, it will copy the user supplied buffers to the target device
  /// if not already done so.
  /// </summary>
  /// <typeparam name="OutNameIter"></typeparam>
  /// <typeparam name="OutParamIter"></typeparam>
  /// <param name="model"></param>
  /// <param name="out_name"></param>
  /// <param name="out_param"></param>
  template <typename OutNameIter, typename OutParamIter>
  void GetParameters(const Model* model, OutNameIter& out_name, OutParamIter& out_param) const {
    std::shared_lock lock(mutex_);
    for (const auto& p : parameters_) {
      *out_name = p.name_;
      ++out_name;
      const auto& result_val = (p.ort_device_value_) ? p.ort_device_value_ : p.ort_user_supplied_value_;
      *out_param = result_val;
      ++out_param;
    }
  }

  /// <summary>
  /// Returns empty parameters for all the parameters in the adapter.
  /// </summary>
  /// <typeparam name="OutNameIter"></typeparam>
  /// <typeparam name="OutParamIter"></typeparam>
  /// <param name="out_name"></param>
  /// <param name="out_param"></param>
  template <typename OutNameIter, typename OutParamIter>
  void GetEmptyParameters(OutNameIter& out_name, OutParamIter& out_param) const {
    std::shared_lock lock(mutex_);
    // Must return the parameters in the same order they were added
    for (const auto& p : parameters_) {
      *out_name = p.name_;
      ++out_name;
      const auto& result_val = (p.ort_device_value_) ? p.ort_device_value_ : p.ort_user_supplied_value_;
      *out_param = CreateEmptyInput(*result_val);
      ++out_param;
    }
  }


 private:
  /// <summary>
  /// Create a copy if the parameter on device if not already done so.
  /// </summary>
  /// <param name="model"></param>
  /// <param name="param"></param>
  void MakeDeviceCopyIfNeeded(const Model& model, LoraParam& param);

  std::string name_;
  mutable std::shared_mutex mutex_;
  std::vector<LoraParam> parameters_;
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
  /// Outputs pointers to names and its corresponding OrtValue params
  /// for all active adapters.
  /// </summary>
  /// <typeparam name="NamesOutputIter"></typeparam>
  /// <typeparam name="TensorOutputIter"></typeparam>
  /// <param name="active_adapters">Set of active adapter names for this generator</param>
  /// <param name="names_out">Output Iterator for std::string</param>
  /// <param name="ort_values_out">Output Iterator for std::shared_ptr of OrtValue</param>
  template <class NamesOutputIter, class TensorOutputIter>
  void OutputAdaptersParameters(const std::set<std::string>& adapter_names, NamesOutputIter names_out,
                                TensorOutputIter params_out) const {
    std::shared_lock lock(mutex_);
    // Output Parameters for adapters in adapter_names
    // otherwise output empty parameters
    for (const auto& [name, adapter] : adapters_) {
      if (adapter_names.find(name) == adapter_names.end()) {
        adapter.GetEmptyParameters(names_out, params_out);
      } else {
        adapter.GetParameters(model_, names_out, params_out);
      }
    }
  }

  /// <summary>
  /// Returns total number of parameters across all adapters
  /// </summary>
  /// <returns></returns>
  size_t GetParamNum() const noexcept {
    size_t result = 0;
    std::shared_lock lock(mutex_);
    for (const auto& [_, adapter] : adapters_) {
      result += adapter.GetParamNum();
    }
    return result;
  }

  /// <summary>
  /// Checks if the adapter exists
  /// </summary>
  /// <param name="adapter_name"></param>
  /// <returns>true if so</returns>
  bool HasAdapter(const std::string& adapter_name) const noexcept {
    std::shared_lock lock(mutex_);
    return adapters_.find(adapter_name) != adapters_.end();
  }

 private:
  const Model* model_;  // optional
  mutable std::shared_mutex mutex_;
  using AdapterMap = std::unordered_map<std::string, details::LoraAdapter>;
  AdapterMap adapters_;
};

}  // namespace Generators
