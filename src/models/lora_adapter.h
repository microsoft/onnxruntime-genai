// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_api.h"

#include "../filesystem.h"
#include "../tensor.h"

#include <algorithm>
#include <iosfwd>
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

struct Config;

namespace lora_parameters {
struct Parameters;
} // namespace lora_parameters

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
  LoraParam(std::string name, std::shared_ptr<OrtValue> parameter);

  std::string name_;
  // We need this as a shared_ptr so we can share while inference is running.
  // This is created over the same user supplied buffer as the originally
  // passed in tensor.
  std::shared_ptr<OrtValue> ort_user_supplied_value_;
  // Copy on device if needed.
  // XXX: Insert caching logic and convert to weak_ptr when cache is implemented
  std::shared_ptr<OrtValue> ort_device_value_;
};

// This class takes hold of the serialized parameters that
// are either loaded from disk or mapped from disk (coming in the future)
// This data is always in host memory.
class BinaryFormatHolder {
 public:
  BinaryFormatHolder() = default;
  BinaryFormatHolder(const BinaryFormatHolder&) = delete;
  BinaryFormatHolder& operator=(const BinaryFormatHolder&) = delete;

  /// <summary>
  /// Load parameters from a flatbuffer file.
  /// </summary>
  /// <param name="file_name">file name that can be opened</param>
  void Load(const std::string& file_name);

  // Get the buffer
  const lora_parameters::Parameters* GetParameters() const noexcept { return parameters_; }

  // Get the size of the buffer
  size_t GetSize() const noexcept { return buffer_.size(); }

 private:
  std::vector<uint8_t> buffer_;
  const lora_parameters::Parameters* parameters_;
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
  /// Loads parameters from flatbuffer format.
  /// </summary>
  /// <param name="file_name"></param>
  void LoadParametersFromFlatBuffer(const std::string& file_name);

  /// <summary>
  /// Returns number of parameters for buffer estimates
  /// </summary>
  size_t GetParamNum() const noexcept {
    std::shared_lock lock(mutex_);
    return parameters_.size();
  }

  void AddParameter(std::string param_name, std::shared_ptr<OrtValue> ort_value) {
    std::unique_lock lock(mutex_);

    auto hit =
        std::find_if(parameters_.begin(), parameters_.end(), [&](const auto& p) { return p.name_ == param_name; });

    if (hit != parameters_.end()) {
      throw std::runtime_error("Adapter: " + name_ + " already has a parameter named: " + param_name);
    }

    parameters_.emplace_back(std::move(param_name), std::move(ort_value));
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
  BinaryFormatHolder format_holder_;
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
  /// This API loads the Lora Adapters as specified in the configuration
  /// </summary>
  /// <param name="config_path">path to where model and lora weights are expected</param>
  /// <param name="config">configuration settings</param>
  void LoadAdaptersFromConfig(const fs::path& model_path, const Config& config);

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
  void AddParameter(const std::string& adapter_name, std::string param_name, std::shared_ptr<OrtValue> p);

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
