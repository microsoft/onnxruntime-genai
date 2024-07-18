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
#include <utility>
#include <vector>

namespace Generators {

struct Config;

namespace lora_parameters {
struct Parameters;
}  // namespace lora_parameters

struct Model;

namespace details {

template <class Iter>
struct Range : std::pair<Iter, Iter> {
  using Base = std::pair<Iter, Iter>;
  using Base::Base;

  Iter begin() const { return this->first; }
  Iter end() const { return this->second; }
};

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
/// <param name="requested_mem_info">memory location where the empty tensor is to be created</param>
/// <returns>A OrtValue over a dummy buffer.</returns>
std::shared_ptr<OrtValue> CreateEmptyInput(const Model& model, const OrtValue& tensor);

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

/// <summary>
/// Copy parameter to a device according to the model's settings
/// </summary>
/// <param name="model"></param>
/// <param name="param"></param>
/// <returns></returns>
std::shared_ptr<OrtValue> MakeDeviceCopyIfNeeded(const Model& model, const LoraParam& param);

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
    return parameters_.size();
  }

  void AddParameter(std::string param_name, std::shared_ptr<OrtValue> ort_value) {
    auto hit =
        std::find_if(parameters_.begin(), parameters_.end(), [&](const auto& p) { return p.name_ == param_name; });

    if (hit != parameters_.end()) {
      throw std::runtime_error("Adapter: " + name_ + " already has a parameter named: " + param_name);
    }

    parameters_.emplace_back(std::move(param_name), std::move(ort_value));
  }

  using ParamContainer = std::vector<LoraParam>;
  using ParamIterator = ParamContainer::const_iterator;

  /// <summary>
  /// Gets access to adapter parameters
  /// </summary>
  /// <returns></returns>
  Range<ParamIterator> GetParameters() const noexcept {
    return Range<ParamIterator>(parameters_.cbegin(), parameters_.cend());
  }

 private:
  std::string name_;
  BinaryFormatHolder format_holder_;
  ParamContainer parameters_;
};

}  // namespace details

/// <summary>
/// This class manages the collection of Lora Adapters
/// </summary>
class LoraAdapterContainer {
 public:
  LoraAdapterContainer() = default;
  ~LoraAdapterContainer() = default;
  LoraAdapterContainer(const LoraAdapterContainer&) = delete;
  LoraAdapterContainer& operator=(const LoraAdapterContainer&) = delete;

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
  /// Returns total number of parameters across all adapters
  /// </summary>
  /// <returns></returns>
  size_t GetParamNum() const noexcept {
    size_t result = 0;
    for (const auto& [_, adapter] : adapters_) {
      result += adapter.GetParamNum();
    }
    return result;
  }

  using AdapterMap = std::unordered_map<std::string, details::LoraAdapter>;
  using AdapterIterator = AdapterMap::const_iterator;

  /// <summary>
  /// Returns iterators to the adapters
  /// </summary>
  /// <returns></returns>
  details::Range<AdapterIterator> GetAdapters() const noexcept {
    return {adapters_.cbegin(), adapters_.cend()};
  }

  /// <summary>
  /// Checks if the adapter exists
  /// </summary>
  /// <param name="adapter_name"></param>
  /// <returns>true if so</returns>
  bool HasAdapter(const std::string& adapter_name) const noexcept {
    return adapters_.find(adapter_name) != adapters_.end();
  }

 private:
  AdapterMap adapters_;
};

template <class NamesOutputIter, class TensorOutputIter>
void OutputAdaptersParameters(const Model& model,
                              const LoraAdapterContainer& lora_container,
                              const std::set<std::string>& adapter_names,
                              NamesOutputIter names_out,
                              TensorOutputIter params_out) {
  for (const auto& [name, adapter] : lora_container.GetAdapters()) {
    if (adapter_names.find(name) == adapter_names.end()) {
      for (const auto& lora_param : adapter.GetParameters()) {
        // Output empty values for inactive adapters
        *names_out = lora_param.name_;
        ++names_out;
        *params_out = details::CreateEmptyInput(model, *lora_param.ort_user_supplied_value_);
        ++params_out;
      }
    } else {
      for (const auto& lora_param : adapter.GetParameters()) {
        *names_out = lora_param.name_;
        ++names_out;
        *params_out = details::MakeDeviceCopyIfNeeded(model, lora_param);
        ++params_out;
      }
    }
  }
}

}  // namespace Generators
