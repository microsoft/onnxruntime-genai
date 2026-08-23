// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Portions of this file consist of AI generated content.
#include "generators.h"
#include "models/model_type.h"
#include "runtime_settings.h"
#include "json.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace Generators {

int64_t SafeDoubleToInt64(double x, std::string_view name);

// Normalizes historical casings, short aliases, and full ORT names (e.g.
// "CUDAExecutionProvider") to the canonical dispatch-table name; unknown names pass through.
std::string_view NormalizeProviderName(std::string_view name) {
  std::string lower_name(name);
  std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });
  // Strip the shared "ExecutionProvider" suffix so full ORT names normalize like the aliases.
  constexpr std::string_view kEpSuffix = "executionprovider";
  if (lower_name.size() > kEpSuffix.size() &&
      lower_name.compare(lower_name.size() - kEpSuffix.size(), kEpSuffix.size(), kEpSuffix) == 0) {
    lower_name.resize(lower_name.size() - kEpSuffix.size());
  }
  if (lower_name == "cpu") {
    return "CPU";
  } else if (lower_name == "cuda") {
    return "cuda";
  } else if (lower_name == "qnn") {
    return "QNN";
  } else if (lower_name == "webgpu") {
    return "WebGPU";
  } else if (lower_name == "dml") {
    return "DML";
  } else if (lower_name == "openvino") {
    return "OpenVINO";
  } else if (lower_name == "vitisai") {
    return "VitisAI";
  } else if (lower_name == "ryzenai") {
    return "RyzenAI";
  } else if (lower_name == "nvtensorrtrtx") {
    return "NvTensorRtRtx";
  } else if (lower_name == "amdgpu" ||
             lower_name == "amdgpuexecutionprovider") {
    // Accept canonical and catalog forms, all route to AMDGPU.
    return "AMDGPU";
  }
  return name;  // Return name unchanged
}

ONNXTensorElementDataType TranslateTensorType(std::string_view value) {
  if (value == "float32") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  if (value == "float16") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  }
  throw std::runtime_error("Invalid tensor type: " + std::string(value));
}

OrtHardwareDeviceType ParseHardwareDeviceType(std::string_view value) {
  std::string lower_value(value);
  std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(),
                 [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });
  if (lower_value == "cpu") {
    return OrtHardwareDeviceType_CPU;
  } else if (lower_value == "gpu") {
    return OrtHardwareDeviceType_GPU;
  } else if (lower_value == "npu") {
    return OrtHardwareDeviceType_NPU;
  } else {
    throw std::runtime_error("Unsupported hardware device type: " + std::string(value));
  }
}

struct NamedStrings_Element : JSON::Element {
  explicit NamedStrings_Element(std::vector<Config::NamedString>& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    v_.emplace_back(name, JSON::Get<std::string_view>(value));
  }

 private:
  std::vector<Config::NamedString>& v_;
};

struct Int_Array_Element : JSON::Element {
  explicit Int_Array_Element(std::vector<int>& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    v_.emplace_back(SafeDoubleToInt(JSON::Get<double>(value), name));
  }

 private:
  std::vector<int>& v_;
};

struct DeviceFilteringOptions_Element : JSON::Element {
  explicit DeviceFilteringOptions_Element(Config::DeviceFilteringOptions& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "hardware_device_type") {
      v_.hardware_device_type = ParseHardwareDeviceType(JSON::Get<std::string_view>(value));
    } else if (name == "hardware_device_id") {
      v_.hardware_device_id = static_cast<uint32_t>(JSON::Get<double>(value));
    } else if (name == "hardware_vendor_id") {
      v_.hardware_vendor_id = static_cast<uint32_t>(JSON::Get<double>(value));
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::DeviceFilteringOptions& v_;
};

struct ProviderOptions_Element : JSON::Element {
  explicit ProviderOptions_Element(Config::ProviderOptions& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    v_.options.emplace_back(name, JSON::Get<std::string_view>(value));
  }

  JSON::Element& OnObject(std::string_view name) override {
    if (name == "device_filtering_options") {
      v_.device_filtering_options = Config::DeviceFilteringOptions{};
      device_filtering_options_element_ = std::make_unique<DeviceFilteringOptions_Element>(*v_.device_filtering_options);
      return *device_filtering_options_element_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::ProviderOptions& v_;
  std::unique_ptr<DeviceFilteringOptions_Element> device_filtering_options_element_;
};

struct ProviderOptionsObject_Element : JSON::Element {
  explicit ProviderOptionsObject_Element(std::vector<Config::ProviderOptions>& v) : v_{v} {}

  JSON::Element& OnObject(std::string_view name) override {
    for (auto& v : v_) {
      if (v.name == name) {
        options_element_ = std::make_unique<ProviderOptions_Element>(v);
        return *options_element_;
      }
    }

    auto& options = v_.emplace_back();
    options.name = name;
    options_element_ = std::make_unique<ProviderOptions_Element>(options);
    return *options_element_;
  }

 private:
  std::vector<Config::ProviderOptions>& v_;
  std::unique_ptr<ProviderOptions_Element> options_element_;
};

struct ProviderOptionsArray_Element : JSON::Element {
  explicit ProviderOptionsArray_Element(std::vector<Config::ProviderOptions>& v) : v_{v} {}

  JSON::Element& OnObject(std::string_view name) override { return object_; }

  void OnComplete(bool /*empty*/) override {
    // For backwards compatibility turn our old names like 'qnn' into 'QNN', and 'webgpu' to 'WebGPU'
    for (auto& v : v_) {
      v.name = NormalizeProviderName(v.name);
    }
  }

 private:
  std::vector<Config::ProviderOptions>& v_;
  ProviderOptionsObject_Element object_{v_};
};

GraphOptimizationLevel GetGraphOptimizationLevel(std::string_view name) {
  if (name == "ORT_DISABLE_ALL") {
    return ORT_DISABLE_ALL;
  } else if (name == "ORT_ENABLE_BASIC") {
    return ORT_ENABLE_BASIC;
  } else if (name == "ORT_ENABLE_EXTENDED") {
    return ORT_ENABLE_EXTENDED;
  } else if (name == "ORT_ENABLE_ALL") {
    return ORT_ENABLE_ALL;
  } else {
    throw std::runtime_error("Unrecognized value:" + std::string(name));
  }
}

struct SessionOptions_Element : JSON::Element {
  explicit SessionOptions_Element(Config::SessionOptions& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "log_id") {
      v_.log_id = JSON::Get<std::string_view>(value);
    } else if (name == "enable_profiling") {
      v_.enable_profiling = JSON::Get<std::string_view>(value);
    } else if (name == "intra_op_num_threads") {
      v_.intra_op_num_threads = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "inter_op_num_threads") {
      v_.inter_op_num_threads = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "log_severity_level") {
      v_.log_severity_level = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "log_verbosity_level") {
      v_.log_verbosity_level = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "enable_cpu_mem_arena") {
      v_.enable_cpu_mem_arena = JSON::Get<bool>(value);
    } else if (name == "enable_mem_pattern") {
      v_.enable_mem_pattern = JSON::Get<bool>(value);
    } else if (name == "graph_optimization_level") {
      v_.graph_optimization_level = GetGraphOptimizationLevel(JSON::Get<std::string_view>(value));
    } else if (name == "custom_ops_library") {
      v_.custom_ops_library = JSON::Get<std::string_view>(value);
    } else {
      // Session options that are set with AddConfigEntry
      v_.config_entries.emplace_back(name, JSON::Get<std::string_view>(value));
    }
  }

  JSON::Element& OnArray(std::string_view name) override {
    if (name == "provider_options") {
      return provider_options_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::SessionOptions& v_;
  ProviderOptionsArray_Element provider_options_{v_.provider_options};
};

struct RunOptions_Element : JSON::Element {
  explicit RunOptions_Element(Config::RunOptions& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    // Run options that are set with AddConfigEntry
    v_.emplace_back(name, JSON::Get<std::string_view>(value));
  }

 private:
  Config::RunOptions& v_;
};

struct EncoderInputs_Element : JSON::Element {
  explicit EncoderInputs_Element(Config::Model::Encoder::Inputs& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "input_ids") {
      v_.input_ids = JSON::Get<std::string_view>(value);
    } else if (name == "inputs_embeds") {
      v_.embeddings = JSON::Get<std::string_view>(value);
    } else if (name == "attention_mask") {
      v_.attention_mask = JSON::Get<std::string_view>(value);
    } else if (name == "position_ids") {
      v_.position_ids = JSON::Get<std::string_view>(value);
    } else if (name == "audio_features") {
      v_.audio_features = JSON::Get<std::string_view>(value);
    } else if (name == "input_lengths") {
      v_.input_lengths = JSON::Get<std::string_view>(value);
    } else if (name == "cache_last_channel") {
      v_.cache_last_channel = JSON::Get<std::string_view>(value);
    } else if (name == "cache_last_time") {
      v_.cache_last_time = JSON::Get<std::string_view>(value);
    } else if (name == "cache_last_channel_len") {
      v_.cache_last_channel_len = JSON::Get<std::string_view>(value);
    } else if (name == "lang_id") {
      v_.lang_id = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Model::Encoder::Inputs& v_;
};

struct EncoderOutputs_Element : JSON::Element {
  explicit EncoderOutputs_Element(Config::Model::Encoder::Outputs& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "encoder_hidden_states") {
      v_.hidden_states = JSON::Get<std::string_view>(value);
    } else if (name == "encoder_outputs") {
      v_.encoder_outputs = JSON::Get<std::string_view>(value);
    } else if (name == "output_lengths") {
      v_.output_lengths = JSON::Get<std::string_view>(value);
    } else if (name == "cache_last_channel_next") {
      v_.cache_last_channel_next = JSON::Get<std::string_view>(value);
    } else if (name == "cache_last_time_next") {
      v_.cache_last_time_next = JSON::Get<std::string_view>(value);
    } else if (name == "cache_last_channel_len_next") {
      v_.cache_last_channel_len_next = JSON::Get<std::string_view>(value);
    } else if (name == "cross_present_key_names") {
      v_.cross_present_key_names = JSON::Get<std::string_view>(value);
    } else if (name == "cross_present_value_names") {
      v_.cross_present_value_names = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Model::Encoder::Outputs& v_;
};

struct DecoderInputs_Element : JSON::Element {
  explicit DecoderInputs_Element(Config::Model::Decoder::Inputs& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "input_ids") {
      v_.input_ids = JSON::Get<std::string_view>(value);
    } else if (name == "inputs_embeds") {
      v_.embeddings = JSON::Get<std::string_view>(value);
    } else if (name == "attention_mask") {
      v_.attention_mask = JSON::Get<std::string_view>(value);
    } else if (name == "position_ids") {
      v_.position_ids = JSON::Get<std::string_view>(value);
    } else if (name == "past_key_names") {
      v_.past_key_names = JSON::Get<std::string_view>(value);
    } else if (name == "past_value_names") {
      v_.past_value_names = JSON::Get<std::string_view>(value);
    } else if (name == "past_names") {
      v_.past_names = JSON::Get<std::string_view>(value);
    } else if (name == "cross_past_key_names") {
      v_.cross_past_key_names = JSON::Get<std::string_view>(value);
    } else if (name == "cross_past_value_names") {
      v_.cross_past_value_names = JSON::Get<std::string_view>(value);
    } else if (name == "past_sequence_length") {
      v_.past_sequence_length = JSON::Get<std::string_view>(value);
    } else if (name == "current_sequence_length") {
      v_.current_sequence_length = JSON::Get<std::string_view>(value);
    } else if (name == "total_sequence_length") {
      v_.total_sequence_length = JSON::Get<std::string_view>(value);
    } else if (name == "encoder_hidden_states") {
      v_.encoder_hidden_states = JSON::Get<std::string_view>(value);
    } else if (name == "encoder_attention_mask") {
      v_.encoder_attention_mask = JSON::Get<std::string_view>(value);
    } else if (name == "rnn_states_prev") {
      v_.rnn_prev_states = JSON::Get<std::string_view>(value);
    } else if (name == "past_key_values_length") {
      v_.past_key_values_length = JSON::Get<std::string_view>(value);
    } else if (name == "cache_indirection") {
      v_.cache_indirection = JSON::Get<std::string_view>(value);
    } else if (name == "cumulative_sequence_lengths") {
      v_.cumulative_sequence_lengths = JSON::Get<std::string_view>(value);
    } else if (name == "past_sequence_lengths") {
      v_.past_sequence_lengths = JSON::Get<std::string_view>(value);
    } else if (name == "block_table") {
      v_.block_table = JSON::Get<std::string_view>(value);
    } else if (name == "block_table_windowed") {
      v_.block_table_windowed = JSON::Get<std::string_view>(value);
    } else if (name == "attention_metadata") {
      v_.attention_metadata = JSON::Get<std::string_view>(value);
    } else if (name == "past_conv_names") {
      v_.past_conv_names = JSON::Get<std::string_view>(value);
    } else if (name == "hidden_states") {
      v_.hidden_states = JSON::Get<std::string_view>(value);
    } else if (name == "targets") {
      v_.targets = JSON::Get<std::string_view>(value);
    } else if (name == "lstm_hidden_state") {
      v_.lstm_hidden_state = JSON::Get<std::string_view>(value);
    } else if (name == "lstm_cell_state") {
      v_.lstm_cell_state = JSON::Get<std::string_view>(value);
    } else if (name == "per_layer_inputs") {
      v_.per_layer_inputs = JSON::Get<std::string_view>(value);
    } else if (name == "targets_length") {
      v_.targets_length = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Model::Decoder::Inputs& v_;
};

struct DecoderOutputs_Element : JSON::Element {
  explicit DecoderOutputs_Element(Config::Model::Decoder::Outputs& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "logits") {
      v_.logits = JSON::Get<std::string_view>(value);
    } else if (name == "present_key_names") {
      v_.present_key_names = JSON::Get<std::string_view>(value);
    } else if (name == "present_value_names") {
      v_.present_value_names = JSON::Get<std::string_view>(value);
    } else if (name == "present_names") {
      v_.present_names = JSON::Get<std::string_view>(value);
    } else if (name == "output_cross_qk_names") {
      v_.output_cross_qk_names = JSON::Get<std::string_view>(value);
    } else if (name == "rnn_states") {
      v_.rnn_states = JSON::Get<std::string_view>(value);
    } else if (name == "present_conv_names") {
      v_.present_conv_names = JSON::Get<std::string_view>(value);
    } else if (name == "hidden_states") {
      v_.hidden_states = JSON::Get<std::string_view>(value);
    } else if (name == "outputs") {
      v_.outputs = JSON::Get<std::string_view>(value);
    } else if (name == "lstm_hidden_state") {
      v_.lstm_hidden_state = JSON::Get<std::string_view>(value);
    } else if (name == "lstm_cell_state") {
      v_.lstm_cell_state = JSON::Get<std::string_view>(value);
    } else if (name == "outputs_length") {
      v_.outputs_length = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Model::Decoder::Outputs& v_;
};

struct StringArray_Element : JSON::Element {
  explicit StringArray_Element(std::vector<std::string>& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    v_.push_back(std::string{JSON::Get<std::string_view>(value)});
  }

 private:
  std::vector<std::string>& v_;
};

struct IntArray_Element : JSON::Element {
  explicit IntArray_Element(std::vector<int>& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    v_.push_back(SafeDoubleToInt(JSON::Get<double>(value), name));
  }

 private:
  std::vector<int>& v_;
};

struct Int64Array_Element : JSON::Element {
  explicit Int64Array_Element(std::vector<int64_t>& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    v_.push_back(SafeDoubleToInt64(JSON::Get<double>(value), name));
  }

 private:
  std::vector<int64_t>& v_;
};

struct SharedInitializer_Element : JSON::Element {
  explicit SharedInitializer_Element(Config::Model::SharedInitializer& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "name") {
      v_.name = JSON::Get<std::string_view>(value);
    } else if (name == "data_file") {
      v_.data_file = JSON::Get<std::string_view>(value);
    } else if (name == "offset") {
      v_.offset = JSON::Get<std::string_view>(value);
    } else if (name == "length") {
      v_.length = JSON::Get<std::string_view>(value);
    } else if (name == "data_type") {
      v_.data_type = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnArray(std::string_view name) override {
    if (name == "shape") {
      return shape_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Model::SharedInitializer& v_;
  Int64Array_Element shape_{v_.shape};
};

struct SharedInitializers_Element : JSON::Element {
  explicit SharedInitializers_Element(std::vector<Config::Model::SharedInitializer>& v) : v_{v} {}

  Element& OnObject(std::string_view /*name*/) override {
    auto& initializer = v_.emplace_back();
    element_ = std::make_unique<SharedInitializer_Element>(initializer);
    return *element_;
  }

 private:
  std::vector<Config::Model::SharedInitializer>& v_;
  std::unique_ptr<SharedInitializer_Element> element_;
};

struct StringStringMap_Element : JSON::Element {
  explicit StringStringMap_Element(std::unordered_map<std::string, std::string>& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    v_[std::string(name)] = std::string(JSON::Get<std::string_view>(value));
  }

 private:
  std::unordered_map<std::string, std::string>& v_;
};

struct PipelineModel_Element : JSON::Element {
  explicit PipelineModel_Element(Config::Model::Decoder::PipelineModel& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "filename") {
      v_.filename = JSON::Get<std::string_view>(value);
    } else if (name == "run_on_prompt") {
      v_.run_on_prompt = JSON::Get<bool>(value);
    } else if (name == "run_on_token_gen") {
      v_.run_on_token_gen = JSON::Get<bool>(value);
    } else if (name == "is_lm_head") {
      v_.is_lm_head = JSON::Get<bool>(value);
    } else if (name == "inherit_session_options") {
      v_.inherit_session_options = JSON::Get<bool>(value);
    } else if (name == "reset_session_idx") {
      v_.reset_session_idx = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  JSON::Element& OnObject(std::string_view name) override {
    if (name == "session_options") {
      v_.session_options = Config::SessionOptions{};
      session_options_ = std::make_unique<SessionOptions_Element>(*v_.session_options);
      return *session_options_;
    }
    if (name == "run_options") {
      v_.run_options = Config::RunOptions{};
      run_options_ = std::make_unique<RunOptions_Element>(*v_.run_options);
      return *run_options_;
    }
    if (name == "output_names_forwarder") {
      return output_names_forwarder_;
    }
    throw JSON::unknown_value_error{};
  }

  Element& OnArray(std::string_view name) override {
    if (name == "inputs") {
      return inputs_;
    } else if (name == "outputs") {
      return outputs_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Model::Decoder::PipelineModel& v_;
  std::unique_ptr<SessionOptions_Element> session_options_;
  std::unique_ptr<RunOptions_Element> run_options_;
  StringArray_Element inputs_{v_.inputs};
  StringArray_Element outputs_{v_.outputs};
  StringStringMap_Element output_names_forwarder_{v_.output_names_forwarder};
};

struct PipelineModelObject_Element : JSON::Element {
  explicit PipelineModelObject_Element(std::vector<Config::Model::Decoder::PipelineModel>& v) : v_{v} {}

  Element& OnObject(std::string_view name) override {
    auto& model = v_.emplace_back();
    model.model_id = name;
    pipeline_model_elements_.emplace_back(model);
    return pipeline_model_elements_.back();
  }

 private:
  std::vector<Config::Model::Decoder::PipelineModel>& v_;
  std::vector<PipelineModel_Element> pipeline_model_elements_;
};

struct Pipeline_Element : JSON::Element {
  explicit Pipeline_Element(std::vector<Config::Model::Decoder::PipelineModel>& v) : v_{v} {}

  Element& OnObject(std::string_view name) override {
    return object_;
  }

 private:
  std::vector<Config::Model::Decoder::PipelineModel>& v_;
  PipelineModelObject_Element object_{v_};
};

struct SlidingWindow_Element : JSON::Element {
  explicit SlidingWindow_Element(std::optional<Config::Model::Decoder::SlidingWindow>& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "window_size") {
      v_->window_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "pad_value") {
      v_->pad_value = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "alignment") {
      v_->alignment = JSON::Get<std::string_view>(value);
    } else if (name == "slide_key_value_cache") {
      v_->slide_key_value_cache = JSON::Get<bool>(value);
    } else if (name == "slide_inputs") {
      v_->slide_inputs = JSON::Get<bool>(value);
    } else if (name == "cache_slack") {
      v_->cache_slack = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnArray(std::string_view name) override {
    if (name == "layers") {
      // Lazy initialize layers_ when first accessed
      if (!layers_) {
        layers_ = std::make_unique<IntArray_Element>(v_->layers);
      }
      return *layers_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  std::optional<Config::Model::Decoder::SlidingWindow>& v_;
  std::unique_ptr<IntArray_Element> layers_;
};

struct Encoder_Element : JSON::Element {
  explicit Encoder_Element(Config::Model::Encoder& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "filename") {
      v_.filename = JSON::Get<std::string_view>(value);
    } else if (name == "hidden_size") {
      v_.hidden_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "num_attention_heads") {
      v_.num_attention_heads = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "num_hidden_layers") {
      v_.num_hidden_layers = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "num_key_value_heads") {
      v_.num_key_value_heads = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "head_size") {
      v_.head_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnObject(std::string_view name) override {
    if (name == "session_options") {
      v_.session_options = Config::SessionOptions{};
      session_options_ = std::make_unique<SessionOptions_Element>(*v_.session_options);
      return *session_options_;
    }
    if (name == "run_options") {
      v_.run_options = Config::RunOptions{};
      run_options_ = std::make_unique<RunOptions_Element>(*v_.run_options);
      return *run_options_;
    }
    if (name == "inputs") {
      return inputs_;
    }
    if (name == "outputs") {
      return outputs_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Model::Encoder& v_;
  std::unique_ptr<SessionOptions_Element> session_options_;
  std::unique_ptr<RunOptions_Element> run_options_;
  EncoderInputs_Element inputs_{v_.inputs};
  EncoderOutputs_Element outputs_{v_.outputs};
};

struct Decoder_Element : JSON::Element {
  explicit Decoder_Element(Config::Model::Decoder& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "filename") {
      v_.filename = JSON::Get<std::string_view>(value);
    } else if (name == "hidden_size") {
      v_.hidden_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "num_attention_heads") {
      v_.num_attention_heads = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "num_hidden_layers") {
      v_.num_hidden_layers = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "num_key_value_heads") {
      v_.num_key_value_heads = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "head_size") {
      v_.head_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "max_logits_sequence_length") {
      v_.max_logits_sequence_length = SafeDoubleToInt(JSON::Get<double>(value), name);
      if (v_.max_logits_sequence_length < 0) throw std::runtime_error("max_logits_sequence_length must be >= 0");
    } else if (name == "conv_cache_size") {
      v_.conv_cache_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnObject(std::string_view name) override {
    if (name == "session_options") {
      return session_options_;
    }
    if (name == "run_options") {
      v_.run_options = Config::RunOptions{};
      run_options_ = std::make_unique<RunOptions_Element>(*v_.run_options);
      return *run_options_;
    }
    if (name == "inputs") {
      return inputs_;
    }
    if (name == "outputs") {
      return outputs_;
    }
    if (name == "sliding_window") {
      v_.sliding_window = Config::Model::Decoder::SlidingWindow{};
      return sliding_window_;
    }
    // Support object-style pipeline: "pipeline": { "embeddings": { ... }, ... }
    if (name == "pipeline") {
      pipeline_object_ = std::make_unique<PipelineModelObject_Element>(v_.pipeline);
      return *pipeline_object_;
    }
    throw JSON::unknown_value_error{};
  }

  Element& OnArray(std::string_view name) override {
    if (name == "pipeline") {
      return pipeline_;
    }
    if (name == "layer_types") {
      layer_types_ = std::make_unique<StringArray_Element>(v_.layer_types);
      return *layer_types_;
    }
    if (name == "shared_initializers") {
      return shared_initializers_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Model::Decoder& v_;
  SessionOptions_Element session_options_{v_.session_options};
  std::unique_ptr<RunOptions_Element> run_options_;
  DecoderInputs_Element inputs_{v_.inputs};
  DecoderOutputs_Element outputs_{v_.outputs};
  Pipeline_Element pipeline_{v_.pipeline};
  SlidingWindow_Element sliding_window_{v_.sliding_window};
  std::unique_ptr<PipelineModelObject_Element> pipeline_object_;  // object-style pipeline support
  std::unique_ptr<StringArray_Element> layer_types_;
  SharedInitializers_Element shared_initializers_{v_.shared_initializers};
};

struct MtpInputs_Element : JSON::Element {
  explicit MtpInputs_Element(Config::Model::Mtp::Inputs& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "input_ids") {
      v_.input_ids = JSON::Get<std::string_view>(value);
    } else if (name == "hidden_states") {
      v_.hidden_states = JSON::Get<std::string_view>(value);
    } else if (name == "attention_mask") {
      v_.attention_mask = JSON::Get<std::string_view>(value);
    } else if (name == "position_ids") {
      v_.position_ids = JSON::Get<std::string_view>(value);
    } else if (name == "past_key_names") {
      v_.past_key_names = JSON::Get<std::string_view>(value);
    } else if (name == "past_value_names") {
      v_.past_value_names = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnArray(std::string_view name) override {
    if (name == "shared_key_names") {
      return shared_key_names_;
    }
    if (name == "shared_value_names") {
      return shared_value_names_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Model::Mtp::Inputs& v_;
  StringArray_Element shared_key_names_{v_.shared_key_names};
  StringArray_Element shared_value_names_{v_.shared_value_names};
};

struct MtpOutputs_Element : JSON::Element {
  explicit MtpOutputs_Element(Config::Model::Mtp::Outputs& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "logits") {
      v_.logits = JSON::Get<std::string_view>(value);
    } else if (name == "hidden_states") {
      v_.hidden_states = JSON::Get<std::string_view>(value);
    } else if (name == "present_key_names") {
      v_.present_key_names = JSON::Get<std::string_view>(value);
    } else if (name == "present_value_names") {
      v_.present_value_names = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Model::Mtp::Outputs& v_;
};

struct Mtp_Element : JSON::Element {
  explicit Mtp_Element(Config::Model::Mtp& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "filename") {
      v_.filename = JSON::Get<std::string_view>(value);
    } else if (name == "num_hidden_layers") {
      v_.num_hidden_layers = SafeDoubleToInt(JSON::Get<double>(value), name);
      if (v_.num_hidden_layers <= 0) throw std::out_of_range("num_hidden_layers must be > 0");
    } else if (name == "num_key_value_heads") {
      v_.num_key_value_heads = SafeDoubleToInt(JSON::Get<double>(value), name);
      if (v_.num_key_value_heads <= 0) throw std::out_of_range("num_key_value_heads must be > 0");
    } else if (name == "head_size") {
      v_.head_size = SafeDoubleToInt(JSON::Get<double>(value), name);
      if (v_.head_size <= 0) throw std::out_of_range("head_size must be > 0");
    } else if (name == "main_hidden_states") {
      v_.main_hidden_states = JSON::Get<std::string_view>(value);
    } else if (name == "main_inputs_embeds") {
      v_.main_inputs_embeds = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnArray(std::string_view name) override {
    if (name == "shared_kv_layers") {
      return shared_kv_layers_;
    }
    if (name == "shared_initializers") {
      return shared_initializers_;
    }
    throw JSON::unknown_value_error{};
  }

  Element& OnObject(std::string_view name) override {
    if (name == "session_options") {
      v_.session_options = Config::SessionOptions{};
      session_options_ = std::make_unique<SessionOptions_Element>(*v_.session_options);
      return *session_options_;
    }
    if (name == "run_options") {
      v_.run_options = Config::RunOptions{};
      run_options_ = std::make_unique<RunOptions_Element>(*v_.run_options);
      return *run_options_;
    }
    if (name == "inputs") {
      return inputs_;
    }
    if (name == "outputs") {
      return outputs_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Model::Mtp& v_;
  std::unique_ptr<SessionOptions_Element> session_options_;
  std::unique_ptr<RunOptions_Element> run_options_;
  IntArray_Element shared_kv_layers_{v_.shared_kv_layers};
  MtpInputs_Element inputs_{v_.inputs};
  MtpOutputs_Element outputs_{v_.outputs};
  SharedInitializers_Element shared_initializers_{v_.shared_initializers};
};

struct VisionInputs_Element : JSON::Element {
  explicit VisionInputs_Element(Config::Model::Vision::Inputs& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "pixel_values") {
      v_.pixel_values = JSON::Get<std::string_view>(value);
    } else if (name == "pixel_position_ids") {
      v_.pixel_position_ids = JSON::Get<std::string_view>(value);
    } else if (name == "image_sizes") {
      v_.image_sizes = JSON::Get<std::string_view>(value);
    } else if (name == "image_grid_thw") {
      v_.image_grid_thw = JSON::Get<std::string_view>(value);
    } else if (name == "attention_mask") {
      v_.attention_mask = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Model::Vision::Inputs& v_;
};

struct VisionOutputs_Element : JSON::Element {
  explicit VisionOutputs_Element(Config::Model::Vision::Outputs& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "image_features") {
      v_.image_features = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Model::Vision::Outputs& v_;
};

// Vision pipeline support structures
struct VisionPipelineModel_Element : JSON::Element {
  explicit VisionPipelineModel_Element(Config::Model::Vision::PipelineModel& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "filename") {
      v_.filename = JSON::Get<std::string_view>(value);
    } else if (name == "run_on_cpu") {
      v_.run_on_cpu = JSON::Get<bool>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnObject(std::string_view name) override {
    if (name == "session_options") {
      v_.session_options = Config::SessionOptions{};
      session_options_ = std::make_unique<SessionOptions_Element>(*v_.session_options);
      return *session_options_;
    }
    if (name == "run_options") {
      v_.run_options = Config::RunOptions{};
      run_options_ = std::make_unique<RunOptions_Element>(*v_.run_options);
      return *run_options_;
    }
    throw JSON::unknown_value_error{};
  }

  Element& OnArray(std::string_view name) override {
    if (name == "inputs") {
      return inputs_;
    }
    if (name == "outputs") {
      return outputs_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Model::Vision::PipelineModel& v_;
  std::unique_ptr<SessionOptions_Element> session_options_;
  std::unique_ptr<RunOptions_Element> run_options_;
  StringArray_Element inputs_{v_.inputs};
  StringArray_Element outputs_{v_.outputs};
};

struct VisionPipelineModelObject_Element : JSON::Element {
  explicit VisionPipelineModelObject_Element(std::vector<Config::Model::Vision::PipelineModel>& v) : v_{v} {}

  Element& OnObject(std::string_view name) override {
    auto& model = v_.emplace_back();
    model.model_id = name;
    elements_.emplace_back(model);
    return elements_.back();
  }

 private:
  std::vector<Config::Model::Vision::PipelineModel>& v_;
  std::vector<VisionPipelineModel_Element> elements_;
};

struct VisionPipeline_Element : JSON::Element {
  explicit VisionPipeline_Element(std::vector<Config::Model::Vision::PipelineModel>& v) : v_{v} {}

  Element& OnObject(std::string_view name) override { return object_; }

 private:
  std::vector<Config::Model::Vision::PipelineModel>& v_;
  VisionPipelineModelObject_Element object_{v_};
};

struct Vision_Element : JSON::Element {
  explicit Vision_Element(Config::Model::Vision& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "filename") {
      v_.filename = JSON::Get<std::string_view>(value);
    } else if (name == "config_filename") {
      v_.config_filename = JSON::Get<std::string_view>(value);
    } else if (name == "adapter_filename") {
      v_.adapter_filename = JSON::Get<std::string_view>(value);
    } else if (name == "spatial_merge_size") {
      v_.spatial_merge_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "tokens_per_second") {
      v_.tokens_per_second = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "patch_size") {
      v_.patch_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "num_visual_tokens") {
      v_.num_visual_tokens = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "window_size") {
      v_.window_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnObject(std::string_view name) override {
    if (name == "session_options") {
      v_.session_options = Config::SessionOptions{};
      session_options_ = std::make_unique<SessionOptions_Element>(*v_.session_options);
      return *session_options_;
    }
    if (name == "run_options") {
      v_.run_options = Config::RunOptions{};
      run_options_ = std::make_unique<RunOptions_Element>(*v_.run_options);
      return *run_options_;
    }
    if (name == "inputs") {
      return inputs_;
    }
    if (name == "outputs") {
      return outputs_;
    }
    // Support object-style pipeline for vision: "pipeline": { "patch_embed": { ... }, ... }
    if (name == "pipeline") {
      vision_pipeline_object_ = std::make_unique<VisionPipelineModelObject_Element>(v_.pipeline);
      return *vision_pipeline_object_;
    }
    throw JSON::unknown_value_error{};
  }

  Element& OnArray(std::string_view name) override {
    if (name == "pipeline") {
      return pipeline_element_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Model::Vision& v_;
  std::unique_ptr<SessionOptions_Element> session_options_;
  std::unique_ptr<RunOptions_Element> run_options_;
  VisionInputs_Element inputs_{v_.inputs};
  VisionOutputs_Element outputs_{v_.outputs};
  VisionPipeline_Element pipeline_element_{v_.pipeline};
  std::unique_ptr<VisionPipelineModelObject_Element> vision_pipeline_object_;  // object-style pipeline support
};

struct SpeechInputs_Element : JSON::Element {
  explicit SpeechInputs_Element(Config::Model::Speech::Inputs& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "audio_embeds") {
      v_.audio_embeds = JSON::Get<std::string_view>(value);
    } else if (name == "attention_mask") {
      v_.attention_mask = JSON::Get<std::string_view>(value);
    } else if (name == "audio_sizes") {
      v_.audio_sizes = JSON::Get<std::string_view>(value);
    } else if (name == "audio_projection_mode") {
      v_.audio_projection_mode = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Model::Speech::Inputs& v_;
};

struct SpeechOutputs_Element : JSON::Element {
  explicit SpeechOutputs_Element(Config::Model::Speech::Outputs& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "audio_features") {
      v_.audio_features = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Model::Speech::Outputs& v_;
};

struct Speech_Element : JSON::Element {
  explicit Speech_Element(Config::Model::Speech& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "filename") {
      v_.filename = JSON::Get<std::string_view>(value);
    } else if (name == "config_filename") {
      v_.config_filename = JSON::Get<std::string_view>(value);
    } else if (name == "adapter_filename") {
      v_.adapter_filename = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnObject(std::string_view name) override {
    if (name == "session_options") {
      v_.session_options = Config::SessionOptions{};
      session_options_ = std::make_unique<SessionOptions_Element>(*v_.session_options);
      return *session_options_;
    }
    if (name == "run_options") {
      v_.run_options = Config::RunOptions{};
      run_options_ = std::make_unique<RunOptions_Element>(*v_.run_options);
      return *run_options_;
    }
    if (name == "inputs") {
      return inputs_;
    }
    if (name == "outputs") {
      return outputs_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Model::Speech& v_;
  std::unique_ptr<SessionOptions_Element> session_options_;
  std::unique_ptr<RunOptions_Element> run_options_;
  SpeechInputs_Element inputs_{v_.inputs};
  SpeechOutputs_Element outputs_{v_.outputs};
};

struct JoinerInputs_Element : JSON::Element {
  explicit JoinerInputs_Element(Config::Model::Joiner::Inputs& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "encoder_outputs") {
      v_.encoder_outputs = JSON::Get<std::string_view>(value);
    } else if (name == "decoder_outputs") {
      v_.decoder_outputs = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Model::Joiner::Inputs& v_;
};

struct JoinerOutputs_Element : JSON::Element {
  explicit JoinerOutputs_Element(Config::Model::Joiner::Outputs& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "logits") {
      v_.logits = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Model::Joiner::Outputs& v_;
};

struct Joiner_Element : JSON::Element {
  explicit Joiner_Element(Config::Model::Joiner& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "filename") {
      v_.filename = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnObject(std::string_view name) override {
    if (name == "session_options") {
      v_.session_options = Config::SessionOptions{};
      session_options_ = std::make_unique<SessionOptions_Element>(*v_.session_options);
      return *session_options_;
    }
    if (name == "run_options") {
      v_.run_options = Config::RunOptions{};
      run_options_ = std::make_unique<RunOptions_Element>(*v_.run_options);
      return *run_options_;
    }
    if (name == "inputs") {
      return inputs_;
    }
    if (name == "outputs") {
      return outputs_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Model::Joiner& v_;
  std::unique_ptr<SessionOptions_Element> session_options_;
  std::unique_ptr<RunOptions_Element> run_options_;
  JoinerInputs_Element inputs_{v_.inputs};
  JoinerOutputs_Element outputs_{v_.outputs};
};

struct VAD_Element : JSON::Element {
  explicit VAD_Element(Config::Model::VAD& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "filename") {
      v_.filename = JSON::Get<std::string_view>(value);
    } else if (name == "threshold") {
      v_.threshold = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "silence_duration_ms") {
      v_.silence_duration_ms = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "prefix_padding_ms") {
      v_.prefix_padding_ms = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnObject(std::string_view name) override {
    if (name == "session_options") {
      v_.session_options = Config::SessionOptions{};
      session_options_ = std::make_unique<SessionOptions_Element>(*v_.session_options);
      return *session_options_;
    }
    if (name == "run_options") {
      v_.run_options = Config::RunOptions{};
      run_options_ = std::make_unique<RunOptions_Element>(*v_.run_options);
      return *run_options_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Model::VAD& v_;
  std::unique_ptr<SessionOptions_Element> session_options_;
  std::unique_ptr<RunOptions_Element> run_options_;
};

struct EmbeddingInputs_Element : JSON::Element {
  explicit EmbeddingInputs_Element(Config::Model::Embedding::Inputs& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "input_ids") {
      v_.input_ids = JSON::Get<std::string_view>(value);
    } else if (name == "image_features") {
      v_.image_features = JSON::Get<std::string_view>(value);
    } else if (name == "audio_features") {
      v_.audio_features = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Model::Embedding::Inputs& v_;
};

struct EmbeddingOutputs_Element : JSON::Element {
  explicit EmbeddingOutputs_Element(Config::Model::Embedding::Outputs& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "inputs_embeds") {
      v_.embeddings = JSON::Get<std::string_view>(value);
    } else if (name == "per_layer_inputs") {
      v_.per_layer_inputs = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Model::Embedding::Outputs& v_;
};

struct Embedding_Element : JSON::Element {
  explicit Embedding_Element(Config::Model::Embedding& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "filename") {
      v_.filename = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnObject(std::string_view name) override {
    if (name == "session_options") {
      v_.session_options = Config::SessionOptions{};
      session_options_ = std::make_unique<SessionOptions_Element>(*v_.session_options);
      return *session_options_;
    }
    if (name == "run_options") {
      v_.run_options = Config::RunOptions{};
      run_options_ = std::make_unique<RunOptions_Element>(*v_.run_options);
      return *run_options_;
    }
    if (name == "inputs") {
      return inputs_;
    }
    if (name == "outputs") {
      return outputs_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Model::Embedding& v_;
  std::unique_ptr<SessionOptions_Element> session_options_;
  std::unique_ptr<RunOptions_Element> run_options_;
  EmbeddingInputs_Element inputs_{v_.inputs};
  EmbeddingOutputs_Element outputs_{v_.outputs};
};

struct Model_Element : JSON::Element {
  explicit Model_Element(Config::Model& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "type") {
      v_.type = JSON::Get<std::string_view>(value);
    } else if (name == "tokenizer_dir") {
      v_.tokenizer_dir = JSON::Get<std::string_view>(value);
    } else if (name == "vocab_size") {
      v_.vocab_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "context_length") {
      v_.context_length = SafeDoubleToInt(JSON::Get<double>(value), name);
      if (v_.context_length <= 0)
        throw std::out_of_range("context_length must be > 0, got " + std::to_string(v_.context_length));
    } else if (name == "pad_token_id") {
      v_.pad_token_id = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "eos_token_id") {
      v_.eos_token_id.assign(1, SafeDoubleToInt(JSON::Get<double>(value), name));
    } else if (name == "bos_token_id") {
      v_.bos_token_id = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "decoder_start_token_id") {
      v_.decoder_start_token_id = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "sep_token_id") {
      v_.sep_token_id = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "image_token_id") {
      v_.image_token_id = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "audio_token_id") {
      v_.audio_token_id = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "boa_token_id") {
      v_.boa_token_id = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "video_token_id") {
      v_.video_token_id = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "vision_start_token_id") {
      v_.vision_start_token_id = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "num_mels") {
      v_.num_mels = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "fft_size") {
      v_.fft_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "hop_length") {
      v_.hop_length = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "win_length") {
      v_.win_length = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "preemph") {
      v_.preemph = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "log_eps") {
      v_.log_eps = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "norm_eps") {
      v_.norm_eps = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "subsampling_factor") {
      v_.subsampling_factor = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "left_context") {
      v_.left_context = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "conv_context") {
      v_.conv_context = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "pre_encode_cache_size") {
      v_.pre_encode_cache_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "sample_rate") {
      v_.sample_rate = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "chunk_samples") {
      v_.chunk_samples = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "blank_id") {
      v_.blank_id = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "max_symbols_per_step") {
      v_.max_symbols_per_step = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "left_context_samples") {
      v_.left_context_samples = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "right_context_samples") {
      v_.right_context_samples = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == Config::Defaults::BotTokenIdName) {
      v_.bot_token_id = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == Config::Defaults::EotTokenIdName) {
      v_.eot_token_id = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == Config::Defaults::BorTokenIdName) {
      v_.bor_token_id = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == Config::Defaults::EorTokenIdName) {
      v_.eor_token_id = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnArray(std::string_view name) override {
    if (name == "eos_token_id")
      return eos_token_id_;
    if (name == "tdt_durations")
      return tdt_durations_;
    throw JSON::unknown_value_error{};
  }

  Element& OnObject(std::string_view name) override {
    if (name == "encoder") {
      return encoder_;
    }
    if (name == "decoder") {
      return decoder_;
    }
    if (name == "draft") {
      if (!v_.draft)
        v_.draft = Config::Model::Decoder{};
      if (!draft_)
        draft_ = std::make_unique<Decoder_Element>(*v_.draft);
      return *draft_;
    }
    if (name == "vision") {
      return vision_;
    }
    if (name == "embedding") {
      return embedding_;
    }
    if (name == "speech") {
      return speech_;
    }
    if (name == "joiner") {
      return joiner_;
    }
    if (name == "vad") {
      return vad_;
    }
    if (name == "mtp") {
      return mtp_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Model& v_;
  Encoder_Element encoder_{v_.encoder};
  Decoder_Element decoder_{v_.decoder};
  std::unique_ptr<Decoder_Element> draft_;
  Int_Array_Element eos_token_id_{v_.eos_token_id};
  Int_Array_Element tdt_durations_{v_.tdt_durations};
  Vision_Element vision_{v_.vision};
  Embedding_Element embedding_{v_.embedding};
  Speech_Element speech_{v_.speech};
  Joiner_Element joiner_{v_.joiner};
  VAD_Element vad_{v_.vad};
  Mtp_Element mtp_{v_.mtp};
};

// Throws std::runtime_error (rather than std::overflow_error/std::invalid_argument) on failure.
// JSON::Parse_Value only re-throws with the offending field's path prepended when it sees a
// std::runtime_error, and the public C API boundary reports std::runtime_error, so every config
// parsing error must use that single type to keep messages (and behavior) consistent.
int SafeDoubleToInt(double x, std::string_view name) {
  // 1. Check for non-finite values (NaN, infinity)
  if (!std::isfinite(x)) {
    std::stringstream ss;
    ss << "Field '" << name << "' cannot be converted to int32 (NaN or Inf)";
    throw std::runtime_error(ss.str());
  }

  // 2. Check if the value is outside the representable range of an integer.
  constexpr double min_int_val = static_cast<double>(std::numeric_limits<int>::min());
  constexpr double max_int_val = static_cast<double>(std::numeric_limits<int>::max());

  if (x < min_int_val || x > max_int_val) {
    std::stringstream ss;
    ss << "Field '" << name << "' value " << x << " is out of int32 range ["
       << std::numeric_limits<int>::min() << ", " << std::numeric_limits<int>::max() << "]";
    throw std::runtime_error(ss.str());
  }

  // 3. Reject fractional values — these fields must be integral.
  if (x != std::trunc(x)) {
    std::stringstream ss;
    ss << "Field '" << name << "' value " << x << " is not an integer";
    throw std::runtime_error(ss.str());
  }

  // 4. Perform the cast.
  return static_cast<int>(x);
}

int64_t SafeDoubleToInt64(double x, std::string_view name) {
  if (!std::isfinite(x)) {
    throw std::runtime_error("Field '" + std::string(name) +
                             "' cannot be converted to int64 (NaN or Inf)");
  }

  // int64_t::max() rounds up to 2^63 as a double, so use an exclusive upper bound.
  constexpr double min_int64_val = -9223372036854775808.0;
  constexpr double max_int64_exclusive = 9223372036854775808.0;
  if (x < min_int64_val || x >= max_int64_exclusive) {
    std::stringstream ss;
    ss << "Field '" << name << "' value " << x << " is out of int64 range ["
       << std::numeric_limits<int64_t>::min() << ", " << std::numeric_limits<int64_t>::max() << "]";
    throw std::runtime_error(ss.str());
  }

  if (x != std::trunc(x)) {
    std::stringstream ss;
    ss << "Field '" << name << "' value " << x << " is not an integer";
    throw std::runtime_error(ss.str());
  }

  return static_cast<int64_t>(x);
}

struct Search_Element : JSON::Element {
  explicit Search_Element(Config::Search& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "min_length") {
      v_.min_length = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "max_length") {
      v_.max_length = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "batch_size") {
      v_.batch_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "num_beams") {
      v_.num_beams = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "num_return_sequences") {
      v_.num_return_sequences = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "top_k") {
      v_.top_k = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "top_p") {
      v_.top_p = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "temperature") {
      v_.temperature = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "repetition_penalty") {
      v_.repetition_penalty = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "length_penalty") {
      v_.length_penalty = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "no_repeat_ngram_size") {
      v_.no_repeat_ngram_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "diversity_penalty") {
      v_.diversity_penalty = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "length_penalty") {
      v_.length_penalty = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "random_seed") {
      v_.random_seed = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "chunk_size") {
      double chunk_value = JSON::Get<double>(value);
      if (chunk_value > 0) {
        v_.chunk_size = static_cast<size_t>(chunk_value);
      } else {
        v_.chunk_size = std::nullopt;
      }
    } else if (name == "do_sample") {
      v_.do_sample = JSON::Get<bool>(value);
    } else if (name == "past_present_share_buffer") {
      v_.past_present_share_buffer = JSON::Get<bool>(value);
    } else if (name == "early_stopping") {
      v_.early_stopping = JSON::Get<bool>(value);
    } else if (name == "blank_penalty") {
      v_.blank_penalty = static_cast<float>(JSON::Get<double>(value));
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Search& v_;
};

struct Speculative_Element : JSON::Element {
  explicit Speculative_Element(Config::Speculative& v) : v_{v} {}

  // K (draft tokens per round) must be within [kMinK, kMaxK].
  static constexpr int kMinK = 1;
  static constexpr int kMaxK = 16;

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "max_draft_tokens") {
      int k = SafeDoubleToInt(JSON::Get<double>(value), name);
      if (k < kMinK || k > kMaxK)
        throw std::runtime_error(
            "speculative.max_draft_tokens must be between " + std::to_string(kMinK) + " and " +
            std::to_string(kMaxK) + " Got: " + std::to_string(k) + ".");
      v_.max_draft_tokens = k;
    } else if (name == "ngram_size") {
      const int ngram_size = SafeDoubleToInt(JSON::Get<double>(value), name);
      if (ngram_size != 0 && (ngram_size < 2 || ngram_size > kMaxK))
        throw std::runtime_error(
            "speculative.ngram_size must be 0 or between 2 and " + std::to_string(kMaxK) +
            ". Got: " + std::to_string(ngram_size) + ".");
      v_.ngram_size = ngram_size;
    } else if (name == "ngram_chained_lookup") {
      v_.ngram_chained_lookup = JSON::Get<bool>(value);
    } else if (name == "min_adaptive_k") {
      const int min_adaptive_k = SafeDoubleToInt(JSON::Get<double>(value), name);
      if (min_adaptive_k < 0 || min_adaptive_k > kMaxK)
        throw std::runtime_error(
            "speculative.min_adaptive_k must be 0 or between " + std::to_string(kMinK) +
            " and " + std::to_string(kMaxK) + ". Got: " +
            std::to_string(min_adaptive_k) + ".");
      v_.min_adaptive_k = min_adaptive_k;
    } else if (name == "cooldown") {
      v_.cooldown = JSON::Get<bool>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Speculative& v_;
};

struct DynamicBatching_Element : JSON::Element {
  explicit DynamicBatching_Element(std::optional<Config::Engine::DynamicBatching>& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (!v_)
      v_ = Config::Engine::DynamicBatching{};

    if (name == "block_size") {
      const auto parsed_value = SafeDoubleToInt(JSON::Get<double>(value), name);
      if (parsed_value <= 0)
        throw std::out_of_range("block_size must be > 0");
      v_->block_size = static_cast<size_t>(parsed_value);
    } else if (name == "num_blocks") {
      const auto parsed_value = SafeDoubleToInt(JSON::Get<double>(value), name);
      if (parsed_value <= 0)
        throw std::out_of_range("num_blocks must be > 0");
      v_->num_blocks = static_cast<size_t>(parsed_value);
    } else if (name == "gpu_utilization_factor") {
      const auto parsed_value = JSON::Get<double>(value);
      if (!std::isfinite(parsed_value) || parsed_value <= 0 || parsed_value > 1)
        throw std::out_of_range("gpu_utilization_factor must be > 0 and <= 1");
      v_->gpu_utilization_factor = static_cast<float>(parsed_value);
    } else if (name == "max_batch_size") {
      const auto parsed_value = SafeDoubleToInt(JSON::Get<double>(value), name);
      if (parsed_value <= 0)
        throw std::out_of_range("max_batch_size must be > 0");
      v_->max_batch_size = static_cast<size_t>(parsed_value);
    } else if (name == "max_scheduled_tokens") {
      const auto parsed_value = SafeDoubleToInt(JSON::Get<double>(value), name);
      if (parsed_value <= 0)
        throw std::out_of_range("max_scheduled_tokens must be > 0");
      v_->max_scheduled_tokens = static_cast<size_t>(parsed_value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  std::optional<Config::Engine::DynamicBatching>& v_;
};

struct StaticBatching_Element : JSON::Element {
  explicit StaticBatching_Element(std::optional<Config::Engine::StaticBatching>& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "max_batch_size") {
      v_->max_batch_size = static_cast<size_t>(JSON::Get<double>(value));
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  std::optional<Config::Engine::StaticBatching>& v_;
};

struct Engine_Element : JSON::Element {
  explicit Engine_Element(Config::Engine& v) : v_{v} {}

  Element& OnObject(std::string_view name) override {
    if (name == "dynamic_batching") {
      if (v_.static_batching)
        v_.static_batching.reset();
      return dynamic_batching_;
    } else if (name == "static_batching") {
      if (v_.dynamic_batching)
        v_.dynamic_batching.reset();
      return static_batching_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Engine& v_;
  DynamicBatching_Element dynamic_batching_{v_.dynamic_batching};
  StaticBatching_Element static_batching_{v_.static_batching};
};

void SetSearchNumber(Config::Search& search, std::string_view name, double value) {
  try {
    Search_Element(search).OnValue(name, value);
  } catch (...) {
    JSON::TranslateException(name);
  }
}

void SetSearchBool(Config::Search& search, std::string_view name, bool value) {
  try {
    Search_Element(search).OnValue(name, value);
  } catch (...) {
    JSON::TranslateException(name);
  }
}

void SetSpeculativeNumber(Config::Speculative& speculative, std::string_view name, double value) {
  try {
    Speculative_Element(speculative).OnValue(name, value);
  } catch (...) {
    JSON::TranslateException(name);
  }
}

void SetSpeculativeBool(Config::Speculative& speculative, std::string_view name, bool value) {
  try {
    Speculative_Element(speculative).OnValue(name, value);
  } catch (...) {
    JSON::TranslateException(name);
  }
}

void ClearProviders(Config& config) {
  config.model.decoder.session_options.providers.clear();
}

// Escape a string for safe embedding inside a JSON string literal. Prevents JSON
// injection when caller-supplied values are concatenated into a JSON document that
// will subsequently be parsed (e.g. in SetProviderOption below). Handles the
// mandatory JSON escapes: quote, backslash, and the C0 control-character shortcuts
// that the local JSON parser understands (\b \f \n \r \t).
//
// Other C0 control characters (< 0x20) have no shortcut and would require a
// \uXXXX escape, which the local JSON parser in src/json.cpp does not support
// (it throws "Unsupported uXXXX code used"). Since provider option names and
// values are configuration strings that are not expected to contain raw
// control characters, reject them here with a clear error rather than
// producing JSON that the parser cannot consume.
static std::string EscapeJsonString(std::string_view s) {
  std::string result;
  result.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '"':
        result += "\\\"";
        break;
      case '\\':
        result += "\\\\";
        break;
      case '\b':
        result += "\\b";
        break;
      case '\f':
        result += "\\f";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          throw std::runtime_error(
              "Unsupported control character in provider option string (code " +
              std::to_string(static_cast<unsigned char>(c)) + ")");
        }
        result += c;
        break;
    }
  }
  return result;
}

void SetProviderOption(Config& config, std::string_view provider_name, std::string_view option_name, std::string_view option_value) {
  // Normalize the provider name once
  auto normalized_provider = NormalizeProviderName(provider_name);

  // Ensure provider is in the providers list
  if (!contains(config.model.decoder.session_options.providers, normalized_provider)) {
    config.model.decoder.session_options.providers.push_back(std::string(normalized_provider));
  }

  // Remove any existing options with the same name to avoid duplicates
  for (auto& provider_options : config.model.decoder.session_options.provider_options) {
    if (provider_options.name == normalized_provider && !option_name.empty()) {
      provider_options.options.erase(
          std::remove_if(provider_options.options.begin(),
                         provider_options.options.end(),
                         [&option_name](const Config::NamedString& opt) {
                           return opt.first == option_name;
                         }),
          provider_options.options.end());
    }
  }

  // JSON-escape all caller-supplied string fragments before concatenating them into the
  // JSON document. Without escaping, quote/backslash characters in provider_name,
  // option_name, or option_value would let a caller inject arbitrary JSON structure
  // (sibling keys, new provider entries, etc.) into the parsed configuration.
  std::ostringstream json;
  json << R"({")" << EscapeJsonString(provider_name) << R"(":{)";
  if (!option_name.empty()) {
    json << R"(")" << EscapeJsonString(option_name) << R"(":")" << EscapeJsonString(option_value) << R"(")";
  }
  json << R"(}})";

  ProviderOptionsArray_Element element{config.model.decoder.session_options.provider_options};
  JSON::Parse(element, json.str());
}

bool IsGraphCaptureEnabled(const Config::SessionOptions& session_options) {
  for (const auto& provider : session_options.providers) {
    const auto provider_options = std::find_if(session_options.provider_options.begin(),
                                               session_options.provider_options.end(),
                                               [&provider](const Config::ProviderOptions& po) {
                                                 return po.name == provider;
                                               });
    if (provider_options != session_options.provider_options.end()) {
      if (provider_options->name == "cuda") {
        for (const auto& value : provider_options->options) {
          if (value.first == "enable_cuda_graph" && value.second == "1") {
            return true;
          }
        }
        return false;
      } else if (provider_options->name == "DML") {
        // Graph capture defaults to ON for DML but can be opted out via the
        // provider option "enable_graph_capture": "0". Captured-command-list
        // replay computes wrong logits on some D3D12 devices (observed on the
        // Xbox Series S Dev-Mode driver: deterministic garbage from the same
        // model that is correct on CPU EP and on non-captured ORT sessions).
        for (const auto& value : provider_options->options) {
          if (value.first == "enable_graph_capture" && value.second == "0") {
            return false;
          }
        }
        return true;
      } else if (provider_options->name == "AMDGPU") {
        return true;
      } else if (provider_options->name == "WebGPU") {
        for (const auto& value : provider_options->options) {
          if (value.first == "enableGraphCapture" && value.second == "1") {
            return true;
          }
        }
        return false;
      } else if (provider_options->name == "NvTensorRtRtx") {
        for (const auto& value : provider_options->options) {
          if (value.first == "enable_cuda_graph" && value.second == "1") {
            return true;
          }
        }
        return false;
      }
    }
  }

  return false;
}

bool IsMultiProfileEnabled(const Config::SessionOptions& session_options) {
  for (const auto& provider : session_options.providers) {
    const auto provider_options = std::find_if(session_options.provider_options.begin(),
                                               session_options.provider_options.end(),
                                               [&provider](const Config::ProviderOptions& po) {
                                                 return po.name == provider;
                                               });
    if (provider_options != session_options.provider_options.end()) {
      if (provider_options->name == "NvTensorRtRtx") {
        for (const auto& value : provider_options->options) {
          if (value.first == "nv_multi_profile_enable" && value.second == "1") {
            return true;
          }
        }
      }
    }
  }
  return false;
}

void SetDecoderProviderOptionsHardwareDeviceType(Config& config, std::string_view provider_name, std::string_view hardware_device_type) {
  auto normalized_provider = NormalizeProviderName(provider_name);
  for (auto& provider_option : config.model.decoder.session_options.provider_options) {
    if (provider_option.name == normalized_provider) {
      if (!provider_option.device_filtering_options) {
        provider_option.device_filtering_options = Config::DeviceFilteringOptions{};
      }
      provider_option.device_filtering_options->hardware_device_type = ParseHardwareDeviceType(hardware_device_type);
    }
  }
}

void SetDecoderProviderOptionsHardwareDeviceId(Config& config, std::string_view provider_name, uint32_t hardware_device_id) {
  auto normalized_provider = NormalizeProviderName(provider_name);
  for (auto& provider_option : config.model.decoder.session_options.provider_options) {
    if (provider_option.name == normalized_provider) {
      if (!provider_option.device_filtering_options) {
        provider_option.device_filtering_options = Config::DeviceFilteringOptions{};
      }
      provider_option.device_filtering_options->hardware_device_id = hardware_device_id;
    }
  }
}

void SetDecoderProviderOptionsHardwareVendorId(Config& config, std::string_view provider_name, uint32_t hardware_vendor_id) {
  auto normalized_provider = NormalizeProviderName(provider_name);
  for (auto& provider_option : config.model.decoder.session_options.provider_options) {
    if (provider_option.name == normalized_provider) {
      if (!provider_option.device_filtering_options) {
        provider_option.device_filtering_options = Config::DeviceFilteringOptions{};
      }
      provider_option.device_filtering_options->hardware_vendor_id = hardware_vendor_id;
    }
  }
}

void ClearDecoderProviderOptionsHardwareDeviceType(Config& config, std::string_view provider_name) {
  auto normalized_provider = NormalizeProviderName(provider_name);
  for (auto& provider_option : config.model.decoder.session_options.provider_options) {
    if (provider_option.name == normalized_provider && provider_option.device_filtering_options) {
      provider_option.device_filtering_options->hardware_device_type = std::nullopt;
    }
  }
}

void ClearDecoderProviderOptionsHardwareDeviceId(Config& config, std::string_view provider_name) {
  auto normalized_provider = NormalizeProviderName(provider_name);
  for (auto& provider_option : config.model.decoder.session_options.provider_options) {
    if (provider_option.name == normalized_provider && provider_option.device_filtering_options) {
      provider_option.device_filtering_options->hardware_device_id = std::nullopt;
    }
  }
}

void ClearDecoderProviderOptionsHardwareVendorId(Config& config, std::string_view provider_name) {
  auto normalized_provider = NormalizeProviderName(provider_name);
  for (auto& provider_option : config.model.decoder.session_options.provider_options) {
    if (provider_option.name == normalized_provider && provider_option.device_filtering_options) {
      provider_option.device_filtering_options->hardware_vendor_id = std::nullopt;
    }
  }
}

struct Root_Element : JSON::Element {
  explicit Root_Element(Config& config) : config_{config} {}

  void OnValue(std::string_view /*name*/, JSON::Value /*value*/) override {
    // No top-level scalar values currently supported
  }

  Element& OnObject(std::string_view name) override {
    if (name == "model") return model_element_;
    if (name == "search") return search_element_;
    if (name == "speculative") return speculative_element_;
    if (name == "engine") return engine_element_;
    throw JSON::unknown_value_error{};
  }

  Config& config_;
  Model_Element model_element_{config_.model};
  Search_Element search_element_{config_.search};
  Speculative_Element speculative_element_{config_.speculative};
  Engine_Element engine_element_{config_.engine};
};

struct RootObject_Element : JSON::Element {
  explicit RootObject_Element(JSON::Element& t) : t_{t} {}

  Element& OnObject(std::string_view /*name*/) override {
    return t_;
  }

  JSON::Element& t_;
};

void ParseConfig(const fs::path& filename, std::string_view json_overlay, Config& config) {
  std::ifstream file = filename.open(std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Error opening " + filename.string());
  }
  std::streamsize const size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    throw std::runtime_error("Error reading " + filename.string());
  }

  Root_Element root{config};
  RootObject_Element root_object{root};
  try {
    JSON::Parse(root_object, std::string_view(buffer.data(), buffer.size()));
  } catch (const std::exception& message) {
    std::ostringstream oss;
    oss << "Error encountered while parsing '" << filename.string() << "' " << message.what();
    throw std::runtime_error(oss.str());
  }

  if (!json_overlay.empty()) {
    try {
      JSON::Parse(root_object, json_overlay);
    } catch (const std::exception& message) {
      std::ostringstream oss;
      oss << "Error encountered while parsing config overlay: " << message.what();
      throw std::runtime_error(oss.str());
    }
  }
}

void OverlayConfig(Config& config, std::string_view json) {
  Root_Element root{config};
  RootObject_Element element{root};
  JSON::Parse(element, json);
}

fs::path Config::ResolvePath(std::string_view value) const {
  if (value.empty()) {
    return config_path;
  }
  if (package_resolver) {
    // Loaded from a model package: ORT owns all path-reference resolution (sha256: shared
    // assets with manifest overrides, relative paths, confinement).
    return package_resolver(config_path, value);
  }
  // Flat directory: sha256: shared-asset references are only meaningful inside a model package.
  constexpr std::string_view kSharedAssetPrefix = "sha256:";
  if (value.substr(0, kSharedAssetPrefix.size()) == kSharedAssetPrefix) {
    throw std::runtime_error(
        "\"" + std::string{value} +
        "\" is a sha256: shared-asset reference, which is only valid when loading from a model "
        "package; this model was loaded from a plain directory.");
  }
  return config_path / std::string{value};
}

// Validates every config-driven filename/path field after parsing so downstream code
// (model/processor/adapter loading) can rely on paths being safe. Centralising the checks
// here keeps individual model families free of path-validation calls.
namespace {

// Validates that a config-specified filename/path stays inside the model directory.
// Throws std::runtime_error if the path is absolute, contains a Windows drive/UNC root,
// or contains a ".." path traversal component. Empty paths are allowed (no-op). The
// optional context label is prepended to error messages so callers can identify which
// config field caused the failure.
void ValidateConfigPath(const std::string& path, std::string_view context = {}) {
  if (path.empty()) return;

  auto make_error = [&](const std::string& msg) -> std::string {
    return context.empty() ? msg : (std::string{context} + ": " + msg);
  };

  // Reject absolute paths: Unix "/" or Windows drive letters "C:" / "C:\" or UNC "\\"
  if (path[0] == '/' || path[0] == '\\') {
    throw std::runtime_error(make_error("Config path must be a relative path under the model directory, got: " + path));
  }
#ifdef _WIN32
  if (path.size() >= 2 && std::isalpha(static_cast<unsigned char>(path[0])) && path[1] == ':') {
    throw std::runtime_error(make_error("Config path must be a relative path under the model directory, got: " + path));
  }
#endif

  // Reject path traversal ".." components. Split on '/' and '\\' and check each component.
  std::string component;
  for (size_t i = 0; i <= path.size(); ++i) {
    if (i == path.size() || path[i] == '/' || path[i] == '\\') {
      if (component == "..") {
        throw std::runtime_error(make_error("Config path must not contain path traversal (..): " + path));
      }
      component.clear();
    } else {
      component += path[i];
    }
  }
}

void ValidateModelPaths(const Config& config) {
  const auto& m = config.model;
  ValidateConfigPath(m.encoder.filename, "model.encoder.filename");
  ValidateConfigPath(m.embedding.filename, "model.embedding.filename");

  ValidateConfigPath(m.vision.filename, "model.vision.filename");
  ValidateConfigPath(m.vision.config_filename, "model.vision.config_filename");
  if (m.vision.adapter_filename.has_value()) {
    ValidateConfigPath(*m.vision.adapter_filename, "model.vision.adapter_filename");
  }
  for (const auto& stage : m.vision.pipeline) {
    ValidateConfigPath(stage.filename, "model.vision.pipeline.filename");
  }

  ValidateConfigPath(m.speech.filename, "model.speech.filename");
  ValidateConfigPath(m.speech.config_filename, "model.speech.config_filename");
  if (m.speech.adapter_filename.has_value()) {
    ValidateConfigPath(*m.speech.adapter_filename, "model.speech.adapter_filename");
  }

  ValidateConfigPath(m.joiner.filename, "model.joiner.filename");
  ValidateConfigPath(m.vad.filename, "model.vad.filename");

  ValidateConfigPath(m.decoder.filename, "model.decoder.filename");
  for (const auto& stage : m.decoder.pipeline) {
    ValidateConfigPath(stage.filename, "model.decoder.pipeline.filename");
  }
}

}  // namespace

Config::Config(const fs::path& path, std::string_view json_overlay) : config_path{path} {
  ParseConfig(path / "genai_config.json", json_overlay, *this);

  if (model.context_length == 0 && !ModelType::IsRNNT(model.type)) {
    throw std::runtime_error("model context_length is 0 or was not set. It must be greater than 0");
  }

  if (search.max_length == 0) {
    search.max_length = model.context_length;
  }

  // If no eos_token_id was set, set it to the pad token id
  if (model.eos_token_id.empty()) {
    model.eos_token_id.push_back(model.pad_token_id);
  }

  for (const auto& provider_option : model.decoder.session_options.provider_options) {
    model.decoder.session_options.providers.push_back(provider_option.name);
  }

  if (model.draft) {
    for (const auto& provider_option : model.draft->session_options.provider_options) {
      model.draft->session_options.providers.push_back(provider_option.name);
    }
  }

  if (model.encoder.session_options.has_value()) {
    for (const auto& provider_option : model.encoder.session_options->provider_options) {
      model.encoder.session_options->providers.push_back(provider_option.name);
    }
  }

  if (model.vision.session_options.has_value()) {
    for (const auto& provider_option : model.vision.session_options->provider_options) {
      model.vision.session_options->providers.push_back(provider_option.name);
    }
  }

  if (model.speech.session_options.has_value()) {
    for (const auto& provider_option : model.speech.session_options->provider_options) {
      model.speech.session_options->providers.push_back(provider_option.name);
    }
  }

  if (model.embedding.session_options.has_value()) {
    for (const auto& provider_option : model.embedding.session_options->provider_options) {
      model.embedding.session_options->providers.push_back(provider_option.name);
    }
  }

  // Validate all config-specified filenames/paths after parsing so downstream loaders
  // (model/processor/adapter creation) can rely on them being safe.
  ValidateModelPaths(*this);
}

void Config::AddMapping(const std::string& nominal_name, const std::string& graph_name) {
  auto [it, emplaced] = nominal_names_to_graph_names_.emplace(nominal_name, graph_name);
  if (it->second != graph_name) {
    std::ostringstream oss;
    oss << "Duplicate nominal name: " << nominal_name << " with graph names: "
        << graph_name << " and " << it->second;
    throw std::runtime_error(oss.str());
  }
}

std::pair<std::string, bool> Config::GetGraphName(const std::string& nominal_name) const {
  auto it = nominal_names_to_graph_names_.find(nominal_name);
  if (it == nominal_names_to_graph_names_.end()) {
    return {nominal_name, false};
  }
  return {it->second, true};
}

}  // namespace Generators
