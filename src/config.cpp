// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Portions of this file consist of AI generated content.
#include "generators.h"
#include "models/model_type.h"
#include "pipeline_presets.h"
#include "runtime_settings.h"
#include "json.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace Generators {

// Fix casing of certain historical names to match current Onnxruntime names
std::string_view NormalizeProviderName(std::string_view name) {
  std::string lower_name(name);
  std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });
  if (lower_name == "qnn") {
    return "QNN";
  } else if (lower_name == "webgpu") {
    return "WebGPU";
  } else if (lower_name == "dml") {
    return "DML";
  } else if (lower_name == "openvino") {
    return "OpenVINO";
  } else if (lower_name == "vitisai") {
    return "VitisAI";
  } else if (lower_name == "nvtensorrtrtx" || lower_name == "nvtensorrtrtxexecutionprovider") {
    return "NvTensorRtRtx";
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
    v_.emplace_back(static_cast<int>(JSON::Get<double>(value)));
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

// v2.1 (issue #2114, PR-F, design §7): SAX parsers for the runtime-vs-build-time feature namespace
// under session_options. Each sub-element throws JSON::unknown_value_error for unknown keys so a
// typo or a mis-namespaced feature fails fast at parse time rather than being silently ignored.
struct RuntimeKvCache_Element : JSON::Element {
  explicit RuntimeKvCache_Element(Config::RuntimeFeatures::KvCache& v) : v_{v} {}
  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "dtype") {
      v_.dtype = JSON::Get<std::string_view>(value);
    } else if (name == "quant") {
      v_.quant = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::RuntimeFeatures::KvCache& v_;
};

struct RuntimePaging_Element : JSON::Element {
  explicit RuntimePaging_Element(Config::RuntimeFeatures::Paging& v) : v_{v} {}
  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "enabled") {
      v_.enabled = JSON::Get<bool>(value);
    } else if (name == "block_size") {
      v_.block_size = static_cast<int>(JSON::Get<double>(value));
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::RuntimeFeatures::Paging& v_;
};

struct RuntimePrefixCache_Element : JSON::Element {
  explicit RuntimePrefixCache_Element(Config::RuntimeFeatures::PrefixCache& v) : v_{v} {}
  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "enabled") {
      v_.enabled = JSON::Get<bool>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::RuntimeFeatures::PrefixCache& v_;
};

struct RuntimeSlidingWindow_Element : JSON::Element {
  explicit RuntimeSlidingWindow_Element(Config::RuntimeFeatures::SlidingWindow& v) : v_{v} {}
  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "size") {
      v_.size = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "sink_tokens") {
      v_.sink_tokens = static_cast<int>(JSON::Get<double>(value));
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::RuntimeFeatures::SlidingWindow& v_;
};

struct RuntimeChunkedPrefill_Element : JSON::Element {
  explicit RuntimeChunkedPrefill_Element(Config::RuntimeFeatures::ChunkedPrefill& v) : v_{v} {}
  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "max_batched_tokens") {
      v_.max_batched_tokens = static_cast<int>(JSON::Get<double>(value));
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::RuntimeFeatures::ChunkedPrefill& v_;
};

struct RuntimeFeatures_Element : JSON::Element {
  explicit RuntimeFeatures_Element(Config::RuntimeFeatures& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "precision") {
      v_.precision = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnObject(std::string_view name) override {
    if (name == "kv_cache") {
      v_.kv_cache = Config::RuntimeFeatures::KvCache{};
      kv_cache_ = std::make_unique<RuntimeKvCache_Element>(*v_.kv_cache);
      return *kv_cache_;
    }
    if (name == "paging") {
      v_.paging = Config::RuntimeFeatures::Paging{};
      paging_ = std::make_unique<RuntimePaging_Element>(*v_.paging);
      return *paging_;
    }
    if (name == "prefix_cache") {
      v_.prefix_cache = Config::RuntimeFeatures::PrefixCache{};
      prefix_cache_ = std::make_unique<RuntimePrefixCache_Element>(*v_.prefix_cache);
      return *prefix_cache_;
    }
    if (name == "sliding_window") {
      v_.sliding_window = Config::RuntimeFeatures::SlidingWindow{};
      sliding_window_ = std::make_unique<RuntimeSlidingWindow_Element>(*v_.sliding_window);
      return *sliding_window_;
    }
    if (name == "chunked_prefill") {
      v_.chunked_prefill = Config::RuntimeFeatures::ChunkedPrefill{};
      chunked_prefill_ = std::make_unique<RuntimeChunkedPrefill_Element>(*v_.chunked_prefill);
      return *chunked_prefill_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::RuntimeFeatures& v_;
  std::unique_ptr<RuntimeKvCache_Element> kv_cache_;
  std::unique_ptr<RuntimePaging_Element> paging_;
  std::unique_ptr<RuntimePrefixCache_Element> prefix_cache_;
  std::unique_ptr<RuntimeSlidingWindow_Element> sliding_window_;
  std::unique_ptr<RuntimeChunkedPrefill_Element> chunked_prefill_;
};

struct BuildRequires_Element : JSON::Element {
  explicit BuildRequires_Element(Config::BuildRequires& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "attention") {
      v_.attention = JSON::Get<std::string_view>(value);
    } else if (name == "quantization") {
      v_.quantization = JSON::Get<std::string_view>(value);
    } else if (name == "extra_heads") {
      v_.extra_heads = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::BuildRequires& v_;
};

struct SessionOptions_Element : JSON::Element {
  explicit SessionOptions_Element(Config::SessionOptions& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "log_id") {
      v_.log_id = JSON::Get<std::string_view>(value);
    } else if (name == "enable_profiling") {
      v_.enable_profiling = JSON::Get<std::string_view>(value);
    } else if (name == "intra_op_num_threads") {
      v_.intra_op_num_threads = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "inter_op_num_threads") {
      v_.inter_op_num_threads = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "log_severity_level") {
      v_.log_severity_level = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "log_verbosity_level") {
      v_.log_verbosity_level = static_cast<int>(JSON::Get<double>(value));
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

  JSON::Element& OnObject(std::string_view name) override {
    // v2.1 (issue #2114, PR-F, design §7): the runtime-vs-build-time feature namespace. These are the
    // only nested objects recognized under session_options; any other object key still throws (so old
    // configs are unaffected and typos fail fast).
    if (name == "runtime") {
      v_.runtime = Config::RuntimeFeatures{};
      runtime_ = std::make_unique<RuntimeFeatures_Element>(*v_.runtime);
      return *runtime_;
    }
    if (name == "build_requires") {
      v_.build_requires = Config::BuildRequires{};
      build_requires_ = std::make_unique<BuildRequires_Element>(*v_.build_requires);
      return *build_requires_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::SessionOptions& v_;
  ProviderOptionsArray_Element provider_options_{v_.provider_options};
  std::unique_ptr<RuntimeFeatures_Element> runtime_;
  std::unique_ptr<BuildRequires_Element> build_requires_;
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
    } else if (name == "past_conv_names") {
      v_.past_conv_names = JSON::Get<std::string_view>(value);
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
    } else if (name == "outputs") {
      v_.outputs = JSON::Get<std::string_view>(value);
    } else if (name == "lstm_hidden_state") {
      v_.lstm_hidden_state = JSON::Get<std::string_view>(value);
    } else if (name == "lstm_cell_state") {
      v_.lstm_cell_state = JSON::Get<std::string_view>(value);
    } else if (name == "hidden_states") {
      v_.hidden_states = JSON::Get<std::string_view>(value);
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
    v_.push_back(static_cast<int>(JSON::Get<double>(value)));
  }

 private:
  std::vector<int>& v_;
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
    } else if (name == "reset_session_idx") {
      v_.reset_session_idx = static_cast<int>(JSON::Get<double>(value));
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
      v_->window_size = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "pad_value") {
      v_->pad_value = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "alignment") {
      v_->alignment = JSON::Get<std::string_view>(value);
    } else if (name == "slide_key_value_cache") {
      v_->slide_key_value_cache = JSON::Get<bool>(value);
    } else if (name == "slide_inputs") {
      v_->slide_inputs = JSON::Get<bool>(value);
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
      v_.hidden_size = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "num_attention_heads") {
      v_.num_attention_heads = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "num_hidden_layers") {
      v_.num_hidden_layers = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "num_key_value_heads") {
      v_.num_key_value_heads = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "head_size") {
      v_.head_size = static_cast<int>(JSON::Get<double>(value));
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
      v_.hidden_size = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "num_attention_heads") {
      v_.num_attention_heads = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "num_hidden_layers") {
      v_.num_hidden_layers = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "num_key_value_heads") {
      v_.num_key_value_heads = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "head_size") {
      v_.head_size = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "conv_cache_size") {
      v_.conv_cache_size = static_cast<int>(JSON::Get<double>(value));
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
      v_.spatial_merge_size = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "tokens_per_second") {
      v_.tokens_per_second = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "patch_size") {
      v_.patch_size = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "num_visual_tokens") {
      v_.num_visual_tokens = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "window_size") {
      v_.window_size = static_cast<int>(JSON::Get<double>(value));
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
      v_.silence_duration_ms = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "prefix_padding_ms") {
      v_.prefix_padding_ms = static_cast<int>(JSON::Get<double>(value));
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
    } else if (name == "vocab_size") {
      v_.vocab_size = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "context_length") {
      v_.context_length = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "pad_token_id") {
      v_.pad_token_id = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "eos_token_id") {
      v_.eos_token_id.assign(1, static_cast<int>(JSON::Get<double>(value)));
    } else if (name == "bos_token_id") {
      v_.bos_token_id = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "decoder_start_token_id") {
      v_.decoder_start_token_id = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "sep_token_id") {
      v_.sep_token_id = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "image_token_id") {
      v_.image_token_id = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "audio_token_id") {
      v_.audio_token_id = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "boa_token_id") {
      v_.boa_token_id = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "video_token_id") {
      v_.video_token_id = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "vision_start_token_id") {
      v_.vision_start_token_id = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "num_mels") {
      v_.num_mels = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "fft_size") {
      v_.fft_size = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "hop_length") {
      v_.hop_length = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "win_length") {
      v_.win_length = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "preemph") {
      v_.preemph = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "log_eps") {
      v_.log_eps = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "norm_eps") {
      v_.norm_eps = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "subsampling_factor") {
      v_.subsampling_factor = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "left_context") {
      v_.left_context = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "conv_context") {
      v_.conv_context = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "pre_encode_cache_size") {
      v_.pre_encode_cache_size = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "sample_rate") {
      v_.sample_rate = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "chunk_samples") {
      v_.chunk_samples = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "blank_id") {
      v_.blank_id = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "max_symbols_per_step") {
      v_.max_symbols_per_step = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "left_context_samples") {
      v_.left_context_samples = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "right_context_samples") {
      v_.right_context_samples = static_cast<int>(JSON::Get<double>(value));
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
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Model& v_;
  Encoder_Element encoder_{v_.encoder};
  Decoder_Element decoder_{v_.decoder};
  Int_Array_Element eos_token_id_{v_.eos_token_id};
  Int_Array_Element tdt_durations_{v_.tdt_durations};
  Vision_Element vision_{v_.vision};
  Embedding_Element embedding_{v_.embedding};
  Speech_Element speech_{v_.speech};
  Joiner_Element joiner_{v_.joiner};
  VAD_Element vad_{v_.vad};
};

int SafeDoubleToInt(double x, std::string_view name) {
  // 1. Check for non-finite values (NaN, infinity)
  if (!std::isfinite(x)) {
    std::stringstream ss;
    ss << "Field '" << name << "' cannot be converted to int32 (NaN or Inf)";
    throw std::overflow_error(ss.str());
  }

  // 2. Check if the value is outside the representable range of an integer.
  constexpr double min_int_val = static_cast<double>(std::numeric_limits<int>::min());
  constexpr double max_int_val = static_cast<double>(std::numeric_limits<int>::max());

  if (x < min_int_val || x > max_int_val) {
    std::stringstream ss;
    ss << "Field '" << name << "' value " << x << " is out of int32 range ["
       << std::numeric_limits<int>::min() << ", " << std::numeric_limits<int>::max() << "]";
    throw std::overflow_error(ss.str());
  }

  // 3. Perform the cast. This truncates any fractional part (e.g., 3.9 becomes 3).
  // If rounding is desired, use `return static_cast<int>(std::round(x));`
  return static_cast<int>(x);
}

struct Search_Element : JSON::Element {
  explicit Search_Element(Config::Search& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "min_length") {
      v_.min_length = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "max_length") {
      v_.max_length = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "batch_size") {
      v_.batch_size = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "num_beams") {
      v_.num_beams = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "num_return_sequences") {
      v_.num_return_sequences = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "top_k") {
      v_.top_k = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "top_p") {
      v_.top_p = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "temperature") {
      v_.temperature = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "repetition_penalty") {
      v_.repetition_penalty = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "length_penalty") {
      v_.length_penalty = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "no_repeat_ngram_size") {
      v_.no_repeat_ngram_size = static_cast<int>(JSON::Get<double>(value));
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

struct DynamicBatching_Element : JSON::Element {
  explicit DynamicBatching_Element(std::optional<Config::Engine::DynamicBatching>& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (!v_)
      v_ = Config::Engine::DynamicBatching{};

    if (name == "block_size") {
      v_->block_size = static_cast<size_t>(JSON::Get<double>(value));
    } else if (name == "num_blocks") {
      v_->num_blocks = static_cast<size_t>(JSON::Get<double>(value));
    } else if (name == "gpu_utilization_factor") {
      v_->gpu_utilization_factor = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "max_batch_size") {
      v_->max_batch_size = static_cast<size_t>(JSON::Get<double>(value));
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

void ClearProviders(Config& config) {
  config.model.decoder.session_options.providers.clear();
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

  std::ostringstream json;
  json << R"({")" << provider_name << R"(":{)";
  if (!option_name.empty()) {
    json << R"(")" << option_name << R"(":")" << option_value << R"(")";
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

namespace {

// v2.1 (issue #2114, PR-F, design §7) validation helpers. Each allowlist captures the values that are
// SENSIBLE for its namespace. A handful of well-known tokens are explicitly tagged as belonging to the
// *other* namespace so a mis-namespaced feature ("declared, never synthesized") yields a precise error
// instead of a generic "unknown value".
bool Contains(std::initializer_list<std::string_view> set, const std::string& value) {
  return std::find(set.begin(), set.end(), value) != set.end();
}

[[noreturn]] void ThrowFeatureError(std::string_view context, std::string_view field, const std::string& value,
                                    std::string_view detail) {
  std::ostringstream oss;
  oss << "Invalid session_options." << field << " value '" << value << "' for " << context << ": " << detail;
  throw std::runtime_error(oss.str());
}

}  // namespace

void ValidateSessionOptionsFeatures(const Config::SessionOptions& session_options, std::string_view context) {
  // Back-compat: nothing to validate (and nothing changes) unless one of the v2.1 blocks is present.
  if (session_options.runtime.has_value()) {
    const auto& runtime = *session_options.runtime;

    if (runtime.kv_cache.has_value() && runtime.kv_cache->dtype.has_value()) {
      const std::string& dtype = *runtime.kv_cache->dtype;
      // Weight-quantization schemes are a BUILD-TIME property of the exported weights, not a runtime KV
      // element type. Reject them here so the honest runtime/build split is enforced.
      if (Contains({"awq", "gptq"}, dtype)) {
        ThrowFeatureError(context, "runtime.kv_cache.dtype", dtype,
                          "this is a build-time weight quantization; declare it under "
                          "session_options.build_requires.quantization, not the runtime namespace");
      }
      if (!Contains({"fp32", "fp16", "bf16", "fp8", "int8", "int4"}, dtype)) {
        ThrowFeatureError(context, "runtime.kv_cache.dtype", dtype,
                          "unknown KV-cache dtype (expected one of fp32, fp16, bf16, fp8, int8, int4)");
      }
    }
    if (runtime.kv_cache.has_value() && runtime.kv_cache->quant.has_value()) {
      const std::string& quant = *runtime.kv_cache->quant;
      if (!Contains({"none", "per_token", "per_channel", "per_tensor"}, quant)) {
        ThrowFeatureError(context, "runtime.kv_cache.quant", quant,
                          "unknown KV-cache quant scheme (expected none, per_token, per_channel, per_tensor)");
      }
    }
    if (runtime.precision.has_value()) {
      const std::string& precision = *runtime.precision;
      if (!Contains({"fp32", "fp16", "bf16", "fp8"}, precision)) {
        ThrowFeatureError(context, "runtime.precision", precision,
                          "unknown compute precision (expected fp32, fp16, bf16, fp8)");
      }
    }
    if (runtime.paging.has_value() && runtime.paging->block_size.has_value() &&
        *runtime.paging->block_size <= 0) {
      ThrowFeatureError(context, "runtime.paging.block_size", std::to_string(*runtime.paging->block_size),
                        "block_size must be a positive integer");
    }
  }

  if (session_options.build_requires.has_value()) {
    const auto& build = *session_options.build_requires;

    if (build.attention.has_value()) {
      const std::string& attention = *build.attention;
      if (!Contains({"mha", "mqa", "gqa"}, attention)) {
        ThrowFeatureError(context, "build_requires.attention", attention,
                          "unknown attention shape (expected mha, mqa, gqa)");
      }
    }
    if (build.quantization.has_value()) {
      const std::string& quant = *build.quantization;
      if (!Contains({"none", "awq", "gptq", "int8", "int4", "fp8"}, quant)) {
        ThrowFeatureError(context, "build_requires.quantization", quant,
                          "unknown weight quantization (expected none, awq, gptq, int8, int4, fp8)");
      }
    }
    if (build.extra_heads.has_value()) {
      const std::string& heads = *build.extra_heads;
      // Runtime-only knobs declared as build heads are nonsensical -- catch the obvious confusions.
      if (Contains({"paging", "prefix_cache", "sliding_window", "chunked_prefill"}, heads)) {
        ThrowFeatureError(context, "build_requires.extra_heads", heads,
                          "this is a runtime feature; declare it under session_options.runtime, not "
                          "the build_requires namespace");
      }
      if (!Contains({"none", "medusa", "mtp", "eagle", "eagle3", "hydra"}, heads)) {
        ThrowFeatureError(context, "build_requires.extra_heads", heads,
                          "unknown extra-heads recipe (expected none, medusa, mtp, eagle, eagle3, hydra)");
      }
    }
  }
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

// ---------------------------------------------------------------------------
// Pipeline-as-Config (issue #2114) schema v2 parsing elements.
// These mirror the existing per-struct SAX visitor style (one *_Element per JSON object).
// They populate Config::Pipeline; they do not change any runtime behavior in PR1.
// ---------------------------------------------------------------------------

// Swallows an object/array subtree without error. Used for v2 sections not consumed in PR1
// (e.g. "preprocessing") so that forward-looking configs still parse.
struct Ignore_Element : JSON::Element {
  void OnValue(std::string_view /*name*/, JSON::Value /*value*/) override {}
  Element& OnObject(std::string_view /*name*/) override { return *this; }
  Element& OnArray(std::string_view /*name*/) override { return *this; }
};

struct PipelineSession_Element : JSON::Element {
  explicit PipelineSession_Element(Config::Pipeline::Session& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "file") {
      v_.file = JSON::Get<std::string_view>(value);
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
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Pipeline::Session& v_;
  std::unique_ptr<SessionOptions_Element> session_options_;
};

// "sessions" is an object keyed by logical session name: {"decoder": {"file": "..."}}.
struct PipelineSessions_Element : JSON::Element {
  explicit PipelineSessions_Element(std::vector<Config::Pipeline::Session>& v) : v_{v} {}

  Element& OnObject(std::string_view name) override {
    auto& session = v_.emplace_back();
    session.name = name;
    elements_.emplace_back(session);
    return elements_.back();
  }

 private:
  std::vector<Config::Pipeline::Session>& v_;
  std::vector<PipelineSession_Element> elements_;
};

struct FlowStep_Element : JSON::Element {
  explicit FlowStep_Element(Config::Pipeline::FlowStep& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "run") {
      v_.run = JSON::Get<std::string_view>(value);
    } else if (name == "when") {
      v_.when = JSON::Get<std::string_view>(value);
    } else if (name == "loop") {
      v_.loop = JSON::Get<std::string_view>(value);
    } else if (name == "cross_attention_from") {
      v_.cross_attention_from = std::string(JSON::Get<std::string_view>(value));
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Pipeline::FlowStep& v_;
};

// "flow" is an array of step objects.
struct Flow_Element : JSON::Element {
  explicit Flow_Element(std::vector<Config::Pipeline::FlowStep>& v) : v_{v} {}

  Element& OnObject(std::string_view /*name*/) override {
    auto& step = v_.emplace_back();
    elements_.emplace_back(step);
    return elements_.back();
  }

 private:
  std::vector<Config::Pipeline::FlowStep>& v_;
  std::vector<FlowStep_Element> elements_;
};

struct Wire_Element : JSON::Element {
  explicit Wire_Element(Config::Pipeline::Wire& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "from") {
      v_.from = JSON::Get<std::string_view>(value);
    } else if (name == "to") {
      v_.to = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Pipeline::Wire& v_;
};

// "dataflow" is an array of {from,to} wire objects.
struct Dataflow_Element : JSON::Element {
  explicit Dataflow_Element(std::vector<Config::Pipeline::Wire>& v) : v_{v} {}

  Element& OnObject(std::string_view /*name*/) override {
    auto& wire = v_.emplace_back();
    elements_.emplace_back(wire);
    return elements_.back();
  }

 private:
  std::vector<Config::Pipeline::Wire>& v_;
  std::vector<Wire_Element> elements_;
};

struct KvCache_Element : JSON::Element {
  explicit KvCache_Element(Config::Pipeline::State::KvCache& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "format") {
      v_.format = JSON::Get<std::string_view>(value);
    } else if (name == "past_key_pattern") {
      v_.past_key_pattern = JSON::Get<std::string_view>(value);
    } else if (name == "present_key_pattern") {
      v_.present_key_pattern = JSON::Get<std::string_view>(value);
    } else if (name == "past_value_pattern") {
      v_.past_value_pattern = JSON::Get<std::string_view>(value);
    } else if (name == "present_value_pattern") {
      v_.present_value_pattern = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Pipeline::State::KvCache& v_;
};

struct CrossCache_Element : JSON::Element {
  explicit CrossCache_Element(Config::Pipeline::State::CrossCache& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "source") {
      v_.source = std::string(JSON::Get<std::string_view>(value));
    } else if (name == "frozen") {
      v_.frozen = JSON::Get<bool>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Pipeline::State::CrossCache& v_;
};

struct PositionIds_Element : JSON::Element {
  explicit PositionIds_Element(Config::Pipeline::State::PositionIds& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "strategy") {
      v_.strategy = JSON::Get<std::string_view>(value);
    } else if (name == "input_name") {
      v_.input_name = JSON::Get<std::string_view>(value);
    } else if (name == "grid_source") {
      v_.grid_source = std::string(JSON::Get<std::string_view>(value));
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Pipeline::State::PositionIds& v_;
};

struct PipelineState_Element : JSON::Element {
  explicit PipelineState_Element(Config::Pipeline::State& v) : v_{v} {}

  Element& OnObject(std::string_view name) override {
    if (name == "kv_cache") return kv_cache_;
    if (name == "position_ids") return position_ids_;
    if (name == "cross_cache") {
      v_.cross_cache = Config::Pipeline::State::CrossCache{};
      cross_cache_ = std::make_unique<CrossCache_Element>(*v_.cross_cache);
      return *cross_cache_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Pipeline::State& v_;
  KvCache_Element kv_cache_{v_.kv_cache};
  PositionIds_Element position_ids_{v_.position_ids};
  std::unique_ptr<CrossCache_Element> cross_cache_;
};

struct Plugin_Element : JSON::Element {
  explicit Plugin_Element(Config::Pipeline::Plugin& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "library") {
      v_.library = JSON::Get<std::string_view>(value);
    } else if (name == "entry_point") {
      v_.entry_point = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Pipeline::Plugin& v_;
};

// v2.1 (issue #2114 §8): "controller" declares the controller-plugin escape hatch (bucket C). It
// names the plugin library + entry-point symbol and carries an opaque, controller-defined config
// string passed through verbatim. Block-presence gated; an absent block leaves all paths unchanged.
struct Controller_Element : JSON::Element {
  explicit Controller_Element(Config::Pipeline::Controller& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "library") {
      v_.library = JSON::Get<std::string_view>(value);
    } else if (name == "entry_point") {
      v_.entry_point = JSON::Get<std::string_view>(value);
    } else if (name == "config") {
      v_.config = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Pipeline::Controller& v_;
};

// v2.1 (issue #2114): "roles" maps a logical role to a session name, e.g.
// {"target": "target_session", "draft": "draft_session"}.
struct Roles_Element : JSON::Element {
  explicit Roles_Element(Config::Pipeline::Roles& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    auto s = std::string(JSON::Get<std::string_view>(value));
    if (name == "target") {
      v_.target = s;
    } else if (name == "draft") {
      v_.draft = s;
    } else if (name == "amateur") {
      v_.amateur = s;
    } else if (name == "expert") {
      v_.expert = s;
    } else if (name == "unconditional") {
      v_.unconditional = s;
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Pipeline::Roles& v_;
};

struct Ngram_Element : JSON::Element {
  explicit Ngram_Element(Config::Pipeline::Speculative::Draft::Ngram& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "min_match") {
      v_.min_match = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "max_draft") {
      v_.max_draft = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "window") {
      v_.window = static_cast<int>(JSON::Get<double>(value));
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Pipeline::Speculative::Draft::Ngram& v_;
};

struct SpeculativeDraft_Element : JSON::Element {
  explicit SpeculativeDraft_Element(Config::Pipeline::Speculative::Draft& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "producer") {
      v_.producer = JSON::Get<std::string_view>(value);
    } else if (name == "session") {
      v_.session = std::string(JSON::Get<std::string_view>(value));
    } else if (name == "depth") {
      v_.depth = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "heads") {
      v_.heads = std::string(JSON::Get<std::string_view>(value));
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnObject(std::string_view name) override {
    if (name == "ngram") {
      v_.ngram = Config::Pipeline::Speculative::Draft::Ngram{};
      ngram_ = std::make_unique<Ngram_Element>(*v_.ngram);
      return *ngram_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Pipeline::Speculative::Draft& v_;
  std::unique_ptr<Ngram_Element> ngram_;
};

struct SpeculativeVerify_Element : JSON::Element {
  explicit SpeculativeVerify_Element(Config::Pipeline::Speculative::Verify& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "session") {
      v_.session = JSON::Get<std::string_view>(value);
    } else {
      throw JSON::unknown_value_error{};
    }
  }

 private:
  Config::Pipeline::Speculative::Verify& v_;
};

struct SpeculativeTree_Element : JSON::Element {
  explicit SpeculativeTree_Element(Config::Pipeline::Speculative::Tree& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "topology") {
      v_.topology = JSON::Get<std::string_view>(value);
    } else if (name == "max_nodes") {
      v_.max_nodes = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "max_depth") {
      v_.max_depth = static_cast<int>(JSON::Get<double>(value));
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  // medusa_choices is an array-of-int-arrays describing the static token tree (each inner array is a
  // root-to-node path of child indices). PR-C parses it into the struct; the executor degrades the
  // tree to its best linear chain (linear-K fallback) because the in-tree decoder graph cannot
  // express a tree-attention mask (see docs/pipeline-config-v2.1-design.md §11.2 and the PR-C note).
  Element& OnArray(std::string_view name) override {
    if (name == "medusa_choices") return medusa_choices_;
    throw JSON::unknown_value_error{};
  }

 private:
  // Parses the outer array of medusa_choices: each element is itself an int array (a tree path).
  struct MedusaChoices_Element : JSON::Element {
    explicit MedusaChoices_Element(std::vector<std::vector<int>>& v) : v_{v} {}
    Element& OnArray(std::string_view /*name*/) override {
      v_.emplace_back();
      inner_ = std::make_unique<IntArray_Element>(v_.back());
      return *inner_;
    }
    std::vector<std::vector<int>>& v_;
    std::unique_ptr<IntArray_Element> inner_;
  };

  Config::Pipeline::Speculative::Tree& v_;
  MedusaChoices_Element medusa_choices_{v_.medusa_choices};
};

// v2.1 (issue #2114): the `speculative` flow strategy block.
struct Strategy_Element : JSON::Element {
  explicit Strategy_Element(Config::Pipeline::Speculative& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "kind") {
      v_.kind = JSON::Get<std::string_view>(value);
    } else if (name == "num_speculative_tokens") {
      v_.num_speculative_tokens = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "acceptance") {
      v_.acceptance = JSON::Get<std::string_view>(value);
    } else if (name == "typical_threshold") {
      v_.typical_threshold = static_cast<float>(JSON::Get<double>(value));
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnObject(std::string_view name) override {
    if (name == "draft") return draft_;
    if (name == "verify") return verify_;
    if (name == "tree") {
      v_.tree = Config::Pipeline::Speculative::Tree{};
      tree_ = std::make_unique<SpeculativeTree_Element>(*v_.tree);
      return *tree_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Pipeline::Speculative& v_;
  SpeculativeDraft_Element draft_{v_.draft};
  SpeculativeVerify_Element verify_{v_.verify};
  std::unique_ptr<SpeculativeTree_Element> tree_;
};

struct PipelineConfig_Element : JSON::Element {
  explicit PipelineConfig_Element(Config::Pipeline& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "extends") {
      v_.extends = std::string(JSON::Get<std::string_view>(value));
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnObject(std::string_view name) override {
    if (name == "sessions") return sessions_;
    if (name == "state") return state_;
    if (name == "roles") {
      v_.roles = Config::Pipeline::Roles{};
      roles_ = std::make_unique<Roles_Element>(*v_.roles);
      return *roles_;
    }
    if (name == "strategy") {
      v_.strategy = Config::Pipeline::Speculative{};
      strategy_ = std::make_unique<Strategy_Element>(*v_.strategy);
      return *strategy_;
    }
    if (name == "plugin") {
      v_.plugin = Config::Pipeline::Plugin{};
      plugin_ = std::make_unique<Plugin_Element>(*v_.plugin);
      return *plugin_;
    }
    if (name == "controller") {
      v_.controller = Config::Pipeline::Controller{};
      controller_ = std::make_unique<Controller_Element>(*v_.controller);
      return *controller_;
    }
    if (name == "preprocessing") return ignore_;  // Consumed by preprocessor in a later PR.
    throw JSON::unknown_value_error{};
  }

  Element& OnArray(std::string_view name) override {
    if (name == "flow") return flow_;
    if (name == "dataflow") return dataflow_;
    throw JSON::unknown_value_error{};
  }

  void OnComplete(bool /*empty*/) override {
    v_.present = true;
  }

 private:
  Config::Pipeline& v_;
  PipelineSessions_Element sessions_{v_.sessions};
  Flow_Element flow_{v_.flow};
  Dataflow_Element dataflow_{v_.dataflow};
  PipelineState_Element state_{v_.state};
  std::unique_ptr<Roles_Element> roles_;
  std::unique_ptr<Strategy_Element> strategy_;
  std::unique_ptr<Plugin_Element> plugin_;
  std::unique_ptr<Controller_Element> controller_;
  Ignore_Element ignore_;
};

// v2 top-level "tokens" section (issue #2114 §4). Lowers directly into the legacy model.* token ids
// so the rest of the runtime keeps reading config.model.*.
struct Tokens_Element : JSON::Element {
  explicit Tokens_Element(Config::Model& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "eos") {  // Scalar form; the array form is handled by OnArray.
      v_.eos_token_id.assign(1, static_cast<int>(JSON::Get<double>(value)));
    } else if (name == "pad") {
      v_.pad_token_id = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "bos") {
      v_.bos_token_id = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "decoder_start") {
      v_.decoder_start_token_id = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "sep") {
      v_.sep_token_id = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "image_token") {
      v_.image_token_id = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "audio_token") {
      v_.audio_token_id = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "video_token") {
      v_.video_token_id = static_cast<int>(JSON::Get<double>(value));
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnArray(std::string_view name) override {
    if (name == "eos") {
      v_.eos_token_id.clear();
      return eos_token_id_;
    }
    throw JSON::unknown_value_error{};
  }

  Config::Model& v_;
  Int_Array_Element eos_token_id_{v_.eos_token_id};
};

// v2.1 (issue #2114 §6): "logit_bias" map { "<token_id>": <delta>, ... } inside a chain entry.
struct LogitsBias_Element : JSON::Element {
  explicit LogitsBias_Element(std::vector<std::pair<int32_t, float>>& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    v_.emplace_back(static_cast<int32_t>(std::stol(std::string(name))),
                    static_cast<float>(JSON::Get<double>(value)));
  }

 private:
  std::vector<std::pair<int32_t, float>>& v_;
};

// v2.1 (issue #2114 §6): a single typed entry of the ordered logit-processor / sampler chain.
struct LogitsProcessor_Element : JSON::Element {
  explicit LogitsProcessor_Element(Config::Search::LogitsProcessor& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "op") {
      v_.op = JSON::Get<std::string_view>(value);
    } else if (name == "value") {
      v_.value = static_cast<float>(JSON::Get<double>(value));
      v_.int_value = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "backend") {
      v_.backend = std::string(JSON::Get<std::string_view>(value));
    } else if (name == "mode") {
      v_.mode = std::string(JSON::Get<std::string_view>(value));
    } else if (name == "alpha") {
      v_.alpha = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "expert") {
      v_.expert = std::string(JSON::Get<std::string_view>(value));
    } else if (name == "amateur") {
      v_.amateur = std::string(JSON::Get<std::string_view>(value));
    } else if (name == "grammar" || name == "stateful") {
      // Grammar payload keys are consumed by the guidance backend, not stored on the spec here.
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnObject(std::string_view name) override {
    if (name == "map") return bias_;
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Search::LogitsProcessor& v_;
  LogitsBias_Element bias_{v_.bias};
};

// v2.1 (issue #2114 §6): "generation.logits" is an ordered array of typed processor entries.
struct LogitsProcessors_Element : JSON::Element {
  explicit LogitsProcessors_Element(std::vector<Config::Search::LogitsProcessor>& v) : v_{v} {}

  Element& OnObject(std::string_view /*name*/) override {
    auto& entry = v_.emplace_back();
    elements_.emplace_back(entry);
    return elements_.back();
  }

 private:
  std::vector<Config::Search::LogitsProcessor>& v_;
  std::vector<LogitsProcessor_Element> elements_;
};

// v2 "generation.sampling" sub-section -> search.* sampling parameters.
struct Sampling_Element : JSON::Element {
  explicit Sampling_Element(Config::Search& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "temperature") {
      v_.temperature = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "top_k") {
      v_.top_k = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "top_p") {
      v_.top_p = static_cast<float>(JSON::Get<double>(value));
    } else if (name == "do_sample") {
      v_.do_sample = JSON::Get<bool>(value);
    } else if (name == "repetition_penalty") {
      v_.repetition_penalty = static_cast<float>(JSON::Get<double>(value));
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Config::Search& v_;
};

// v2 top-level "generation" section -> search.*.
struct Generation_Element : JSON::Element {
  explicit Generation_Element(Config::Search& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "max_length") {
      v_.max_length = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "min_length") {
      v_.min_length = static_cast<int>(JSON::Get<double>(value));
    } else if (name == "num_beams") {
      v_.num_beams = static_cast<int>(JSON::Get<double>(value));
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnObject(std::string_view name) override {
    if (name == "sampling") return sampling_;
    if (name == "stop") return ignore_;  // Stop-sequence handling is owned by a later PR.
    throw JSON::unknown_value_error{};
  }

  Element& OnArray(std::string_view name) override {
    if (name == "logits") return logits_;  // v2.1 §6: ordered logit-processor / sampler chain.
    throw JSON::unknown_value_error{};
  }

  Config::Search& v_;
  Sampling_Element sampling_{v_};
  LogitsProcessors_Element logits_{v_.logits_processors};
  Ignore_Element ignore_;
};

// v2 top-level "metadata" section. Human-facing; only model_type is consumed (and only to seed the
// legacy model.type used by the CreatePipeline fallback dispatch). All other keys are ignored.
struct Metadata_Element : JSON::Element {
  explicit Metadata_Element(Config::Model& v) : v_{v} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "model_type" && v_.type.empty()) {
      v_.type = JSON::Get<std::string_view>(value);
    }
    // source, export_version, and any other metadata keys are human-only; ignore silently.
  }

  Config::Model& v_;
};

struct Root_Element : JSON::Element {
  explicit Root_Element(Config& config) : config_{config} {}

  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "version") {
      config_.version = static_cast<int>(JSON::Get<double>(value));
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnObject(std::string_view name) override {
    if (name == "model") return model_element_;
    if (name == "search") return search_element_;
    if (name == "engine") return engine_element_;
    if (name == "pipeline") return pipeline_element_;
    if (name == "tokens") return tokens_element_;
    if (name == "generation") return generation_element_;
    if (name == "metadata") return metadata_element_;
    throw JSON::unknown_value_error{};
  }

  Config& config_;
  Model_Element model_element_{config_.model};
  Search_Element search_element_{config_.search};
  Engine_Element engine_element_{config_.engine};
  PipelineConfig_Element pipeline_element_{config_.pipeline};
  Tokens_Element tokens_element_{config_.model};
  Generation_Element generation_element_{config_.search};
  Metadata_Element metadata_element_{config_.model};
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

void ResolvePipelineExtends(Config::Pipeline& pipeline) {
  if (!pipeline.extends.has_value()) {
    return;
  }
  const Config::Pipeline* preset = GetPipelinePreset(*pipeline.extends);
  if (!preset) {
    throw std::runtime_error("Unknown pipeline preset in 'extends': " + *pipeline.extends);
  }
  // Override semantics (documented in pipeline_presets.h): explicit top-level arrays in the config
  // replace the preset's arrays wholesale; omitted arrays inherit the preset's. Sub-objects of
  // `state` are inherited only when the config left them untouched.
  if (pipeline.flow.empty()) {
    pipeline.flow = preset->flow;
  }
  if (pipeline.dataflow.empty()) {
    pipeline.dataflow = preset->dataflow;
  }
  if (!pipeline.state.cross_cache.has_value() && preset->state.cross_cache.has_value()) {
    pipeline.state.cross_cache = preset->state.cross_cache;
  }
}

namespace {

// Best-effort v2 lowering: map pipeline session files onto the legacy config.model.* filenames so
// existing consumers keep working. Only fills fields the config left empty (never clobbers an
// explicit model block).
void LowerPipelineToModel(Config& config) {
  for (const auto& session : config.pipeline.sessions) {
    if (session.file.empty()) {
      continue;
    }
    if (session.name == "decoder" && config.model.decoder.filename.empty()) {
      config.model.decoder.filename = session.file;
    } else if (session.name == "encoder" && config.model.encoder.filename.empty()) {
      config.model.encoder.filename = session.file;
    } else if (session.name == "vision" && config.model.vision.filename.empty()) {
      config.model.vision.filename = session.file;
    } else if (session.name == "speech" && config.model.speech.filename.empty()) {
      config.model.speech.filename = session.file;
    } else if (session.name == "embedding" && config.model.embedding.filename.empty()) {
      config.model.embedding.filename = session.file;
    }
  }

  // Pure-v2 configs (issue #2114 §4.1) omit the legacy `model` block, so model.context_length is
  // never set directly. The v2 schema only carries generation.max_length (lowered into
  // search.max_length). Derive a context_length from it so Config::Config's validation passes and the
  // KV cache / generation loop have a bound to work with. Never clobber an explicit value.
  if (config.model.context_length == 0 && config.search.max_length > 0) {
    config.model.context_length = config.search.max_length;
  }
}

}  // namespace

void TranslateV1ToPipeline(Config& config) {
  auto& pipeline = config.pipeline;
  const auto& model = config.model;
  const std::string& type = model.type;

  const bool is_qwen_vl = ModelType::IsQwenVLFamily(type);
  const bool is_pixtral = ModelType::IsPixtralFamily(type);
  const bool is_pipe = ModelType::IsPipe(type) || !model.decoder.pipeline.empty();
  const bool is_vlm = ModelType::IsVLM(type) || ModelType::IsMMM(type);
  const bool is_encoder_decoder = ModelType::IsALM(type) || type == "marian-ssru";

  auto add_session = [&](const std::string& name, const std::string& file) -> bool {
    if (file.empty()) {
      return false;
    }
    Config::Pipeline::Session session;
    session.name = name;
    session.file = file;
    pipeline.sessions.push_back(std::move(session));
    return true;
  };

  if (is_pipe) {
    // Passthrough: the multi-session pipeline is already enumerated by decoder.pipeline[].
    pipeline.extends = "autoregressive-decoder";
    for (const auto& stage : model.decoder.pipeline) {
      const std::string name = !stage.model_id.empty() ? stage.model_id : stage.filename;
      add_session(name, stage.filename);
      Config::Pipeline::FlowStep step;
      step.run = name;
      // A stage that only runs during prompt processing maps to "init"; otherwise it is part of
      // the per-token "step" loop.
      step.when = (stage.run_on_prompt && !stage.run_on_token_gen) ? "init" : "step";
      pipeline.flow.push_back(std::move(step));
    }
  } else if (is_encoder_decoder) {
    pipeline.extends = "encoder-decoder";
    const bool has_encoder = add_session("encoder", model.encoder.filename);
    add_session("decoder", model.decoder.filename);
    if (has_encoder) {
      Config::Pipeline::FlowStep encoder_step;
      encoder_step.run = "encoder";
      encoder_step.when = "init";
      pipeline.flow.push_back(std::move(encoder_step));
    }
    Config::Pipeline::FlowStep decoder_step;
    decoder_step.run = "decoder";
    decoder_step.when = "step";
    if (has_encoder) {
      decoder_step.cross_attention_from = "encoder";
      Config::Pipeline::State::CrossCache cross_cache;
      cross_cache.source = "encoder";
      cross_cache.frozen = true;
      pipeline.state.cross_cache = cross_cache;
      pipeline.dataflow.push_back({"encoder." + model.encoder.outputs.hidden_states,
                                   "decoder." + model.decoder.inputs.encoder_hidden_states});
    }
    pipeline.flow.push_back(std::move(decoder_step));
  } else if (is_vlm) {
    const bool has_vision = add_session("vision", model.vision.filename);
    const bool has_speech = add_session("speech", model.speech.filename);
    const bool has_embedding = add_session("embedding", model.embedding.filename);
    add_session("decoder", model.decoder.filename);
    pipeline.extends = has_vision ? "vision-language" : (has_speech ? "speech-language" : "vision-language");

    const std::string vision_loop = (is_qwen_vl || is_pixtral) ? "per_image" : "batched";
    if (has_vision) {
      Config::Pipeline::FlowStep step;
      step.run = "vision";
      step.when = "init";
      step.loop = vision_loop;
      // Pixtral runs a per-image loop with variable image resolution; this structural flag lets
      // CreateVisionState() pick PixtralVisionState vs QwenVisionState without consulting model.type.
      step.variable_resolution = is_pixtral;
      pipeline.flow.push_back(std::move(step));
    }
    if (has_speech) {
      Config::Pipeline::FlowStep step;
      step.run = "speech";
      step.when = "init";
      pipeline.flow.push_back(std::move(step));
    }
    if (has_embedding) {
      Config::Pipeline::FlowStep step;
      step.run = "embedding";
      step.when = "init";
      pipeline.flow.push_back(std::move(step));
    }
    Config::Pipeline::FlowStep decoder_step;
    decoder_step.run = "decoder";
    decoder_step.when = "step";
    pipeline.flow.push_back(std::move(decoder_step));

    if (has_vision && has_embedding) {
      pipeline.dataflow.push_back({"vision." + model.vision.outputs.image_features,
                                   "embedding." + model.embedding.inputs.image_features});
    }
    if (has_speech && has_embedding) {
      pipeline.dataflow.push_back({"speech." + model.speech.outputs.audio_features,
                                   "embedding." + model.embedding.inputs.audio_features});
    }
    if (has_embedding) {
      pipeline.dataflow.push_back({"embedding." + model.embedding.outputs.embeddings,
                                   "decoder." + model.decoder.inputs.embeddings});
    }
  } else {
    // Default: a plain autoregressive decoder (IsLLM / IsLFM2 / gpt2 / etc.).
    pipeline.extends = "autoregressive-decoder";
    add_session("decoder", model.decoder.filename);
    Config::Pipeline::FlowStep step;
    step.run = "decoder";
    step.when = "step";
    pipeline.flow.push_back(std::move(step));
  }

  // KV cache format: gpt2 uses a single combined past/present tensor per layer; everything else uses
  // the standard separate key/value tensors.
  auto& kv_cache = pipeline.state.kv_cache;
  if (type == "gpt2") {
    kv_cache.format = "combined";
    kv_cache.past_key_pattern = model.decoder.inputs.past_names;
    kv_cache.present_key_pattern = model.decoder.outputs.present_names;
  } else {
    kv_cache.format = "separate";
    kv_cache.past_key_pattern = model.decoder.inputs.past_key_names;
    kv_cache.past_value_pattern = model.decoder.inputs.past_value_names;
    kv_cache.present_key_pattern = model.decoder.outputs.present_key_names;
    kv_cache.present_value_pattern = model.decoder.outputs.present_value_names;
  }

  // Position-id strategy mirrors CreatePositionInputs(): 3D mRoPE for the Qwen-VL family, windowed
  // when a sliding window is configured, otherwise standard 1D position ids.
  auto& position_ids = pipeline.state.position_ids;
  position_ids.input_name = model.decoder.inputs.position_ids;
  if (is_qwen_vl) {
    position_ids.strategy = "mrope_3d";
  } else if (model.decoder.sliding_window.has_value()) {
    position_ids.strategy = "windowed";
  } else {
    position_ids.strategy = "default";
  }

  pipeline.present = true;
}

Config::Config(const fs::path& path, std::string_view json_overlay) : config_path{path} {
  ParseConfig(path / "genai_config.json", json_overlay, *this);

  // Pipeline-as-Config (issue #2114): produce a normalized, introspectable Config::Pipeline.
  // For v2 inputs we resolve `extends` and lower session files + top-level tokens/generation/metadata
  // back into config.model.* / config.search.* so existing consumers keep working AND so a pure v2
  // config (no legacy `model` block) populates the fields validated just below. For v1 inputs we only
  // DERIVE the pipeline view (translator) and leave config.model.* exactly as parsed -- the safest
  // backward-compatible path (no behavior change). This runs BEFORE validation so pure-v2 lowering
  // (e.g. context_length from generation.max_length) takes effect in time.
  if (version >= 2 && pipeline.present) {
    ResolvePipelineExtends(pipeline);
    LowerPipelineToModel(*this);
  } else if (version < 2 && !pipeline.present) {
    TranslateV1ToPipeline(*this);
  }

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

  // v2.1 (issue #2114, PR-F, design §7): validate the runtime-vs-build-time feature namespace on every
  // session_options block. No-op (and therefore byte-for-byte back-compatible) unless a config declares
  // a "runtime"/"build_requires" object; otherwise it fails fast on an unknown or mis-namespaced value.
  ValidateSessionOptionsFeatures(model.decoder.session_options, "decoder");
  if (model.encoder.session_options.has_value())
    ValidateSessionOptionsFeatures(*model.encoder.session_options, "encoder");
  if (model.vision.session_options.has_value())
    ValidateSessionOptionsFeatures(*model.vision.session_options, "vision");
  if (model.speech.session_options.has_value())
    ValidateSessionOptionsFeatures(*model.speech.session_options, "speech");
  if (model.embedding.session_options.has_value())
    ValidateSessionOptionsFeatures(*model.embedding.session_options, "embedding");
  if (model.joiner.session_options.has_value())
    ValidateSessionOptionsFeatures(*model.joiner.session_options, "joiner");
  if (model.vad.session_options.has_value())
    ValidateSessionOptionsFeatures(*model.vad.session_options, "vad");
  for (const auto& stage : model.decoder.pipeline) {
    if (stage.session_options.has_value())
      ValidateSessionOptionsFeatures(*stage.session_options, "pipeline model '" + stage.model_id + "'");
  }
  for (const auto& stage : model.vision.pipeline) {
    if (stage.session_options.has_value())
      ValidateSessionOptionsFeatures(*stage.session_options, "vision pipeline model");
  }
  for (const auto& session : pipeline.sessions) {
    if (session.session_options.has_value())
      ValidateSessionOptionsFeatures(*session.session_options, "pipeline session '" + session.name + "'");
  }
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
