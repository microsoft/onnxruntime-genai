// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
#include "generators.h"
#include "models/model_package.h"
#include "models/model_type.h"
#include "runtime_settings.h"
#include "json.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

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
  } else if (lower_name == "nvtensorrtrtx") {
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
    } else if (name == "targets") {
      v_.targets = JSON::Get<std::string_view>(value);
    } else if (name == "lstm_hidden_state") {
      v_.lstm_hidden_state = JSON::Get<std::string_view>(value);
    } else if (name == "lstm_cell_state") {
      v_.lstm_cell_state = JSON::Get<std::string_view>(value);
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
    } else if (name == "outputs") {
      v_.outputs = JSON::Get<std::string_view>(value);
    } else if (name == "lstm_hidden_state") {
      v_.lstm_hidden_state = JSON::Get<std::string_view>(value);
    } else if (name == "lstm_cell_state") {
      v_.lstm_cell_state = JSON::Get<std::string_view>(value);
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
    } else if (name == "component") {
      v_.component = JSON::Get<std::string_view>(value);
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
    } else if (name == "component") {
      v_.component = JSON::Get<std::string_view>(value);
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
    } else if (name == "component") {
      v_.component = JSON::Get<std::string_view>(value);
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
    } else if (name == "component") {
      v_.component = JSON::Get<std::string_view>(value);
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
    } else if (name == "component") {
      v_.component = JSON::Get<std::string_view>(value);
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
    } else {
      throw JSON::unknown_value_error{};
    }
  }

  Element& OnArray(std::string_view name) override {
    if (name == "eos_token_id")
      return eos_token_id_;
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
        // Graph Capture is currently broken for CUDA
        for (const auto& value : provider_options->options) {
          if (value.first == "enable_cuda_graph" && value.second == "1") {
            throw std::runtime_error("Graph Capture is currently unsupported for CUDA");
          }
        }
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
    if (name == "engine") return engine_element_;
    throw JSON::unknown_value_error{};
  }

  Config& config_;
  Model_Element model_element_{config_.model};
  Search_Element search_element_{config_.search};
  Engine_Element engine_element_{config_.engine};
};

struct RootObject_Element : JSON::Element {
  explicit RootObject_Element(JSON::Element& t) : t_{t} {}

  Element& OnObject(std::string_view /*name*/) override {
    return t_;
  }

  JSON::Element& t_;
};

namespace {

// Parse a JSON config text (and optional overlay) into Config via the
// streaming Element interface. Same diagnostic semantics as ParseConfig
// (file-based) but with a caller-supplied source label for error context.
void ParseConfigFromText(std::string_view config_text,
                         std::string_view source_label,
                         std::string_view json_overlay,
                         Config& config) {
  Root_Element root{config};
  RootObject_Element root_object{root};
  try {
    JSON::Parse(root_object, config_text);
  } catch (const std::exception& message) {
    std::ostringstream oss;
    oss << "Error encountered while parsing " << source_label << ": " << message.what();
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

}  // namespace

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

  std::ostringstream label;
  label << "'" << filename.string() << "'";
  ParseConfigFromText(std::string_view(buffer.data(), buffer.size()),
                      label.str(), json_overlay, config);
}

void OverlayConfig(Config& config, std::string_view json) {
  Root_Element root{config};
  RootObject_Element element{root};
  JSON::Parse(element, json);
}

const Config::SessionOptions& EffectiveSessionOptions(
    const Config& config,
    const std::optional<Config::SessionOptions>& component_session_options) {
  return component_session_options.has_value() ? component_session_options.value()
                                               : config.model.decoder.session_options;
}

namespace {

// Read an entire file as binary into a string.
std::string ReadFileBinary(const fs::path& filename) {
  std::ifstream file = filename.open(std::ios::binary | std::ios::ate);
  if (!file.is_open())
    throw std::runtime_error("Error opening " + filename.string());
  std::streamsize const size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::string buffer(static_cast<size_t>(size), '\0');
  if (size > 0 && !file.read(buffer.data(), size))
    throw std::runtime_error("Error reading " + filename.string());
  return buffer;
}

// EP defaulting algorithm (spec Appendix A): intersect the per-component
// `EpsCompatibleWith()` sets and require exactly one survivor.
//
// The intersection order follows the FIRST component's first-seen order, so
// diagnostic output (and any future tie-break) is deterministic.
//
// `user_ep` is the public-API `ep` argument. When non-empty, defaulting
// is bypassed and the user's EP becomes the sole captured priority entry —
// per-component `SelectComponent` will then either find a matching variant
// or throw a clear "no variant for component X under EP Y" diagnostic. We
// intentionally do NOT pre-validate `user_ep` against the global intersection;
// the per-component error surfaces naturally and is more actionable.
//
// Returns a `ModelPackageSelectionOptions` carrying the resolved EP as the
// sole priority entry. Throws on empty intersection (no EP can load every
// component) or, in the no-user-ep case, multi-EP intersection (ambiguous —
// the diagnostic now points at the public `ep` argument as the resolution
// channel).
ModelPackageSelectionOptions ComputeEpDefaulting(const ModelPackageContext& ctx,
                                                 std::string_view user_ep) {
  const std::size_t n = ctx.NumComponents();
  if (n == 0)
    throw std::runtime_error("v4 model package: no components declared");

  if (!user_ep.empty()) {
    ModelPackageSelectionOptions options;
    options.ep_priority.push_back(EpSelection{std::string(user_ep), std::nullopt});
    return options;
  }

  // Start with the first component's EP list (preserve order).
  std::vector<std::string> intersection = ctx.EpsCompatibleWith(0);

  for (std::size_t cix = 1; cix < n; ++cix) {
    auto eps = ctx.EpsCompatibleWith(cix);
    std::unordered_set<std::string> as_set(eps.begin(), eps.end());
    std::vector<std::string> next;
    next.reserve(intersection.size());
    for (const auto& ep : intersection) {
      if (as_set.count(ep) != 0) next.push_back(ep);
    }
    intersection = std::move(next);
    if (intersection.empty()) break;
  }

  if (intersection.empty()) {
    std::ostringstream oss;
    oss << "v4 model package: no execution provider is supported by every component (";
    for (std::size_t cix = 0; cix < n; ++cix) {
      if (cix > 0) oss << "; ";
      oss << ctx.ComponentName(cix) << "=[";
      auto eps = ctx.EpsCompatibleWith(cix);
      for (std::size_t i = 0; i < eps.size(); ++i) {
        if (i > 0) oss << ",";
        oss << eps[i];
      }
      oss << "]";
    }
    oss << "). Re-package or pick per-component EPs (not yet supported).";
    throw std::runtime_error(oss.str());
  }

  if (intersection.size() > 1) {
    std::ostringstream oss;
    oss << "v4 model package: multiple execution providers are compatible with every component (";
    for (std::size_t i = 0; i < intersection.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << intersection[i];
    }
    oss << "). Pass the `ep` argument to og.Model / og.Config (or"
        << " OgaCreateModelWithEp / OgaCreateConfigWithEp in C) to choose one.";
    throw std::runtime_error(oss.str());
  }

  ModelPackageSelectionOptions options;
  options.ep_priority.push_back(EpSelection{intersection[0], std::nullopt});
  return options;
}

// Walk every `model.<role>.component` field on the merged Config and assert
// that any non-empty value names a component that the package actually
// produced. Throw with the offending role/value pair if not. This catches
// producer typos at Config-load time rather than at session-build time.
void ValidateRoleComponentReferences(const Config& config) {
  struct Ref {
    std::string_view role;
    const std::string& component;
  };
  const Ref refs[] = {
      {"encoder", config.model.encoder.component},
      {"decoder", config.model.decoder.component},
      {"vision", config.model.vision.component},
      {"speech", config.model.speech.component},
      {"embedding", config.model.embedding.component},
  };
  for (const auto& r : refs) {
    if (r.component.empty()) continue;
    if (config.component_instances.count(r.component) != 0) continue;
    std::ostringstream oss;
    oss << "v4 model package: model." << r.role << ".component references unknown component '"
        << r.component << "'. Known components: [";
    bool first = true;
    for (const auto& kv : config.component_instances) {
      if (!first) oss << ", ";
      oss << kv.first;
      first = false;
    }
    oss << "].";
    throw std::runtime_error(oss.str());
  }
}

// Load Config from a v4 model package.
//
// 1. Set shared_assets_path to <pkg>/configs/.
// 2. EP defaulting -> ModelPackageSelectionOptions (single-EP for now).
// 3. Read <pkg>/configs/genai_config.json as the base DOM.
// 4. For each component (in manifest order): SelectComponent, pull
//    consumer_metadata, extract `genai_config_overlay`, RFC-7386 merge into
//    the base. Stash the ComponentInstance for later role lookup.
// 5. Serialize the merged DOM and run it through the same strict streaming
//    parser the flat-dir path uses, applying the caller json_overlay last
//    (layer-2 channel: OgaConfigOverlay / RuntimeSettings::GenerateConfigOverlay).
//
// Caveat (intentional): once the merged DOM is round-tripped through
// SerializeDocument/Parse, the strict-parser unknown-key error no longer
// points at the overlaying component. The diagnostic wraps the parse error
// with "merged package genai_config" context so the user knows where to look.
void LoadFromPackage(Config& config,
                     std::shared_ptr<ModelPackageContext> ctx,
                     std::string_view json_overlay,
                     std::string_view user_ep) {
  const fs::path shared = ctx->SharedAssetsPath();
  config.shared_assets_path = shared;

  // EP defaulting first — needed before SelectComponent. When `user_ep`
  // is non-empty, defaulting is bypassed and the user's choice wins.
  ModelPackageSelectionOptions options = ComputeEpDefaulting(*ctx, user_ep);
  const std::string& selected_ep = options.ep_priority.front().ep_name;

  // Load base genai_config.json.
  const fs::path base_path = shared / std::string("genai_config.json");
  if (!base_path.exists()) {
    throw std::runtime_error("v4 model package missing configs/genai_config.json at " +
                             base_path.string());
  }
  std::string base_text = ReadFileBinary(base_path);
  JSON::Document merged;
  try {
    merged = JSON::ParseDocument(base_text);
  } catch (const std::exception& e) {
    throw std::runtime_error("v4 model package: failed to parse " + base_path.string() + ": " +
                             e.what());
  }
  if (!merged.IsObject()) {
    throw std::runtime_error("v4 model package: " + base_path.string() +
                             " must be a JSON object at the top level");
  }

  // Per-component overlay merge (manifest declaration order).
  std::unordered_map<std::string, std::shared_ptr<ComponentInstance>> instances;
  const std::size_t n = ctx->NumComponents();
  for (std::size_t cix = 0; cix < n; ++cix) {
    const std::string cname = ctx->ComponentName(cix);
    std::unique_ptr<ComponentInstance> cinst_unique;
    try {
      cinst_unique = ctx->SelectComponent(cix, options);
    } catch (const std::exception& e) {
      std::ostringstream oss;
      oss << "v4 model package: ORT model-package selection failed for component '" << cname
          << "' with execution provider '" << selected_ep << "': " << e.what();
      auto eps = ctx->EpsCompatibleWith(cix);
      if (!eps.empty()) {
        oss << " Compatible EPs for this component: [";
        for (std::size_t i = 0; i < eps.size(); ++i) {
          if (i > 0) oss << ", ";
          oss << eps[i];
        }
        oss << "].";
      }
      throw std::runtime_error(oss.str());
    }
    if (!cinst_unique) {
      // SelectComponent returns null when no variant of the component
      // matches the captured EP. Under defaulting this is "defensive"
      // (we picked an EP from the intersection so every component has a
      // match). Under user-supplied `ep`, this is the natural failure
      // path — list the component's compatible EPs so the user can fix
      // the typo / re-package / pick a different ep.
      std::ostringstream oss;
      oss << "v4 model package: no variant of component '" << cname
          << "' matches execution provider '" << selected_ep << "'.";
      auto eps = ctx->EpsCompatibleWith(cix);
      if (!eps.empty()) {
        oss << " Compatible EPs for this component: [";
        for (std::size_t i = 0; i < eps.size(); ++i) {
          if (i > 0) oss << ", ";
          oss << eps[i];
        }
        oss << "].";
      }
      throw std::runtime_error(oss.str());
    }
    std::shared_ptr<ComponentInstance> cinst(std::move(cinst_unique));
    const std::string blob = cinst->ConsumerMetadata();
    if (!blob.empty()) {
      JSON::Document cm;
      try {
        cm = JSON::ParseDocument(blob);
      } catch (const std::exception& e) {
        throw std::runtime_error("v4 model package: component '" + cname +
                                 "' consumer_metadata is not valid JSON: " + e.what());
      }
      if (cm.IsObject()) {
        const auto& obj = cm.AsObject();
        auto it = obj.find("genai_config_overlay");
        if (it != obj.end()) {
          if (!it->second.IsObject()) {
            throw std::runtime_error("v4 model package: component '" + cname +
                                     "' consumer_metadata.genai_config_overlay must be a JSON object");
          }
          JSON::MergePatch(merged, it->second);
        }
      }
    }
    instances.emplace(cname, std::move(cinst));
  }

  // Round-trip merged DOM through the strict streaming parser, applying the
  // caller overlay last. Wrap parse errors with a clear "merged package config"
  // label so authors know where to look.
  std::string merged_text = JSON::SerializeDocument(merged);
  ParseConfigFromText(merged_text, "merged v4 package genai_config", json_overlay, config);

  config.component_instances = std::move(instances);
  config.model_package = std::move(ctx);

  ValidateRoleComponentReferences(config);
}

}  // namespace

Config::Config(const fs::path& path, std::string_view json_overlay)
    : Config(path, json_overlay, std::string_view{}) {}

Config::Config(const fs::path& path, std::string_view json_overlay, std::string_view user_ep)
    : config_path{path}, shared_assets_path{path} {
  if (auto ctx_unique = ModelPackageContext::Open(path)) {
    LoadFromPackage(*this, std::shared_ptr<ModelPackageContext>(std::move(ctx_unique)),
                    json_overlay, user_ep);
  } else {
    if (!user_ep.empty()) {
      throw std::runtime_error(
          "The 'ep' argument is only supported for v4 model packages. "
          "For flat-directory models, set providers via "
          "OgaConfigClearProviders / OgaConfigAppendProvider on the "
          "genai_config.json provider list.");
    }
    ParseConfig(path / "genai_config.json", json_overlay, *this);
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
