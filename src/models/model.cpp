// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
#include <algorithm>
#include <climits>
#include <random>
#include <set>
#include <string>
#include <thread>

#include "../generators.h"
#include "../search.h"
#include "../tracing.h"
#include "model.h"
#include "gpt.h"
#include "decoder_only.h"
#include "whisper.h"
#include "multi_modal.h"
#include "marian.h"
#include "decoder_only_pipeline.h"
#include "../dml/interface.h"

#if defined(_WIN32)
#include <direct.h>
#define GETCWD _getcwd
#define CHDIR _wchdir
#include <windows.h>
#else
#include <unistd.h>
#define GETCWD getcwd
#define CHDIR chdir
#include <limits.h>
#endif

namespace Generators {

namespace {

class DirGuard {
 private:
  fs::path original_dir_;

 public:
  DirGuard() {
    char buffer[PATH_MAX];
    if (GETCWD(buffer, sizeof(buffer))) {
      original_dir_ = fs::path(buffer);
    } else {
      throw std::runtime_error("Failed to get current working directory");
    }
  }

  DirGuard(const DirGuard&) = delete;
  DirGuard& operator=(const DirGuard&) = delete;
  DirGuard(DirGuard&&) = delete;

  void ChangeTo(const fs::path& new_dir) {
    if (CHDIR(new_dir.c_str()) != 0) {
      throw std::runtime_error("Failed to change directory to: " + new_dir.string());
    }
  }

  ~DirGuard() {
    if (CHDIR(original_dir_.c_str()) != 0) {
      Log("warning", "Failed to change back to original directory: " + original_dir_.string());
    }
  }
};

}  // namespace

State::State(const GeneratorParams& params, const Model& model)
    : model_{model},
      params_{params.shared_from_this()},
      run_options_{OrtRunOptions::Create()},
      extra_outputs_{*this} {
  // Generate a random id for graph capture
  if (params_->use_graph_capture) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, INT_MAX);
    graph_id_ = std::to_string(dis(gen));
  }
}

void State::Run(OrtSession& session, bool graph_capture_this_run) {
  DurationTrace trace{"State::Run"};

  if (params_->use_graph_capture) {
    if (graph_capture_this_run)
      run_options_->AddConfigEntry("gpu_graph_id", graph_id_.c_str());
    else
      run_options_->AddConfigEntry("gpu_graph_id", "-1");
  }

  if (first_run_) {
    extra_outputs_.Add(session.GetOutputNames());
    if (params_->use_multi_profile) {
      // Run the context phase profile for the first run
      run_options_->AddConfigEntry("nv_profile_index", "0");
    }
    first_run_ = false;
  } else {
    extra_outputs_.Update();
    if (params_->use_multi_profile) {
      run_options_->AddConfigEntry("nv_profile_index", "1");
    }
  }

  if (g_log.enabled && g_log.model_input_values) {
    auto& stream = Log("model_input_values");
    stream << std::endl;
    DumpTensors(model_, stream, inputs_.data(), input_names_.data(), input_names_.size(), true);
  }

  if (g_log.enabled && g_log.model_output_shapes) {
    auto& stream = Log("model_output_shapes");
    stream << std::endl;
    DumpTensors(model_, stream, outputs_.data(), output_names_.data(), output_names_.size(), false);
  }

  if (!ep_dynamic_options_next_run_.empty()) {
    std::vector<const char*> keys;
    std::vector<const char*> values;
    for (auto& kv_pair : ep_dynamic_options_next_run_) {
      keys.push_back(kv_pair.first.c_str());
      values.push_back(kv_pair.second.c_str());
    }
    session.SetEpDynamicOptions(keys.data(), values.data(), ep_dynamic_options_next_run_.size());
    ep_dynamic_options_next_run_.clear();
  }

  session.Run(run_options_.get(), input_names_.data(), inputs_.data(), input_names_.size(),
              output_names_.data(), outputs_.data(), output_names_.size());

  extra_outputs_.RegisterOutputs();

  if (g_log.enabled && g_log.model_output_values) {
    auto& stream = Log("model_output_values");
    stream << std::endl;
    DumpTensors(model_, stream, outputs_.data(), output_names_.data(), output_names_.size(), true);
  }
}

void State::SetTerminate() {
  session_terminated_ = true;
  run_options_->SetTerminate();
}

void State::UnsetTerminate() {
  session_terminated_ = false;
  run_options_->UnsetTerminate();
}

OrtValue* State::GetInput(const char* name) {
  ThrowErrorIfSessionTerminated(session_terminated_);
  for (size_t i = 0; i < input_names_.size(); i++) {
    if (std::strcmp(input_names_[i], name) == 0) {
      return inputs_[i];
    }
  }
  return nullptr;
}

OrtValue* State::GetOutput(const char* name) {
  ThrowErrorIfSessionTerminated(session_terminated_);
  for (size_t i = 0; i < output_names_.size(); i++) {
    if (std::strcmp(output_names_[i], name) == 0) {
      return outputs_[i];
    }
  }
  return nullptr;
}

void State::ClearIO() {
  input_names_.clear();
  output_names_.clear();
  inputs_.clear();
  outputs_.clear();
}

void State::SetActiveAdapter(Adapters* adapters, const std::string& adapter_name) {
  if (!adapters_) {
    adapters_ = adapters->shared_from_this();
  } else if (adapters_.get() != adapters) {
    // Two different instances of Adapters are being used. The Generator state can only manage
    // active adapters from a single Adapters container.
    throw std::runtime_error("Generator state can only register a single Adapters container.");
  }

  run_options_->AddActiveLoraAdapter(*adapters_->AcquireAdapter(adapter_name));
  adapter_names_.push_back(adapter_name);
}

State::~State() {
  if (adapters_) {
    for (const auto& adapter_name : adapter_names_) {
      adapters_->ReleaseAdapter(adapter_name);
    }
  }
}

std::vector<int32_t> PadInputs(std::span<std::span<const int32_t>> sequences, int32_t pad_token_id) {
  bool pad_right_{true};

  size_t max_length = 0;
  for (auto& sequence : sequences)
    max_length = std::max(max_length, sequence.size());

  std::vector<int32_t> result(max_length * sequences.size());
  std::span<int32_t> result_span(result);

  // Copy and pad the sequences with pad_token_id
  for (size_t i = 0; i < sequences.size(); i++) {
    auto output_span = result_span.subspan(i * max_length, max_length);
    auto input_span = sequences[i];

    auto pad_count = max_length - input_span.size();
    if (pad_right_) {
      std::copy(input_span.begin(), input_span.end(), output_span.begin());
      std::fill(output_span.end() - pad_count, output_span.end(), pad_token_id);
    } else {
      std::fill(output_span.begin(), output_span.begin() + pad_count, pad_token_id);
      std::copy(input_span.begin(), input_span.end(), output_span.begin() + pad_count);
    }
  }

  return result;
}

void CheckResult(extError_t error) {
  if (error != kOrtxOK)
    throw std::runtime_error(OrtxGetLastErrorMessage());
}

TokenizerStream::TokenizerStream(const Tokenizer& tokenizer)
    : tokenizer_{tokenizer.shared_from_this()} {
  CheckResult(OrtxCreate(kOrtxKindDetokenizerCache, cache_.Address()));
}

const std::string& TokenizerStream::Decode(int32_t token) {
  const char* string;
  CheckResult(OrtxDetokenizeCached(tokenizer_->tokenizer_, cache_, token, &string));
  chunk_ = string;
  return chunk_;
}

Tokenizer::Tokenizer(Config& config) : pad_token_id_{config.model.pad_token_id} {
  CheckResult(OrtxCreateTokenizer(tokenizer_.Address(), config.config_path.string().c_str()));
}

std::unique_ptr<TokenizerStream> Tokenizer::CreateStream() const {
  return std::make_unique<TokenizerStream>(*this);
}

std::vector<int32_t> Tokenizer::Encode(const char* text) const {
  OrtxPtr<OrtxTokenId2DArray> ids;
  CheckResult(OrtxTokenizeWithOptions(tokenizer_, &text, 1, ids.Address(), false /* add_special_tokens */));

  const extTokenId_t* tokens;
  size_t count;
  CheckResult(OrtxTokenId2DArrayGetItem(ids, 0, &tokens, &count));
  return {tokens, tokens + count};
}

std::string Tokenizer::Decode(std::span<const int32_t> tokens) const {
  OrtxPtr<OrtxStringArray> ortx_string_array;
  CheckResult(OrtxDetokenize1D(tokenizer_, reinterpret_cast<const uint32_t*>(tokens.data()), tokens.size(), ortx_string_array.Address()));

  const char* string;
  CheckResult(OrtxStringArrayGetItem(ortx_string_array, 0, &string));
  return string;
}

std::string Tokenizer::ApplyChatTemplate(const char* template_str, const char* messages, const char* tools, bool add_generation_prompt) const {
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> templated_text;
  CheckResult(OrtxApplyChatTemplate(tokenizer_, template_str, messages, tools, templated_text.ToBeAssigned(), add_generation_prompt, false /*tokenize*/));

  ort_extensions::OrtxObjectPtr<OrtxTensor> tensor;
  CheckResult(OrtxTensorResultGetAt(templated_text.get(), 0, tensor.ToBeAssigned()));

  const char* text_ptr{};
  CheckResult(OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&text_ptr), nullptr, nullptr));

  return text_ptr;
}

std::vector<int32_t> Tokenizer::EncodeBatch(std::span<const std::string> strings) const {
  std::vector<std::vector<int32_t>> sequences;
  std::vector<std::span<const int32_t>> span_sequences;
  for (size_t i = 0; i < strings.size(); i++) {
    sequences.emplace_back(Encode(strings[i].c_str()));
    span_sequences.emplace_back(sequences.back());
  }

  return PadInputs(span_sequences, pad_token_id_);
}

std::shared_ptr<Tensor> Tokenizer::EncodeBatch(std::span<const char*> strings) const {
  std::vector<std::vector<int32_t>> sequences;
  std::vector<std::span<const int32_t>> span_sequences;
  for (size_t i = 0; i < strings.size(); i++) {
    sequences.emplace_back(Encode(strings[i]));
    span_sequences.emplace_back(sequences.back());
  }

  auto encoded = PadInputs(span_sequences, pad_token_id_);  // TODO: Pad directly into tensor vs copying?

  auto shape = std::array<int64_t, 2>{static_cast<int64_t>(strings.size()), static_cast<int64_t>(encoded.size() / strings.size())};
  auto ort_tensor_ = OrtValue::CreateTensor<int32_t>(Ort::Allocator::GetWithDefaultOptions(), shape);
  auto tensor = std::make_shared<Tensor>(std::move(ort_tensor_));
  std::copy(encoded.begin(), encoded.end(), tensor->GetMutableData<int32_t>());

  return tensor;
}

std::vector<std::string> Tokenizer::DecodeBatch(std::span<const int32_t> sequences, size_t count) const {
  if (sequences.size() % count != 0)
    throw std::runtime_error("DecodeBatch: sequences must be evenly divisible by the count");
  size_t sequence_length = sequences.size() / count;
  std::vector<std::string> strings;
  for (size_t i = 0; i < count; i++)
    strings.emplace_back(Decode(sequences.subspan(sequence_length * i, sequence_length)));
  return strings;
}

int32_t Tokenizer::TokenToTokenId(const char* token) const {
  extTokenId_t token_id;
  CheckResult(OrtxConvertTokenToId(tokenizer_, token, &token_id));
  return token_id;
}

/**
 * @brief Creates profile shapes for NvTensorRtRtx execution provider optimization.
 *
 * This function generates profiles for TensorRT execution provider optimization.
 * If multi-profile is enabled, it creates separate profiles for context and generation phases.
 * If multi-profile is disabled, it creates a single profile with simple shapes.
 *
 */
void ConfigureNvTensorRtRTxProfile(const Config& config, OrtSessionOptions& session_options, bool is_multi_profile_enabled) {
  // Get model parameters from decoder config
  const int num_layers = config.model.decoder.num_hidden_layers;
  const int num_kv_heads = config.model.decoder.num_key_value_heads;
  const int head_dim = config.model.decoder.head_size;
  const int batch_size = config.search.batch_size;

  // Get max context length from config
  const int max_context_len = config.model.context_length;

  // Extract KV cache name patterns from decoder config
  std::string_view past_key_pattern = config.model.decoder.inputs.past_key_names;
  std::string_view past_value_pattern = config.model.decoder.inputs.past_value_names;

  // Helper function to add KV cache with sequence length
  const auto add_key_value_cache_shapes = [](std::ostringstream& shapes,
                                             int batch_size,
                                             std::string_view key_pattern,
                                             std::string_view value_pattern,
                                             int seq_len,
                                             int num_layers,
                                             int num_kv_heads,
                                             int head_dim) {
    for (int i = 0; i < num_layers; i++) {
      // Use the existing function to format the key/value names
      const std::string key_name = ComposeKeyValueName(std::string(key_pattern), i);
      const std::string value_name = ComposeKeyValueName(std::string(value_pattern), i);

      shapes << "," << key_name << ":" << batch_size << "x" << num_kv_heads << "x" << seq_len << "x" << head_dim;
      shapes << "," << value_name << ":" << batch_size << "x" << num_kv_heads << "x" << seq_len << "x" << head_dim;
    }
  };

  if (is_multi_profile_enabled) {
    // Multi-profile mode: existing logic for context and generation phases
    const int opt_context_len = config.model.context_length / 2;
    const int min_seq_len = 1;

    // Helper function to add input shapes (input_ids, attention_mask, position_ids)
    const auto add_input_shapes = [](std::ostringstream& shapes, int batch_size, int seq_len, bool append = false) {
      if (append) shapes << ",";
      shapes << Config::Defaults::InputIdsName << ":" << batch_size << "x" << seq_len << ","
             << Config::Defaults::AttentionMaskName << ":" << batch_size << "x" << seq_len;
    };

    // Helper function to add generation phase input shapes
    const auto add_generation_input_shapes = [](std::ostringstream& shapes, int batch_size, int context_len) {
      shapes << "," << Config::Defaults::AttentionMaskName << ":" << batch_size << "x" << context_len << ","
             << Config::Defaults::InputIdsName << ":" << batch_size << "x1";
    };

    // Helper function to add empty KV cache shapes for all layers
    const auto add_empty_key_value_cache_shapes = [](std::ostringstream& shapes,
                                                     int batch_size,
                                                     std::string_view key_pattern,
                                                     std::string_view value_pattern,
                                                     int num_layers,
                                                     int num_kv_heads,
                                                     int head_dim) {
      for (int i = 0; i < num_layers; i++) {
        // Use the existing function to format the key/value names
        const std::string key_name = ComposeKeyValueName(std::string(key_pattern), i);
        const std::string value_name = ComposeKeyValueName(std::string(value_pattern), i);

        shapes << "," << key_name << ":" << batch_size << "x" << num_kv_heads << "x0x" << head_dim;
        shapes << "," << value_name << ":" << batch_size << "x" << num_kv_heads << "x0x" << head_dim;
      }
    };

    std::ostringstream min_shapes, opt_shapes, max_shapes;

    // MIN SHAPES (context phase and first token generation)
    add_input_shapes(min_shapes, batch_size, min_seq_len);
    add_empty_key_value_cache_shapes(min_shapes, batch_size, past_key_pattern, past_value_pattern, num_layers, num_kv_heads, head_dim);
    add_generation_input_shapes(min_shapes, batch_size, min_seq_len);
    add_key_value_cache_shapes(min_shapes, batch_size, past_key_pattern, past_value_pattern, min_seq_len, num_layers, num_kv_heads, head_dim);

    // OPT SHAPES (prefill with medium context and generation after medium context)
    add_input_shapes(opt_shapes, batch_size, opt_context_len);
    add_empty_key_value_cache_shapes(opt_shapes, batch_size, past_key_pattern, past_value_pattern, num_layers, num_kv_heads, head_dim);
    add_generation_input_shapes(opt_shapes, batch_size, opt_context_len);
    add_key_value_cache_shapes(opt_shapes, batch_size, past_key_pattern, past_value_pattern, opt_context_len - 1, num_layers, num_kv_heads, head_dim);

    // MAX SHAPES (prefill with maximum context and generation after maximum context)
    add_input_shapes(max_shapes, batch_size, max_context_len);
    add_key_value_cache_shapes(max_shapes, batch_size, past_key_pattern, past_value_pattern, max_context_len - 1, num_layers, num_kv_heads, head_dim);
    add_generation_input_shapes(max_shapes, batch_size, max_context_len);
    add_key_value_cache_shapes(max_shapes, batch_size, past_key_pattern, past_value_pattern, max_context_len - 1, num_layers, num_kv_heads, head_dim);

    // Add the constructed profiles to session options
    session_options.AddConfigEntry("ep.nvtensorrtrtxexecutionprovider.nv_profile_min_shapes", min_shapes.str().c_str());
    session_options.AddConfigEntry("ep.nvtensorrtrtxexecutionprovider.nv_profile_opt_shapes", opt_shapes.str().c_str());
    session_options.AddConfigEntry("ep.nvtensorrtrtxexecutionprovider.nv_profile_max_shapes", max_shapes.str().c_str());
  } else {
    // Single profile mode: simple shapes with batch_dim=[0,1,1] and seq_dim=[0,1,max_context_len]
    std::ostringstream min_shapes, opt_shapes, max_shapes;

    // MIN SHAPES: batch_dim=0, seq_dim=0
    min_shapes << Config::Defaults::InputIdsName << ":0x0,"
               << Config::Defaults::AttentionMaskName << ":0x0";
    add_key_value_cache_shapes(min_shapes, 0, past_key_pattern, past_value_pattern, 0, num_layers, num_kv_heads, head_dim);

    // OPT SHAPES: batch_dim=1, seq_dim=1
    opt_shapes << Config::Defaults::InputIdsName << ":1x1,"
               << Config::Defaults::AttentionMaskName << ":1x1";
    add_key_value_cache_shapes(opt_shapes, 1, past_key_pattern, past_value_pattern, 1, num_layers, num_kv_heads, head_dim);

    // MAX SHAPES: batch_dim=1, seq_dim=max_context_len
    max_shapes << Config::Defaults::InputIdsName << ":" << batch_size << "x" << max_context_len << ","
               << Config::Defaults::AttentionMaskName << ":" << batch_size << "x" << max_context_len;
    add_key_value_cache_shapes(max_shapes, batch_size, past_key_pattern, past_value_pattern, max_context_len, num_layers, num_kv_heads, head_dim);

    // Add the constructed profiles to session options
    session_options.AddConfigEntry("ep.nvtensorrtrtxexecutionprovider.nv_profile_min_shapes", min_shapes.str().c_str());
    session_options.AddConfigEntry("ep.nvtensorrtrtxexecutionprovider.nv_profile_opt_shapes", opt_shapes.str().c_str());
    session_options.AddConfigEntry("ep.nvtensorrtrtxexecutionprovider.nv_profile_max_shapes", max_shapes.str().c_str());
  }
}

DeviceInterface* SetProviderSessionOptions(OrtSessionOptions& session_options,
                                           const std::vector<std::string>& providers,
                                           const std::vector<Config::ProviderOptions>& provider_options_list,
                                           bool is_primary_session_options,
                                           bool disable_graph_capture,
                                           const Config& config) {
  DeviceInterface* p_device{};

  auto providers_list = providers;
  if (!is_primary_session_options) {
    // Providers specified in a non-primary provider options list are added
    // to the primary providers. They are considered immutable and implicitly
    // added as providers.
    std::transform(provider_options_list.begin(), provider_options_list.end(), std::back_inserter(providers_list),
                   [](const auto& provider_options) { return provider_options.name; });
  }

  for (auto& provider : providers_list) {
    auto provider_options_it = std::find_if(provider_options_list.begin(), provider_options_list.end(),
                                            [&provider](const Config::ProviderOptions& po) { return po.name == provider; });

    if (provider_options_it == provider_options_list.end()) {
      throw std::runtime_error("Provider options not found for provider: " + provider);
    }
    const auto& provider_options = *provider_options_it;

    if (provider_options.name == "cuda") {
      auto ort_provider_options = OrtCUDAProviderOptionsV2::Create();
      std::vector<const char*> keys, values;
      for (auto& option : provider_options.options) {
        keys.emplace_back(option.first.c_str());
        values.emplace_back(option.second.c_str());
      }
      ort_provider_options->Update(keys.data(), values.data(), keys.size());

      // Device type determines the scoring device.
      // Only use the primary session options to determine the device type
      if (is_primary_session_options) {
        p_device = GetDeviceInterface(DeviceType::CUDA);

        // Create and set our cudaStream_t
        ort_provider_options->UpdateValue("user_compute_stream", p_device->GetCudaStream());
      }

      session_options.AppendExecutionProvider_CUDA_V2(*ort_provider_options);
    } else if (provider_options.name == "rocm") {
      OrtROCMProviderOptions ort_provider_options;

      std::vector<const char*> keys, values;
      for (auto& option : provider_options.options) {
        keys.emplace_back(option.first.c_str());
        values.emplace_back(option.second.c_str());
      }

      Ort::ThrowOnError(Ort::api->UpdateROCMProviderOptions(&ort_provider_options, keys.data(), values.data(), keys.size()));
      session_options.AppendExecutionProvider_ROCM(ort_provider_options);
    } else if (provider_options.name == "DML") {
#if USE_DML
      if (!GetDmlInterface()) {
        LUID device_luid{};
        LUID* p_device_luid{};
        uint32_t device_index{};
        uint32_t* p_device_index{};
        for (const auto& [name, value] : provider_options.options) {
          if (name == "luid") {
            if (auto separator_position = value.find(":"); separator_position != std::string::npos) {
              device_luid.HighPart = std::stol(value.substr(0, separator_position));
              device_luid.LowPart = std::stol(value.substr(separator_position + 1));
              p_device_luid = &device_luid;
            }
          } else if (name == "device_index") {
            device_index = std::stoi(value);
            p_device_index = &device_index;
          }
        }

        InitDmlInterface(p_device_luid, p_device_index);
      }

      if (!disable_graph_capture) {
        session_options.AddConfigEntry("ep.dml.enable_graph_capture", "1");
      }

      SetDmlProvider(session_options);

      if (is_primary_session_options)
        p_device = GetDeviceInterface(DeviceType::DML);  // We use a DML allocator for input/output caches, but other tensors will use CPU tensors
#else
      throw std::runtime_error("DML provider requested, but the installed GenAI has not been built with DML support");
#endif
    } else {
      // For providers that go through the extensible AppendExecutionProvider API:
      if (provider_options.name == "QNN") {
        session_options.AddConfigEntry("ep.share_ep_contexts", "1");
        // TODO set device_type_ in a less hacky way.
        // now, all QNN EP enable_htp_shared_memory_allocator option values had better be consistent...
        // on the other hand, not sure if is_primary_session_options is the right thing to check here.
        if (const auto opt_it = std::find_if(provider_options.options.begin(), provider_options.options.end(),
                                             [](const auto& pair) { return pair.first == "enable_htp_shared_memory_allocator"; });
            opt_it != provider_options.options.end() && opt_it->second == "1") {
          p_device = GetDeviceInterface(DeviceType::QNN);
        }
      } else if (provider_options.name == "WebGPU")
        p_device = GetDeviceInterface(DeviceType::WEBGPU);
      else if (provider_options.name == "OpenVINO")
        p_device = GetDeviceInterface(DeviceType::OpenVINO);
      else if (provider_options.name == "VitisAI") {
        session_options.AddConfigEntry("session.inter_op.allow_spinning", "0");
        session_options.AddConfigEntry("session.intra_op.allow_spinning", "0");
      } else if (provider_options.name == "NvTensorRtRtx") {
        bool is_multi_profile_enabled = IsMultiProfileEnabled(config.model.decoder.session_options);
        ConfigureNvTensorRtRTxProfile(config, session_options, is_multi_profile_enabled);
        if (IsGraphCaptureEnabled(config.model.decoder.session_options)) {
          session_options.AddConfigEntry("ep.nvtensorrtrtxexecutionprovider.nv_cuda_graph_enable", "1");
        }
        p_device = GetDeviceInterface(DeviceType::NvTensorRtRtx);
      }

      std::vector<const char*> keys, values;
      for (auto& option : provider_options.options) {
        keys.emplace_back(option.first.c_str());
        values.emplace_back(option.second.c_str());
      }
      session_options.AppendExecutionProvider(provider_options.name.c_str(), keys.data(), values.data(), keys.size());
    }
  }
  return p_device;
}

// Trivial ONNX model that just returns a single float constant. Used below to create an OrtSession that
// lets us get a device Ort::Allocator for each device type. This is necessary because the Ort::Allocator
// needs to persist and is valid for the lifetime of this OrtSession.
static const uint8_t g_trivial_model[] = {
    0x08, 0x0a, 0x12, 0x01, 0x61, 0x3a, 0x53, 0x0a, 0x38, 0x12, 0x06, 0x76, 0x61, 0x6c, 0x75, 0x65,
    0x73, 0x22, 0x08, 0x43, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x2a, 0x24, 0x0a, 0x05, 0x76,
    0x61, 0x6c, 0x75, 0x65, 0x2a, 0x18, 0x08, 0x01, 0x10, 0x01, 0x42, 0x0c, 0x63, 0x6f, 0x6e, 0x73,
    0x74, 0x5f, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x4a, 0x04, 0x00, 0x00, 0x00, 0x00, 0xa0, 0x01,
    0x04, 0x12, 0x01, 0x62, 0x62, 0x14, 0x0a, 0x06, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x73, 0x12, 0x0a,
    0x0a, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x01, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x15};

// Since Python/Others can and will hold onto a generator object past the model object's lifetime we need to ensure
// the allocator used is not destroyed until last. This keeps the allocator around until exit, after all other memory
// has been destroyed. Without this, we will crash in the Onnxruntime BFCArena code when deleting tensors due to the
// arena already being destroyed.
void EnsureDeviceOrtInit(DeviceInterface& device, const Config& config) {
  // CPU Allocator is a special case, it's not in the owned 'allocator_device_' table below so we handle it separately
  // OpenVINO delegates to the CPU device allocator
  auto type = device.GetType();
  if (type == DeviceType::CPU || type == DeviceType::OpenVINO)
    return;

  auto& allocator = GetOrtGlobals()->device_allocators_[static_cast<int>(type)];
  if (allocator.allocator_)
    return;

  // Allocator lifetime is tied to the execution provider lifetime, which is typically per Session.
  // We create a global dummy Session using a trivial model with the required EP in order to get the allocator so we can
  // re-use it for all models.
  // This ensures memory allocated on-device for model inputs/outputs is valid for the lifetime of GenAI.

  // Names for the device types used by 'SetProviderSessionOptions'
  static const char* device_type_names[] = {"CPU (Not used, see above)", "cuda", "DML", "WebGPU", "QNN", "OpenVINO (Not used, see above)", "NvTensorRtRtx"};
  static_assert(std::size(device_type_names) == static_cast<size_t>(DeviceType::MAX));

  // Create an OrtSessionOptions and set the options to use the DeviceType we're using here
  auto session_options = OrtSessionOptions::Create();
  std::vector<Config::ProviderOptions> provider_options_list;
  provider_options_list.emplace_back(Config::ProviderOptions{device_type_names[static_cast<int>(type)], {}});
  // QnnHtpShared is a special case. This allocator is only made available when the provider option
  // 'enable_htp_shared_memory_allocator' is set to 1.
  if (type == DeviceType::QNN) {
    provider_options_list.back().options.emplace_back("enable_htp_shared_memory_allocator", "1");
  }
  const std::vector<std::string> providers{device_type_names[static_cast<int>(type)]};
  SetProviderSessionOptions(*session_options, providers, provider_options_list, true, false, config);
  session_options->SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);  // Errors only here, as warnings are not useful to the user

  allocator.session_ = OrtSession::Create(GetOrtEnv(), g_trivial_model, sizeof(g_trivial_model), session_options.get());

  // Names for the device memory types used by 'OrtMemoryInfo::Create'
  static const char* device_memory_type_names[] = {"CPU (Not used, see above)", "Cuda", "DML", "WebGPU_Buffer", "QnnHtpShared", "OpenVINO (Not used, see above)", "Cuda"};
  static_assert(std::size(device_memory_type_names) == static_cast<size_t>(DeviceType::MAX));

  // Get the allocator from the OrtSession for the DeviceType (it's called 'AllocatorCreate' but it's really 'AllocatorGet')
  auto name = device_memory_type_names[static_cast<int>(type)];
  auto memory_info = OrtMemoryInfo::Create(name, OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
  allocator.allocator_ = Ort::Allocator::Create(*allocator.session_, *memory_info);
  if (!allocator.allocator_) {
    allocator = {};  // Reset everything just to be safe
    throw std::runtime_error("Unexpected failure to create device memory allocator for " + std::string(name));
  }
  device.InitOrt(*Ort::api, *allocator.allocator_);
}

void SessionInfo::Add(OrtSession& session) {
  auto input_names = session.GetInputNames();
  for (size_t i = 0; i < input_names.size(); i++) {
    auto type_info = session.GetInputTypeInfo(i);
    auto input_type = type_info->GetTensorTypeAndShapeInfo().GetElementType();
    auto found_input = inputs_.find(input_names[i]);
    if (found_input != inputs_.end() && found_input->second->GetTensorTypeAndShapeInfo().GetElementType() != input_type)
      throw std::runtime_error("Model input type mismatch: " + input_names[i] + " expected " +
                               std::to_string(found_input->second->GetTensorTypeAndShapeInfo().GetElementType()) +
                               " got " + std::to_string(input_type));
    inputs_.emplace(std::make_pair(std::move(input_names[i]), std::move(type_info)));
  }

  auto output_names = session.GetOutputNames();
  for (size_t i = 0; i < output_names.size(); i++) {
    auto type_info = session.GetOutputTypeInfo(i);
    outputs_.emplace(std::make_pair(std::move(output_names[i]), std::move(type_info)));
  }
}

bool SessionInfo::HasInput(const std::string& name) const {
  return inputs_.find(name) != inputs_.end();
}

bool SessionInfo::HasOutput(const std::string& name) const {
  return outputs_.find(name) != outputs_.end();
}

ONNXTensorElementDataType SessionInfo::GetInputDataType(const std::string& name) const {
  auto result = inputs_.find(name);
  if (result == inputs_.end())
    throw std::runtime_error("Model input was not found: " + name);
  return result->second->GetTensorTypeAndShapeInfo().GetElementType();
}

ONNXTensorElementDataType SessionInfo::GetOutputDataType(const std::string& name) const {
  auto result = outputs_.find(name);
  if (result == outputs_.end())
    throw std::runtime_error("Model output was not found: " + name);
  return result->second->GetTensorTypeAndShapeInfo().GetElementType();
}

std::vector<std::string> SessionInfo::GetInputNames() const {
  std::vector<std::string> names;
  names.reserve(inputs_.size());
  for (const auto& input : inputs_)
    names.push_back(input.first);
  return names;
}

std::vector<const char*> SessionInfo::GetInputSymbolicShape(const std::string& name) const {
  auto type_info = inputs_.find(name);
  if (type_info == inputs_.end())
    throw std::runtime_error("Model input was not found: " + name);
  return type_info->second->GetTensorTypeAndShapeInfo().GetSymbolicDimensions();
}

std::vector<const char*> SessionInfo::GetOutputSymbolicShape(const std::string& name) const {
  auto type_info = outputs_.find(name);
  if (type_info == outputs_.end())
    throw std::runtime_error("Model output was not found: " + name);
  return type_info->second->GetTensorTypeAndShapeInfo().GetSymbolicDimensions();
}

Model::Model(std::unique_ptr<Config> config) : config_{std::move(config)} {
  CreateSessionOptions();
  EnsureDeviceOrtInit(*p_device_, *config_);

  // Only CUDA, TRT-RTX and DML does every input on the device
  if (p_device_->GetType() == DeviceType::CUDA || p_device_->GetType() == DeviceType::DML || p_device_->GetType() == DeviceType::NvTensorRtRtx)
    p_device_inputs_ = p_device_;
  else
    p_device_inputs_ = GetDeviceInterface(DeviceType::CPU);

  // The kvcache is always allocated in device memory
  p_device_kvcache_ = p_device_;
}

Model::~Model() {
#if USE_DML
  if (p_device_->GetType() == DeviceType::DML) {
    auto& allocator = GetOrtGlobals()->device_allocators_[static_cast<int>(DeviceType::DML)];
    allocator.session_.reset();
    allocator.allocator_.reset();
    session_options_.reset();
    // DML objects are globally scoped and launch background threads that retain hardware resources.
    // These threads persist beyond the lifetime of a Model, preventing proper cleanup and potentially causing deadlocks.
    // To avoid blocking driver threads, we explicitly destroy DML objects when the Model is destroyed.
    // They will be recreated as needed when a new Model is initialized.
    CloseDmlInterface();
  }
#endif
}

void Model::CreateSessionOptionsFromConfig(const Config::SessionOptions& config_session_options,
                                           OrtSessionOptions& session_options,
                                           bool is_primary_session_options,
                                           bool disable_graph_capture) {
  // Default to a limit of 16 threads to optimize performance
  constexpr int min_thread_nums = 1;
  constexpr int max_thread_nums = 16;
  int num_of_cores = std::max(min_thread_nums, static_cast<int>(std::thread::hardware_concurrency() / 2));
  session_options.SetIntraOpNumThreads(std::min(num_of_cores, max_thread_nums));

  if (config_session_options.intra_op_num_threads.has_value()) {
    session_options.SetIntraOpNumThreads(config_session_options.intra_op_num_threads.value());
  }

  if (config_session_options.inter_op_num_threads.has_value()) {
    session_options.SetInterOpNumThreads(config_session_options.inter_op_num_threads.value());
  }

  if (config_session_options.enable_cpu_mem_arena.has_value()) {
    if (config_session_options.enable_cpu_mem_arena.value())
      session_options.EnableCpuMemArena();
    else
      session_options.DisableCpuMemArena();
  }

  if (config_session_options.enable_mem_pattern.has_value()) {
    if (config_session_options.enable_mem_pattern.value())
      session_options.EnableMemPattern();
    else
      session_options.DisableMemPattern();
  }

  if (config_session_options.log_id.has_value()) {
    session_options.SetLogId(config_session_options.log_id.value().c_str());
  }

  if (config_session_options.log_severity_level.has_value()) {
    session_options.SetLogSeverityLevel(config_session_options.log_severity_level.value());
  }

  if (config_session_options.enable_profiling.has_value()) {
    fs::path profile_file_prefix{config_session_options.enable_profiling.value()};
    session_options.EnableProfiling(profile_file_prefix.c_str());
  }

  if (config_session_options.disable_cpu_ep_fallback.has_value()) {
    if (config_session_options.disable_cpu_ep_fallback.value())
      session_options.DisableCpuEpFallback();
    else
      session_options.EnableCpuEpFallback();
  }

  if (config_session_options.disable_quant_qdq.has_value()) {
    if (config_session_options.disable_quant_qdq.value())
      session_options.DisableQuantQdq();
    else
      session_options.EnableQuantQdq();
  }

  if (config_session_options.enable_quant_qdq_cleanup.has_value()) {
    if (config_session_options.enable_quant_qdq_cleanup.value())
      session_options.EnableQuantQdqCleanup();
    else
      session_options.DisableQuantQdqCleanup();
  }

  if (config_session_options.ep_context_enable.has_value()) {
    if (config_session_options.ep_context_enable.value())
      session_options.SetEpContextEnable();
  }

  if (config_session_options.ep_context_embed_mode.has_value()) {
    session_options.SetEpContextEmbedMode(config_session_options.ep_context_embed_mode.value().c_str());
  }

  if (config_session_options.ep_context_file_path.has_value()) {
    session_options.SetEpContextFilePath(config_session_options.ep_context_file_path.value().c_str());
  }

  if (config_session_options.provider_options.empty() && config_session_options.use_env_allocators) {
    // Share env allocators across sessions that only use the CPU provider
    session_options.AddConfigEntry("session.use_env_allocators", "1");
  }

  for (auto& config_entry : config_session_options.config_entries) {
    session_options.AddConfigEntry(config_entry.first.c_str(), config_entry.second.c_str());
  }

  if (config_session_options.custom_ops_library.has_value()) {
    fs::path custom_library_file_prefix{config_session_options.custom_ops_library.value()};
    session_options.RegisterCustomOpsLibrary(custom_library_file_prefix.c_str());
  }

  if (config_session_options.graph_optimization_level.has_value()) {
    session_options.SetGraphOptimizationLevel(config_session_options.graph_optimization_level.value());
  }

  auto session_device = SetProviderSessionOptions(session_options, config_session_options.providers,
                                                  config_session_options.provider_options, is_primary_session_options,
                                                  disable_graph_capture, *config_);

  if (!p_device_) {
    p_device_ = session_device;
  } else if (session_device != nullptr && session_device->GetType() != p_device_->GetType()) {
    throw std::runtime_error("Running a model with multiple providers is not supported. Encountered " +
                             to_string(session_device->GetType()) + " and " + to_string(p_device_->GetType()));
  }
}

void Model::CreateSessionOptions() {
  session_options_ = OrtSessionOptions::Create();

  CreateSessionOptionsFromConfig(config_->model.decoder.session_options, *session_options_, true, false);

  for (auto& pipeline_model : config_->model.decoder.pipeline) {
    if (pipeline_model.session_options.has_value()) {
      auto emplaced = pipeline_session_options_.emplace(pipeline_model.model_id, OrtSessionOptions::Create());
      CreateSessionOptionsFromConfig(*pipeline_model.session_options, *emplaced.first->second, false, false);
    }
  }

  // Fallback to CPU if no provider specific interface was set
  if (!p_device_)
    p_device_ = GetDeviceInterface(DeviceType::CPU);
}

OrtSessionOptions* Model::GetSessionOptions(const std::string& model_id) const {
  auto session_options = pipeline_session_options_.find(model_id);
  // Use the pipeline model session options id config defined it.
  if (session_options != pipeline_session_options_.end())
    return session_options->second.get();

  // Else fallback to the main session options.
  return session_options_.get();
}

std::unique_ptr<OrtSession> Model::CreateSession(OrtEnv& ort_env, const std::string& model_filename, OrtSessionOptions* session_options) {
  if (auto model_data_it = config_->model_data_spans_.find(model_filename);
      model_data_it != config_->model_data_spans_.end()) {
    // If model data was provided, load the model from memory
    if (model_data_it->second.empty()) {
      throw std::runtime_error("Failed to load model data from memory for " + model_filename);
    }
    // TODO (baijumeswani): Loading ONNX models from memory that hold references to data stored in external files
    // is not supported at the moment. This limitation stems from the fact that ONNX models typically
    // reference these external files using relative paths to the model file. When loading a model from memory,
    // the relative paths may not resolve correctly, leading to issues in locating the referenced data.
    // To work around this, we change the current working directory to the model's config path
    // before creating the session. This allows the model to resolve relative paths correctly.
    // Note that this is not a problem for models that do not reference external files.
    // This is a temporary solution and can be potentially addressed by exposing means to set a working directory
    // for the OrtSession through the ONNX Runtime API.
    // This solution is not ideal since it modifies the global state of the process, and is hence not thread-safe.
    DirGuard dir_guard;
    dir_guard.ChangeTo(config_->config_path);
    auto session = OrtSession::Create(ort_env, model_data_it->second.data(), model_data_it->second.size(), session_options);

    return session;
  }

  // Otherwise, load the model from the file system
  return OrtSession::Create(ort_env, (config_->config_path / fs::path(model_filename)).c_str(), session_options);
}

std::shared_ptr<Tokenizer> Model::CreateTokenizer() const {
  return std::make_shared<Tokenizer>(*config_);
}

std::shared_ptr<MultiModalProcessor> Model::CreateMultiModalProcessor() const {
  return std::make_shared<MultiModalProcessor>(*config_, session_info_);
}

std::shared_ptr<Model> CreateModel(OrtEnv& ort_env, const char* config_path, const RuntimeSettings* settings /*= nullptr*/) {
  std::string config_overlay;
  if (settings) {
    config_overlay = settings->GenerateConfigOverlay();
  }
  auto config = std::make_unique<Config>(fs::path(config_path), config_overlay);
  return CreateModel(ort_env, std::move(config));
}

std::shared_ptr<Model> CreateModel(OrtEnv& ort_env, std::unique_ptr<Config> config) {
  if (config->model.type == "gpt2")
    return std::make_shared<Gpt_Model>(std::move(config), ort_env);
  if (ModelType::IsLLM(config->model.type))
    return std::make_shared<DecoderOnly_Model>(std::move(config), ort_env);
  if (ModelType::IsALM(config->model.type))
    return std::make_shared<WhisperModel>(std::move(config), ort_env);
  if (ModelType::IsVLM(config->model.type))
    return std::make_shared<MultiModalLanguageModel>(std::move(config), ort_env, true, false);
  if (ModelType::IsPipe(config->model.type))
    return std::make_shared<DecoderOnlyPipelineModel>(std::move(config), ort_env);
  if (ModelType::IsMMM(config->model.type))
    return std::make_shared<MultiModalLanguageModel>(std::move(config), ort_env, true, true);
  if (config->model.type == "marian-ssru")
    return std::make_shared<MarianModel>(std::move(config), ort_env);

  throw std::runtime_error("Unsupported model_type in config.json: " + config->model.type);
}

std::shared_ptr<GeneratorParams> CreateGeneratorParams(const Model& model) {
  return std::make_shared<GeneratorParams>(model);
}

// Used by benchmarking tests only, should not be used normally
std::shared_ptr<GeneratorParams> CreateGeneratorParams(const Config& config) {
  return std::make_shared<GeneratorParams>(config);
}

void Cast(OrtValue& input, std::unique_ptr<OrtValue>& output, DeviceInterface& device, ONNXTensorElementDataType output_type) {
  auto input_info = input.GetTensorTypeAndShapeInfo();
  auto shape = input_info->GetShape();

  if (output && shape != output->GetTensorTypeAndShapeInfo()->GetShape())
    output = nullptr;
  if (!output)
    output = OrtValue::CreateTensor(device.GetAllocator(), shape, output_type);

  auto input_type = input_info->GetElementType();
  auto element_count = input_info->GetElementCount();

  if (element_count != output->GetTensorTypeAndShapeInfo()->GetElementCount())
    throw std::runtime_error("Cast: input and output element count mismatch");

  void* input_data = input.GetTensorMutableRawData();
  void* output_data = output->GetTensorMutableRawData();
  if (!device.Cast(input_data, output_data, input_type, output_type, element_count)) {
    auto input_span = ByteWrapTensor(device, input);
    auto output_span = ByteWrapTensor(device, *output);
    input_data = input_span.CopyDeviceToCpu().data();
    output_data = output_span.CopyDeviceToCpu().data();
    GetDeviceInterface(DeviceType::CPU)->Cast(input_data, output_data, input_type, output_type, element_count);
    output_span.CopyCpuToDevice();
  }
}

std::unique_ptr<OrtValue> Model::ExpandInputs(std::unique_ptr<OrtValue>& input, int num_beams) const {
  // Input shape (batch_size, sequence_length). The input is required with data type T.
  // Output shape (batch_size * num_beams, sequence_length)

  // When num_beams == 1, we don't need to expand the input, but the expand has a side effect of copying from
  // CPU memory to device memory, so we can skip if the p_device_inputs_ is the CPU device
  if (num_beams == 1 && p_device_inputs_ == GetDeviceInterface(DeviceType::CPU))
    return std::move(input);

  auto input_type_info = input->GetTensorTypeAndShapeInfo();
  auto element_type = input_type_info->GetElementType();
  auto input_shape = input_type_info->GetShape();
  const int64_t batch_size = input_shape[0];
  const int64_t data_size_bytes = input_type_info->GetElementCount() * Ort::SizeOf(element_type) / batch_size;

  input_shape[0] *= num_beams;

  auto input_span = ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *input);
  auto expanded = OrtValue::CreateTensor(p_device_inputs_->GetAllocator(), input_shape, element_type);
  auto expanded_span = ByteWrapTensor(*p_device_inputs_, *expanded);

  // Detect fast & simple copy case
  if (num_beams == 1) {
    expanded_span.CopyFrom(input_span);
  } else {
    // TODO (RyanHill): To avoid cuda uninitialized memory warnings, we should copy input_span to device memory first
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < num_beams; j++) {
        expanded_span.subspan((i * num_beams + j) * data_size_bytes, data_size_bytes).CopyFrom(input_span.subspan(i * data_size_bytes, data_size_bytes));
      }
    }
  }
  return expanded;
}

MultiModalProcessor::MultiModalProcessor(Config& config, const SessionInfo& session_info)
    : tokenizer_{std::make_shared<Tokenizer>(config)},
      processor_factory_{
          {"phi3v", Processor::Create<PhiImageProcessor>},
          {"whisper", Processor::Create<WhisperProcessor>},
          {"phi4mm", Processor::Create<PhiMultiModalProcessor>},
          {"gemma3", Processor::Create<GemmaImageProcessor>}} {
  auto processor = processor_factory_.find(config.model.type);
  if (processor != processor_factory_.end()) {
    processor_ = processor->second(config, session_info);
  } else {
    throw std::runtime_error("MultiModalProcessor cannot be created. " + config.model.type + " is not a registered multi-modal model type.");
  }
}

std::unique_ptr<NamedTensors> MultiModalProcessor::Process(const std::string& prompt, const Images* images, const Audios* audios) const {
  Payload payload{prompt, {}, images, audios};
  return processor_->Process(*tokenizer_, payload);
}

std::unique_ptr<NamedTensors> MultiModalProcessor::Process(std::span<const char*> prompts, const Images* images, const Audios* audios) const {
  Payload payload{"", prompts, images, audios};
  return processor_->Process(*tokenizer_, payload);
}

}  // namespace Generators
