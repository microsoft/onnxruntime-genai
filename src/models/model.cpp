// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "nemotron_speech.h"
#include "multi_modal.h"
#include "marian.h"
#include "decoder_only_pipeline.h"
#include "qwen_vl_model.h"
#include "qwen2_5_vl_image_processor.h"
#include "mistral3_image_processor.h"
#include "../dml/interface.h"
#include "../openvino/interface.h"
#include "../ryzenai/interface.h"
#include "session_options.h"

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
    if (CHDIR(original_dir_.c_str()) != 0 && g_log.enabled) {
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

void State::DumpInputs() {
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
}

void State::DumpOutputs() {
  if (g_log.enabled && g_log.model_output_values) {
    auto& stream = Log("model_output_values");
    stream << std::endl;
    DumpTensors(model_, stream, outputs_.data(), output_names_.data(), output_names_.size(), true);
  }
}

void State::Run(OrtSession& session, bool graph_capture_this_run) {
  DurationTrace trace{"State::Run"};

  if (params_->use_graph_capture) {
    if (graph_capture_this_run) {
      run_options_->AddConfigEntry("gpu_graph_id", graph_id_.c_str());
    } else {
      run_options_->AddConfigEntry("gpu_graph_id", "-1");
    }
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

  DumpInputs();

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

  if (model_.p_device_ && model_.p_device_->GetType() == DeviceType::NvTensorRtRtx) {
    run_options_->AddConfigEntry("disable_synchronize_execution_providers", "1");
  }

  std::unique_ptr<OrtValue> new_input_ids;

  if (prompt_gen_) {
    for (int i = 0; i < inputs_.size(); i++) {
      std::string input_name = input_names_[i];

      if (input_name == "input_ids") {
        int64_t batch_size = params_->search.batch_size;
        int64_t padded_seq_len = static_cast<int64_t>(params_->search.max_length);
        std::vector<int64_t> new_shape{batch_size, padded_seq_len};

        OrtValue* value = inputs_[i];
        auto info = value->GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType elem_type = info->GetElementType();
        std::vector<int64_t> dims = info->GetShape();
        int64_t orig_seq_len = std::min(dims[1], padded_seq_len);

        new_input_ids = OrtValue::CreateTensor(model_.allocator_cpu_, new_shape, elem_type);

        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          auto* src = value->GetTensorMutableData<int64_t>();
          auto* dst = new_input_ids->GetTensorMutableData<int64_t>();
          std::fill(dst, dst + batch_size * padded_seq_len, int64_t{0});
          for (int64_t b = 0; b < batch_size; ++b) {
            std::copy(src + b * orig_seq_len, src + b * orig_seq_len + orig_seq_len,
                      dst + b * padded_seq_len);
          }
        } else {
          auto* src = value->GetTensorMutableData<int32_t>();
          auto* dst = new_input_ids->GetTensorMutableData<int32_t>();
          std::fill(dst, dst + batch_size * padded_seq_len, int32_t{0});
          for (int64_t b = 0; b < batch_size; ++b) {
            std::copy(src + b * orig_seq_len, src + b * orig_seq_len + orig_seq_len,
                      dst + b * padded_seq_len);
          }
        }
        inputs_[i] = new_input_ids.get();
        break;
      }
    }
  }
  session.Run(run_options_.get(), input_names_.data(), inputs_.data(), input_names_.size(),
              output_names_.data(), outputs_.data(), output_names_.size());

  new_input_ids.reset();

  extra_outputs_.RegisterOutputs();

  DumpOutputs();
}

void State::SetRunOption(const char* key, const char* value) {
  if (strcmp(key, "terminate_session") == 0) {
    if (strcmp(value, "0") == 0) {
      session_terminated_ = false;
      run_options_->UnsetTerminate();
    } else if (strcmp(value, "1") == 0) {
      session_terminated_ = true;
      run_options_->SetTerminate();
    } else {
      // Value not expected
      throw std::runtime_error(std::string("terminate_session key value unexpected: ") + value);
    }
    return;
  } else if (strcmp(key, "enable_profiling") == 0) {
#if ORT_API_VERSION >= 25
    if (strcmp(value, "0") == 0) {
      run_options_->DisableProfiling();
    } else {
      // Enable run-level profiling. The profiling output file is named: <prefix>_<timestamp>.json
      // Value "1" uses the default prefix; any other value is treated as a custom prefix.
      constexpr const char* default_profile_prefix = "onnxruntime_run_profile";
      const char* prefix = (strcmp(value, "1") == 0) ? default_profile_prefix : value;
      run_options_->EnableProfiling(fs::path(prefix).c_str());
    }
#else
    throw std::runtime_error("enable_profiling requires ONNX Runtime 1.25 or later");
#endif
    return;
  }
  run_options_->AddConfigEntry(key, value);
}

/*
 * Set all run options that are key-value pairs of strings.
 * Reference: https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h
 */
void State::SetRunOptions(const Config::RunOptions& config_run_options) {
  for (auto& config_entry : config_run_options) {
    SetRunOption(config_entry.first.c_str(), config_entry.second.c_str());
  }
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

Tokenizer::Tokenizer(Config& config) : bos_token_id_{config.model.bos_token_id},
                                       eos_token_id_{config.model.eos_token_id},
                                       pad_token_id_{config.model.pad_token_id} {
  // Default tokenizer options
  const char* keys[] = {"add_special_tokens", "skip_special_tokens"};
  const char* values[] = {"false", "true"};

  CheckResult(OrtxCreateTokenizerWithOptions(tokenizer_.Address(), config.config_path.string().c_str(), keys, values, 2));
}

std::unique_ptr<TokenizerStream> Tokenizer::CreateStream() const {
  return std::make_unique<TokenizerStream>(*this);
}

void Tokenizer::UpdateOptions(const char* const* keys, const char* const* values, size_t num_options) {
  // Tap into ORT Extensions API
  CheckResult(OrtxUpdateTokenizerOptions(tokenizer_, const_cast<const char**>(keys), const_cast<const char**>(values), num_options));
}

std::vector<int32_t> Tokenizer::Encode(const char* text) const {
  OrtxPtr<OrtxTokenId2DArray> ids;
  CheckResult(OrtxTokenize(tokenizer_, &text, 1, ids.Address()));

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
// has been destroyed. Without this, we will crash in the OnnxRuntime BFCArena code when deleting tensors due to the
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
  static const char* device_type_names[] = {"CPU (Not used, see above)", "cuda", "DML", "WebGPU", "QNN", "OpenVINO (Not used, see above)", "NvTensorRtRtx", "RyzenAI"};
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
  SetProviderSessionOptions(*session_options, providers, provider_options_list, true, config);
  session_options->SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);  // Errors only here, as warnings are not useful to the user

  allocator.session_ = OrtSession::Create(GetOrtEnv(), g_trivial_model, sizeof(g_trivial_model), session_options.get());

  // Names for the device memory types used by 'OrtMemoryInfo::Create'
  static const char* device_memory_type_names[] = {"CPU (Not used, see above)", "Cuda", "DML", "WebGPU_Buf", "QnnHtpShared", "OpenVINO (Not used, see above)", "Cuda", "Cpu"};
  static_assert(std::size(device_memory_type_names) == static_cast<size_t>(DeviceType::MAX));

  // Get the allocator from the OrtSession for the DeviceType (it's called 'AllocatorCreate' but it's really 'AllocatorGet')
  auto name = device_memory_type_names[static_cast<int>(type)];
  try {
    auto memory_info = OrtMemoryInfo::Create(name, OrtAllocatorType::OrtDeviceAllocator,
                                             0, OrtMemType::OrtMemTypeDefault);
    allocator.allocator_ = Ort::Allocator::Create(*allocator.session_, *memory_info);
  } catch (const Ort::Exception& e) {
    // WebGPU memory type name changed from "WebGPU_Buffer" to "WebGPU_Buf" in ORT 1.24.3.
    // Try the old name before giving up.
    if (type == DeviceType::WEBGPU) {
      auto fallback_info = OrtMemoryInfo::Create("WebGPU_Buffer", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
      try {
        allocator.allocator_ = Ort::Allocator::Create(*allocator.session_, *fallback_info);
      } catch (const Ort::Exception& fallback_e) {
        throw std::runtime_error(
            "Failed to create allocator for WebGPU. "
            "Primary name '" +
            std::string(name) + "' error: " + std::string(e.what()) +
            "; fallback 'WebGPU_Buffer' error: " + std::string(fallback_e.what()));
      }
    } else {
      throw std::runtime_error("Failed to create allocator for " + std::string(name) + ": " + std::string(e.what()));
    }
  }
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

std::vector<int64_t> SessionInfo::GetInputShape(const std::string& name) const {
  auto type_info = inputs_.find(name);
  if (type_info == inputs_.end())
    throw std::runtime_error("Model input was not found: " + name);
  return type_info->second->GetTensorTypeAndShapeInfo().GetShape();
}

std::vector<int64_t> SessionInfo::GetOutputShape(const std::string& name) const {
  auto type_info = outputs_.find(name);
  if (type_info == outputs_.end())
    throw std::runtime_error("Model output was not found: " + name);
  return type_info->second->GetTensorTypeAndShapeInfo().GetShape();
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

  // Only CUDA, TRT-RTX, RyzenAI and DML does every input on the device
  // For WebGPU, use device memory only if graph capture is enabled, otherwise use CPU
  if (p_device_->GetType() == DeviceType::CUDA || p_device_->GetType() == DeviceType::DML || p_device_->GetType() == DeviceType::NvTensorRtRtx ||
      p_device_->GetType() == DeviceType::RyzenAI ||
      (p_device_->GetType() == DeviceType::WEBGPU && IsGraphCaptureEnabled(config_->model.decoder.session_options)))
    p_device_inputs_ = p_device_;
  else
    p_device_inputs_ = GetDeviceInterface(DeviceType::CPU);

  // Search and sampling are performed on the CPU for all device types,
  // except for CUDA and NvTensorRtRtx, where this is performed on the device.
  if (p_device_->GetType() == DeviceType::CUDA ||
      p_device_->GetType() == DeviceType::NvTensorRtRtx)
    p_device_scoring_ = p_device_;
  else
    p_device_scoring_ = GetDeviceInterface(DeviceType::CPU);

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

  if (config_session_options.log_verbosity_level.has_value()) {
    session_options.SetLogVerbosityLevel(config_session_options.log_verbosity_level.value());
  }

  if (config_session_options.enable_profiling.has_value()) {
    fs::path profile_file_prefix{config_session_options.enable_profiling.value()};
    session_options.EnableProfiling(profile_file_prefix.c_str());
  }

  /*
   * Set all session options that are key-value pairs of strings.
   * Reference: https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h
   */
  for (auto& config_entry : config_session_options.config_entries) {
    session_options.AddConfigEntry(config_entry.first.c_str(), config_entry.second.c_str());
  }

  // Register custom ops libraries only if explicitly configured
  if (config_session_options.custom_ops_library.has_value()) {
    // Reference: https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_ep_device_ep_metadata_keys.h
    constexpr const char* const library_path_metadata_key_name = "library_path";

    std::string custom_library_file_prefix = config_session_options.custom_ops_library.value();

    // If relative path, try to resolve using multiple search locations
    fs::path custom_library_path{custom_library_file_prefix};
    if (custom_library_path.is_relative()) {
      bool resolved = false;

      // First try: resolve relative to GenAI model folder (most intuitive for users)
      fs::path model_relative_path = config_->config_path / custom_library_path;
      if (fs::exists(model_relative_path)) {
        custom_library_file_prefix = model_relative_path.string();
        resolved = true;
      }

      // Second try: resolve relative to EP library directory (for system-wide installations)
      if (!resolved) {
        size_t num_devices = 0;
        const OrtEpDevice* const* device_ptrs = nullptr;
        Ort::GetEpDevices(&GetOrtEnv(), &device_ptrs, &num_devices);

        for (size_t i = 0; i < num_devices && !resolved; ++i) {
          const OrtKeyValuePairs* keyvals = Ort::GetEpDeviceMetadata(device_ptrs[i]);
          size_t num_entries = 0;
          const char* const* keys = nullptr;
          const char* const* values = nullptr;
          Ort::GetKeyValuePairs(keyvals, &keys, &values, &num_entries);

          for (size_t kvi = 0; kvi < num_entries; kvi++) {
            const std::string key = keys[kvi];
            const std::string val = values[kvi];
            if (key == library_path_metadata_key_name) {
              fs::path ep_library_dir = fs::path(val).parent_path();
              fs::path resolved_path = ep_library_dir / custom_library_path;
              if (fs::exists(resolved_path)) {
                custom_library_file_prefix = resolved_path.string();
                resolved = true;
                break;
              }
            }
          }
        }
      }

      // Third try: resolve relative to current working directory (for development/portable apps)
      if (!resolved) {
        char cwd_buffer[PATH_MAX];
        if (GETCWD(cwd_buffer, sizeof(cwd_buffer))) {
          fs::path cwd_relative_path = fs::path(cwd_buffer) / custom_library_path;
          if (fs::exists(cwd_relative_path)) {
            custom_library_file_prefix = cwd_relative_path.string();
            resolved = true;
          }
        }
      }
    }

    // Convert to fs::path for proper wide string handling on Windows
    fs::path custom_ops_lib_path(custom_library_file_prefix);
    session_options.RegisterCustomOpsLibrary(custom_ops_lib_path.c_str());
  }

  if (config_session_options.graph_optimization_level.has_value()) {
    session_options.SetGraphOptimizationLevel(config_session_options.graph_optimization_level.value());
  }

  auto session_device = SetProviderSessionOptions(session_options, config_session_options.providers,
                                                  config_session_options.provider_options, is_primary_session_options,
                                                  *config_, disable_graph_capture);

  if (!p_device_) {
    p_device_ = session_device;
  } else if (session_device != nullptr && session_device->GetType() != p_device_->GetType()) {
    throw std::runtime_error("Running a model with multiple providers is not supported. Encountered " +
                             to_string(session_device->GetType()) + " and " + to_string(p_device_->GetType()));
  }
}

void Model::CreateSessionOptions() {
  session_options_ = OrtSessionOptions::Create();

  CreateSessionOptionsFromConfig(config_->model.decoder.session_options, *session_options_, true);

  for (auto& pipeline_model : config_->model.decoder.pipeline) {
    if (pipeline_model.session_options.has_value()) {
      auto emplaced = pipeline_session_options_.emplace(pipeline_model.model_id, OrtSessionOptions::Create());
      CreateSessionOptionsFromConfig(*pipeline_model.session_options, *emplaced.first->second, false);
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

bool Model::IsPruned() const {
  const auto& logits_name = config_->model.decoder.outputs.logits;
  if (!session_info_.HasOutput(logits_name))
    return false;
  const auto logits_shape = session_info_.GetOutputShape(logits_name);
  return logits_shape[1] == 1;
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
  // Check if it's a pipeline model by checking if decoder.pipeline is configured
  if ((config->model.type == "fara" || config->model.type == "qwen2_5_vl" || config->model.type == "qwen3_vl") && !config->model.decoder.pipeline.empty())
    return std::make_shared<Qwen2_5_VL_PipelineModel>(std::move(config), ort_env);
  if (config->model.type == "gpt2")
    return std::make_shared<Gpt_Model>(std::move(config), ort_env);
  if (ModelType::IsLLM(config->model.type))
    return std::make_shared<DecoderOnly_Model>(std::move(config), ort_env);
  if (ModelType::IsRNNT(config->model.type))
    return std::make_shared<NemotronSpeechModel>(std::move(config), ort_env);
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
          {"gemma3", Processor::Create<GemmaImageProcessor>},
          {"mistral3", Processor::Create<Mistral3ImageProcessor>},
          {"fara", Processor::Create<QwenImageProcessor>},
          {"qwen2_5_vl", Processor::Create<QwenImageProcessor>},
          {"qwen3_vl", Processor::Create<QwenImageProcessor>},
          {"qwen3_5", Processor::Create<QwenImageProcessor>}} {
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
