// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "multi_decoder_pipeline_modal.h"
#include "../logging.h"
#include "../tracing.h"
#include "windowed_kv_cache.h"

namespace Generators {

namespace {

int64_t GetNumImageTokens(const std::vector<ExtraInput>& extra_inputs) {
  for (size_t i = 0; i < extra_inputs.size(); ++i) {
    if (extra_inputs[i].name == Config::Defaults::NumImageTokens) {
      assert(extra_inputs[i].tensor->ort_tensor_);
      const int32_t* num_image_tokens_data = extra_inputs[i].tensor->ort_tensor_->GetTensorData<int32_t>();
      return std::accumulate(num_image_tokens_data,
                             num_image_tokens_data + extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetElementCount(),
                             0LL);
    }
  }

  return 0;
}

int64_t GetNumAudioTokens(const std::vector<ExtraInput>& extra_inputs,
                          const std::string& audio_sizes_name) {
  for (size_t i = 0; i < extra_inputs.size(); ++i) {
    if (extra_inputs[i].name == audio_sizes_name) {
      assert(extra_inputs[i].tensor->ort_tensor_);
      auto type_and_shape_info = extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
      const auto element_count = type_and_shape_info->GetElementCount();
      if (type_and_shape_info->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        const int64_t* audio_sizes_data = extra_inputs[i].tensor->ort_tensor_->GetTensorData<int64_t>();
        return std::accumulate(audio_sizes_data, audio_sizes_data + element_count, 0LL);
      } else {
        throw std::runtime_error("Unsupported data type " + std::to_string(static_cast<int64_t>(type_and_shape_info->GetElementType())) + " for audio_sizes tensor. Only int64 is supported.");
      }
    }
  }

  return 0;
}

int64_t GetImageFeatureBatchSize(const std::vector<ExtraInput>& extra_inputs) {
  for (size_t i = 0; i < extra_inputs.size(); ++i) {
    if (extra_inputs[i].name == Config::Defaults::PixelValuesName) {
      assert(extra_inputs[i].tensor->ort_tensor_);
      const auto num_dims = extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape().size();
      if (num_dims < 3) {
        return 0;
      }
      // If image features have rank 3, the batch size is the first dimension
      return extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape().front();
    }
  }

  return 0;
}

}  // namespace

MultiModalPipelineLanguageModel::MultiModalPipelineLanguageModel(std::unique_ptr<Config> config, OrtEnv& ort_env, bool vision, bool speech)
    : Model(std::move(config)),
      ort_env_{ort_env} {
  // The non-decoder models don't support graph capture because of control flow nodes, so disable graph capture for them
  if (vision) {
    auto vision_session_options = OrtSessionOptions::Create();
    //if (config_->model.vision.session_options.has_value()) {
    CreateSessionOptionsFromConfig((config_->model.vision.session_options), *vision_session_options, true, true);
    //} else {
    //  CreateSessionOptionsFromConfig(config_->model.decoder.session_options, *vision_session_options, true, true);
    //}
    vision_session_ = CreateSession(ort_env, config_->model.vision.filename, vision_session_options.get());
  }

  if (speech) {
    auto speech_session_options = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(config_->model.decoder.session_options, *speech_session_options, true, true);
    speech_session_ = CreateSession(ort_env, config_->model.speech.filename, speech_session_options.get());
  }

  auto embedding_session_options = OrtSessionOptions::Create();
  CreateSessionOptionsFromConfig(config_->model.decoder.session_options, *embedding_session_options, true, true);
  embedding_session_ = CreateSession(ort_env, config_->model.embedding.filename, embedding_session_options.get());

  for (const auto& model : config_->model.decoder.pipeline) {
    decoder_pipeline_sessions_.emplace_back(CreateSession(ort_env, model.filename, GetSessionOptions(model.model_id)));
  }

  for (auto& session : decoder_pipeline_sessions_) {
    session_info_.Add(*session);
  }

  session_info_.Add(*embedding_session_);
  if (speech) {
    session_info_.Add(*speech_session_);
  }
  if (vision) {
    session_info_.Add(*vision_session_);
  }
}

std::unique_ptr<State> MultiModalPipelineLanguageModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<MultiModalDecoderPipelineState>(*this, sequence_lengths, params);
}

VisionPipelineState::VisionPipelineState(const MultiModalPipelineLanguageModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {}

void VisionPipelineState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_images, const int64_t num_image_tokens) {
  num_image_tokens_ = num_image_tokens;
  num_images_ = num_images;

  image_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Output,  // model output
                                                         model_.config_->model.vision.outputs.image_features,
                                                         num_images_, num_image_tokens_);
  for (const auto& ei : extra_inputs) {
    if (ei.name == "pixel_values") {
      pixel_values_tensor_ = ei.tensor;
      break;
    }
  }
  extra_inputs_.Add(extra_inputs, model_.vision_session_->GetInputNames());
}

// Create a [1, C, H, W] tensor and copy the i-th image from a [N, C, H, W] tensor.
static std::shared_ptr<Tensor> MakeSingleImagePixelValues(const std::shared_ptr<Tensor>& full,
                                                          int64_t index,
                                                          DeviceInterface* device) {
  if (!full || !full->GetOrtTensor()) {
    throw std::runtime_error("MakeSingleImagePixelValues: source tensor is null");
  }
  const auto full_shape = full->GetShape();  // expected [N, C, H, W]
  if (full_shape.size() != 4) {
    throw std::runtime_error("MakeSingleImagePixelValues: expected [N, C, H, W] shape");
  }
  const int64_t N = full_shape[0];
  const int64_t C = full_shape[1];
  const int64_t H = full_shape[2];
  const int64_t W = full_shape[3];
  if (index < 0 || index >= N) {
    throw std::runtime_error("MakeSingleImagePixelValues: index out of range");
  }

  // Destination shape [1, C, H, W]
  std::vector<int64_t> dst_shape = {1, C, H, W};

  auto dst = std::make_shared<Tensor>(device, full->GetType());
  dst->CreateTensor(dst_shape, /*make_static=*/false);

  // Compute byte ranges and copy
  const size_t elem_size = Ort::SizeOf(full->GetType());
  const size_t per_image_bytes = static_cast<size_t>(C) * static_cast<size_t>(H) * static_cast<size_t>(W) * elem_size;
  const size_t offset_bytes = static_cast<size_t>(index) * per_image_bytes;

  auto src_bytes = full->GetByteSpan();
  auto dst_bytes = dst->GetByteSpan();

  if (offset_bytes + per_image_bytes > src_bytes.size() || per_image_bytes > dst_bytes.size()) {
    throw std::runtime_error("MakeSingleImagePixelValues: copy bounds exceeded");
  }

  dst_bytes.CopyFrom(src_bytes.subspan(offset_bytes, per_image_bytes));
  return dst;
}

DeviceSpan<float> VisionPipelineState::Run(int current_length,
                                           DeviceSpan<int32_t>& next_tokens,
                                           DeviceSpan<int32_t> next_indices) {
  if (!model_.vision_session_ || !image_features_ || !pixel_values_tensor_) {
    return {};
  }

  const int64_t total_images = num_images_;
  const size_t bytes_per_image = image_features_->BytesPerImage();

  // Flat destination bytes of the global features buffer
  auto dst_all_bytes = image_features_->AsByteSpan();

  // Bind a single-image output features object once and reuse across runs
  std::unique_ptr<MultiModalFeatures> run_features =
      std::make_unique<MultiModalFeatures>(*this,
                                           MultiModalFeatures::Mode::Output,
                                           model_.config_->model.vision.outputs.image_features,
                                           /*batch_size=*/1,
                                           /*num_feature_tokens=*/num_image_tokens_);
  run_features->Add();

  for (int64_t i = 0; i < total_images; ++i) {
    auto pixel_values_i = MakeSingleImagePixelValues(pixel_values_tensor_, i, model_.p_device_);
    extra_inputs_.Replace("pixel_values", pixel_values_i);

    State::Run(*model_.vision_session_);

    auto src_bytes = run_features->AsByteSpan();

    const size_t dst_offset = static_cast<size_t>(i) * bytes_per_image;
    if (dst_offset + bytes_per_image <= dst_all_bytes.size() && bytes_per_image <= src_bytes.size()) {
      dst_all_bytes.subspan(dst_offset, bytes_per_image).CopyFrom(src_bytes.subspan(0, bytes_per_image));
    } else {
      throw std::runtime_error("VisionPipelineState::Run: features copy out of bounds");
    }
  }

  return {};
}

SpeechPipelineState::SpeechPipelineState(const MultiModalPipelineLanguageModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {}

void SpeechPipelineState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_audio_tokens) {
  audio_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Output,  // Model output
                                                         model_.config_->model.speech.outputs.audio_features,
                                                         -1, num_audio_tokens_);
  audio_features_->Add();
  extra_inputs_.Add(extra_inputs, model_.speech_session_->GetInputNames());
}

DeviceSpan<float> SpeechPipelineState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  State::Run(*model_.speech_session_);
  return {};
}

EmbeddingPipelineState::EmbeddingPipelineState(const MultiModalPipelineLanguageModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      inputs_embeds_{
          std::make_unique<WindowedEmbeddings>(
              *this, Embeddings::Mode::Output, model.config_->model.embedding.outputs.embeddings)}
{
  input_ids_.Add();
  inputs_embeds_->Add();
}

void EmbeddingPipelineState::SetExtraInputs(const int64_t num_images, const int64_t num_image_tokens, const int64_t num_audio_tokens) {
  num_image_tokens_ = num_image_tokens;
  num_audio_tokens_ = num_audio_tokens;

  if (model_.vision_session_) {
    image_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Input,  // Optional model input
                                                           model_.config_->model.embedding.inputs.image_features,
                                                           num_images, num_image_tokens_);
    image_features_->Add();
  }
  if (model_.speech_session_) {
    audio_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Input,  // Optional model input
                                                           model_.config_->model.embedding.inputs.audio_features,
                                                           -1, num_audio_tokens_);
    audio_features_->Add();
  }
}

void DecoderPipelineState::SetExtraInputs(const int64_t num_images, const int64_t num_image_tokens, const std::vector<ExtraInput>& extra_inputs) {
  for (auto& session : model_.decoder_pipeline_sessions_) {
    extra_inputs_.Add(extra_inputs, session->GetInputNames());
  }
}

void EmbeddingPipelineState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, bool is_prompt) {
  input_ids_.Update(next_tokens);
  if (model_.vision_session_) image_features_->Update(is_prompt);
  if (model_.speech_session_) audio_features_->Update(is_prompt);

  if (!is_prompt && image_features_->GetShape()[1] > 0) {
    inputs_.erase(inputs_.begin() + image_features_->GetIndex());
    input_names_.erase(input_names_.begin() + image_features_->GetIndex());
  }
}

DeviceSpan<float> EmbeddingPipelineState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  State::Run(*model_.embedding_session_);
  return {};
}

IntermediateDecoderPipelineState::IntermediateDecoderPipelineState(const MultiModalPipelineLanguageModel& model, const GeneratorParams& params,
                                                     size_t pipeline_state_index)
    : State{params, model},
      id_{pipeline_state_index},
      model_{model} {}

bool IntermediateDecoderPipelineState::HasInput(std::string_view name) const {
  return std::any_of(model_.config_->model.decoder.pipeline[id_].inputs.begin(),
                     model_.config_->model.decoder.pipeline[id_].inputs.end(),
                     [&name](const std::string& elem) { return elem == name; });
}

bool IntermediateDecoderPipelineState::HasOutput(std::string_view name) const {
  return std::any_of(model_.config_->model.decoder.pipeline[id_].outputs.begin(),
                     model_.config_->model.decoder.pipeline[id_].outputs.end(),
                     [&name](const std::string& elem) { return elem == name; });
}

bool IntermediateDecoderPipelineState::SupportsPrimaryDevice() const {
  if (model_.p_device_->GetType() == DeviceType::CPU || model_.p_device_->GetType() == DeviceType::QNN) {
    return true;
  } else if (model_.p_device_->GetType() == DeviceType::CUDA) {
    if (!model_.config_->model.decoder.pipeline[id_].session_options.has_value()) {
      // No session options, so this session uses the default session options.
      // Default session options supports the cuda device type.
      return true;
    } else if (auto& provider_options = (*model_.config_->model.decoder.pipeline[id_].session_options).provider_options;
               std::any_of(provider_options.begin(), provider_options.end(),
                           [](const Config::ProviderOptions& elem) { return elem.name == "cuda"; })) {
      // cuda is listed as one of the providers. This session supports the cuda device type.
      return true;
    } else {
      // cuda is not listed as one of the providers. This session does not support the cuda device type.
      return false;
    }
  }

  return false;
}

DeviceSpan<float> IntermediateDecoderPipelineState::Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                                                 DeviceSpan<int32_t> next_indices) {
  if (!model_.decoder_pipeline_sessions_[id_]) {
    const_cast<MultiModalPipelineLanguageModel*>(&model_)->decoder_pipeline_sessions_[id_] =
        OrtSession::Create(model_.ort_env_, (model_.config_->config_path / fs::path(model_.config_->model.decoder.pipeline[id_].filename)).c_str(),
                           model_.GetSessionOptions(model_.config_->model.decoder.pipeline[id_].model_id));
  }
  State::Run(*model_.decoder_pipeline_sessions_[id_]);
  return {};
}

using NameToLayerIdxMap = std::unordered_map<std::string, size_t>;

static NameToLayerIdxMap GeneratePastKeyNameToLayerIdxMap(const Config& config) {
  const size_t num_layers = config.model.decoder.num_hidden_layers;
  const std::string& past_key_name_template = config.model.decoder.inputs.past_key_names;
  NameToLayerIdxMap m{};
  for (size_t i = 0; i < num_layers; ++i) {
    m.emplace(ComposeKeyValueName(past_key_name_template, static_cast<int>(i)), i);
  }
  return m;
}

static std::vector<size_t> GetLayerIndicesSetFromPastKeyNameInputs(
    const NameToLayerIdxMap& past_key_name_to_layer_idx, std::span<const std::string> inputs) {
  std::vector<size_t> layer_indices{};
  for (const auto& input_name : inputs) {
    const auto it = past_key_name_to_layer_idx.find(input_name);
    if (it != past_key_name_to_layer_idx.end()) {
      layer_indices.push_back(it->second);
    }
  }
  // sort and remove duplicates
  std::sort(layer_indices.begin(), layer_indices.end());
  layer_indices.erase(std::unique(layer_indices.begin(), layer_indices.end()),
                      layer_indices.end());
  return layer_indices;
}

DecoderPipelineState::DecoderPipelineState(const MultiModalPipelineLanguageModel& model,
                                                   DeviceSpan<int32_t> sequence_lengths,
                                                   const GeneratorParams& params)
    : State{params, model},
      full_inputs_embeds_{std::make_unique<WindowedEmbeddings>(*this, Embeddings::Mode::Input,
                                                               model.config_->model.embedding.outputs.embeddings)},
      model_{model},
      input_ids_{CreateInputIDs(*this)},
      inputs_embeds_ {
          std::make_unique<WindowedEmbeddings>(
              *this, Embeddings::Mode::Input, model.config_->model.embedding.outputs.embeddings)},
      key_value_cache_{CreateKeyValueCache(*this)},
      do_key_value_cache_partial_update_{key_value_cache_ && key_value_cache_->IsPartialUpdateSupported()},
      position_inputs_{CreatePositionInputs(*this, sequence_lengths, model_.config_->model.decoder.inputs.attention_mask)} {
  input_ids_->Add();
  inputs_embeds_->Add();
  position_inputs_->Add();
  logits_.Add();
  if (key_value_cache_) {
    key_value_cache_->Add();
  }

  const auto& config_pipeline = model_.config_->model.decoder.pipeline;

  for (size_t i = 0; i < config_pipeline.size(); ++i) {
    auto pipeline_model_state = std::make_unique<IntermediateDecoderPipelineState>(model_, params, pipeline_states_.size());
    pipeline_states_.emplace_back(std::move(pipeline_model_state));
  }

  if (do_key_value_cache_partial_update_) {
    const auto past_key_name_to_layer_idx = GeneratePastKeyNameToLayerIdxMap(*model_.config_);

    std::map<std::vector<size_t>, size_t> layer_indices_to_update_record_idx{};
    std::unordered_set<size_t> layer_indices_encountered{};

    for (size_t i = 0; i < config_pipeline.size(); ++i) {
      const auto& pipeline_model = config_pipeline[i];

      const auto layer_indices = GetLayerIndicesSetFromPastKeyNameInputs(past_key_name_to_layer_idx,
                                                                         pipeline_model.inputs);

      if (layer_indices.empty()) {
        continue;
      }

      size_t record_idx{};

      if (auto layer_indices_to_update_record_it = layer_indices_to_update_record_idx.find(layer_indices);
          layer_indices_to_update_record_it != layer_indices_to_update_record_idx.end()) {
        // we have seen this exact set of layer indices before. reuse the existing record.
        record_idx = layer_indices_to_update_record_it->second;
      } else {
        // verify that the new set of layer indices is valid.
        // i.e., it is disjoint with the set of all layer indices we've seen so far.
        const bool layer_indices_valid =
            std::all_of(layer_indices.begin(), layer_indices.end(),
                        [&layer_indices_encountered](size_t layer_idx) {
                          return layer_indices_encountered.find(layer_idx) == layer_indices_encountered.end();
                        });

        if (!layer_indices_valid) {
          throw std::runtime_error(
              "Invalid layer indices. Layer index sets for partial key value cache update must be either an exact "
              "match with another set or disjoint with all other sets.");
        }

        // add a new record
        auto record = PartialKeyValueCacheUpdateRecord{};
        record.layer_indices = layer_indices;

        partial_kv_cache_update_records_.emplace_back(std::move(record));
        record_idx = partial_kv_cache_update_records_.size() - 1;

        // add layer_indices to what we've seen so far
        layer_indices_encountered.insert(layer_indices.begin(), layer_indices.end());
        layer_indices_to_update_record_idx.emplace(layer_indices, record_idx);
      }

      pipeline_state_id_to_partial_kv_cache_update_record_idx_.emplace(i, record_idx);
    }

    if (!partial_kv_cache_update_records_.empty()) {
      key_value_cache_update_worker_thread_.emplace();
    }
  }
}




void DecoderPipelineState::RunPipeline(int total_length, DeviceSpan<int32_t>& next_tokens,
                                           DeviceSpan<int32_t> next_indices) {
  for (auto& pipeline_state : pipeline_states_) {
    if (first_run_ && !model_.config_->model.decoder.pipeline[pipeline_state->id_].run_on_prompt) {
      continue;
    } else if (!first_run_ && !model_.config_->model.decoder.pipeline[pipeline_state->id_].run_on_token_gen) {
      continue;
    }

    DurationTrace trace{MakeString("DecoderPipelineState::RunPipeline[", pipeline_state->id_, "]")};

    if (model_.config_->model.decoder.pipeline[pipeline_state->id_].reset_session_idx > -1) {
      if (model_.config_->model.decoder.pipeline[pipeline_state->id_].reset_session_idx >=
          static_cast<int>(model_.decoder_pipeline_sessions_.size())) {
        throw std::runtime_error(
            MakeString("Invalid reset_session_idx ", model_.config_->model.decoder.pipeline[pipeline_state->id_].reset_session_idx,
                       " for pipeline model ", model_.config_->model.decoder.pipeline[pipeline_state->id_].model_id));
      }
      (const_cast<MultiModalPipelineLanguageModel*>(&model_))->decoder_pipeline_sessions_[model_.config_->model.decoder.pipeline[pipeline_state->id_].reset_session_idx].reset();
    }

    auto* const partial_kv_cache_update_record = [&]() -> PartialKeyValueCacheUpdateRecord* {
      auto it = pipeline_state_id_to_partial_kv_cache_update_record_idx_.find(pipeline_state->id_);
      if (it != pipeline_state_id_to_partial_kv_cache_update_record_idx_.end()) {
        return &partial_kv_cache_update_records_[it->second];
      }
      return nullptr;
    }();

    // If there is any outstanding partial KV cache update, wait for it to finish.
    // It is important to synchronize at this point, before setting input/output tensors for this pipeline state run,
    // because a KV cache update may replace the KV cache input/output tensors.
    if (partial_kv_cache_update_record) {
      if (partial_kv_cache_update_record->outstanding_update.valid()) {
        partial_kv_cache_update_record->outstanding_update.get();
      }
    }

    // Clear the intermediate pipeline state outputs from the previous runs.
    // These outputs will be replaced by the outputs from the current run.
    for (const auto& output_name : pipeline_state->output_names_) {
      if (auto iter = ortvalue_store_.find(output_name); iter != ortvalue_store_.end()) {
        ortvalue_store_.erase(iter);
      }
    }
    pipeline_state->ClearIO();

    // Managed inputs and outputs are those inputs and outputs that the
    // Model knows how to create and update from one run to the next.

    // Add all the managed inputs to the intermediate pipeline state
    for (const auto& input_name : input_names_) {
      if (pipeline_state->HasInput(input_name)) {
        if (!pipeline_state->SupportsPrimaryDevice()) {
          throw std::runtime_error(
              MakeString("Managed input ", input_name, " resides on the primary device type (",
                         static_cast<int>(model_.p_device_->GetType()), "). But the pipeline model ",
                         model_.config_->model.decoder.pipeline[pipeline_state->id_].model_id,
                         " is expecting it to reside elsewhere."));
        }
        pipeline_state->input_names_.push_back(input_name);
        pipeline_state->inputs_.push_back(State::GetInput(input_name));
      }
    }

    // Add outputs from the previous pipeline states to the current pipeline state
    for (auto& [name, ortvalue] : ortvalue_store_) {
      if (pipeline_state->HasInput(name)) {
        pipeline_state->input_names_.push_back(name.c_str());
        pipeline_state->inputs_.push_back(ortvalue.get());
      }
    }

    // Add all the managed outputs to the intermediate pipeline state
    for (const auto& output_name : output_names_) {
      if (pipeline_state->HasOutput(output_name)) {
        if (!pipeline_state->SupportsPrimaryDevice()) {
          throw std::runtime_error(
              MakeString("Managed output ", output_name, " resides on the primary device type (",
                         static_cast<int>(model_.p_device_->GetType()), "). But the pipeline model ",
                         model_.config_->model.decoder.pipeline[pipeline_state->id_].model_id,
                         " is expecting it to reside elsewhere."));
        }
        pipeline_state->output_names_.push_back(output_name);
        pipeline_state->outputs_.push_back(State::GetOutput(output_name));
      }
    }

    // Output of pipeline models could also be managed inputs.
    // For example, the output of a pipeline model could be the key-value cache.
    // In such cases, use the managed output buffers and register them with the pipeline model as outputs.
    for (const auto& input_name : input_names_) {
      if (pipeline_state->HasOutput(input_name)) {
        if (!pipeline_state->SupportsPrimaryDevice()) {
          throw std::runtime_error(
              MakeString("Managed input ", input_name, " resides on the primary device type (",
                         static_cast<int>(model_.p_device_->GetType()), "). But the pipeline model ",
                         model_.config_->model.decoder.pipeline[pipeline_state->id_].model_id,
                         " is expecting it to reside elsewhere."));
        }
        pipeline_state->output_names_.push_back(input_name);
        pipeline_state->outputs_.push_back(State::GetInput(input_name));
      }
    }

    // Add all the remaining outputs for the intermediate pipeline state
    for (const auto& output_name : model_.config_->model.decoder.pipeline[pipeline_state->id_].outputs) {
      if (std::none_of(pipeline_state->output_names_.begin(), pipeline_state->output_names_.end(),
                       [&](const std::string& elem) { return elem == output_name; })) {
        pipeline_state->output_names_.push_back(output_name.c_str());
        pipeline_state->outputs_.push_back(nullptr);
      }
    }

    // Run the intermediate pipeline state
    pipeline_state->Run(total_length, next_tokens, next_indices);

    // If there is any partial KV cache update to start, enqueue it.
    if (partial_kv_cache_update_record) {
      assert(key_value_cache_update_worker_thread_.has_value());
      auto update_fn = [&key_value_cache = *key_value_cache_.get(),
                        layer_indices = partial_kv_cache_update_record->layer_indices,
                        next_indices, total_length]() {
        key_value_cache.PartialUpdate(next_indices, total_length, layer_indices);
      };
      partial_kv_cache_update_record->outstanding_update = key_value_cache_update_worker_thread_->Enqueue(update_fn);
    }

    // Transfer ownership of all the non-managed outputs from the current pipeline state to the ortvalue store.
    // All non managed outputs are assumed to be on CPU
    for (size_t i = 0; i < pipeline_state->output_names_.size(); ++i) {
      if (std::none_of(output_names_.begin(), output_names_.end(),
                       [&](const std::string& elem) { return elem == pipeline_state->output_names_[i]; }) &&
          std::none_of(input_names_.begin(), input_names_.end(),
                       [&](const std::string& elem) { return elem == pipeline_state->output_names_[i]; })) {
        auto forwarded_output = model_.config_->model.decoder.pipeline[pipeline_state->id_].output_names_forwarder.find(pipeline_state->output_names_[i]);
        if (forwarded_output != model_.config_->model.decoder.pipeline[pipeline_state->id_].output_names_forwarder.end()) {
          ortvalue_store_[forwarded_output->second] = std::unique_ptr<OrtValue>(pipeline_state->outputs_[i]);
        } else {
          ortvalue_store_[pipeline_state->output_names_[i]] = std::unique_ptr<OrtValue>(pipeline_state->outputs_[i]);
        }
      }
    }
  }
}

  DeviceSpan<float> DecoderPipelineState::Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                                                DeviceSpan<int32_t> next_indices) {
  DurationTrace trace{"DecoderPipelineState::Run"};

  UpdateInputsOutputs(next_tokens, next_indices, *full_inputs_embeds_, total_length);

  size_t num_chunks{1};
  if (first_run_ && model_.config_->model.decoder.sliding_window.has_value()) {
    int window_size = model_.config_->model.decoder.sliding_window->window_size;
    num_chunks = (next_tokens.size() + window_size - 1) / window_size;
  }

  for (size_t i = 0; i < num_chunks; ++i) {
    RunPipeline(total_length, next_tokens, next_indices);

    if (model_.config_->model.decoder.sliding_window.has_value() && i < num_chunks - 1) {
      // Sliding the window over the input_ids, key_cache, and value_cache, position_ids, and attention_mask
      input_ids_->Update(next_tokens);
      inputs_embeds_->Update(*full_inputs_embeds_);
      UpdateKeyValueCache(next_indices, total_length);
      position_inputs_->Update(next_tokens, total_length, static_cast<int>(input_ids_->GetShape()[1]));
      logits_.Update(WrapTensor<int32_t>(*model_.p_device_inputs_, *input_ids_->Get()),
                     static_cast<int>(input_ids_->GetShape()[1]));
    }
  }

  // Clear the outputs of the pipeline models that are only run on prompt since this cannot happen earlier.
  if (!first_run_) {
    for (auto& pipeline_state : pipeline_states_) {
      if (!model_.config_->model.decoder.pipeline[pipeline_state->id_].run_on_token_gen) {
        for (const auto& output_name : pipeline_state->output_names_) {
          if (auto iter = ortvalue_store_.find(output_name); iter != ortvalue_store_.end()) {
            ortvalue_store_.erase(iter);
          }
        }
      }
    }
  }

  first_run_ = false;

  return logits_.Get();
}

void DecoderPipelineState::UpdateKeyValueCache(DeviceSpan<int32_t> beam_indices, int total_length) {
  if (key_value_cache_) {
    const bool outstanding_key_value_cache_partial_update =
        do_key_value_cache_partial_update_ &&
        std::any_of(partial_kv_cache_update_records_.rbegin(),
                    partial_kv_cache_update_records_.rend(),
                    [](const PartialKeyValueCacheUpdateRecord& record) {
                      return record.outstanding_update.valid();
                    });

    if (outstanding_key_value_cache_partial_update) {
      // If there is any outstanding partial KV cache update, don't update the KV cache here.
    } else {
      key_value_cache_->Update(beam_indices, total_length);
    }
  }
}

void DecoderPipelineState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens,
                                               DeviceSpan<int32_t> beam_indices, Embeddings& embeddings, int total_length) {
  input_ids_->Update(next_tokens);
  inputs_embeds_->Update(embeddings);
  size_t new_length = input_ids_->GetShape()[1];
  position_inputs_->Update(next_tokens, total_length, static_cast<int>(new_length));
  UpdateKeyValueCache(beam_indices, total_length);

  auto next_windowed_tokens = WrapTensor<int32_t>(*model_.p_device_inputs_, *input_ids_->Get());
  logits_.Update(next_windowed_tokens, new_length);
}

OrtValue* DecoderPipelineState::GetOutput(const char* name) {
  // Check the ortvalue store to search if name is one of the non-managed output.
  auto it = ortvalue_store_.find(name);
  if (it != ortvalue_store_.end()) {
    return it->second.get();
  }

  // Search managed outputs saved in this State.
  return State::GetOutput(name);
}


MultiModalDecoderPipelineState::MultiModalDecoderPipelineState(const MultiModalPipelineLanguageModel& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      adapters_{std::make_shared<Adapters>(&model_)} {
  if (model_.vision_session_) {
    vision_state_ = std::make_unique<VisionPipelineState>(model_, params);
  }
  if (model_.speech_session_) {
    speech_state_ = std::make_unique<SpeechPipelineState>(model_, params);
  }
  decoder_pipeline_state_ = std::make_unique<DecoderPipelineState>(model, sequence_lengths, params);
  embedding_state_ = std::make_unique<EmbeddingPipelineState>(model, params);

  if (vision_state_ != nullptr && model_.config_->model.vision.adapter_filename.has_value() && num_image_tokens_ > 0) {
    const auto lora_adapter = (model_.config_->config_path / fs::path(*model_.config_->model.vision.adapter_filename)).string();
    adapters_->LoadAdapter(lora_adapter.c_str(), vision_adapter_name_);
    decoder_pipeline_state_->SetActiveAdapter(adapters_.get(), vision_adapter_name_);
  } else if (speech_state_ != nullptr && model_.config_->model.speech.adapter_filename.has_value() && num_audio_tokens_ > 0) {
    const auto lora_adapter = (model_.config_->config_path / fs::path(*model_.config_->model.speech.adapter_filename)).string();
    adapters_->LoadAdapter(lora_adapter.c_str(), speech_adapter_name_);
    decoder_pipeline_state_->SetActiveAdapter(adapters_.get(), speech_adapter_name_);
  }
}

void MultiModalDecoderPipelineState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  num_image_tokens_ = GetNumImageTokens(extra_inputs);
  num_audio_tokens_ = GetNumAudioTokens(extra_inputs, model_.config_->model.speech.inputs.audio_sizes);
  num_images_ = GetImageFeatureBatchSize(extra_inputs);

  if (model_.vision_session_) {
    vision_state_->SetExtraInputs(extra_inputs, num_images_, num_image_tokens_);
  }
  if (model_.speech_session_) {
    speech_state_->SetExtraInputs(extra_inputs, num_audio_tokens_);
  }
  embedding_state_->SetExtraInputs(num_images_, num_image_tokens_, num_audio_tokens_);
  decoder_pipeline_state_->SetExtraInputs(num_images_, num_image_tokens_, extra_inputs);
}

DeviceSpan<float> MultiModalDecoderPipelineState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  // Pipeline state defines the pipeline of the execution of the models
  // Prompt stage:
  //   - pixel_values -> |vision_model| -> image_features
  //   - audio_embeds, audio_sizes, audio_projection_mode -> |audio_model| -> audio_features
  //   - input_ids (1, 320), image_features (1, 256, 2560), audio_features -> |embeddings_model| -> inputs_embeds (1, 320, 2560)
  //   - inputs_embeds -> |decoder_model| -> logits
  // Generation stage:
  //   - input_ids -> |decoder_model| -> logits

  embedding_state_->UpdateInputsOutputs(next_tokens, is_prompt_);
  int batch_size = static_cast<int>(decoder_pipeline_state_->full_inputs_embeds_->GetShape()[0]);
  size_t new_length = next_tokens.size() / batch_size;
  decoder_pipeline_state_->full_inputs_embeds_->UpdateSequenceLength(new_length);

  if (is_prompt_) {
    if (num_image_tokens_ > 0 && vision_state_) {
      vision_state_->Run(current_length, next_tokens, next_indices);
    }
    if (num_audio_tokens_ > 0 && speech_state_) {
      speech_state_->Run(current_length, next_tokens, next_indices);
    }
    vision_state_->image_features_->Add();
    if (vision_state_) embedding_state_->image_features_->ReuseFeaturesBuffer(*vision_state_->image_features_);
    if (speech_state_) embedding_state_->audio_features_->ReuseFeaturesBuffer(*speech_state_->audio_features_);
    embedding_state_->inputs_embeds_->ReuseEmbeddingsBuffer(*decoder_pipeline_state_->full_inputs_embeds_);
    embedding_state_->Run(current_length, next_tokens, next_indices);

    auto logits = decoder_pipeline_state_->Run(current_length, next_tokens, next_indices);

    is_prompt_ = false;
    if (vision_state_) vision_state_.reset();  // The vision state is no longer needed in generation stage
    if (speech_state_) speech_state_.reset();  // The speech state is no longer needed in generation stage

    return logits;
  }

  embedding_state_->inputs_embeds_->ReuseEmbeddingsBuffer(*decoder_pipeline_state_->full_inputs_embeds_);
  embedding_state_->Run(current_length, next_tokens, next_indices);
  return decoder_pipeline_state_->Run(current_length, next_tokens, next_indices);
}

OrtValue* MultiModalDecoderPipelineState::GetInput(const char* name) {
  if (vision_state_) {
    // Check if input name is in vision state's inputs
    for (size_t i = 0; i < vision_state_->input_names_.size(); i++) {
      if (std::strcmp(vision_state_->input_names_[i], name) == 0) {
        return vision_state_->inputs_[i];
      }
    }
  }

  if (speech_state_) {
    // Check if input name is in speech state's inputs
    for (size_t i = 0; i < speech_state_->input_names_.size(); i++) {
      if (std::strcmp(speech_state_->input_names_[i], name) == 0) {
        return speech_state_->inputs_[i];
      }
    }
  }

  // Check if input name is in embedding state's inputs
  for (size_t i = 0; i < embedding_state_->input_names_.size(); i++) {
    if (std::strcmp(embedding_state_->input_names_[i], name) == 0) {
      return embedding_state_->inputs_[i];
    }
  }

  // Check if input name is in decoder state's inputs
  for (size_t i = 0; i < decoder_pipeline_state_->input_names_.size(); i++) {
    if (std::strcmp(decoder_pipeline_state_->input_names_[i], name) == 0) {
      return decoder_pipeline_state_->inputs_[i];
    }
  }

  return State::GetInput(name);
};

OrtValue* MultiModalDecoderPipelineState::GetOutput(const char* name) {
  if (vision_state_) {
    // Check if output name is in vision state's outputs
    for (size_t i = 0; i < vision_state_->output_names_.size(); i++) {
      if (std::strcmp(vision_state_->output_names_[i], name) == 0) {
        return vision_state_->outputs_[i];
      }
    }
  }

  if (speech_state_) {
    // Check if output name is in speech state's outputs
    for (size_t i = 0; i < speech_state_->output_names_.size(); i++) {
      if (std::strcmp(speech_state_->output_names_[i], name) == 0) {
        return speech_state_->outputs_[i];
      }
    }
  }

  // Check if output name is in embedding state's outputs
  for (size_t i = 0; i < embedding_state_->output_names_.size(); i++) {
    if (std::strcmp(embedding_state_->output_names_[i], name) == 0) {
      return embedding_state_->outputs_[i];
    }
  }

  // Check if output name is in decoder state's outputs
  for (size_t i = 0; i < decoder_pipeline_state_->output_names_.size(); i++) {
    if (std::strcmp(decoder_pipeline_state_->output_names_[i], name) == 0) {
      return decoder_pipeline_state_->outputs_[i];
    }
  }

  return State::GetOutput(name);
};

}  // namespace Generators
