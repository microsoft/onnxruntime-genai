// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "gemma4_vision_state.h"

#include <array>
#include <numeric>

namespace Generators {

namespace {

struct OrtValuePointerRestore {
  OrtValue*& slot;
  OrtValue* value;

  ~OrtValuePointerRestore() { slot = value; }
};

std::unique_ptr<OrtValue> SliceImage(OrtValue& source, int64_t image) {
  auto shape = source.GetTensorTypeAndShapeInfo()->GetShape();
  const auto type = source.GetTensorTypeAndShapeInfo()->GetElementType();
  int64_t elements = 1;
  for (size_t i = 1; i < shape.size(); ++i) elements *= shape[i];
  const size_t bytes = static_cast<size_t>(elements) * Ort::SizeOf(type);
  shape[0] = 1;
  auto* data = static_cast<uint8_t*>(source.GetTensorMutableRawData());
  return OrtValue::CreateTensor(source.GetTensorMemoryInfo(), data + static_cast<size_t>(image) * bytes,
                                bytes, shape, type);
}

}  // namespace

void Gemma4VisionState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs,
                                       const int64_t num_images,
                                       const int64_t num_image_tokens) {
  image_token_counts_.clear();
  for (const auto& input : extra_inputs) {
    if (input.name == Config::Defaults::NumImageTokens && input.tensor->ort_tensor_) {
      const auto info = input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
      const auto count = info->GetElementCount();
      const int64_t* data = input.tensor->ort_tensor_->GetTensorData<int64_t>();
      image_token_counts_.assign(data, data + count);
      break;
    }
  }
  VisionState::SetExtraInputs(extra_inputs, num_images, num_image_tokens);

  const std::string& pixel_values_name = model_.config_->model.vision.inputs.pixel_values;
  const std::string& position_ids_name = model_.config_->model.vision.inputs.pixel_position_ids;
  pixel_values_index_ = SIZE_MAX;
  position_ids_index_ = SIZE_MAX;
  for (size_t i = 0; i < input_names_.size(); ++i) {
    if (input_names_[i] == pixel_values_name) pixel_values_index_ = i;
    if (input_names_[i] == position_ids_name) position_ids_index_ = i;
  }

  if (model_.vision_projector_session_ && num_image_tokens > 0) {
    const auto encoder_outputs = model_.vision_session_->GetOutputNames();
    projector_input_names_ = model_.vision_projector_session_->GetInputNames();
    if (encoder_outputs.size() != 1) {
      throw std::runtime_error("Gemma 4 vision encoder must have exactly one output");
    }
    const auto found = std::find(projector_input_names_.begin(), projector_input_names_.end(), encoder_outputs[0]);
    if (found == projector_input_names_.end()) {
      throw std::runtime_error("Gemma 4 vision projector must consume the encoder's single output");
    }
    encoder_output_index_ = static_cast<size_t>(std::distance(projector_input_names_.begin(), found));
    const auto projector_outputs = model_.vision_projector_session_->GetOutputNames();
    if (std::find(projector_outputs.begin(), projector_outputs.end(),
                  model_.config_->model.vision.outputs.image_features) == projector_outputs.end()) {
      throw std::runtime_error("Gemma 4 vision projector must output '" +
                               model_.config_->model.vision.outputs.image_features + "'");
    }
    projector_source_inputs_.clear();
    for (size_t i = 0; i < projector_input_names_.size(); ++i) {
      if (i == encoder_output_index_) {
        projector_source_inputs_.push_back(nullptr);
        continue;
      }
      const auto& name = projector_input_names_[i];
      const auto source = std::find_if(extra_inputs.begin(), extra_inputs.end(),
                                       [&](const ExtraInput& input) { return input.name == name; });
      if (source == extra_inputs.end()) {
        throw std::runtime_error("Vision projector: required input '" + name + "' was not produced by the processor");
      }
      projector_source_inputs_.push_back(source->tensor->GetOrtTensor());
    }
  }
}

DeviceSpan<float> Gemma4VisionState::Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                                         DeviceSpan<int32_t> next_indices) {
  if (model_.vision_projector_session_) {
    return RunSplitVision();
  }
  if (model_.config_->model.vision.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.vision.run_options.value());
  }

  if (num_images_ <= 1) {
    State::Run(*model_.vision_session_);
    return {};
  }

  if (pixel_values_index_ == SIZE_MAX || position_ids_index_ == SIZE_MAX) {
    throw std::runtime_error(
        "Gemma 4 multi-image vision requires pixel_values and pixel_position_ids inputs");
  }
  if (image_token_counts_.size() != static_cast<size_t>(num_images_)) {
    throw std::runtime_error("Gemma 4 multi-image vision requires one num_image_tokens value per image");
  }

  OrtValue* pixel_values = inputs_[pixel_values_index_];
  OrtValue* position_ids = inputs_[position_ids_index_];
  OrtValue* image_features = outputs_[0];
  const OrtValuePointerRestore restore_pixel_values{inputs_[pixel_values_index_], pixel_values};
  const OrtValuePointerRestore restore_position_ids{inputs_[position_ids_index_], position_ids};
  const OrtValuePointerRestore restore_image_features{outputs_[0], image_features};
  const auto pixel_info = pixel_values->GetTensorTypeAndShapeInfo();
  const auto position_info = position_ids->GetTensorTypeAndShapeInfo();
  const auto feature_info = image_features->GetTensorTypeAndShapeInfo();
  const auto pixel_shape = pixel_info->GetShape();
  const auto position_shape = position_info->GetShape();
  const auto feature_shape = feature_info->GetShape();
  if (pixel_shape.size() != 3 || position_shape.size() != 3 ||
      pixel_shape[0] != num_images_ || position_shape[0] != num_images_ ||
      pixel_shape[1] != position_shape[1]) {
    throw std::runtime_error("Gemma 4 multi-image vision inputs must have matching [num_images, num_patches, ...] shapes");
  }
  if (feature_shape.size() != 2) {
    throw std::runtime_error("Gemma 4 image_features must have rank 2 [num_image_tokens, hidden_size]");
  }

  const int64_t num_patches = pixel_shape[1];
  const int64_t patch_dim = pixel_shape[2];
  const int64_t position_dim = position_shape[2];
  const int64_t hidden_size = feature_shape[1];
  const auto pixel_type = pixel_info->GetElementType();
  const auto position_type = position_info->GetElementType();
  const auto feature_type = feature_info->GetElementType();
  const size_t pixel_bytes = Ort::SizeOf(pixel_type);
  const size_t position_bytes = Ort::SizeOf(position_type);
  const size_t feature_bytes = Ort::SizeOf(feature_type);
  auto* pixel_data = static_cast<uint8_t*>(pixel_values->GetTensorMutableRawData());
  auto* position_data = static_cast<uint8_t*>(position_ids->GetTensorMutableRawData());
  auto* feature_data = static_cast<uint8_t*>(image_features->GetTensorMutableRawData());
  const auto& pixel_memory = pixel_values->GetTensorMemoryInfo();
  const auto& position_memory = position_ids->GetTensorMemoryInfo();
  const auto& feature_memory = image_features->GetTensorMemoryInfo();

  const int64_t total_image_tokens =
      std::accumulate(image_token_counts_.begin(), image_token_counts_.end(), 0LL);
  if (total_image_tokens != feature_shape[0]) {
    throw std::runtime_error("Gemma 4 image token counts total " + std::to_string(total_image_tokens) +
                             " but image_features has space for " + std::to_string(feature_shape[0]) + " tokens");
  }

  int64_t feature_offset = 0;
  for (int64_t image = 0; image < num_images_; ++image) {
    const int64_t image_tokens = image_token_counts_[static_cast<size_t>(image)];
    if (image_tokens <= 0) {
      throw std::runtime_error("Gemma 4 image token counts must be positive");
    }
    const std::array<int64_t, 3> image_pixel_shape{1, num_patches, patch_dim};
    const std::array<int64_t, 3> image_position_shape{1, num_patches, position_dim};
    const std::array<int64_t, 2> image_feature_shape{image_tokens, hidden_size};

    auto image_pixel_values = OrtValue::CreateTensor(
        pixel_memory, pixel_data + static_cast<size_t>(image * num_patches * patch_dim) * pixel_bytes,
        static_cast<size_t>(num_patches * patch_dim) * pixel_bytes, image_pixel_shape, pixel_type);
    auto image_position_ids = OrtValue::CreateTensor(
        position_memory, position_data + static_cast<size_t>(image * num_patches * position_dim) * position_bytes,
        static_cast<size_t>(num_patches * position_dim) * position_bytes, image_position_shape, position_type);
    auto image_feature_values = OrtValue::CreateTensor(
        feature_memory, feature_data + static_cast<size_t>(feature_offset * hidden_size) * feature_bytes,
        static_cast<size_t>(image_tokens * hidden_size) * feature_bytes, image_feature_shape, feature_type);

    inputs_[pixel_values_index_] = image_pixel_values.get();
    inputs_[position_ids_index_] = image_position_ids.get();
    outputs_[0] = image_feature_values.get();
    State::Run(*model_.vision_session_);
    feature_offset += image_tokens;
  }

  return {};
}

DeviceSpan<float> Gemma4VisionState::RunSplitVision() {
  if (pixel_values_index_ == SIZE_MAX) {
    throw std::runtime_error("Gemma 4 split vision requires pixel_values in the encoder");
  }
  const auto& vision = model_.config_->model.vision;
  auto stage_run_options = [&](size_t index) {
    auto options = OrtRunOptions::Create();
    const auto& configured = vision.pipeline[index].run_options.has_value()
                                 ? vision.pipeline[index].run_options
                                 : vision.run_options;
    if (configured) {
      for (const auto& [key, value] : *configured) {
        options->AddConfigEntry(key.c_str(), value.c_str());
      }
    }
    return options;
  };
  auto encoder_options = stage_run_options(0);
  auto projector_options = stage_run_options(1);
  const auto encoder_outputs = model_.vision_session_->GetOutputNames();
  const char* encoder_output_name = encoder_outputs[0].c_str();
  const char* projector_output_name = vision.outputs.image_features.c_str();
  std::vector<const char*> projector_names;
  for (const auto& name : projector_input_names_) projector_names.push_back(name.c_str());

  const auto feature_shape = outputs_[0]->GetTensorTypeAndShapeInfo()->GetShape();
  if (num_images_ > 1) {
    if (image_token_counts_.size() != static_cast<size_t>(num_images_) || feature_shape.size() != 2) {
      throw std::runtime_error("Gemma 4 split vision requires one token count per image and rank-2 image_features");
    }
    const auto total_tokens = std::accumulate(image_token_counts_.begin(), image_token_counts_.end(), 0LL);
    if (feature_shape[0] != total_tokens) {
      throw std::runtime_error("Gemma 4 image token counts do not match the image_features output");
    }
  }

  std::vector<const OrtValue*> encoder_inputs(inputs_.begin(), inputs_.end());
  std::vector<const OrtValue*> projector_inputs(projector_source_inputs_.begin(), projector_source_inputs_.end());
  int64_t feature_offset = 0;
  for (int64_t image = 0; image < num_images_; ++image) {
    std::vector<std::unique_ptr<OrtValue>> slices;
    if (num_images_ > 1) {
      for (size_t i = 0; i < inputs_.size(); ++i) {
        const auto shape = inputs_[i]->GetTensorTypeAndShapeInfo()->GetShape();
        if (shape.size() >= 2 && shape[0] == num_images_) {
          slices.push_back(SliceImage(*inputs_[i], image));
          encoder_inputs[i] = slices.back().get();
        }
      }
      for (size_t i = 0; i < projector_source_inputs_.size(); ++i) {
        auto* source = projector_source_inputs_[i];
        if (!source) continue;
        const auto shape = source->GetTensorTypeAndShapeInfo()->GetShape();
        if (shape.size() >= 2 && shape[0] == num_images_) {
          slices.push_back(SliceImage(*source, image));
          projector_inputs[i] = slices.back().get();
        }
      }
    }

    auto encoded = model_.vision_session_->Run(encoder_options.get(), input_names_.data(),
                                               encoder_inputs.data(), encoder_inputs.size(),
                                               &encoder_output_name, 1);
    projector_inputs[encoder_output_index_] = encoded[0].get();
    OrtValue* output = outputs_[0];
    std::unique_ptr<OrtValue> output_slice;
    if (num_images_ > 1) {
      const int64_t tokens = image_token_counts_[static_cast<size_t>(image)];
      if (tokens <= 0) {
        throw std::runtime_error("Gemma 4 image token counts must be positive");
      }
      const auto type = outputs_[0]->GetTensorTypeAndShapeInfo()->GetElementType();
      const size_t bytes = static_cast<size_t>(tokens * feature_shape[1]) * Ort::SizeOf(type);
      auto* data = static_cast<uint8_t*>(outputs_[0]->GetTensorMutableRawData());
      const std::array<int64_t, 2> shape{tokens, feature_shape[1]};
      output_slice = OrtValue::CreateTensor(outputs_[0]->GetTensorMemoryInfo(),
                                            data + static_cast<size_t>(feature_offset * feature_shape[1]) * Ort::SizeOf(type),
                                            bytes, shape, type);
      output = output_slice.get();
      feature_offset += tokens;
    }
    model_.vision_projector_session_->Run(projector_options.get(), projector_names.data(),
                                          projector_inputs.data(), projector_names.size(),
                                          &projector_output_name, &output, 1);
  }
  return {};
}

}  // namespace Generators