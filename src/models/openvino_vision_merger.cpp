// Copyright (c) Intel® Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>

#include "../generators.h"
#include "model.h"
#include "multi_modal_features.h"
#include "openvino_multi_modal.h"
#include "openvino_vision_merger.h"

namespace Generators {

namespace {

// Returns the OrtValue for a named extra input, or nullptr if absent.
OrtValue* FindExtraInput(const std::vector<ExtraInput>& extra_inputs, std::string_view name) {
  for (const auto& input : extra_inputs) {
    if (input.name == name) return input.tensor->GetOrtTensor();
  }
  return nullptr;
}

// Gemma3: single vision session (pixel_values -> image_features). Image placeholder tokens are marked
// by token_type_ids == 1 (produced by GemmaImageProcessor); Gemma configs carry no image_token_id, so the
// token_type_ids channel is used to locate the image rows for host-side feature injection.
struct Gemma3OpenVINOVisionMerger : OpenVINOVisionMerger {
  explicit Gemma3OpenVINOVisionMerger(const OpenVINOMultiModalModel& model) : model_{model} {}

  void RunVision(const std::vector<ExtraInput>& extra_inputs) override;
  void Merge(OrtValue& inputs_embeds, const OrtValue& input_ids) override;

 private:
  const OpenVINOMultiModalModel& model_;
  std::unique_ptr<OrtValue> image_features_;  // produced for the current prompt (null when no image)
  std::vector<int32_t> token_type_ids_;       // host copy of token_type_ids for the current prompt
};

void Gemma3OpenVINOVisionMerger::RunVision(const std::vector<ExtraInput>& extra_inputs) {
  image_features_.reset();
  token_type_ids_.clear();

  OrtValue* pixel_values = FindExtraInput(extra_inputs, Config::Defaults::PixelValuesName);
  if (!pixel_values || !model_.ov_vision_session_) return;  // text-only prompt

  // Ensure pixel_values are float32 for the vision graph.
  std::unique_ptr<OrtValue> pixel_values_fp32;
  const OrtValue* pixel_input = pixel_values;
  if (pixel_values->GetTensorTypeAndShapeInfo()->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    Cast(*pixel_values, pixel_values_fp32, *model_.p_device_inputs_, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    pixel_input = pixel_values_fp32.get();
  }

  const std::string& pixel_input_name = model_.config_->model.vision.inputs.pixel_values;
  const std::string& image_features_name = model_.config_->model.vision.outputs.image_features;
  const char* input_names[] = {pixel_input_name.c_str()};
  const OrtValue* input_values[] = {pixel_input};
  const char* output_names[] = {image_features_name.c_str()};

  auto outputs = model_.ov_vision_session_->Run(nullptr, input_names, input_values, 1, output_names, 1);
  image_features_ = std::move(outputs[0]);

  // Cache token_type_ids (int32) so Merge can locate the image rows after the text embedding runs.
  if (OrtValue* token_type_ids = FindExtraInput(extra_inputs, Config::Defaults::TokenTypeIdsName)) {
    const size_t count = token_type_ids->GetTensorTypeAndShapeInfo()->GetElementCount();
    const int32_t* data = token_type_ids->GetTensorData<int32_t>();
    token_type_ids_.assign(data, data + count);
  }
}

void Gemma3OpenVINOVisionMerger::Merge(OrtValue& inputs_embeds, const OrtValue& /*input_ids*/) {
  if (!image_features_) return;  // text-only prompt

  // Gemma marks image placeholder tokens with token_type_ids == 1.
  std::vector<int64_t> image_token_rows;
  image_token_rows.reserve(token_type_ids_.size());
  for (size_t i = 0; i < token_type_ids_.size(); ++i) {
    if (token_type_ids_[i] == 1) image_token_rows.push_back(static_cast<int64_t>(i));
  }

  MergeImageFeaturesIntoEmbeddings(inputs_embeds, *image_features_, image_token_rows);
}

}  // namespace

std::unique_ptr<OpenVINOVisionMerger> CreateOpenVINOVisionMerger(const OpenVINOMultiModalModel& model,
                                                                 const GeneratorParams& /*params*/) {
  const std::string& type = model.config_->model.type;
  if (type == "gemma3") {
    return std::make_unique<Gemma3OpenVINOVisionMerger>(model);
  }
  throw std::runtime_error("No OpenVINO vision merger registered for model type: " + type);
}

}  // namespace Generators
