// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"

namespace Generators {

namespace {

// Simple literal string count (no regex overhead for fixed tokens)
size_t CountOccurrences(const std::string& text, const std::string& token) {
  size_t count = 0;
  size_t pos = 0;
  while ((pos = text.find(token, pos)) != std::string::npos) {
    ++count;
    pos += token.size();
  }
  return count;
}

// Replace all occurrences of a literal string (avoids std::regex compilation cost)
void ReplaceAll(std::string& text, const std::string& from, const std::string& to) {
  size_t pos = 0;
  while ((pos = text.find(from, pos)) != std::string::npos) {
    text.replace(pos, from.size(), to);
    pos += to.size();
  }
}

std::tuple<std::unique_ptr<OrtValue>, std::unique_ptr<OrtValue>, std::unique_ptr<OrtValue>>
ProcessGemma4ImagePrompt(const Generators::Tokenizer& tokenizer, const std::string& prompt,
                         OrtxTensor* pixel_values, Ort::Allocator& allocator,
                         size_t vision_soft_tokens_per_image,
                         int64_t num_audio_tokens = 0) {
  constexpr char boi_token[] = "<|image>";
  constexpr char image_token[] = "<|image|>";
  constexpr char eoi_token[] = "<image|>";
  constexpr size_t boi_token_len = sizeof(boi_token) - 1;
  constexpr size_t image_token_len = sizeof(image_token) - 1;

  int64_t num_images{};
  if (pixel_values) {
    const float* pixel_values_data{};
    const int64_t* pixel_values_shape{};
    size_t pixel_values_num_dims;
    CheckResult(OrtxGetTensorData(pixel_values, reinterpret_cast<const void**>(&pixel_values_data),
                                  &pixel_values_shape, &pixel_values_num_dims));
    num_images = pixel_values_shape[0];
  }

  std::string text = prompt;
  if (num_images > 0) {
    // Count existing boi tokens in prompt
    auto existing_boi_count = static_cast<int64_t>(CountOccurrences(text, boi_token));

    // The chat template may insert <|image|> (image_token) instead of <|image> (boi_token).
    // If we find standalone <|image|> tokens that aren't part of <|image>, treat each as one image.
    if (existing_boi_count == 0) {
      auto image_token_count = static_cast<int64_t>(CountOccurrences(text, image_token));
      if (image_token_count > 0 && image_token_count == num_images) {
        // Replace each standalone <|image|> with <|image> so the expansion logic works
        ReplaceAll(text, image_token, boi_token);
        existing_boi_count = image_token_count;
      }
    }

    if (existing_boi_count == 0) {
      // No image tokens in prompt — auto-insert them before the text
      std::string prefix;
      prefix.reserve(static_cast<size_t>(num_images) * (boi_token_len + 1));
      for (int64_t i = 0; i < num_images; ++i) {
        prefix += boi_token;
        if (i < num_images - 1) prefix += ' ';
      }
      text = prefix + (text.empty() ? "" : " ") + text;
    }
  }

  // Count and validate boi tokens using simple string search
  const auto boi_count = static_cast<int64_t>(CountOccurrences(text, boi_token));
  if (num_images != boi_count) {
    throw std::runtime_error("Prompt contained " + std::to_string(boi_count) + " image tokens but received " +
                             std::to_string(num_images) + " images.");
  }

  // Build the expanded image token sequence with pre-allocated buffer
  std::string image_tokens_expanded;
  image_tokens_expanded.reserve(vision_soft_tokens_per_image * image_token_len);
  for (size_t i = 0; i < vision_soft_tokens_per_image; ++i) {
    image_tokens_expanded += image_token;
  }
  const std::string full_image_sequence = "\n\n" + std::string(boi_token) + image_tokens_expanded + eoi_token + "\n\n";
  ReplaceAll(text, boi_token, full_image_sequence);

  // Expand audio tokens: replace single <|audio|> from chat template with N audio soft tokens
  if (num_audio_tokens > 0) {
    constexpr char boa_token[] = "<|audio>";
    constexpr char audio_token[] = "<|audio|>";
    constexpr char eoa_token[] = "<audio|>";

    std::string audio_tokens_expanded;
    audio_tokens_expanded.reserve(static_cast<size_t>(num_audio_tokens) * (sizeof(audio_token) - 1));
    for (int64_t i = 0; i < num_audio_tokens; ++i) {
      audio_tokens_expanded += audio_token;
    }
    const std::string full_audio_sequence = "\n\n" + std::string(boa_token) + audio_tokens_expanded + eoa_token + "\n\n";

    // Chat template inserts <|audio|> per audio clip — replace with expanded sequence
    auto audio_marker_count = CountOccurrences(text, audio_token);
    if (audio_marker_count > 0) {
      // Replace only the first standalone <|audio|> (avoid replacing expanded image tokens
      // that might share a substring). Use find to locate the first occurrence.
      auto pos = text.find(audio_token);
      if (pos != std::string::npos) {
        text.replace(pos, sizeof(audio_token) - 1, full_audio_sequence);
      }
    } else {
      // No audio marker in prompt — look for <|audio> (boa_token)
      auto boa_count = CountOccurrences(text, boa_token);
      if (boa_count > 0) {
        ReplaceAll(text, boa_token, full_audio_sequence);
      } else {
        // No audio tokens at all — append before the text
        text = full_audio_sequence + text;
      }
    }
  }

  const std::vector<int32_t> input_ids = tokenizer.Encode(text.c_str());
  const auto seq_len = static_cast<int64_t>(input_ids.size());

  auto input_ids_value = OrtValue::CreateTensor<int32_t>(allocator, std::vector<int64_t>{1, seq_len});
  std::copy(input_ids.begin(), input_ids.end(), input_ids_value->GetTensorMutableData<int32_t>());

  auto token_type_ids = OrtValue::CreateTensor<int32_t>(allocator, std::vector<int64_t>{1, seq_len});
  const auto image_token_id = tokenizer.TokenToTokenId(image_token);
  auto* token_type_data = token_type_ids->GetTensorMutableData<int32_t>();
  for (size_t i = 0; i < input_ids.size(); ++i) {
    token_type_data[i] = (input_ids[i] == image_token_id) ? 1 : 0;
  }

  auto num_img_tokens = OrtValue::CreateTensor<int64_t>(allocator, std::vector<int64_t>{1});
  num_img_tokens->GetTensorMutableData<int64_t>()[0] = static_cast<int64_t>(vision_soft_tokens_per_image);

  return {std::move(input_ids_value), std::move(token_type_ids), std::move(num_img_tokens)};
}

// Helper to avoid repeating the float/bf16/fp16 type dispatch for every tensor
void EmplaceProcessedTensor(NamedTensors& tensors, std::string_view name,
                            OrtxTensor* tensor, ONNXTensorElementDataType type,
                            Ort::Allocator& allocator) {
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    tensors.emplace(std::string(name), std::make_shared<Tensor>(ProcessTensor<float>(tensor, allocator)));
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    tensors.emplace(std::string(name), std::make_shared<Tensor>(ProcessTensor<Ort::BFloat16_t>(tensor, allocator)));
  } else {
    tensors.emplace(std::string(name), std::make_shared<Tensor>(ProcessTensor<Ort::Float16_t>(tensor, allocator)));
  }
}

}  // namespace

Gemma4MultiModalProcessor::Gemma4MultiModalProcessor(Config& config, const SessionInfo& session_info)
    : pixel_values_type_{session_info.GetInputDataType(config.model.vision.inputs.pixel_values)} {
  const auto image_processor_config = (config.config_path / fs::path(config.model.vision.config_filename)).string();
  CheckResult(OrtxCreateProcessor(image_processor_.ToBeAssigned(), image_processor_config.c_str()));

  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.embedding.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::PixelValuesName), config.model.vision.inputs.pixel_values);
  config.AddMapping(std::string(Config::Defaults::PixelPositionIdsName), config.model.vision.inputs.pixel_position_ids);

  // Initialize speech/audio processor if config is present
  if (!config.model.speech.config_filename.empty()) {
    auto speech_config_path = config.config_path / fs::path(config.model.speech.config_filename);
    if (fs::exists(speech_config_path)) {
      has_speech_ = true;
      audio_features_type_ = session_info.GetInputDataType(config.model.speech.inputs.audio_embeds);
      CheckResult(OrtxCreateSpeechFeatureExtractor(audio_processor_.ToBeAssigned(), speech_config_path.string().c_str()));

      config.AddMapping(std::string(Config::Defaults::AudioEmbedsName), config.model.speech.inputs.audio_embeds);
    }
  }
}

std::unique_ptr<NamedTensors> Gemma4MultiModalProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  // Text-only path: no images and no audio
  if (!payload.images && !payload.audios) {
    auto [input_ids, token_type_ids, num_img_tokens] =
        ProcessGemma4ImagePrompt(tokenizer, std::string(payload.prompt), nullptr, allocator, vision_soft_tokens_per_image_);
    named_tensors->emplace(Config::Defaults::InputIdsName, std::make_shared<Tensor>(std::move(input_ids)));
    return named_tensors;
  }

  // Process images if present
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> image_result;
  OrtxTensor* pixel_values = nullptr;
  OrtxTensor* pixel_position_ids = nullptr;
  size_t actual_soft_tokens = vision_soft_tokens_per_image_;
  if (payload.images) {
    CheckResult(OrtxImagePreProcess(image_processor_.get(), payload.images->images_.get(), image_result.ToBeAssigned()));
    CheckResult(OrtxTensorResultGetAt(image_result.get(), 0, &pixel_values));

    // pixel_position_ids is the second output from the Gemma4 image preprocessor
    OrtxTensor* temp_tensor = nullptr;
    if (OrtxTensorResultGetAt(image_result.get(), 1, &temp_tensor) == kOrtxOK && temp_tensor) {
      pixel_position_ids = temp_tensor;
    }

    // num_soft_tokens is the third output — the actual number of vision tokens after pooling
    OrtxTensor* num_soft_tokens_tensor = nullptr;
    if (OrtxTensorResultGetAt(image_result.get(), 2, &num_soft_tokens_tensor) == kOrtxOK && num_soft_tokens_tensor) {
      const int64_t* nst_data{};
      const int64_t* nst_shape{};
      size_t nst_dims;
      CheckResult(OrtxGetTensorData(num_soft_tokens_tensor, reinterpret_cast<const void**>(&nst_data),
                                    &nst_shape, &nst_dims));
      if (nst_data && nst_data[0] > 0) {
        actual_soft_tokens = static_cast<size_t>(nst_data[0]);
      }
    }
  }

  // Process audio FIRST to compute num_audio_tokens (needed for prompt token expansion)
  int64_t num_audio_tokens = 0;
  if (payload.audios && has_speech_) {
    ort_extensions::OrtxObjectPtr<OrtxTensorResult> audio_result;
    CheckResult(OrtxFeatureExtraction(audio_processor_.get(), payload.audios->audios_.get(), audio_result.ToBeAssigned()));

    OrtxTensor* audio_features = nullptr;
    CheckResult(OrtxTensorResultGetAt(audio_result.get(), 0, &audio_features));

    EmplaceProcessedTensor(*named_tensors, Config::Defaults::AudioEmbedsName, audio_features, audio_features_type_, allocator);

    // Create input_features_mask: all-True for single-clip inference (no padding)
    // Shape matches audio features: [batch, time] bool
    const float* audio_data{};
    const int64_t* audio_shape{};
    size_t audio_dims;
    CheckResult(OrtxGetTensorData(audio_features, reinterpret_cast<const void**>(&audio_data),
                                  &audio_shape, &audio_dims));
    int64_t time_dim = (audio_dims == 3) ? audio_shape[1] : audio_shape[0];
    int64_t batch_dim = (audio_dims == 3) ? audio_shape[0] : 1;
    {
      auto mask = OrtValue::CreateTensor<bool>(allocator, std::vector<int64_t>{batch_dim, time_dim});
      std::fill_n(mask->GetTensorMutableData<bool>(), batch_dim * time_dim, true);
      named_tensors->emplace(std::string("input_features_mask"),
                             std::make_shared<Tensor>(std::move(mask)));
    }

    // Compute audio_sizes: the speech encoder uses 2-stage Conv2d with stride=2 each
    int64_t t_after_1 = (time_dim - 1) / 2 + 1;
    int64_t t_after_2 = (t_after_1 - 1) / 2 + 1;
    num_audio_tokens = t_after_2;
    std::array<int64_t, 1> audio_sizes_shape = {1};
    auto audio_sizes = OrtValue::CreateTensor<int64_t>(allocator, audio_sizes_shape);
    audio_sizes->GetTensorMutableData<int64_t>()[0] = num_audio_tokens;
    named_tensors->emplace(std::string(Config::Defaults::AudioSizesName),
                           std::make_shared<Tensor>(std::move(audio_sizes)));
  }

  // Process prompt: expand image and audio tokens, then encode
  auto [input_ids, token_type_ids, num_img_tokens] =
      ProcessGemma4ImagePrompt(tokenizer, std::string(payload.prompt), pixel_values, allocator, actual_soft_tokens, num_audio_tokens);
  named_tensors->emplace(std::string(Config::Defaults::InputIdsName), std::make_shared<Tensor>(std::move(input_ids)));
  named_tensors->emplace(std::string(Config::Defaults::TokenTypeIdsName), std::make_shared<Tensor>(std::move(token_type_ids)));

  if (payload.images) {
    // The Gemma4ImageTransform pads pixel_values and position_ids to max_patches.
    // The vision ONNX model expects the actual (unpadded) number of patches.
    // Trim the tensors to actual_patches = actual_soft_tokens * pooling_kernel_size².
    constexpr int64_t kPoolingKernelSize = 3;
    const int64_t actual_patches = static_cast<int64_t>(actual_soft_tokens) * kPoolingKernelSize * kPoolingKernelSize;

    // Get padded pixel_values shape
    const float* pv_data{};
    const int64_t* pv_shape{};
    size_t pv_dims;
    CheckResult(OrtxGetTensorData(pixel_values, reinterpret_cast<const void**>(&pv_data), &pv_shape, &pv_dims));

    // Determine the patches dimension and patch_dim based on tensor rank
    // 2D: (num_patches, patch_dim) or 3D: (batch, num_patches, patch_dim)
    const int64_t num_padded_patches = (pv_dims == 3) ? pv_shape[1] : pv_shape[0];
    const int64_t patch_dim = (pv_dims == 3) ? pv_shape[2] : pv_shape[1];

    if (actual_patches < num_padded_patches) {
      // Trim: copy only the first actual_patches from the padded tensor
      auto trimmed_shape = (pv_dims == 3)
          ? std::vector<int64_t>{pv_shape[0], actual_patches, patch_dim}
          : std::vector<int64_t>{actual_patches, patch_dim};
      auto trimmed_pv = OrtValue::CreateTensor<float>(allocator, trimmed_shape);
      std::memcpy(trimmed_pv->GetTensorMutableData<float>(), pv_data,
                  static_cast<size_t>(actual_patches * patch_dim) * sizeof(float));
      named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                             std::make_shared<Tensor>(std::move(trimmed_pv)));
    } else {
      EmplaceProcessedTensor(*named_tensors, Config::Defaults::PixelValuesName, pixel_values, pixel_values_type_, allocator);
    }

    named_tensors->emplace(std::string(Config::Defaults::NumImageTokens), std::make_shared<Tensor>(std::move(num_img_tokens)));

    // Trim pixel_position_ids similarly
    if (pixel_position_ids) {
      const int64_t* pos_data{};
      const int64_t* pos_shape{};
      size_t pos_dims;
      CheckResult(OrtxGetTensorData(pixel_position_ids, reinterpret_cast<const void**>(&pos_data), &pos_shape, &pos_dims));

      const int64_t num_padded_pos = (pos_dims == 3) ? pos_shape[1] : pos_shape[0];
      const int64_t pos_last_dim = (pos_dims == 3) ? pos_shape[2] : pos_shape[1];

      if (actual_patches < num_padded_pos) {
        auto trimmed_pos_shape = (pos_dims == 3)
            ? std::vector<int64_t>{pos_shape[0], actual_patches, pos_last_dim}
            : std::vector<int64_t>{actual_patches, pos_last_dim};
        auto trimmed_pos = OrtValue::CreateTensor<int64_t>(allocator, trimmed_pos_shape);
        std::memcpy(trimmed_pos->GetTensorMutableData<int64_t>(), pos_data,
                    static_cast<size_t>(actual_patches * pos_last_dim) * sizeof(int64_t));
        named_tensors->emplace(std::string(Config::Defaults::PixelPositionIdsName),
                               std::make_shared<Tensor>(std::move(trimmed_pos)));
      } else {
        named_tensors->emplace(std::string(Config::Defaults::PixelPositionIdsName),
                               std::make_shared<Tensor>(ProcessTensor<int64_t>(pixel_position_ids, allocator)));
      }
    }
  }

  return named_tensors;
}

}  // namespace Generators
