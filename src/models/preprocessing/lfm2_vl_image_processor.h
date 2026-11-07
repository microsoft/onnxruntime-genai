// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "models/preprocessing/processor.h"

namespace Generators {

// LFM2-VL special tokens. The image token is repeated once per projected vision feature and the
// whole run is wrapped in the start/end markers, matching `Lfm2VlProcessor._build_image_tokens`.
inline constexpr char kLfm2VlImageToken[] = "<image>";
inline constexpr char kLfm2VlImageStartToken[] = "<|image_start|>";
inline constexpr char kLfm2VlImageEndToken[] = "<|image_end|>";

// Patch/token geometry of a single image after the ort-extensions smart resize.
//
// The SigLIP2 NaViT encoder consumes a flat sequence of `encoder_patch_size` squared patches, and
// the multi-modal projector then pixel-unshuffles that grid by `downsample_factor` in both
// directions, so one decoder token covers a downsample_factor x downsample_factor block of patches.
struct Lfm2VlImageGeometry {
  int64_t patch_rows{};   // image height in encoder patches
  int64_t patch_cols{};   // image width in encoder patches
  int64_t num_patches{};  // patch_rows * patch_cols: sequence length seen by the vision encoder
  int64_t num_tokens{};   // decoder tokens the projector emits for this image
};

// Throws if the resized image is not a whole number of patches; smart resize always rounds to a
// multiple of encoder_patch_size * downsample_factor, so a violation means the preprocessing
// pipeline in processor_config.json disagrees with the vision config in genai_config.json.
Lfm2VlImageGeometry ComputeLfm2VlImageGeometry(int64_t image_height, int64_t image_width,
                                               int64_t encoder_patch_size, int64_t downsample_factor);

// Flattens one image into its patch sequence, the layout `convert_image_to_patches` produces:
// patch (row, col) holds the encoder_patch_size square at that grid position, ordered [y][x][channel].
// `image` points at this image inside the zero-padded [N, C, padded_height, padded_width] batch, so
// rows are padded_width apart; `destination` receives num_patches * patch_size^2 * channels floats.
void WriteLfm2VlImagePatches(ThreadPool* thread_pool, const float* image, int64_t channels,
                             int64_t padded_height, int64_t padded_width,
                             const Lfm2VlImageGeometry& geometry, int64_t encoder_patch_size,
                             float* destination);

// "<|image_start|>" + "<image>" * num_tokens + "<|image_end|>"
std::string BuildLfm2VlImagePlaceholder(int64_t num_tokens);

// Replaces the i-th "<image>" in the prompt with the placeholder run for image i. Throws unless the
// prompt holds exactly one "<image>" per image, the rule Lfm2VlProcessor.validate_inputs applies, so
// the vision features and the placeholder runs can never be misaligned or silently dropped.
std::string ExpandLfm2VlImageTokens(const std::string& prompt, const std::vector<int64_t>& tokens_per_image);

// Helpers shared with Lfm3VlImageProcessor. `processor_name` starts every error message.

std::unique_ptr<OrtValue> MakeInt64Tensor(const std::vector<int64_t>& values, std::vector<int64_t> shape,
                                          Ort::Allocator& allocator);
std::unique_ptr<OrtValue> MakeInputIds(const std::vector<int32_t>& input_ids, Ort::Allocator& allocator);

// The vision graph's input names come from genai_config.json; a wrong name would otherwise be
// dropped silently by ExtraInputs and only surface as ORT's "Missing Input" at run time.
ONNXTensorElementDataType VisionInputType(const SessionInfo& session_info, const std::string& name,
                                          const char* config_field, const char* processor_name);

// Integer side inputs are always emitted as int64, so fail at load rather than at the first image
// if the graph was exported with another integer type.
void RequireInt64VisionInput(const SessionInfo& session_info, const std::string& name, const char* config_field,
                             const char* processor_name);

// The list overload of MultiModalProcessor::Process fills payload.prompts and leaves payload.prompt
// empty. Batching is not supported, so a single-entry list is the prompt and anything longer is an error
// rather than silently dropped text.
std::string ResolvePrompt(const Payload& payload, const char* processor_name);

struct Lfm2VlImageProcessor : Processor {
  Lfm2VlImageProcessor(Config& config, const SessionInfo& session_info);

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ort_extensions::OrtxObjectPtr<OrtxProcessor> processor_;

  ONNXTensorElementDataType pixel_values_type_;

  int64_t encoder_patch_size_{};
  int64_t downsample_factor_{};
  int64_t max_num_patches_{};
};

}  // namespace Generators
