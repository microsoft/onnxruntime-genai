// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "processor.h"
#include "vision_preparation.h"
#include "vision_media_prompt.h"
#include "json.h"

namespace Generators {

struct QwenImageProcessor;

struct PatchProcessingLimits : JSON::Element {
  int64_t min_pixels{}, max_pixels{}, patch_size{}, temporal_patch_size{}, merge_size{}, smart_resize{};
  int64_t max_video_tokens{}, total_pixels{};
  bool bgr_normalization{};
  std::string coefficient_type, interpolation;
  std::vector<std::string> operations;
  Element& OnObject(std::string_view) override { return *this; }
  Element& OnArray(std::string_view) override { return *this; }
  void OnValue(std::string_view name, JSON::Value value) override;
};

std::string ReadPatchProcessor(const Config& config, const std::string& filename, PatchProcessingLimits& limits);

struct PatchVisionPreparation : VisionPreparation {
  PatchVisionPreparation(const Config& config, const SessionInfo& info,
                         const Tokenizer& tokenizer, const QwenImageProcessor& images, ThreadPool* thread_pool);
  size_t Estimate(const VisionPreparationInput& input, size_t host_limit) const override;
  std::unique_ptr<const VisionPlan> Plan(const Tokenizer& tokenizer,
                                         const VisionPreparationInput& input, size_t host_limit) const override;
  std::vector<NamedTensors> Materialize(const VisionPlan& plan) const override;
  void ValidateTextTokens(std::span<const int32_t> tokens) const override;
  virtual std::array<int64_t, 3> ImageGrid(uint32_t width, uint32_t height) const = 0;
  virtual std::array<int64_t, 3> VideoGrid(uint32_t width, uint32_t height, size_t frames) const = 0;
  virtual size_t PromptBytes(std::string_view prompt, const VisionMediaPromptSource& source, size_t maximum_bytes) const = 0;
  virtual std::string ExpandPrompt(std::string_view prompt, std::span<const std::array<int64_t, 3>> images,
                                   std::span<const VisionVideoPrompt> videos, size_t maximum_bytes,
                                   std::vector<size_t>& unit_order) const = 0;
  virtual std::shared_ptr<const VisionPosition> Position(std::array<int64_t, 3> grid) const = 0;

 protected:
  void InitializeVideo(const std::string& definition);
  virtual std::shared_ptr<Tensor> ConvertVideoPixels(OrtxTensor& pixels) const;
  const Config& config_;
  const QwenImageProcessor& images_;
  ThreadPool* thread_pool_;
  int64_t min_pixels_{}, max_pixels_{};
  int64_t pixel_width_{};
  ONNXTensorElementDataType pixel_type_{};
  int32_t vision_end_token_{}, video_token_{};
  ort_extensions::OrtxObjectPtr<OrtxProcessor> videos_;
};

}  // namespace Generators
