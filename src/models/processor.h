// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "image_processor.h"
#include "ortx_cpp_helper.h"
#include "speech_extractor.h"
#include "utils.h"
#include "../generators.h"
#include "model.h"

namespace Generators {

struct Images {
  Images() = delete;
  Images(ort_extensions::OrtxObjectPtr<OrtxRawImages> images, size_t num_images)
      : images_(std::move(images)), num_images_{num_images} {}

  ort_extensions::OrtxObjectPtr<OrtxRawImages> images_;
  size_t num_images_{};
};

std::unique_ptr<Images> LoadImages(std::span<const char* const> image_paths);
std::unique_ptr<Images> LoadImagesFromBuffers(std::span<const void*> image_data, std::span<const size_t> image_data_sizes);

struct Audios {
  Audios(ort_extensions::OrtxObjectPtr<OrtxRawAudios> audios, size_t num_audios)
      : audios_(std::move(audios)), num_audios_{num_audios} {}

  Audios() = delete;
  Audios(const Audios&) = delete;
  Audios& operator=(const Audios&) = delete;

  ort_extensions::OrtxObjectPtr<OrtxRawAudios> audios_;
  size_t num_audios_{};
};

std::unique_ptr<Audios> LoadAudios(const std::span<const char* const>& audio_paths);
std::unique_ptr<Audios> LoadAudiosFromBuffers(std::span<const void*> audio_data, std::span<const size_t> audio_data_sizes);

struct Payload {
  const std::string& prompt;
  std::span<const char*> prompts;
  const Images* images;
  const Audios* audios;
};

struct Config;
struct SessionInfo;

template <typename T>
std::unique_ptr<OrtValue> ProcessTensor(OrtxTensor* tensor, Ort::Allocator& allocator);

template <typename SrcT, typename DstT>
std::unique_ptr<OrtValue> ProcessTensor(OrtxTensor* tensor, Ort::Allocator& allocator);
struct Processor {
  Processor() = default;
  Processor(const Processor&) = delete;
  Processor& operator=(const Processor&) = delete;

  template <typename ProcessorType>
  static std::shared_ptr<Processor> Create(Config& config, const SessionInfo& session_info) {
    return std::make_shared<ProcessorType>(config, session_info);
  }

  virtual std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const = 0;
};

}  // namespace Generators
