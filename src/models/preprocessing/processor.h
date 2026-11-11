// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <array>
#include <cstdint>

#include "image_processor.h"
#include "ortx_cpp_helper.h"
#include "speech_extractor.h"
#include "models/utils.h"
#include "generator/generators.h"
#include "models/model.h"

namespace Generators {

struct ImagePositionLayout {
  // Relative to the next text coordinate; extent controls text after the image.
  std::array<int64_t, 3> origin{};
  int64_t width{};
  int64_t extent{};
};

inline constexpr size_t kMaxEncodedImageBytes = 512 * 1024 * 1024;

struct Images {
  struct Info {
    size_t encoded_bytes{};
    uint32_t width{};
    uint32_t height{};
  };
  Images() = delete;
  Images(ort_extensions::OrtxObjectPtr<OrtxRawImages> images, size_t num_images, std::vector<Info> info = {})
      : images_(std::move(images)), num_images_{num_images}, info_{std::move(info)} {}

  ort_extensions::OrtxObjectPtr<OrtxRawImages> images_;
  size_t num_images_{};
  std::vector<Info> info_;
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
class ThreadPool;

// A null pool keeps same-type copies on the direct sequential path.
template <typename T>
std::unique_ptr<OrtValue> ProcessTensor(ThreadPool* thread_pool, OrtxTensor* tensor,
                                        Ort::Allocator& allocator);

template <typename SrcT, typename DstT>
std::unique_ptr<OrtValue> ProcessTensor(ThreadPool* thread_pool, OrtxTensor* tensor,
                                        Ort::Allocator& allocator);

// Helper to emplace a processed tensor with correct type dispatch (float, bf16, fp16)
void EmplaceProcessedTensor(ThreadPool* thread_pool, NamedTensors& tensors,
                            std::string_view name,
                            OrtxTensor* tensor, ONNXTensorElementDataType type,
                            Ort::Allocator& allocator);

struct Processor {
  explicit Processor(ThreadPool* thread_pool = nullptr) : thread_pool_{thread_pool} {}
  Processor(const Processor&) = delete;
  Processor& operator=(const Processor&) = delete;

  template <typename ProcessorType>
  static std::shared_ptr<Processor> Create(Config& config, const SessionInfo& session_info,
                                           ThreadPool* thread_pool) {
    auto processor = std::make_shared<ProcessorType>(config, session_info);
    processor->thread_pool_ = thread_pool;
    return processor;
  }

  virtual void ConfigureTokenizer(Tokenizer&) const {}
  virtual std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const = 0;
  virtual std::vector<int32_t> ExpandImagePrompt(const Tokenizer&, const std::string&, std::span<const int64_t>) const {
    throw std::runtime_error("This processor does not support packed image prompts.");
  }
  virtual ImagePositionLayout GetImagePositionLayout(const std::array<int64_t, 3>&) const {
    throw std::runtime_error("This processor does not support packed image positions.");
  }

 protected:
  ThreadPool* thread_pool_{};
};

}  // namespace Generators
