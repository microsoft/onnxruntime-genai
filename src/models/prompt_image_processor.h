// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ortx_processor.h"
#include "image_processor.h"
#include "utils.h"

namespace Generators {

struct Images {
  Images() = delete;
  Images(std::unique_ptr<ort_extensions::ImageRawData[]> images, size_t num_images)
      : images_(std::move(images)), num_images_{num_images} {}

  std::unique_ptr<ort_extensions::ImageRawData[]> images_;
  size_t num_images_{};
};

std::unique_ptr<Images> LoadImages(const std::span<const char* const>& image_paths);

struct ImageProcessor {
  ImageProcessor(Config& config, const SessionInfo& session_info);

  // Returned NamedTensors own the OrtValue and are not owned by the caller.
  // OrtValue memory will be released when the NamedTensors are destroyed.
  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const std::string& prompt, const Images* images);

 private:
  OrtxPtr<OrtxProcessor> processor_;

  std::string input_ids_name_;
  std::string pixel_values_name_;
  ONNXTensorElementDataType pixel_values_type_;
  std::string image_sizes_name_;
};

}  // namespace Generators
