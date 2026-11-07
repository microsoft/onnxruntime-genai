// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <array>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "generator/generators.h"

namespace Generators {

struct DecoderOnly_Model;
struct Images;

struct MediaAccounting {
  size_t host_bytes{};
  size_t device_bytes{};
};

struct MediaItem {
  ~MediaItem();

  std::array<int64_t, 3> grid;
  std::shared_ptr<Tensor> pixels;
  std::shared_ptr<Tensor> features;
  std::shared_ptr<MediaAccounting> accounting;
  size_t host_bytes{};
  size_t feature_bytes{};
  size_t rows{};
  int32_t prefix_base{};
  void ReleaseFeatures();
};

struct MediaSpan {
  size_t start{};
  std::shared_ptr<MediaItem> item;
};

struct PreparedMediaTurn {
  std::vector<int32_t> tokens;
  std::vector<MediaSpan> media;
};

std::array<int64_t, 3> MediaPosition(std::span<const MediaSpan> media, size_t row, size_t merge_size);
void MediaPositions(std::span<const MediaSpan> media, size_t first_row, size_t merge_size,
                    const std::array<std::span<int64_t>, 3>& positions);
std::array<int64_t, 3> MediaImageGrid(uint32_t width, uint32_t height, size_t patch, size_t merge,
                                      int64_t min_pixels, int64_t max_pixels);

class MediaStore {
 public:
  MediaStore(std::shared_ptr<DecoderOnly_Model> model, size_t host_limit, size_t device_limit);
  PreparedMediaTurn Prepare(const std::string& prompt, std::span<const int32_t> raw_tokens,
                            const Images* images, size_t first_row, size_t remaining_rows);
  void ValidateTextTokens(std::span<const int32_t> tokens) const;
  void ReserveItems(size_t count);
  void Retain(std::span<const MediaSpan> media) noexcept;
  void Encode(const std::shared_ptr<MediaItem>& item);
  void ReleaseFeatures();

 private:
  void MakeHostRoom(size_t bytes);
  std::shared_ptr<DecoderOnly_Model> model_;
  std::shared_ptr<MediaAccounting> accounting_{std::make_shared<MediaAccounting>()};
  std::vector<std::shared_ptr<MediaItem>> items_;
  size_t host_limit_;
  size_t device_limit_;
  int64_t next_prefix_row_{1};
  int64_t min_pixels_{};
  int64_t max_pixels_{};
  int32_t vision_end_token_{};
};

}  // namespace Generators
