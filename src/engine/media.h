// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <array>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "generator/generators.h"
#include "ortx_cpp_helper.h"
#include "ortx_processor.h"
#include "models/preprocessing/vision_position.h"

namespace Generators {

struct DecoderOnly_Model;
struct Images;
struct Videos;
struct MediaPrefix;

struct MediaAccounting {
  size_t host_bytes{};
  size_t device_bytes{};
};

struct MediaAllocation {
  ~MediaAllocation();
  std::shared_ptr<MediaAccounting> accounting;
  size_t host_bytes{};
};

struct MediaItem {
  ~MediaItem();

  NamedTensors inputs;
  std::shared_ptr<Tensor> features;
  std::shared_ptr<MediaAccounting> accounting;
  size_t host_bytes{};
  size_t feature_bytes{};
  size_t rows{};
  void ReleaseFeatures();
};

struct MediaSpan {
  MediaSpan(size_t start, std::shared_ptr<MediaItem> item,
            std::shared_ptr<const VisionPosition> positions = {},
            size_t rows = 0, size_t feature_offset = 0)
      : start{start}, item{std::move(item)}, positions{std::move(positions)}, rows{rows ? rows : (this->item ? this->item->rows : 0)}, feature_offset{feature_offset} {
    Validate();
  }

  void Validate() const {
    if (!item || rows == 0 ||
        rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        feature_offset > item->rows ||
        rows > item->rows - feature_offset ||
        start > std::numeric_limits<size_t>::max() - rows ||
        start > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - rows ||
        (positions && positions->rows != rows))
      throw std::invalid_argument("Media span does not match its feature or position range.");
  }

  size_t End() const noexcept { return start + rows; }

  size_t start{};
  std::shared_ptr<MediaItem> item;
  std::shared_ptr<const VisionPosition> positions;
  std::shared_ptr<const MediaPrefix> prefix;
  std::shared_ptr<MediaAllocation> position_storage;
  size_t rows{};
  size_t feature_offset{};
};

struct MediaPrefix {
  ~MediaPrefix();
  bool Matches(const MediaSpan& span, std::span<const int32_t> values, int64_t base) const;

  std::shared_ptr<MediaItem> item;
  std::shared_ptr<const VisionPosition> positions;
  std::vector<int32_t> tokens;
  size_t feature_offset{};
  int64_t position_base{};
  int32_t first_id{};
  std::shared_ptr<MediaAccounting> accounting;
  size_t host_bytes{};
};

struct PreparedMediaTurn {
  PreparedMediaTurn() = default;
  PreparedMediaTurn(PreparedMediaTurn&&) = default;
  PreparedMediaTurn& operator=(PreparedMediaTurn&&) = default;
  PreparedMediaTurn(const PreparedMediaTurn&) = delete;
  PreparedMediaTurn& operator=(const PreparedMediaTurn&) = delete;

  std::vector<int32_t> tokens;
  std::vector<MediaSpan> media;
  std::shared_ptr<MediaAllocation> storage;
};

std::array<int64_t, 3> MediaPosition(std::span<const MediaSpan> media, size_t row);
void MediaPositions(std::span<const MediaSpan> media, size_t first_row,
                    const std::array<std::span<int64_t>, 3>& positions);
void ApplyMediaFeatures(std::span<const MediaSpan> media, size_t first_row, size_t rows,
                        Tensor& embeddings, size_t packed_row);
class MediaStore {
 public:
  MediaStore(std::shared_ptr<DecoderOnly_Model> model, size_t host_limit, size_t device_limit);
  PreparedMediaTurn Prepare(const std::string& prompt, std::span<const int32_t> raw_tokens,
                            const Images* images, size_t first_row, size_t remaining_rows,
                            const Videos* videos = nullptr);
  void ValidateTextTokens(std::span<const int32_t> tokens) const;
  void Identify(PreparedMediaTurn& turn, size_t first_row, std::span<const MediaSpan> history);
  void ReserveItems(size_t count);
  void Retain(std::span<const MediaSpan> media) noexcept;
  void Encode(const std::shared_ptr<MediaItem>& item);
  void ReleaseFeatures();
  std::shared_ptr<MediaAllocation> ReserveHost(size_t bytes);

 private:
  void MakeHostRoom(size_t bytes);
  std::shared_ptr<MediaAllocation> TransferHost(size_t bytes, size_t& reservation);
  void ValidateInput(const std::string& name, ONNXTensorElementDataType type,
                     std::span<const int64_t> shape) const;
  void ValidateInputs(const NamedTensors& inputs) const;
  std::shared_ptr<MediaItem> Intern(NamedTensors inputs, size_t rows,
                                    std::span<const MediaSpan> pending, size_t& reservation);
  std::shared_ptr<DecoderOnly_Model> model_;
  std::shared_ptr<MediaAccounting> accounting_{std::make_shared<MediaAccounting>()};
  std::shared_ptr<MediaAllocation> item_storage_;
  std::shared_ptr<MediaAllocation> prefix_storage_;
  std::vector<std::shared_ptr<MediaItem>> items_;
  std::vector<std::shared_ptr<const MediaPrefix>> prefixes_;
  size_t host_limit_;
  size_t device_limit_;
  int64_t next_prefix_row_{1};
};

}  // namespace Generators
