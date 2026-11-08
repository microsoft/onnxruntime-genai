// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qwen_vision_processing.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace Generators {
namespace {

size_t AddBytes(size_t left, size_t right) {
  if (right > std::numeric_limits<size_t>::max() - left)
    throw std::overflow_error("Media allocation size overflows size_t.");
  return left + right;
}

size_t MultiplyBytes(size_t left, size_t right) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
    throw std::overflow_error("Media allocation size overflows size_t.");
  return left * right;
}

size_t RoundToEvenMultiple(size_t value, size_t factor) {
  const size_t quotient = value / factor, remainder = value % factor;
  const bool increment = remainder > factor - remainder || (remainder == factor - remainder && quotient % 2 != 0);
  return MultiplyBytes(quotient + increment, factor);
}

size_t CheckedDimension(double value) {
  if (!std::isfinite(value) || value < 0 || value >= static_cast<double>(std::numeric_limits<size_t>::max()))
    throw std::overflow_error("Video resize dimension overflows size_t.");
  return static_cast<size_t>(value);
}

size_t MergedGridRows(const std::array<int64_t, 3>& grid, size_t merge) {
  if (merge == 0 || merge > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      grid[0] <= 0 || grid[1] <= 0 || grid[2] <= 0 ||
      static_cast<uint64_t>(grid[1]) > std::numeric_limits<size_t>::max() ||
      static_cast<uint64_t>(grid[2]) > std::numeric_limits<size_t>::max() ||
      grid[1] % static_cast<int64_t>(merge) != 0 || grid[2] % static_cast<int64_t>(merge) != 0)
    throw std::invalid_argument("Media prompt requires a complete spatially merged grid.");
  return MultiplyBytes(static_cast<size_t>(grid[1]) / merge, static_cast<size_t>(grid[2]) / merge);
}

struct SpanPromptSource : QwenMediaPromptSource {
  SpanPromptSource(std::span<const std::array<int64_t, 3>> images, std::span<const QwenVideoPrompt> videos)
      : images{images}, videos{videos} {}
  size_t ImageCount() const override { return images.size(); }
  size_t VideoCount() const override { return videos.size(); }
  std::array<int64_t, 3> ImageGrid(size_t image) const override { return images[image]; }
  std::array<int64_t, 3> VideoGrid(size_t video) const override { return videos[video].grid; }
  size_t FrameCount(size_t video) const override { return videos[video].timestamps.size(); }
  double Timestamp(size_t video, size_t frame) const override { return videos[video].timestamps[frame]; }
  std::span<const std::array<int64_t, 3>> images;
  std::span<const QwenVideoPrompt> videos;
};

size_t BuildQwenMediaPrompt(std::string_view prompt, const QwenMediaPromptSource& source, size_t merge_size,
                            size_t maximum_bytes, std::string* expanded) {
  constexpr std::string_view start = "<|vision_start|>", end = "<|vision_end|>";
  constexpr std::string_view image = "<|image_pad|>", video = "<|video_pad|>";
  constexpr std::string_view image_marker = "<|vision_start|><|image_pad|><|vision_end|>";
  constexpr std::string_view video_marker = "<|vision_start|><|video_pad|><|vision_end|>";
  size_t bytes = 0;
  const auto append = [&](std::string_view text, size_t repeats) {
    bytes = AddBytes(bytes, MultiplyBytes(text.size(), repeats));
    if (bytes > maximum_bytes)
      throw std::invalid_argument("Expanded media prompt exceeds its host byte limit.");
    if (expanded)
      for (size_t index = 0; index < repeats; ++index) expanded->append(text);
  };
  size_t image_index = 0, video_index = 0;
  while (!prompt.empty()) {
    const auto marker = prompt.find(start);
    const auto text = prompt.substr(0, marker);
    if (text.find(image) != std::string_view::npos || text.find(video) != std::string_view::npos ||
        text.find(end) != std::string_view::npos)
      throw std::invalid_argument("Media markers must form complete ordered items.");
    append(text, 1);
    if (marker == std::string_view::npos) break;
    prompt.remove_prefix(marker);
    if (prompt.substr(0, image_marker.size()) == image_marker) {
      if (image_index == source.ImageCount() || source.ImageGrid(image_index)[0] != 1)
        throw std::invalid_argument("Image marker count or temporal grid does not match the ordered inputs.");
      const auto rows = MergedGridRows(source.ImageGrid(image_index++), merge_size);
      append(start, 1);
      append(image, rows);
      append(end, 1);
      prompt.remove_prefix(image_marker.size());
    } else if (prompt.substr(0, video_marker.size()) == video_marker) {
      if (video_index == source.VideoCount())
        throw std::invalid_argument("Video marker count does not match the ordered inputs.");
      const auto clip = video_index++;
      const auto grid = source.VideoGrid(clip);
      const auto frames = source.FrameCount(clip);
      const auto rows = MergedGridRows(grid, merge_size);
      if (frames < 2 ||
          static_cast<uint64_t>(grid[0]) != frames / 2 + frames % 2)
        throw std::invalid_argument("Video temporal grid does not match its selected frame count.");
      double previous = 0;
      for (size_t index = 0; index < frames; ++index) {
        const auto timestamp = source.Timestamp(clip, index);
        if (!std::isfinite(timestamp) || timestamp < 0 ||
            (index != 0 && timestamp < previous))
          throw std::invalid_argument("Video presentation timestamps must be finite, nonnegative and ordered.");
        previous = timestamp;
      }
      append(start, 1);
      for (size_t group = 0; group < static_cast<size_t>(grid[0]); ++group) {
        const auto timestamp =
            (source.Timestamp(clip, group * 2) + source.Timestamp(clip, std::min(group * 2 + 1, frames - 1))) / 2;
        if (!std::isfinite(timestamp))
          throw std::invalid_argument("Video group timestamp overflows.");
        std::ostringstream label;
        label.imbue(std::locale::classic());
        label << '<' << std::fixed << std::setprecision(1) << timestamp << " seconds>";
        append(label.str(), 1);
        append(start, 1);
        append(video, rows);
        append(end, 1);
      }
      append(end, 1);
      prompt.remove_prefix(video_marker.size());
    } else {
      throw std::invalid_argument("Each media item requires a complete vision-start/pad/vision-end marker.");
    }
  }
  if (image_index != source.ImageCount() || video_index != source.VideoCount())
    throw std::invalid_argument("Media marker counts do not match the ordered inputs.");
  return bytes;
}

}  // namespace

size_t QwenMediaPromptBytes(std::string_view prompt,
                            std::span<const std::array<int64_t, 3>> image_grids,
                            std::span<const QwenVideoPrompt> videos, size_t merge_size,
                            size_t maximum_bytes) {
  return QwenMediaPromptBytes(prompt, SpanPromptSource{image_grids, videos}, merge_size, maximum_bytes);
}

size_t QwenMediaPromptBytes(std::string_view prompt, const QwenMediaPromptSource& source,
                            size_t merge_size, size_t maximum_bytes) {
  return BuildQwenMediaPrompt(prompt, source, merge_size, maximum_bytes, nullptr);
}

std::string ExpandQwenMediaPrompt(std::string_view prompt,
                                  std::span<const std::array<int64_t, 3>> image_grids,
                                  std::span<const QwenVideoPrompt> videos, size_t merge_size,
                                  size_t maximum_bytes) {
  const auto bytes = QwenMediaPromptBytes(prompt, image_grids, videos, merge_size, maximum_bytes);
  std::string expanded;
  expanded.reserve(bytes);
  BuildQwenMediaPrompt(prompt, SpanPromptSource{image_grids, videos}, merge_size, maximum_bytes, &expanded);
  return expanded;
}

std::array<int64_t, 3> QwenImageGrid(uint32_t width, uint32_t height, size_t patch, size_t merge,
                                     int64_t min_pixels, int64_t max_pixels) {
  if (width == 0 || height == 0 || width > INT32_MAX || height > INT32_MAX ||
      patch == 0 || merge == 0 || min_pixels <= 0 || max_pixels < min_pixels)
    throw std::invalid_argument("Invalid dimensions or resize limits for packed vision.");
  if (static_cast<double>(std::max(width, height)) / std::min(width, height) > 200.0)
    throw std::invalid_argument("Image aspect ratio exceeds the Qwen preprocessing limit.");
  const size_t factor = MultiplyBytes(patch, merge);
  const auto rounded = [factor](size_t dimension) { return std::max(factor, RoundToEvenMultiple(dimension, factor)); };
  size_t resized_height = rounded(height), resized_width = rounded(width);
  const size_t source_pixels = MultiplyBytes(width, height);
  if (MultiplyBytes(resized_height, resized_width) > static_cast<size_t>(max_pixels)) {
    const double scale = std::sqrt(static_cast<double>(source_pixels) / max_pixels);
    resized_height = static_cast<size_t>(std::floor(static_cast<size_t>(height / scale) / static_cast<double>(factor))) * factor;
    resized_width = static_cast<size_t>(std::floor(static_cast<size_t>(width / scale) / static_cast<double>(factor))) * factor;
  } else if (MultiplyBytes(resized_height, resized_width) < static_cast<size_t>(min_pixels)) {
    const double scale = std::sqrt(static_cast<double>(min_pixels) / source_pixels);
    resized_height = static_cast<size_t>(std::ceil(height * scale / factor)) * factor;
    resized_width = static_cast<size_t>(std::ceil(width * scale / factor)) * factor;
  }
  if (resized_height == 0 || resized_width == 0)
    throw std::invalid_argument("Image geometry cannot form complete patches under the resize limits.");
  return {1, static_cast<int64_t>(resized_height / patch), static_cast<int64_t>(resized_width / patch)};
}

std::array<int64_t, 3> QwenVideoGrid(uint32_t width, uint32_t height, size_t frames,
                                     size_t patch, size_t merge, int64_t min_pixels,
                                     int64_t max_pixels, size_t max_video_tokens) {
  constexpr size_t temporal = 2;
  if (width == 0 || height == 0 || width > INT32_MAX || height > INT32_MAX ||
      frames < temporal || patch == 0 || merge == 0 || min_pixels <= 0 ||
      max_pixels < min_pixels || max_video_tokens == 0)
    throw std::invalid_argument("Invalid dimensions or resize limits for packed video.");
  if constexpr (sizeof(size_t) < sizeof(uint64_t)) {
    if (static_cast<uint64_t>(max_pixels) > std::numeric_limits<size_t>::max())
      throw std::invalid_argument("Video pixel limit exceeds size_t.");
  }
  const size_t factor = MultiplyBytes(patch, merge);
  if (factor > INT32_MAX)
    throw std::invalid_argument("Video patch geometry exceeds the processor dimension limit.");
  size_t source_height = height, source_width = width;
  if (source_height < factor || source_width < factor) {
    const double scale = std::max(static_cast<double>(factor) / source_height,
                                  static_cast<double>(factor) / source_width);
    source_height = CheckedDimension(source_height * scale);
    source_width = CheckedDimension(source_width * scale);
  }
  if (static_cast<double>(std::max(source_height, source_width)) / std::min(source_height, source_width) > 200.0)
    throw std::invalid_argument("Video aspect ratio exceeds the Qwen preprocessing limit.");
  const size_t frame_cap = MultiplyBytes(max_video_tokens, MultiplyBytes(factor, factor));
  const size_t per_frame = std::max(std::min(frame_cap, static_cast<size_t>(max_pixels) / frames),
                                    CheckedDimension(static_cast<double>(min_pixels) * 1.05));
  const size_t clip_cap = MultiplyBytes(per_frame, frames);
  size_t resized_height = RoundToEvenMultiple(source_height, factor);
  size_t resized_width = RoundToEvenMultiple(source_width, factor);
  // Resize budgeting rounds the temporal count to even; patch padding instead rounds up.
  const size_t rounded_frames = RoundToEvenMultiple(frames, temporal);
  const size_t rounded_pixels = MultiplyBytes(rounded_frames, MultiplyBytes(resized_height, resized_width));
  const size_t source_pixels = MultiplyBytes(frames, MultiplyBytes(source_height, source_width));
  if (rounded_pixels > clip_cap) {
    const double scale = std::sqrt(static_cast<double>(source_pixels) / clip_cap);
    resized_height = std::max(factor, MultiplyBytes(CheckedDimension(std::floor(source_height / scale / factor)), factor));
    resized_width = std::max(factor, MultiplyBytes(CheckedDimension(std::floor(source_width / scale / factor)), factor));
  } else if (rounded_pixels < static_cast<size_t>(min_pixels)) {
    const double scale = std::sqrt(static_cast<double>(min_pixels) / source_pixels);
    resized_height = MultiplyBytes(CheckedDimension(std::ceil(source_height * scale / factor)), factor);
    resized_width = MultiplyBytes(CheckedDimension(std::ceil(source_width * scale / factor)), factor);
  }
  if (resized_height == 0 || resized_width == 0 || resized_height > INT32_MAX || resized_width > INT32_MAX / 4)
    throw std::invalid_argument("Video resize exceeds the processor dimension limit.");
  return {static_cast<int64_t>(frames / temporal + (frames % temporal != 0)),
          static_cast<int64_t>(resized_height / patch), static_cast<int64_t>(resized_width / patch)};
}

}  // namespace Generators
