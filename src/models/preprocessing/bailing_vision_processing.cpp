// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bailing_vision_processing.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace Generators {
namespace {

size_t Add(size_t left, size_t right) {
  if (right > std::numeric_limits<size_t>::max() - left)
    throw std::overflow_error("Bailing media size overflows size_t.");
  return left + right;
}

size_t Multiply(size_t left, size_t right) {
  if (left && right > std::numeric_limits<size_t>::max() / left)
    throw std::overflow_error("Bailing media size overflows size_t.");
  return left * right;
}

size_t Rounded(size_t value, size_t factor) {
  const auto quotient = value / factor, remainder = value % factor;
  return Multiply(quotient + (remainder > factor - remainder ||
                              (remainder == factor - remainder && quotient % 2)),
                  factor);
}

size_t Dimension(double value, size_t factor) {
  if (!std::isfinite(value) || value < 0 || value > INT32_MAX / static_cast<double>(factor))
    throw std::invalid_argument("Bailing resize exceeds the processor dimension limit.");
  return Multiply(static_cast<size_t>(value), factor);
}

std::array<int64_t, 3> SpatialGrid(uint32_t width, uint32_t height, size_t patch, size_t merge,
                                   double min_pixels, double max_pixels, bool video) {
  if (!width || !height || width > INT32_MAX || height > INT32_MAX || !patch || !merge ||
      min_pixels <= 0 || max_pixels < min_pixels || !std::isfinite(max_pixels))
    throw std::invalid_argument("Invalid Bailing geometry or pixel limits.");
  if (static_cast<double>(std::max(width, height)) / std::min(width, height) > 200)
    throw std::invalid_argument("Bailing media aspect ratio exceeds 200.");
  const auto factor = Multiply(patch, merge);
  if (factor > INT32_MAX)
    throw std::invalid_argument("Bailing patch geometry exceeds the processor dimension limit.");
  auto h = Rounded(height, factor), w = Rounded(width, factor);
  if (video) {
    h = std::max(factor, h);
    w = std::max(factor, w);
  }
  const auto pixels = Multiply(width, height);
  const auto rounded_pixels = Multiply(h, w);
  if (rounded_pixels > max_pixels) {
    const auto beta = std::sqrt(pixels / max_pixels);
    h = Dimension(std::floor(height / beta / factor), factor);
    w = Dimension(std::floor(width / beta / factor), factor);
    if (!video) {
      h = std::max(factor, h);
      w = std::max(factor, w);
    }
  } else if (rounded_pixels < min_pixels) {
    const auto beta = std::sqrt(min_pixels / pixels);
    h = Dimension(std::ceil(height * beta / factor), factor);
    w = Dimension(std::ceil(width * beta / factor), factor);
  }
  if (!h || !w || h > INT32_MAX || w > INT32_MAX / 4)
    throw std::invalid_argument("Bailing geometry cannot form complete processor patches.");
  return {1, static_cast<int64_t>(h / patch), static_cast<int64_t>(w / patch)};
}

size_t Rows(const std::array<int64_t, 3>& grid, size_t merge) {
  if (!merge || merge > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      grid[0] <= 0 || grid[1] <= 0 || grid[2] <= 0 ||
      grid[1] % static_cast<int64_t>(merge) || grid[2] % static_cast<int64_t>(merge) ||
      static_cast<uint64_t>(grid[1]) > std::numeric_limits<size_t>::max() ||
      static_cast<uint64_t>(grid[2]) > std::numeric_limits<size_t>::max())
    throw std::invalid_argument("Bailing prompt requires complete merged spatial grids.");
  return Multiply(static_cast<size_t>(grid[1]) / merge, static_cast<size_t>(grid[2]) / merge);
}

size_t Count(std::string_view prompt, std::string_view marker) {
  size_t count = 0, index = 0;
  while ((index = prompt.find(marker, index)) != std::string_view::npos) {
    ++count;
    index += marker.size();
  }
  return count;
}

size_t BuildPrompt(std::string_view prompt, const VisionMediaPromptSource& source, size_t merge,
                   size_t maximum_bytes, std::string* expanded, std::vector<size_t>* unit_order) {
  constexpr std::string_view start = "<|vision_start|>", end = "<|vision_end|>";
  constexpr std::string_view image = "<|image_pad|>", video = "<|video_pad|>";
  constexpr std::string_view video_start = "<|video_start|>", video_end = "<|video_end|>";
  constexpr std::string_view image_wrapper = "<|vision_start|><|image_pad|><|vision_end|>";
  constexpr std::string_view vision_wrapper = "<|vision_start|><|video_pad|><|vision_end|>";
  constexpr std::string_view video_wrapper = "<|video_start|><|video_pad|><|video_end|>";
  if (Count(prompt, image) != source.ImageCount() || Count(prompt, video) != source.VideoCount())
    throw std::invalid_argument("Bailing media marker counts do not match the ordered inputs.");
  const auto wrapped_images = Count(prompt, image_wrapper);
  const auto wrapped_videos = Count(prompt, video_wrapper);
  const auto vision_videos = Count(prompt, vision_wrapper);
  size_t image_full = 0, image_bare = 0, video_full = 0, video_vision = 0, video_bare = 0, bytes = 0;
  const auto append = [&](std::string_view text, size_t repeats = 1) {
    bytes = Add(bytes, Multiply(text.size(), repeats));
    if (bytes > maximum_bytes)
      throw std::invalid_argument("Expanded Bailing prompt exceeds its host byte limit.");
    if (expanded)
      for (size_t index = 0; index < repeats; ++index) expanded->append(text);
  };
  while (!prompt.empty()) {
    size_t next = std::string_view::npos;
    for (const auto marker : {start, end, image, video, video_start, video_end})
      next = std::min(next, prompt.find(marker));
    append(prompt.substr(0, next));
    if (next == std::string_view::npos) break;
    prompt.remove_prefix(next);
    const bool wrapped_image = prompt.substr(0, image_wrapper.size()) == image_wrapper;
    const bool full_video = prompt.substr(0, video_wrapper.size()) == video_wrapper;
    const bool vision_video = prompt.substr(0, vision_wrapper.size()) == vision_wrapper;
    if (wrapped_image || prompt.substr(0, image.size()) == image) {
      // The publisher consumes all wrapped occurrences before bare occurrences.
      const auto index = wrapped_image ? image_full++ : Add(wrapped_images, image_bare++);
      if (index >= source.ImageCount())
        throw std::invalid_argument("Bailing image marker has no ordered input.");
      const auto grid = source.ImageGrid(index);
      if (grid[0] != 1) throw std::invalid_argument("Bailing image grids must have T=1.");
      const auto rows = Rows(grid, merge);
      append(start);
      append(image, rows);
      append(end);
      append("\n");
      if (unit_order) unit_order->push_back(index);
      prompt.remove_prefix(wrapped_image ? image_wrapper.size() : image.size());
    } else if (full_video || vision_video || prompt.substr(0, video.size()) == video) {
      const auto index = full_video ? video_full++ : vision_video ? Add(wrapped_videos, video_vision++)
                                                                  : Add(Add(wrapped_videos, vision_videos), video_bare++);
      if (index >= source.VideoCount())
        throw std::invalid_argument("Bailing video marker has no ordered input.");
      const auto grid = source.VideoGrid(index);
      const auto frames = source.FrameCount(index);
      const auto rows = Rows(grid, merge);
      if (frames < 2 || static_cast<uint64_t>(grid[0]) != frames / 2 + frames % 2)
        throw std::invalid_argument("Bailing video grid does not match its selected frames.");
      double previous = 0;
      for (size_t frame = 0; frame < frames; ++frame) {
        const auto timestamp = source.Timestamp(index, frame);
        if (!std::isfinite(timestamp) || timestamp < previous)
          throw std::invalid_argument("Bailing timestamps must be finite, nonnegative and ordered.");
        previous = timestamp;
      }
      size_t first_unit = source.ImageCount();
      if (unit_order)
        for (size_t clip = 0; clip < index; ++clip)
          first_unit = Add(first_unit, static_cast<size_t>(source.VideoGrid(clip)[0]));
      if (full_video) append(video_start);
      for (size_t group = 0; group < static_cast<size_t>(grid[0]); ++group) {
        const auto timestamp = (source.Timestamp(index, group * 2) +
                                source.Timestamp(index, std::min(group * 2 + 1, frames - 1))) /
                               2;
        if (!std::isfinite(timestamp)) throw std::invalid_argument("Bailing group timestamp overflows.");
        std::ostringstream label;
        label.imbue(std::locale::classic());
        label << '<' << std::fixed << std::setprecision(1) << timestamp << " seconds>";
        append(label.str());
        append(start);
        append(video, rows);
        append(end);
        if (unit_order) unit_order->push_back(Add(first_unit, group));
      }
      if (full_video) append(video_end);
      prompt.remove_prefix(full_video ? video_wrapper.size() : vision_video ? vision_wrapper.size()
                                                                            : video.size());
    } else {
      throw std::invalid_argument("Bailing media markers must form supported complete wrappers or bare pads.");
    }
  }
  return bytes;
}

struct PromptSource : VisionMediaPromptSource {
  PromptSource(std::span<const std::array<int64_t, 3>> images, std::span<const VisionVideoPrompt> videos)
      : images{images}, videos{videos} {}
  size_t ImageCount() const override { return images.size(); }
  size_t VideoCount() const override { return videos.size(); }
  std::array<int64_t, 3> ImageGrid(size_t image) const override { return images[image]; }
  std::array<int64_t, 3> VideoGrid(size_t video) const override { return videos[video].grid; }
  size_t FrameCount(size_t video) const override { return videos[video].timestamps.size(); }
  double Timestamp(size_t video, size_t frame) const override { return videos[video].timestamps[frame]; }
  std::span<const std::array<int64_t, 3>> images;
  std::span<const VisionVideoPrompt> videos;
};

}  // namespace

std::array<int64_t, 3> BailingImageGrid(uint32_t width, uint32_t height, size_t patch, size_t merge,
                                        int64_t min_pixels, int64_t max_pixels) {
  return SpatialGrid(width, height, patch, merge, static_cast<double>(min_pixels), static_cast<double>(max_pixels), false);
}

std::array<int64_t, 3> BailingVideoGrid(uint32_t width, uint32_t height, size_t frames, size_t patch, size_t merge,
                                        int64_t min_pixels, int64_t total_pixels) {
  if (frames < 2 || total_pixels <= 0 ||
      frames / 2 + frames % 2 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    throw std::invalid_argument("Bailing video requires at least two selected frames and a positive clip budget.");
  // Sizing uses the original selected count, before the final frame is padded.
  const auto max_pixels = std::max(static_cast<double>(total_pixels) / frames * 2,
                                   std::floor(static_cast<double>(min_pixels) * 1.05));
  auto grid = SpatialGrid(width, height, patch, merge, static_cast<double>(min_pixels), max_pixels, true);
  grid[0] = static_cast<int64_t>(frames / 2 + frames % 2);
  return grid;
}

size_t BailingMediaPromptBytes(std::string_view prompt, const VisionMediaPromptSource& source,
                               size_t merge, size_t maximum_bytes) {
  return BuildPrompt(prompt, source, merge, maximum_bytes, nullptr, nullptr);
}

std::string ExpandBailingMediaPrompt(std::string_view prompt, std::span<const std::array<int64_t, 3>> images,
                                     std::span<const VisionVideoPrompt> videos, size_t merge, size_t maximum_bytes,
                                     std::vector<size_t>* unit_order) {
  const PromptSource source{images, videos};
  const auto bytes = BuildPrompt(prompt, source, merge, maximum_bytes, nullptr, nullptr);
  std::string expanded;
  expanded.reserve(bytes);
  if (unit_order) unit_order->clear();
  BuildPrompt(prompt, source, merge, maximum_bytes, &expanded, unit_order);
  return expanded;
}

}  // namespace Generators
