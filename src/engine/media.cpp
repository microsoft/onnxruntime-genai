// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "media.h"
#include "json.h"
#include "models/decoder_only.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/preprocessing/multi_modal_processor.h"
#include "models/preprocessing/processor.h"

#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>

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

struct ProcessingLimits : JSON::Element {
  int64_t min_pixels{}, max_pixels{}, patch_size{}, temporal_patch_size{}, merge_size{}, smart_resize{};
  int64_t max_video_tokens{};
  bool bgr_normalization{};
  std::string coefficient_type;
  std::vector<std::string> operations;

  Element& OnObject(std::string_view) override { return *this; }
  Element& OnArray(std::string_view) override { return *this; }
  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "type") {
      operations.emplace_back(JSON::Get<std::string_view>(value));
    } else if (name == "min_pixels") {
      min_pixels = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "max_pixels") {
      max_pixels = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "patch_size") {
      patch_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "temporal_patch_size") {
      temporal_patch_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "merge_size") {
      merge_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "smart_resize") {
      smart_resize = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "max_video_tokens") {
      max_video_tokens = SafeDoubleToInt(JSON::Get<double>(value), name);
    } else if (name == "coefficient_type") {
      coefficient_type = JSON::Get<std::string_view>(value);
    } else if (name == "qwen2_5_vl" || name == "qwen3_vl") {
      bgr_normalization = bgr_normalization || SafeDoubleToInt(JSON::Get<double>(value), name) != 0;
    }
  }
};

struct HostReservation {
  std::shared_ptr<MediaAccounting> accounting;
  size_t bytes{};
  ~HostReservation() { accounting->host_bytes -= bytes; }
};

void AppendPrompt(std::string& output, std::string_view text, size_t repeats, size_t limit) {
  const auto bytes = MultiplyBytes(text.size(), repeats);
  if (output.size() > limit || bytes > limit - output.size())
    throw std::invalid_argument("Expanded media prompt exceeds its host byte limit.");
  for (size_t index = 0; index < repeats; ++index) output.append(text);
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

size_t BuildQwenMediaPrompt(std::string_view prompt,
                            std::span<const std::array<int64_t, 3>> image_grids,
                            std::span<const QwenVideoPrompt> videos, size_t merge_size,
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
    if (expanded) AppendPrompt(*expanded, text, repeats, maximum_bytes);
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
      if (image_index == image_grids.size() || image_grids[image_index][0] != 1)
        throw std::invalid_argument("Image marker count or temporal grid does not match the ordered inputs.");
      const auto rows = MergedGridRows(image_grids[image_index++], merge_size);
      append(start, 1);
      append(image, rows);
      append(end, 1);
      prompt.remove_prefix(image_marker.size());
    } else if (prompt.substr(0, video_marker.size()) == video_marker) {
      if (video_index == videos.size())
        throw std::invalid_argument("Video marker count does not match the ordered inputs.");
      const auto& clip = videos[video_index++];
      const auto rows = MergedGridRows(clip.grid, merge_size);
      if (clip.timestamps.size() < 2 ||
          static_cast<uint64_t>(clip.grid[0]) != clip.timestamps.size() / 2 + clip.timestamps.size() % 2)
        throw std::invalid_argument("Video temporal grid does not match its selected frame count.");
      for (size_t index = 0; index < clip.timestamps.size(); ++index) {
        if (!std::isfinite(clip.timestamps[index]) || clip.timestamps[index] < 0 ||
            (index != 0 && clip.timestamps[index] < clip.timestamps[index - 1]))
          throw std::invalid_argument("Video presentation timestamps must be finite, nonnegative and ordered.");
      }
      append(start, 1);
      for (size_t group = 0; group < static_cast<size_t>(clip.grid[0]); ++group) {
        const auto timestamp =
            (clip.timestamps[group * 2] + clip.timestamps[std::min(group * 2 + 1, clip.timestamps.size() - 1)]) / 2;
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
  if (image_index != image_grids.size() || video_index != videos.size())
    throw std::invalid_argument("Media marker counts do not match the ordered inputs.");
  return bytes;
}

}  // namespace

std::string ExpandQwenMediaPrompt(std::string_view prompt,
                                  std::span<const std::array<int64_t, 3>> image_grids,
                                  std::span<const QwenVideoPrompt> videos, size_t merge_size,
                                  size_t maximum_bytes) {
  const auto bytes = BuildQwenMediaPrompt(prompt, image_grids, videos, merge_size, maximum_bytes, nullptr);
  std::string expanded;
  expanded.reserve(bytes);
  BuildQwenMediaPrompt(prompt, image_grids, videos, merge_size, maximum_bytes, &expanded);
  return expanded;
}

MediaItem::~MediaItem() {
  if (accounting) {
    accounting->host_bytes -= host_bytes;
    if (features) accounting->device_bytes -= feature_bytes;
  }
}

void MediaItem::ReleaseFeatures() {
  if (!features) return;
  features->p_device_->Synchronize();
  features.reset();
  accounting->device_bytes -= feature_bytes;
}

std::array<int64_t, 3> MediaPosition(std::span<const MediaSpan> media, size_t row, size_t merge_size) {
  std::array<int64_t, 3> result;
  MediaPositions(media, row, merge_size, {std::span<int64_t>{&result[0], 1}, std::span<int64_t>{&result[1], 1}, std::span<int64_t>{&result[2], 1}});
  return result;
}

void MediaPositions(std::span<const MediaSpan> media, size_t first_row, size_t merge_size,
                    const std::array<std::span<int64_t>, 3>& positions) {
  const size_t count = positions[0].size();
  if (positions[1].size() != count || positions[2].size() != count)
    throw std::invalid_argument("Media position planes must have equal lengths.");
  int64_t delta = 0;
  size_t span_index = 0;
  size_t index = 0;
  while (index < count) {
    const size_t row = first_row + index;
    while (span_index < media.size() && row >= media[span_index].start + media[span_index].item->rows) {
      const auto& span = media[span_index++];
      delta += std::max(span.item->grid[1], span.item->grid[2]) / static_cast<int64_t>(merge_size) -
               static_cast<int64_t>(span.item->rows);
    }
    if (span_index == media.size() || row < media[span_index].start) {
      const size_t end = span_index == media.size() ? count : std::min(count, media[span_index].start - first_row);
      for (auto plane : positions)
        std::iota(plane.begin() + index, plane.begin() + end, static_cast<int64_t>(row) + delta);
      index = end;
      continue;
    }
    const auto& span = media[span_index];
    const auto width = span.item->grid[2] / static_cast<int64_t>(merge_size);
    const auto base = static_cast<int64_t>(span.start) + delta;
    const size_t end = index + std::min(count - index, span.item->rows - (row - span.start));
    for (; index < end; ++index) {
      const auto offset = static_cast<int64_t>(first_row + index - span.start);
      positions[0][index] = base;
      positions[1][index] = base + offset / width;
      positions[2][index] = base + offset % width;
    }
  }
}

std::array<int64_t, 3> MediaImageGrid(uint32_t width, uint32_t height, size_t patch, size_t merge,
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

std::array<int64_t, 3> MediaVideoGrid(uint32_t width, uint32_t height, size_t frames,
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

MediaStore::MediaStore(std::shared_ptr<DecoderOnly_Model> model, size_t host_limit, size_t device_limit)
    : model_{std::move(model)}, host_limit_{host_limit}, device_limit_{device_limit} {
  const auto& config = model_->config_->model.vision;
  auto file = (model_->config_->config_path / fs::path(config.config_filename)).open(std::ios::binary);
  if (!file) throw std::runtime_error("Cannot read the packed vision processor configuration.");
  std::ostringstream document;
  document << file.rdbuf();
  ProcessingLimits limits;
  JSON::Parse(limits, document.str());
  if (limits.operations != std::vector<std::string>{"DecodeImage", "ConvertRGB", "Resize", "Rescale", "Normalize", "PatchImage"} ||
      limits.min_pixels <= 0 || limits.max_pixels < limits.min_pixels || limits.patch_size != config.patch_size ||
      limits.temporal_patch_size != 2 || limits.merge_size != config.spatial_merge_size || limits.smart_resize != 1) {
    throw std::runtime_error("Packed vision requires the bounded Qwen PatchImage preprocessing pipeline.");
  }
  min_pixels_ = limits.min_pixels;
  max_pixels_ = limits.max_pixels;
  const auto marker = model_->vision_->processor->tokenizer_->Encode("<|vision_end|>");
  if (marker.size() != 1) throw std::logic_error("The packed vision tokenizer has no vision-end token.");
  vision_end_token_ = marker[0];
  const auto video_marker = model_->vision_->processor->tokenizer_->Encode("<|video_pad|>");
  if (video_marker.size() == 1 && video_marker[0] > 0) video_token_ = video_marker[0];
  if (config.video_config_filename) {
    const auto& type = model_->config_->model.type;
    if (type != "qwen3_5" && type != "qwen3_5_moe" && type != "qwen4_exp")
      throw std::runtime_error("Packed video requires a verified Qwen3.5 or Qwen4Exp position contract.");
    auto video_file = model_->config_->ResolvePath(*config.video_config_filename).open(std::ios::binary);
    if (!video_file) throw std::runtime_error("Cannot read the packed video processor configuration.");
    std::ostringstream video_document;
    video_document << video_file.rdbuf();
    const auto definition = video_document.str();
    ProcessingLimits video;
    JSON::Parse(video, definition);
    if (video.operations != std::vector<std::string>{"Resize", "Rescale", "Normalize", "PatchImage"} ||
        video.min_pixels <= 0 || video.max_pixels < video.min_pixels || video.patch_size != config.patch_size ||
        video.temporal_patch_size != 2 || video.merge_size != config.spatial_merge_size ||
        video.smart_resize != 0 || video.coefficient_type != "int16" || video.max_video_tokens <= 0 ||
        video.bgr_normalization) {
      throw std::runtime_error("Packed video requires the bounded Qwen3-VL temporal preprocessing pipeline.");
    }
    if (video_token_ == 0 || video_token_ != model_->config_->model.video_token_id)
      throw std::runtime_error("Packed video requires the source video-pad token.");
    video_min_pixels_ = video.min_pixels;
    video_max_pixels_ = video.max_pixels;
    max_video_tokens_ = static_cast<size_t>(video.max_video_tokens);
    CheckResult(OrtxCreateProcessor(video_processor_.ToBeAssigned(), definition.c_str()));
  }
}

void MediaStore::MakeHostRoom(size_t bytes) {
  if (bytes > host_limit_) throw std::runtime_error("Media input exceeds engine.dynamic_batching.media_host_bytes.");
  for (auto item = items_.begin(); accounting_->host_bytes > host_limit_ - bytes && item != items_.end();) {
    if (item->use_count() == 1) {
      item = items_.erase(item);
    } else {
      ++item;
    }
  }
  if (accounting_->host_bytes > host_limit_ - bytes)
    throw std::runtime_error("Retained media leaves insufficient host capacity for this turn.");
}

void MediaStore::ValidateTextTokens(std::span<const int32_t> tokens) const {
  const auto& config = model_->config_->model;
  if (video_token_ != 0 && std::find(tokens.begin(), tokens.end(), video_token_) != tokens.end())
    throw std::invalid_argument("Video markers require the formatted prompt and videos overload.");
  if (std::any_of(tokens.begin(), tokens.end(), [&](int32_t token) {
        return token == config.image_token_id || token == config.vision_start_token_id || token == vision_end_token_;
      }))
    throw std::invalid_argument("Image markers require the formatted prompt and images overload.");
}

PreparedMediaTurn MediaStore::Prepare(const std::string& prompt, std::span<const int32_t> raw_tokens,
                                      const Images* images, size_t first_row, size_t remaining_rows,
                                      const Videos* videos) {
  if (videos && !videos->clips_.empty())
    return PrepareVideos(prompt, raw_tokens, images, *videos, first_row, remaining_rows);
  const auto& config = model_->config_->model;
  if (video_token_ != 0 && std::find(raw_tokens.begin(), raw_tokens.end(), video_token_) != raw_tokens.end())
    throw std::invalid_argument("Video markers require ordered video inputs.");
  const size_t count = images ? images->num_images_ : 0;
  size_t markers = 0;
  for (size_t index = 0; index < raw_tokens.size(); ++index) {
    if (raw_tokens[index] == config.vision_start_token_id) {
      if (index + 2 >= raw_tokens.size() || raw_tokens[index + 1] != config.image_token_id ||
          raw_tokens[index + 2] != vision_end_token_)
        throw std::invalid_argument("Each image must use a complete vision-start/image-pad/vision-end marker.");
      ++markers;
      index += 2;
    } else if (raw_tokens[index] == config.image_token_id || raw_tokens[index] == vision_end_token_) {
      throw std::invalid_argument("Image markers must form complete ordered image items.");
    }
  }
  if (markers != count) throw std::invalid_argument("Image marker count does not match the ordered image inputs.");
  if (raw_tokens.empty() || raw_tokens.size() > remaining_rows)
    throw std::invalid_argument("The formatted turn does not fit the remaining session rows.");
  if (count == 0) return {std::vector<int32_t>{raw_tokens.begin(), raw_tokens.end()}, {}};
  if (images->info_.size() != count) throw std::invalid_argument("Image header metadata is unavailable.");
  const size_t merge = static_cast<size_t>(config.vision.spatial_merge_size);
  const size_t patch = static_cast<size_t>(config.vision.patch_size);
  const auto pixel_shape = model_->vision_->session_info.GetInputShape(config.vision.inputs.pixel_values);
  // Extensions output, the FP32 copy and the converted tensor coexist during preprocessing.
  const size_t patch_bytes = MultiplyBytes(static_cast<size_t>(pixel_shape[1]),
                                           2 * sizeof(float) + Ort::SizeOf(model_->cpu_embedding_->type_));
  size_t estimate = 0, maximum_patches = 0, expected_rows = raw_tokens.size();
  for (const auto& info : images->info_) {
    if (info.width == 0 || info.height == 0)
      throw std::invalid_argument("Packed vision requires PNG or JPEG dimensions before preprocessing.");
    const size_t source_pixels = MultiplyBytes(info.width, info.height);
    const auto grid = MediaImageGrid(info.width, info.height, patch, merge, min_pixels_, max_pixels_);
    const size_t height = MultiplyBytes(static_cast<size_t>(grid[1]), patch);
    const size_t width = MultiplyBytes(static_cast<size_t>(grid[2]), patch);
    const size_t patches = MultiplyBytes(static_cast<size_t>(grid[1]), static_cast<size_t>(grid[2]));
    maximum_patches = std::max(maximum_patches, patches);
    const size_t rows = patches / MultiplyBytes(merge, merge);
    expected_rows = AddBytes(expected_rows, rows - 1);
    if (MultiplyBytes(rows, MultiplyBytes(static_cast<size_t>(model_->cpu_embedding_->hidden_size_),
                                          Ort::SizeOf(model_->cpu_embedding_->type_))) > device_limit_)
      throw std::invalid_argument("An image cannot fit the configured device media budget.");
    estimate = AddBytes(estimate, AddBytes(MultiplyBytes(source_pixels, 16), MultiplyBytes(MultiplyBytes(height, width), 64)));
  }
  if (expected_rows > remaining_rows) throw std::invalid_argument("Expanded image rows exceed the remaining session capacity.");
  estimate = AddBytes(estimate, MultiplyBytes(MultiplyBytes(count, maximum_patches), patch_bytes));
  estimate = AddBytes(estimate, MultiplyBytes(expected_rows, 128));
  MakeHostRoom(estimate);
  accounting_->host_bytes += estimate;
  HostReservation reservation{accounting_, estimate};
  auto processed = model_->vision_->processor->Process(prompt, images, nullptr);
  const auto ids = processed->at(std::string{Config::Defaults::InputIdsName});
  PreparedMediaTurn result;
  result.tokens.assign(ids->GetData<int32_t>(), ids->GetData<int32_t>() + ids->GetElementCount());
  if (result.tokens.size() != expected_rows) throw std::runtime_error("Image preprocessing violated its expanded-row estimate.");
  const auto grids = processed->at(std::string{Config::Defaults::ImageGridThwName});
  const auto pixels = processed->at(std::string{Config::Defaults::PixelValuesName});
  const auto shape = pixels->GetShape();
  if (grids->GetType() != Ort::TypeToTensorType<int64_t> || grids->GetElementCount() != count * 3 ||
      pixels->GetType() != model_->cpu_embedding_->type_ || (shape.size() != 2 && shape.size() != 3) ||
      shape.back() != pixel_shape[1] || (shape.size() == 3 && shape[0] != static_cast<int64_t>(count)))
    throw std::runtime_error("Image preprocessing returned an unsupported patch/grid layout.");
  const size_t physical_patches = pixels->GetElementCount() / static_cast<size_t>(shape.back());
  size_t logical_patches = 0;
  for (size_t image = 0; image < count; ++image) {
    const auto* grid = grids->GetData<int64_t>() + image * 3;
    if (grid[0] != 1 || grid[1] <= 0 || grid[2] <= 0 || grid[1] % merge != 0 || grid[2] % merge != 0)
      throw std::runtime_error("Packed vision supports complete spatially merged image grids, not video.");
    logical_patches = AddBytes(logical_patches, MultiplyBytes(static_cast<size_t>(grid[1]), static_cast<size_t>(grid[2])));
  }
  const bool compact = physical_patches == logical_patches;
  if (!compact && physical_patches % count != 0)
    throw std::runtime_error("Image patch padding is not a uniform per-image stride.");
  size_t image_index = 0;
  size_t patch_offset = 0;
  for (size_t index = 0; index < result.tokens.size(); ++index) {
    if (result.tokens[index] != config.image_token_id) continue;
    if (image_index >= count) throw std::runtime_error("Expanded image rows have no corresponding grid.");
    const auto* grid = grids->GetData<int64_t>() + image_index * 3;
    const size_t patches = MultiplyBytes(static_cast<size_t>(grid[1]), static_cast<size_t>(grid[2]));
    const size_t rows = patches / MultiplyBytes(merge, merge);
    const size_t stride = compact ? patches : physical_patches / count;
    if (patches > stride || rows > result.tokens.size() - index ||
        !std::all_of(result.tokens.begin() + index, result.tokens.begin() + index + rows,
                     [&](int32_t token) { return token == config.image_token_id; }))
      throw std::runtime_error("Expanded image span does not match its processed geometry.");
    auto item = Intern({grid[0], grid[1], grid[2]}, config.image_token_id, *pixels, patch_offset,
                       result.media, reservation.bytes);
    result.media.push_back({first_row + index, std::move(item)});
    patch_offset += stride;
    index += rows - 1;
    ++image_index;
  }
  if (image_index != count) throw std::runtime_error("Image preprocessing omitted an image.");
  return result;
}

std::shared_ptr<MediaItem> MediaStore::Intern(const std::array<int64_t, 3>& grid, int32_t marker,
                                              const Tensor& pixels, size_t patch_offset,
                                              std::span<const MediaSpan> pending, size_t& reservation) {
  const size_t patches = MultiplyBytes(static_cast<size_t>(grid[1]), static_cast<size_t>(grid[2]));
  const size_t rows = MergedGridRows(grid, static_cast<size_t>(model_->config_->model.vision.spatial_merge_size));
  const auto width = pixels.GetShape().back();
  if (AddBytes(patch_offset, patches) > pixels.GetElementCount() / static_cast<size_t>(width))
    throw std::logic_error("Canonical media exceeds its processed patch buffer.");
  const size_t row_bytes = MultiplyBytes(static_cast<size_t>(width), Ort::SizeOf(pixels.GetType()));
  const size_t bytes = MultiplyBytes(patches, row_bytes);
  const auto* data = static_cast<const uint8_t*>(pixels.GetRawData()) + MultiplyBytes(patch_offset, row_bytes);
  const auto matches = [&](const std::shared_ptr<MediaItem>& cached) {
    return cached->marker_token == marker && cached->grid == grid && cached->pixels->GetType() == pixels.GetType() &&
           MultiplyBytes(cached->pixels->GetElementCount(), Ort::SizeOf(cached->pixels->GetType())) == bytes &&
           std::memcmp(cached->pixels->GetRawData(), data, bytes) == 0;
  };
  for (const auto& cached : items_) {
    if (matches(cached)) return cached;
  }
  for (const auto& span : pending) {
    if (matches(span.item)) return span.item;
  }
  if (rows > static_cast<uint64_t>(std::numeric_limits<int32_t>::max() - next_prefix_row_))
    throw std::overflow_error("Engine media prefix identity space is exhausted.");
  const auto retained_bytes = AddBytes(bytes, sizeof(MediaItem) + sizeof(std::shared_ptr<MediaItem>));
  if (retained_bytes > reservation) throw std::logic_error("Canonical media exceeded its host reservation.");
  auto item = std::make_shared<MediaItem>();
  item->grid = grid;
  item->marker_token = marker;
  item->rows = rows;
  item->prefix_base = static_cast<int32_t>(next_prefix_row_);
  item->pixels = std::make_shared<Tensor>(GetDeviceInterface(DeviceType::CPU), pixels.GetType());
  const std::array<int64_t, 2> item_shape{static_cast<int64_t>(patches), width};
  item->pixels->CreateTensor(item_shape);
  std::memcpy(item->pixels->GetMutableRawData(), data, bytes);
  item->feature_bytes = MultiplyBytes(MultiplyBytes(rows, static_cast<size_t>(model_->cpu_embedding_->hidden_size_)),
                                      Ort::SizeOf(model_->cpu_embedding_->type_));
  next_prefix_row_ += rows;
  item->accounting = accounting_;
  item->host_bytes = retained_bytes;
  reservation -= retained_bytes;
  return item;
}

PreparedMediaTurn MediaStore::PrepareVideos(const std::string& prompt, std::span<const int32_t> raw_tokens,
                                            const Images* images, const Videos& videos, size_t first_row,
                                            size_t remaining_rows) {
  if (!video_processor_) throw std::invalid_argument("This package has no supported video preprocessing contract.");
  if (raw_tokens.empty() || raw_tokens.size() > remaining_rows)
    throw std::invalid_argument("The formatted turn does not fit the remaining session rows.");
  const auto& config = model_->config_->model;
  const auto merge = static_cast<size_t>(config.vision.spatial_merge_size);
  const auto patch = static_cast<size_t>(config.vision.patch_size);
  const size_t image_count = images ? images->num_images_ : 0;
  if (image_count > remaining_rows || videos.clips_.size() > remaining_rows ||
      (images && images->info_.size() != image_count))
    throw std::invalid_argument("Media input metadata exceeds the remaining session capacity.");
  const auto pixel_width = model_->vision_->session_info.GetInputShape(config.vision.inputs.pixel_values)[1];
  const auto patch_bytes = MultiplyBytes(static_cast<size_t>(pixel_width),
                                         2 * sizeof(float) + 2 * Ort::SizeOf(model_->cpu_embedding_->type_));
  size_t estimate = 0, minimum_rows = 0, maximum_image_patches = 0;
  std::vector<std::array<int64_t, 3>> image_grids;
  const auto image_info = images ? std::span<const Images::Info>{images->info_} : std::span<const Images::Info>{};
  for (const auto& info : image_info) {
    if (info.width == 0 || info.height == 0)
      throw std::invalid_argument("Packed vision requires PNG or JPEG dimensions before preprocessing.");
    const auto grid = MediaImageGrid(info.width, info.height, patch, merge, min_pixels_, max_pixels_);
    const auto patches = MultiplyBytes(static_cast<size_t>(grid[1]), static_cast<size_t>(grid[2]));
    maximum_image_patches = std::max(maximum_image_patches, patches);
    minimum_rows = AddBytes(minimum_rows, MergedGridRows(grid, merge));
    estimate = AddBytes(estimate, AddBytes(MultiplyBytes(MultiplyBytes(info.width, info.height), 16),
                                           MultiplyBytes(MultiplyBytes(patches, MultiplyBytes(patch, patch)), 64)));
    image_grids.push_back(grid);
  }
  estimate = AddBytes(estimate, MultiplyBytes(MultiplyBytes(image_count, maximum_image_patches), patch_bytes));
  std::vector<std::vector<double>> timestamps;
  std::vector<QwenVideoPrompt> clips;
  timestamps.reserve(videos.clips_.size());
  clips.reserve(videos.clips_.size());
  for (const auto& clip : videos.clips_) {
    size_t count{};
    CheckResult(OrtxGetRawFrameCount(clip.get(), &count));
    if (count < 2 || count / 2 + count % 2 > remaining_rows)
      throw std::invalid_argument("Video selected frame count is incompatible with the remaining session capacity.");
    OrtxFrame first{};
    CheckResult(OrtxGetRawFrame(clip.get(), 0, &first));
    if (first.width > UINT32_MAX || first.height > UINT32_MAX)
      throw std::invalid_argument("Video source geometry exceeds the model dimension limit.");
    const auto grid = MediaVideoGrid(static_cast<uint32_t>(first.width), static_cast<uint32_t>(first.height),
                                     count, patch, merge, video_min_pixels_, video_max_pixels_, max_video_tokens_);
    const auto rows = MergedGridRows(grid, merge);
    minimum_rows = AddBytes(minimum_rows, MultiplyBytes(static_cast<size_t>(grid[0]), rows));
    if (minimum_rows > remaining_rows)
      throw std::invalid_argument("Expanded media rows exceed the remaining session capacity.");
    estimate = AddBytes(estimate, MultiplyBytes(count, MultiplyBytes(MultiplyBytes(
                                                                         static_cast<size_t>(first.width), static_cast<size_t>(first.height)),
                                                                     16)));
    const auto patches = MultiplyBytes(static_cast<size_t>(grid[0]),
                                       MultiplyBytes(static_cast<size_t>(grid[1]), static_cast<size_t>(grid[2])));
    estimate = AddBytes(estimate, AddBytes(MultiplyBytes(patches, patch_bytes),
                                           MultiplyBytes(MultiplyBytes(patches, MultiplyBytes(patch, patch)), 64)));
    if (estimate > host_limit_) throw std::invalid_argument("Video preprocessing exceeds the host media budget.");
    timestamps.emplace_back();
    timestamps.back().reserve(count);
    for (size_t frame = 0; frame < count; ++frame) {
      OrtxFrame descriptor{};
      CheckResult(OrtxGetRawFrame(clip.get(), frame, &descriptor));
      if (descriptor.width != first.width || descriptor.height != first.height)
        throw std::invalid_argument("Qwen video requires uniform source geometry within each clip.");
      timestamps.back().push_back(descriptor.timestamp_seconds);
    }
    clips.push_back({grid, timestamps.back()});
  }
  for (const auto& grid : image_grids) {
    const auto feature_bytes = MultiplyBytes(MergedGridRows(grid, merge), MultiplyBytes(
                                                                              static_cast<size_t>(model_->cpu_embedding_->hidden_size_), Ort::SizeOf(model_->cpu_embedding_->type_)));
    if (feature_bytes > device_limit_) throw std::invalid_argument("An image cannot fit the device media budget.");
  }
  for (const auto& clip : clips) {
    const auto feature_bytes = MultiplyBytes(MergedGridRows(clip.grid, merge), MultiplyBytes(
                                                                                   static_cast<size_t>(model_->cpu_embedding_->hidden_size_), Ort::SizeOf(model_->cpu_embedding_->type_)));
    if (feature_bytes > device_limit_) throw std::invalid_argument("A video group cannot fit the device media budget.");
  }
  const auto prompt_bytes = BuildQwenMediaPrompt(prompt, image_grids, clips, merge, host_limit_, nullptr);
  // Original tokens bound normalized text; expanded ASCII bytes also bound new tokens and their scratch.
  estimate = AddBytes(estimate, AddBytes(AddBytes(prompt_bytes, 1),
                                         MultiplyBytes(AddBytes(raw_tokens.size(), prompt_bytes), 128)));
  MakeHostRoom(estimate);
  accounting_->host_bytes += estimate;
  HostReservation reservation{accounting_, estimate};
  const auto expanded = ExpandQwenMediaPrompt(prompt, image_grids, clips, merge, prompt_bytes);
  PreparedMediaTurn result;
  result.tokens = model_->vision_->processor->tokenizer_->Encode(expanded.c_str());
  if (result.tokens.size() >= remaining_rows)
    throw std::invalid_argument("Expanded media tokens exceed the remaining session capacity.");
  std::vector<MediaSpan> pending;
  std::vector<std::shared_ptr<MediaItem>> image_items, video_items;
  if (image_count != 0) {
    std::string image_prompt;
    AppendPrompt(image_prompt, "<|vision_start|><|image_pad|><|vision_end|>", image_count, host_limit_);
    auto processed = model_->vision_->processor->Process(image_prompt, images, nullptr);
    const auto& grids = processed->at(std::string{Config::Defaults::ImageGridThwName});
    const auto& pixels = processed->at(std::string{Config::Defaults::PixelValuesName});
    const auto shape = pixels->GetShape();
    if (grids->GetType() != Ort::TypeToTensorType<int64_t> || grids->GetElementCount() != image_count * 3 ||
        pixels->GetType() != model_->cpu_embedding_->type_ || (shape.size() != 2 && shape.size() != 3) ||
        shape.back() != pixel_width || (shape.size() == 3 && shape[0] != static_cast<int64_t>(image_count)))
      throw std::runtime_error("Image preprocessing returned an unsupported patch/grid layout.");
    const auto physical = pixels->GetElementCount() / static_cast<size_t>(pixel_width);
    size_t logical = 0;
    for (size_t image = 0; image < image_count; ++image) {
      if (!std::equal(image_grids[image].begin(), image_grids[image].end(), grids->GetData<int64_t>() + image * 3))
        throw std::runtime_error("Image preprocessing violated its resolved geometry.");
      logical = AddBytes(logical, MultiplyBytes(static_cast<size_t>(image_grids[image][1]),
                                                static_cast<size_t>(image_grids[image][2])));
    }
    if (physical != logical && physical % image_count != 0)
      throw std::runtime_error("Image patch padding is not a uniform per-image stride.");
    size_t offset = 0;
    for (const auto& grid : image_grids) {
      const auto patches = MultiplyBytes(static_cast<size_t>(grid[1]), static_cast<size_t>(grid[2]));
      const auto stride = physical == logical ? patches : physical / image_count;
      if (patches > stride) throw std::runtime_error("Image patches exceed their physical stride.");
      auto item = Intern(grid, config.image_token_id, *pixels, offset, pending, reservation.bytes);
      image_items.push_back(item);
      pending.push_back({0, std::move(item)});
      offset += stride;
    }
  }
  for (size_t video = 0; video < clips.size(); ++video) {
    const auto& grid = clips[video].grid;
    const OrtxFramePreProcessOptions options{sizeof(OrtxFramePreProcessOptions),
                                             grid[1] * static_cast<int64_t>(patch),
                                             grid[2] * static_cast<int64_t>(patch)};
    ort_extensions::OrtxObjectPtr<OrtxTensorResult> processed;
    CheckResult(OrtxFramesPreProcess(video_processor_.get(), videos.clips_[video].get(), &options,
                                     processed.ToBeAssigned()));
    ort_extensions::OrtxObjectPtr<OrtxTensor> pixels, actual_grid;
    CheckResult(OrtxTensorResultGetAt(processed.get(), 0, pixels.ToBeAssigned()));
    CheckResult(OrtxTensorResultGetAt(processed.get(), 1, actual_grid.ToBeAssigned()));
    extDataType_t pixel_type{}, grid_type{};
    CheckResult(OrtxGetTensorType(pixels.get(), &pixel_type));
    CheckResult(OrtxGetTensorType(actual_grid.get(), &grid_type));
    const void *pixel_data{}, *grid_data{};
    const int64_t *shape{}, *grid_shape{};
    size_t rank{}, grid_rank{};
    CheckResult(OrtxGetTensorData(pixels.get(), &pixel_data, &shape, &rank));
    CheckResult(OrtxGetTensorData(actual_grid.get(), &grid_data, &grid_shape, &grid_rank));
    const auto patches = MultiplyBytes(static_cast<size_t>(grid[1]), static_cast<size_t>(grid[2]));
    if (pixel_type != kOrtxFloat || grid_type != kOrtxInt64 || rank != 2 || grid_rank != 2 ||
        grid_shape[0] != 1 || grid_shape[1] != 3 || shape[1] != pixel_width ||
        shape[0] != static_cast<int64_t>(MultiplyBytes(static_cast<size_t>(grid[0]), patches)) ||
        !std::equal(grid.begin(), grid.end(), static_cast<const int64_t*>(grid_data)))
      throw std::runtime_error("Video preprocessing violated its resolved patch/grid layout.");
    NamedTensors converted;
    EmplaceProcessedTensor(nullptr, converted, Config::Defaults::PixelValuesName, pixels.get(),
                           model_->cpu_embedding_->type_, Ort::Allocator::GetWithDefaultOptions());
    for (size_t group = 0; group < static_cast<size_t>(grid[0]); ++group) {
      auto item = Intern({1, grid[1], grid[2]}, config.video_token_id,
                         *converted.at(std::string{Config::Defaults::PixelValuesName}),
                         MultiplyBytes(group, patches), pending, reservation.bytes);
      video_items.push_back(item);
      pending.push_back({0, std::move(item)});
    }
  }
  size_t image = 0, video = 0;
  for (size_t index = 0; index < result.tokens.size(); ++index) {
    const auto token = result.tokens[index];
    if (token != config.image_token_id && token != config.video_token_id) continue;
    const auto& collection = token == config.image_token_id ? image_items : video_items;
    auto& item_index = token == config.image_token_id ? image : video;
    if (item_index == collection.size()) throw std::runtime_error("Expanded media has no corresponding encoder unit.");
    const auto& item = collection[item_index++];
    if (item->rows > result.tokens.size() - index ||
        !std::all_of(result.tokens.begin() + index, result.tokens.begin() + index + item->rows,
                     [&](int32_t value) { return value == token; }))
      throw std::runtime_error("Expanded media span does not match its processed geometry.");
    result.media.push_back({AddBytes(first_row, index), item});
    index += item->rows - 1;
  }
  if (image != image_items.size() || video != video_items.size())
    throw std::runtime_error("Media preprocessing omitted an encoder unit.");
  return result;
}

void MediaStore::ReserveItems(size_t count) {
  items_.reserve(AddBytes(items_.size(), count));
}

void MediaStore::Retain(std::span<const MediaSpan> media) noexcept {
  for (const auto& span : media) {
    if (std::find(items_.begin(), items_.end(), span.item) == items_.end()) items_.push_back(span.item);
  }
}

void MediaStore::Encode(const std::shared_ptr<MediaItem>& item) {
  if (item->features) return;
  if (item->feature_bytes > device_limit_) throw std::runtime_error("An image cannot fit the device media budget.");
  for (auto& cached : items_) {
    if (accounting_->device_bytes <= device_limit_ - item->feature_bytes) break;
    cached->ReleaseFeatures();
  }
  const auto& config = model_->config_->model.vision;
  const auto& memory = GetDeviceInterface(DeviceType::CPU)->GetAllocator().GetInfo();
  const std::array<int64_t, 2> grid_shape{1, 3};
  auto grid = OrtValue::CreateTensor(memory, item->grid.data(), sizeof(item->grid), grid_shape, Ort::TypeToTensorType<int64_t>);
  const char* names[]{config.inputs.pixel_values.c_str(), config.inputs.image_grid_thw.c_str()};
  OrtValue* inputs[]{item->pixels->GetOrtTensor(), grid.get()};
  auto features = std::make_shared<Tensor>(model_->vision_->device, model_->cpu_embedding_->type_);
  const std::array<int64_t, 2> feature_shape{static_cast<int64_t>(item->rows), model_->cpu_embedding_->hidden_size_};
  features->CreateTensor(feature_shape);
  const char* output_name = config.outputs.image_features.c_str();
  OrtValue* output = features->GetOrtTensor();
  auto options = OrtRunOptions::Create();
  if (config.run_options) {
    for (const auto& [key, value] : *config.run_options) options->AddConfigEntry(key.c_str(), value.c_str());
  }
  // Component sessions can inherit CUDA graph settings from the decoder. Image shapes and
  // output addresses vary per item, so encoding must never capture or replay the default graph.
  options->AddConfigEntry("gpu_graph_id", "-1");
  model_->vision_->session->Run(options.get(), names, inputs, 2, &output_name, &output, 1);
  model_->vision_->device->Synchronize();
  item->features = std::move(features);
  accounting_->device_bytes += item->feature_bytes;
}

void MediaStore::ReleaseFeatures() {
  for (auto& item : items_) item->ReleaseFeatures();
}

}  // namespace Generators
