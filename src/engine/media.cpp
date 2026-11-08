// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "media.h"
#include "json.h"
#include "models/decoder_only.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/preprocessing/multi_modal_processor.h"
#include "models/preprocessing/processor.h"
#include "models/preprocessing/qwen2_5_vl_image_processor.h"
#include "models/preprocessing/qwen_vision_position.h"
#include "models/preprocessing/qwen_vision_processing.h"

#include <algorithm>
#include <cstring>
#include <limits>
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

int64_t AddPosition(int64_t left, int64_t right) {
  if ((right > 0 && left > std::numeric_limits<int64_t>::max() - right) ||
      (right < 0 && left < std::numeric_limits<int64_t>::min() - right))
    throw std::overflow_error("Media coordinate exceeds int64.");
  return left + right;
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

}  // namespace

MediaItem::~MediaItem() {
  if (accounting) {
    accounting->host_bytes -= host_bytes;
    if (features) accounting->device_bytes -= feature_bytes;
  }
}

MediaPrefix::~MediaPrefix() {
  if (accounting) accounting->host_bytes -= host_bytes;
}

bool MediaPrefix::Matches(const MediaSpan& span, std::span<const int32_t> values, int64_t base) const {
  return item == span.item && feature_offset == span.feature_offset && position_base == base &&
         tokens.size() == span.rows && values.size() == tokens.size() &&
         std::equal(tokens.begin(), tokens.end(), values.begin()) &&
         (positions == span.positions || (positions && span.positions && positions->Equals(*span.positions)));
}

void MediaItem::ReleaseFeatures() {
  if (!features) return;
  features->p_device_->Synchronize();
  features.reset();
  accounting->device_bytes -= feature_bytes;
}

std::array<int64_t, 3> MediaPosition(std::span<const MediaSpan> media, size_t row) {
  std::array<int64_t, 3> result;
  MediaPositions(media, row, {std::span<int64_t>{&result[0], 1}, std::span<int64_t>{&result[1], 1}, std::span<int64_t>{&result[2], 1}});
  return result;
}

void ApplyMediaFeatures(std::span<const MediaSpan> media, size_t first_row, size_t rows,
                        Tensor& embeddings, size_t packed_row) {
  const auto shape = embeddings.GetShape();
  if (shape.size() != 2 || shape[0] < 0 || shape[1] <= 0 ||
      packed_row > static_cast<uint64_t>(shape[0]) || rows > static_cast<uint64_t>(shape[0]) - packed_row)
    throw std::logic_error("Media replacement exceeds its packed embedding range.");
  const auto row_bytes = MultiplyBytes(static_cast<size_t>(shape[1]), Ort::SizeOf(embeddings.GetType()));
  const auto end_row = AddBytes(first_row, rows);
  for (const auto& span : media) {
    const auto begin = std::max(first_row, span.start);
    const auto end = std::min(end_row, span.End());
    if (begin >= end) continue;
    if (!span.item->features) throw std::logic_error("Decoder plan references unprepared media features.");
    const auto feature_shape = span.item->features->GetShape();
    if (feature_shape.size() != 2 || feature_shape[0] != static_cast<int64_t>(span.item->rows) ||
        feature_shape[1] != shape[1] || span.item->features->GetType() != embeddings.GetType())
      throw std::logic_error("Media features do not match the decoder embedding contract.");
    embeddings.GetByteSpan()
        .subspan(MultiplyBytes(packed_row + begin - first_row, row_bytes), MultiplyBytes(end - begin, row_bytes))
        .CopyFrom(span.item->features->GetByteSpan().subspan(
            MultiplyBytes(span.feature_offset + begin - span.start, row_bytes), MultiplyBytes(end - begin, row_bytes)));
  }
}

void MediaPositions(std::span<const MediaSpan> media, size_t first_row,
                    const std::array<std::span<int64_t>, 3>& positions) {
  const size_t count = positions[0].size();
  if (positions[1].size() != count || positions[2].size() != count)
    throw std::invalid_argument("Media position planes must have equal lengths.");
  if (first_row > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - first_row)
    throw std::overflow_error("Media position range exceeds int64.");
  size_t previous_end = 0;
  for (const auto& span : media) {
    span.Validate();
    if (span.start < previous_end)
      throw std::invalid_argument("Media position spans must be ordered and nonoverlapping.");
    previous_end = span.End();
  }
  int64_t delta = 0;
  size_t span_index = 0;
  size_t index = 0;
  while (index < count) {
    const size_t row = first_row + index;
    while (span_index < media.size() && row >= media[span_index].End()) {
      const auto& span = media[span_index++];
      const auto advance = span.positions ? span.positions->text_advance : static_cast<int64_t>(span.rows);
      delta = AddPosition(delta, advance - static_cast<int64_t>(span.rows));
    }
    if (span_index == media.size() || row < media[span_index].start) {
      const size_t end = span_index == media.size() ? count : std::min(count, media[span_index].start - first_row);
      for (auto plane : positions)
        for (size_t offset = index; offset < end; ++offset)
          plane[offset] = AddPosition(static_cast<int64_t>(first_row + offset), delta);
      index = end;
      continue;
    }
    const auto& span = media[span_index];
    const auto base = AddPosition(static_cast<int64_t>(span.start), delta);
    const size_t size = std::min(count - index, span.rows - (row - span.start));
    if (span.positions) {
      span.positions->Fill(row - span.start, base,
                           {positions[0].subspan(index, size), positions[1].subspan(index, size),
                            positions[2].subspan(index, size)});
    } else {
      for (auto plane : positions)
        for (size_t offset = index; offset < index + size; ++offset)
          plane[offset] = AddPosition(static_cast<int64_t>(first_row + offset), delta);
    }
    index += size;
  }
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
  for (auto prefix = prefixes_.begin(); accounting_->host_bytes > host_limit_ - bytes && prefix != prefixes_.end();) {
    if (prefix->use_count() == 1)
      prefix = prefixes_.erase(prefix);
    else
      ++prefix;
  }
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

void MediaStore::Identify(PreparedMediaTurn& turn, size_t first_row, std::span<const MediaSpan> history) {
  if (turn.media.empty()) return;
  auto delta = AddPosition(MediaPosition(history, first_row)[0], -static_cast<int64_t>(first_row));
  const auto identity_bytes = [](const MediaSpan& span) {
    return AddBytes(AddBytes(sizeof(MediaPrefix) + 3 * sizeof(std::shared_ptr<MediaPrefix>),
                             MultiplyBytes(span.rows, sizeof(int32_t))),
                    span.positions ? span.positions->SizeBytes() : 0);
  };
  size_t estimate = 0;
  size_t previous_end = first_row;
  for (const auto& span : turn.media) {
    span.Validate();
    if (span.start < previous_end)
      throw std::invalid_argument("Media prefix spans must be ordered and nonoverlapping.");
    if (span.start - first_row > turn.tokens.size() ||
        span.rows > turn.tokens.size() - (span.start - first_row))
      throw std::invalid_argument("Media prefix range does not match the expanded tokens.");
    previous_end = span.End();
    estimate = AddBytes(estimate, identity_bytes(span));
  }
  MakeHostRoom(estimate);
  accounting_->host_bytes += estimate;
  HostReservation reservation{accounting_, estimate};
  prefixes_.reserve(AddBytes(prefixes_.size(), turn.media.size()));
  for (auto& span : turn.media) {
    const auto values = std::span<const int32_t>{turn.tokens}.subspan(span.start - first_row, span.rows);
    const auto base = AddPosition(static_cast<int64_t>(span.start), delta);
    const auto advance = span.positions ? span.positions->text_advance : static_cast<int64_t>(span.rows);
    delta = AddPosition(delta, advance - static_cast<int64_t>(span.rows));
    const auto match = std::find_if(prefixes_.begin(), prefixes_.end(), [&](const auto& prefix) {
      return prefix->Matches(span, values, base);
    });
    if (match != prefixes_.end()) {
      span.prefix = *match;
      span.positions = (*match)->positions;
      continue;
    }
    if (span.rows > static_cast<uint64_t>(std::numeric_limits<int32_t>::max() - next_prefix_row_))
      throw std::overflow_error("Engine media prefix identity space is exhausted.");
    auto prefix = std::make_shared<MediaPrefix>();
    prefix->item = span.item;
    prefix->positions = span.positions;
    prefix->tokens.assign(values.begin(), values.end());
    prefix->feature_offset = span.feature_offset;
    prefix->position_base = base;
    prefix->first_id = static_cast<int32_t>(next_prefix_row_);
    const auto bytes = identity_bytes(span);
    prefix->accounting = accounting_;
    prefix->host_bytes = bytes;
    reservation.bytes -= bytes;
    next_prefix_row_ += span.rows;
    prefixes_.push_back(prefix);
    span.prefix = std::move(prefix);
  }
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
    const auto grid = QwenImageGrid(info.width, info.height, patch, merge, min_pixels_, max_pixels_);
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
  estimate = AddBytes(estimate, MultiplyBytes(AddBytes(expected_rows, prompt.size()), 128));
  MakeHostRoom(estimate);
  accounting_->host_bytes += estimate;
  HostReservation reservation{accounting_, estimate};
  std::vector<std::array<int64_t, 3>> planned_grids;
  planned_grids.reserve(count);
  for (const auto& info : images->info_)
    planned_grids.push_back(QwenImageGrid(info.width, info.height, patch, merge, min_pixels_, max_pixels_));
  const auto expanded = ExpandQwenMediaPrompt(prompt, planned_grids, {}, merge, host_limit_);
  PreparedMediaTurn result;
  result.tokens = model_->vision_->processor->tokenizer_->Encode(expanded.c_str());
  if (result.tokens.size() != expected_rows)
    throw std::runtime_error("Image prompt expansion violated its resolved row count.");
  if (result.tokens.size() >= remaining_rows)
    throw std::invalid_argument("Expanded image tokens exceed the remaining session capacity.");
  const auto processor = std::dynamic_pointer_cast<QwenImageProcessor>(model_->vision_->processor->processor_);
  if (!processor) throw std::logic_error("Packed Qwen image preparation has no image transform.");
  auto processed = processor->ProcessImages(*images);
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
    if (!std::equal(planned_grids[image].begin(), planned_grids[image].end(), grid))
      throw std::runtime_error("Image preprocessing violated its resolved geometry.");
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
    auto item = Intern({grid[0], grid[1], grid[2]}, *pixels, patch_offset,
                       result.media, reservation.bytes);
    result.media.push_back({first_row + index, std::move(item),
                            MakeQwenVisionPosition({grid[0], grid[1], grid[2]}, merge)});
    patch_offset += stride;
    index += rows - 1;
    ++image_index;
  }
  if (image_index != count) throw std::runtime_error("Image preprocessing omitted an image.");
  return result;
}

void MediaStore::ValidateInputs(const NamedTensors& inputs) const {
  const auto& info = model_->vision_->session_info;
  if (inputs.size() != info.GetInputNames().size())
    throw std::invalid_argument("Media inputs do not cover the encoder session.");
  for (const auto& [name, tensor] : inputs) {
    if (!tensor || tensor->p_device_ != GetDeviceInterface(DeviceType::CPU) ||
        !info.HasInput(name) || tensor->GetType() != info.GetInputDataType(name))
      throw std::invalid_argument("Media input does not match its configured CPU encoder feed.");
    const auto shape = tensor->GetShape();
    const auto expected = info.GetInputShape(name);
    if (shape.size() != expected.size())
      throw std::invalid_argument("Media input rank does not match its encoder feed.");
    for (size_t axis = 0; axis < shape.size(); ++axis) {
      if (shape[axis] < 0 || (expected[axis] >= 0 && shape[axis] != expected[axis]))
        throw std::invalid_argument("Media input shape does not match its encoder feed.");
    }
  }
}

std::shared_ptr<MediaItem> MediaStore::Intern(NamedTensors inputs, size_t rows,
                                              std::span<const MediaSpan> pending, size_t& reservation) {
  ValidateInputs(inputs);
  if (rows == 0 || rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    throw std::invalid_argument("Media encoder output rows must be positive and fit int64.");
  const auto matches = [&](const std::shared_ptr<MediaItem>& cached) {
    if (cached->rows != rows || cached->inputs.size() != inputs.size()) return false;
    for (const auto& [name, tensor] : inputs) {
      const auto found = cached->inputs.find(name);
      if (found == cached->inputs.end() || found->second->GetType() != tensor->GetType() ||
          found->second->GetShape() != tensor->GetShape())
        return false;
      const auto bytes = MultiplyBytes(tensor->GetElementCount(), Ort::SizeOf(tensor->GetType()));
      if (bytes != 0 && std::memcmp(found->second->GetRawData(), tensor->GetRawData(), bytes) != 0)
        return false;
    }
    return true;
  };
  for (const auto& cached : items_) {
    if (matches(cached)) return cached;
  }
  for (const auto& span : pending) {
    if (matches(span.item)) return span.item;
  }
  size_t retained_bytes = AddBytes(sizeof(MediaItem) + sizeof(std::shared_ptr<MediaItem>),
                                   MultiplyBytes(inputs.bucket_count(), sizeof(void*)));
  for (const auto& [name, tensor] : inputs) {
    const auto metadata = AddBytes(sizeof(NamedTensors::value_type) + sizeof(Tensor) + 4 * sizeof(void*),
                                   AddBytes(AddBytes(name.capacity(), 1), MultiplyBytes(tensor->GetShape().size(), sizeof(int64_t))));
    retained_bytes = AddBytes(retained_bytes, AddBytes(
                                                  metadata, MultiplyBytes(tensor->GetElementCount(), Ort::SizeOf(tensor->GetType()))));
  }
  if (retained_bytes > reservation) throw std::logic_error("Canonical media exceeded its host reservation.");
  auto item = std::make_shared<MediaItem>();
  item->inputs = std::move(inputs);
  item->rows = rows;
  item->feature_bytes = MultiplyBytes(MultiplyBytes(rows, static_cast<size_t>(model_->cpu_embedding_->hidden_size_)),
                                      Ort::SizeOf(model_->cpu_embedding_->type_));
  item->accounting = accounting_;
  item->host_bytes = retained_bytes;
  reservation -= retained_bytes;
  return item;
}

std::shared_ptr<MediaItem> MediaStore::Intern(const std::array<int64_t, 3>& grid,
                                              const Tensor& pixels, size_t patch_offset,
                                              std::span<const MediaSpan> pending, size_t& reservation) {
  if (grid[0] != 1 || pixels.p_device_ != GetDeviceInterface(DeviceType::CPU))
    throw std::logic_error("Qwen encoder units require CPU patches and a single temporal group.");
  const size_t patches = MultiplyBytes(static_cast<size_t>(grid[1]), static_cast<size_t>(grid[2]));
  const size_t rows = MergedGridRows(grid, static_cast<size_t>(model_->config_->model.vision.spatial_merge_size));
  const auto width = pixels.GetShape().back();
  if (AddBytes(patch_offset, patches) > pixels.GetElementCount() / static_cast<size_t>(width))
    throw std::logic_error("Canonical media exceeds its processed patch buffer.");
  const size_t row_bytes = MultiplyBytes(static_cast<size_t>(width), Ort::SizeOf(pixels.GetType()));
  const size_t bytes = MultiplyBytes(patches, row_bytes);
  if (AddBytes(bytes, sizeof(grid)) > reservation)
    throw std::logic_error("Canonical media inputs exceed their host reservation.");
  const auto* data = static_cast<const uint8_t*>(pixels.GetRawData()) + MultiplyBytes(patch_offset, row_bytes);
  auto owned_pixels = std::make_shared<Tensor>(GetDeviceInterface(DeviceType::CPU), pixels.GetType());
  const std::array<int64_t, 2> item_shape{static_cast<int64_t>(patches), width};
  owned_pixels->CreateTensor(item_shape);
  std::memcpy(owned_pixels->GetMutableRawData(), data, bytes);
  auto owned_grid = std::make_shared<Tensor>(GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<int64_t>);
  owned_grid->CreateTensor(std::array<int64_t, 2>{1, 3});
  std::copy(grid.begin(), grid.end(), owned_grid->GetMutableData<int64_t>());
  const auto& config = model_->config_->model.vision;
  NamedTensors inputs;
  inputs.emplace(config.inputs.pixel_values, std::move(owned_pixels));
  inputs.emplace(config.inputs.image_grid_thw, std::move(owned_grid));
  return Intern(std::move(inputs), rows, pending, reservation);
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
    const auto grid = QwenImageGrid(info.width, info.height, patch, merge, min_pixels_, max_pixels_);
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
    const auto grid = QwenVideoGrid(static_cast<uint32_t>(first.width), static_cast<uint32_t>(first.height),
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
  const auto prompt_bytes = QwenMediaPromptBytes(prompt, image_grids, clips, merge, host_limit_);
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
  std::vector<std::shared_ptr<const VisionPosition>> image_positions, video_positions;
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
      auto item = Intern(grid, *pixels, offset, pending, reservation.bytes);
      image_items.push_back(item);
      image_positions.push_back(MakeQwenVisionPosition(grid, merge));
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
      auto item = Intern({1, grid[1], grid[2]},
                         *converted.at(std::string{Config::Defaults::PixelValuesName}),
                         MultiplyBytes(group, patches), pending, reservation.bytes);
      video_items.push_back(item);
      video_positions.push_back(MakeQwenVisionPosition({1, grid[1], grid[2]}, merge));
      pending.push_back({0, std::move(item)});
    }
  }
  size_t image = 0, video = 0;
  for (size_t index = 0; index < result.tokens.size(); ++index) {
    const auto token = result.tokens[index];
    if (token != config.image_token_id && token != config.video_token_id) continue;
    const auto& collection = token == config.image_token_id ? image_items : video_items;
    const auto& positions = token == config.image_token_id ? image_positions : video_positions;
    auto& item_index = token == config.image_token_id ? image : video;
    if (item_index == collection.size()) throw std::runtime_error("Expanded media has no corresponding encoder unit.");
    const auto unit = item_index++;
    const auto& item = collection[unit];
    if (item->rows > result.tokens.size() - index ||
        !std::all_of(result.tokens.begin() + index, result.tokens.begin() + index + item->rows,
                     [&](int32_t value) { return value == token; }))
      throw std::runtime_error("Expanded media span does not match its processed geometry.");
    result.media.push_back({AddBytes(first_row, index), item, positions[unit]});
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
  ValidateInputs(item->inputs);
  if (item->feature_bytes > device_limit_) throw std::runtime_error("An encoder unit cannot fit the device media budget.");
  for (auto& cached : items_) {
    if (accounting_->device_bytes <= device_limit_ - item->feature_bytes) break;
    cached->ReleaseFeatures();
  }
  const auto& config = model_->config_->model.vision;
  const auto input_names = model_->vision_->session->GetInputNames();
  std::vector<const char*> names;
  std::vector<OrtValue*> inputs;
  names.reserve(input_names.size());
  inputs.reserve(input_names.size());
  for (const auto& name : input_names) {
    names.push_back(name.c_str());
    inputs.push_back(item->inputs.at(name)->GetOrtTensor());
  }
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
  model_->vision_->session->Run(options.get(), names.data(), inputs.data(), inputs.size(), &output_name, &output, 1);
  model_->vision_->device->Synchronize();
  item->features = std::move(features);
  accounting_->device_bytes += item->feature_bytes;
}

void MediaStore::ReleaseFeatures() {
  for (auto& item : items_) item->ReleaseFeatures();
}

}  // namespace Generators
