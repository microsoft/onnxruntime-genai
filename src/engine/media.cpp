// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "media.h"
#include "json.h"
#include "models/decoder_only.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/preprocessing/multi_modal_processor.h"
#include "models/preprocessing/processor.h"

#include <cmath>
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

struct ProcessingLimits : JSON::Element {
  int64_t min_pixels{}, max_pixels{}, patch_size{}, temporal_patch_size{}, merge_size{}, smart_resize{};
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
    }
  }
};

struct HostReservation {
  std::shared_ptr<MediaAccounting> accounting;
  size_t bytes{};
  ~HostReservation() { accounting->host_bytes -= bytes; }
};

}  // namespace

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
  int64_t delta = 0;
  for (const auto& span : media) {
    if (row < span.start) break;
    const auto height = span.item->grid[1] / static_cast<int64_t>(merge_size);
    const auto width = span.item->grid[2] / static_cast<int64_t>(merge_size);
    const auto base = static_cast<int64_t>(span.start) + delta;
    if (row - span.start < span.item->rows) {
      const auto offset = static_cast<int64_t>(row - span.start);
      return {base, base + offset / width, base + offset % width};
    }
    delta += std::max(height, width) - static_cast<int64_t>(span.item->rows);
  }
  const auto position = static_cast<int64_t>(row) + delta;
  return {position, position, position};
}

std::array<int64_t, 3> MediaImageGrid(uint32_t width, uint32_t height, size_t patch, size_t merge,
                                      int64_t min_pixels, int64_t max_pixels) {
  if (width == 0 || height == 0 || width > INT32_MAX || height > INT32_MAX ||
      patch == 0 || merge == 0 || min_pixels <= 0 || max_pixels < min_pixels)
    throw std::invalid_argument("Invalid dimensions or resize limits for packed vision.");
  if (static_cast<double>(std::max(width, height)) / std::min(width, height) > 200.0)
    throw std::invalid_argument("Image aspect ratio exceeds the Qwen preprocessing limit.");
  const size_t factor = MultiplyBytes(patch, merge);
  const auto rounded = [factor](size_t dimension) {
    const size_t quotient = dimension / factor, remainder = dimension % factor;
    const bool increment = remainder > factor - remainder || (remainder == factor - remainder && quotient % 2 != 0);
    return std::max(factor, MultiplyBytes(quotient + increment, factor));
  };
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

PreparedMediaTurn MediaStore::Prepare(const std::string& prompt, const Images* images, size_t first_row, size_t remaining_rows) {
  auto& tokenizer = *model_->vision_->processor->tokenizer_;
  const auto& config = model_->config_->model;
  const size_t count = images ? images->num_images_ : 0;
  const auto marker = tokenizer.Encode("<|vision_end|>");
  if (marker.size() != 1) throw std::logic_error("The packed vision tokenizer has no vision-end token.");
  const auto raw_tokens = tokenizer.Encode(prompt.c_str());
  size_t markers = 0;
  for (size_t index = 0; index < raw_tokens.size(); ++index) {
    if (raw_tokens[index] == config.vision_start_token_id) {
      if (index + 2 >= raw_tokens.size() || raw_tokens[index + 1] != config.image_token_id ||
          raw_tokens[index + 2] != marker[0])
        throw std::invalid_argument("Each image must use a complete vision-start/image-pad/vision-end marker.");
      ++markers;
      index += 2;
    } else if (raw_tokens[index] == config.image_token_id || raw_tokens[index] == marker[0]) {
      throw std::invalid_argument("Image markers must form complete ordered image items.");
    }
  }
  if (markers != count) throw std::invalid_argument("Image marker count does not match the ordered image inputs.");
  if (raw_tokens.empty() || raw_tokens.size() > remaining_rows)
    throw std::invalid_argument("The formatted turn does not fit the remaining session rows.");
  if (count == 0) return {raw_tokens, {}};
  if (images->info_.size() != count) throw std::invalid_argument("Image header metadata is unavailable.");
  const size_t merge = static_cast<size_t>(config.vision.spatial_merge_size);
  const size_t patch = static_cast<size_t>(config.vision.patch_size);
  const auto pixel_shape = model_->vision_->session_info.GetInputShape(config.vision.inputs.pixel_values);
  const size_t patch_bytes = MultiplyBytes(static_cast<size_t>(pixel_shape[1]), sizeof(float));
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
    const size_t bytes = MultiplyBytes(MultiplyBytes(patches, static_cast<size_t>(shape.back())), Ort::SizeOf(pixels->GetType()));
    const auto* data = static_cast<const uint8_t*>(pixels->GetRawData()) +
                       MultiplyBytes(MultiplyBytes(patch_offset, static_cast<size_t>(shape.back())), Ort::SizeOf(pixels->GetType()));
    std::shared_ptr<MediaItem> item;
    const auto matches = [&](const std::shared_ptr<MediaItem>& cached) {
      return cached->grid == std::array<int64_t, 3>{grid[0], grid[1], grid[2]} &&
             cached->pixels->GetElementCount() * Ort::SizeOf(cached->pixels->GetType()) == bytes &&
             std::memcmp(cached->pixels->GetRawData(), data, bytes) == 0;
    };
    for (const auto& cached : items_) {
      if (matches(cached)) {
        item = cached;
        break;
      }
    }
    if (!item) {
      for (const auto& span : result.media) {
        if (matches(span.item)) {
          item = span.item;
          break;
        }
      }
    }
    if (!item) {
      if (static_cast<int64_t>(rows) > std::numeric_limits<int32_t>::max() - next_prefix_row_)
        throw std::overflow_error("Engine media prefix identity space is exhausted.");
      item = std::make_shared<MediaItem>();
      item->grid = {grid[0], grid[1], grid[2]};
      item->rows = rows;
      item->prefix_base = static_cast<int32_t>(next_prefix_row_);
      next_prefix_row_ += rows;
      item->pixels = std::make_shared<Tensor>(GetDeviceInterface(DeviceType::CPU), pixels->GetType());
      const std::array<int64_t, 2> item_shape{static_cast<int64_t>(patches), shape.back()};
      item->pixels->CreateTensor(item_shape);
      std::memcpy(item->pixels->GetMutableRawData(), data, bytes);
      item->feature_bytes = MultiplyBytes(MultiplyBytes(rows, static_cast<size_t>(model_->cpu_embedding_->hidden_size_)),
                                          Ort::SizeOf(model_->cpu_embedding_->type_));
      const auto retained_bytes = AddBytes(bytes, sizeof(MediaItem) + sizeof(std::shared_ptr<MediaItem>));
      if (retained_bytes > reservation.bytes) throw std::logic_error("Canonical media exceeded its host reservation.");
      item->accounting = accounting_;
      item->host_bytes = retained_bytes;
      reservation.bytes -= item->host_bytes;
    }
    result.media.push_back({first_row + index, std::move(item)});
    patch_offset += stride;
    index += rows - 1;
    ++image_index;
  }
  if (image_index != count) throw std::runtime_error("Image preprocessing omitted an image.");
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
