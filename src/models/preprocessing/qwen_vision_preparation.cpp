// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qwen_vision_preparation.h"
#include "genai_tokenizer.h"
#include "qwen2_5_vl_image_processor.h"
#include "qwen_vision_position.h"
#include "qwen_vision_processing.h"
#include "json.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>

namespace Generators {
namespace {

size_t Add(size_t left, size_t right) {
  if (right > std::numeric_limits<size_t>::max() - left)
    throw std::overflow_error("Vision preparation size overflows size_t.");
  return left + right;
}

size_t Multiply(size_t left, size_t right) {
  if (left && right > std::numeric_limits<size_t>::max() / left)
    throw std::overflow_error("Vision preparation size overflows size_t.");
  return left * right;
}

size_t Patches(const std::array<int64_t, 3>& grid) {
  return Multiply(static_cast<size_t>(grid[1]), static_cast<size_t>(grid[2]));
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
    if (name == "type")
      operations.emplace_back(JSON::Get<std::string_view>(value));
    else if (name == "min_pixels")
      min_pixels = SafeDoubleToInt(JSON::Get<double>(value), name);
    else if (name == "max_pixels")
      max_pixels = SafeDoubleToInt(JSON::Get<double>(value), name);
    else if (name == "patch_size")
      patch_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    else if (name == "temporal_patch_size")
      temporal_patch_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    else if (name == "merge_size")
      merge_size = SafeDoubleToInt(JSON::Get<double>(value), name);
    else if (name == "smart_resize")
      smart_resize = SafeDoubleToInt(JSON::Get<double>(value), name);
    else if (name == "max_video_tokens")
      max_video_tokens = SafeDoubleToInt(JSON::Get<double>(value), name);
    else if (name == "coefficient_type")
      coefficient_type = JSON::Get<std::string_view>(value);
    else if (name == "qwen2_5_vl" || name == "qwen3_vl")
      bgr_normalization = bgr_normalization || SafeDoubleToInt(JSON::Get<double>(value), name) != 0;
  }
};

struct Metadata : QwenMediaPromptSource {
  Metadata(const VisionPreparationInput& input, size_t patch, size_t merge,
           int64_t min_pixels, int64_t max_pixels, int64_t video_min, int64_t video_max,
           size_t max_video_tokens)
      : input{input}, patch{patch}, merge{merge}, min_pixels{min_pixels}, max_pixels{max_pixels}, video_min{video_min}, video_max{video_max}, max_video_tokens{max_video_tokens} {
    if (ImageCount() > input.remaining_rows || VideoCount() > input.remaining_rows ||
        (input.images && input.images->info_.size() != ImageCount()))
      throw std::invalid_argument("Media input metadata exceeds the remaining session capacity.");
  }
  size_t ImageCount() const override { return input.images ? input.images->num_images_ : 0; }
  size_t VideoCount() const override { return input.videos ? input.videos->clips_.size() : 0; }
  std::array<int64_t, 3> ImageGrid(size_t image) const override {
    const auto& info = input.images->info_[image];
    if (!info.width || !info.height)
      throw std::invalid_argument("Packed vision requires PNG or JPEG dimensions before preprocessing.");
    return QwenImageGrid(info.width, info.height, patch, merge, min_pixels, max_pixels);
  }
  size_t FrameCount(size_t video) const override {
    size_t count{};
    CheckResult(OrtxGetRawFrameCount(input.videos->clips_[video].get(), &count));
    if (count < 2 || count / 2 + count % 2 > input.remaining_rows)
      throw std::invalid_argument("Video selected frame count is incompatible with the remaining session capacity.");
    return count;
  }
  OrtxFrame Frame(size_t video, size_t frame) const {
    OrtxFrame result{};
    CheckResult(OrtxGetRawFrame(input.videos->clips_[video].get(), frame, &result));
    return result;
  }
  double Timestamp(size_t video, size_t frame) const override { return Frame(video, frame).timestamp_seconds; }
  std::array<int64_t, 3> VideoGrid(size_t video) const override {
    const auto count = FrameCount(video);
    const auto first = Frame(video, 0);
    if (first.width <= 0 || first.height <= 0 || first.width > UINT32_MAX || first.height > UINT32_MAX)
      throw std::invalid_argument("Video source geometry exceeds the model dimension limit.");
    for (size_t frame = 1; frame < count; ++frame) {
      const auto current = Frame(video, frame);
      if (current.width != first.width || current.height != first.height)
        throw std::invalid_argument("Qwen video requires uniform source geometry within each clip.");
    }
    return QwenVideoGrid(static_cast<uint32_t>(first.width), static_cast<uint32_t>(first.height),
                         count, patch, merge, video_min, video_max, max_video_tokens);
  }
  const VisionPreparationInput& input;
  size_t patch, merge;
  int64_t min_pixels, max_pixels, video_min, video_max;
  size_t max_video_tokens;
};

struct QwenPlan : VisionPlan {
  const QwenVisionPreparation* owner{};
  const Images* images{};
  const Videos* videos{};
  std::vector<std::array<int64_t, 3>> image_grids;
  std::vector<std::vector<double>> timestamps;
  std::vector<QwenVideoPrompt> clips;
  std::vector<std::shared_ptr<const VisionPosition>> positions;
};

}  // namespace

QwenVisionPreparation::QwenVisionPreparation(const Config& config, const SessionInfo& info,
                                             const Tokenizer& tokenizer, const QwenImageProcessor& images, ThreadPool* thread_pool)
    : config_{config}, images_{images}, thread_pool_{thread_pool} {
  const auto& vision = config.model.vision;
  auto file = (fs::path{config.config_path} / fs::path(vision.config_filename)).open(std::ios::binary);
  if (!file) throw std::runtime_error("Cannot read the packed vision processor configuration.");
  std::ostringstream document;
  document << file.rdbuf();
  ProcessingLimits limits;
  JSON::Parse(limits, document.str());
  if (limits.operations != std::vector<std::string>{"DecodeImage", "ConvertRGB", "Resize", "Rescale", "Normalize", "PatchImage"} ||
      limits.min_pixels <= 0 || limits.max_pixels < limits.min_pixels || limits.patch_size != vision.patch_size ||
      vision.patch_size <= 0 || vision.spatial_merge_size <= 0 ||
      limits.temporal_patch_size != 2 || limits.merge_size != vision.spatial_merge_size || limits.smart_resize != 1)
    throw std::runtime_error("Packed vision requires the bounded Qwen PatchImage preprocessing pipeline.");
  min_pixels_ = limits.min_pixels;
  max_pixels_ = limits.max_pixels;
  const auto pixels = info.GetInputShape(vision.inputs.pixel_values);
  if (info.GetInputNames().size() != 2 || pixels.size() != 2 || pixels[0] >= 0 ||
      pixels[1] <= 0 || static_cast<uint64_t>(pixels[1]) != Multiply(6, Multiply(static_cast<size_t>(vision.patch_size), static_cast<size_t>(vision.patch_size))) ||
      info.GetInputShape(vision.inputs.image_grid_thw) != std::vector<int64_t>{1, 3} ||
      info.GetInputDataType(vision.inputs.image_grid_thw) != Ort::TypeToTensorType<int64_t>)
    throw std::runtime_error("Packed Qwen encoder does not match its patch/grid input contract.");
  pixel_width_ = pixels[1];
  pixel_type_ = info.GetInputDataType(vision.inputs.pixel_values);
  if (pixel_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && pixel_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
      pixel_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)
    throw std::runtime_error("Packed Qwen encoder requires floating-point patches.");
  const auto marker = tokenizer.Encode("<|vision_end|>");
  if (marker.size() != 1) throw std::logic_error("The packed vision tokenizer has no vision-end token.");
  vision_end_token_ = marker[0];
  const auto video_marker = tokenizer.Encode("<|video_pad|>");
  if (video_marker.size() == 1 && video_marker[0] > 0) video_token_ = video_marker[0];
  if (!vision.video_config_filename) return;
  if (config.model.type != "qwen3_5" && config.model.type != "qwen3_5_moe" && config.model.type != "qwen4_exp")
    throw std::runtime_error("Packed video requires a verified Qwen3.5 or Qwen4Exp position contract.");
  auto video_file = config.ResolvePath(*vision.video_config_filename).open(std::ios::binary);
  if (!video_file) throw std::runtime_error("Cannot read the packed video processor configuration.");
  std::ostringstream video_document;
  video_document << video_file.rdbuf();
  const auto definition = video_document.str();
  ProcessingLimits video;
  JSON::Parse(video, definition);
  if (video.operations != std::vector<std::string>{"Resize", "Rescale", "Normalize", "PatchImage"} ||
      video.min_pixels <= 0 || video.max_pixels < video.min_pixels || video.patch_size != vision.patch_size ||
      video.temporal_patch_size != 2 || video.merge_size != vision.spatial_merge_size ||
      video.smart_resize != 0 || video.coefficient_type != "int16" || video.max_video_tokens <= 0 ||
      video.bgr_normalization)
    throw std::runtime_error("Packed video requires the bounded Qwen3-VL temporal preprocessing pipeline.");
  if (!video_token_ || video_token_ != config.model.video_token_id)
    throw std::runtime_error("Packed video requires the source video-pad token.");
  video_min_pixels_ = video.min_pixels;
  video_max_pixels_ = video.max_pixels;
  max_video_tokens_ = static_cast<size_t>(video.max_video_tokens);
  CheckResult(OrtxCreateProcessor(videos_.ToBeAssigned(), definition.c_str()));
}

void QwenVisionPreparation::ValidateTextTokens(std::span<const int32_t> tokens) const {
  const auto& config = config_.model;
  if (video_token_ && std::find(tokens.begin(), tokens.end(), video_token_) != tokens.end())
    throw std::invalid_argument("Video markers require the formatted prompt and videos overload.");
  if (std::any_of(tokens.begin(), tokens.end(), [&](int32_t token) {
        return token == config.image_token_id || token == config.vision_start_token_id || token == vision_end_token_;
      }))
    throw std::invalid_argument("Image markers require the formatted prompt and images overload.");
}

std::array<int64_t, 2> QwenVisionPreparation::CoordinateRange() const {
  if (config_.model.context_length <= 0)
    throw std::runtime_error("Packed Qwen positions require a positive exported context length.");
  return {0, static_cast<int64_t>(config_.model.context_length) - 1};
}

size_t QwenVisionPreparation::Estimate(const VisionPreparationInput& input, size_t host_limit) const {
  const auto patch = static_cast<size_t>(config_.model.vision.patch_size);
  const auto merge = static_cast<size_t>(config_.model.vision.spatial_merge_size);
  Metadata metadata{input, patch, merge, min_pixels_, max_pixels_, video_min_pixels_, video_max_pixels_, max_video_tokens_};
  if (metadata.VideoCount() && !videos_)
    throw std::invalid_argument("This package has no supported video preprocessing contract.");
  if (!metadata.VideoCount() && video_token_ &&
      input.prompt.find("<|video_pad|>") != std::string::npos)
    throw std::invalid_argument("Video markers require ordered video inputs.");
  if (input.prompt.empty())
    throw std::runtime_error("Expected at least one input token for generation. Received 0.");
  if (!metadata.ImageCount() && !metadata.VideoCount()) {
    const auto bytes = QwenMediaPromptBytes(input.prompt, metadata, merge, std::numeric_limits<size_t>::max());
    return Add(1024, Multiply(Add(bytes, 1), 128));
  }
  size_t estimate = 0, units = 0, rows = 0, maximum_image_patches = 0;
  const auto patch_bytes = Multiply(static_cast<size_t>(pixel_width_), 2 * sizeof(float) + 2 * Ort::SizeOf(pixel_type_));
  for (size_t image = 0; image < metadata.ImageCount(); ++image) {
    const auto grid = metadata.ImageGrid(image);
    const auto& info = input.images->info_[image];
    maximum_image_patches = std::max(maximum_image_patches, Patches(grid));
    rows = Add(rows, Patches(grid) / Multiply(merge, merge));
    estimate = Add(estimate, Add(Multiply(Multiply(info.width, info.height), 16),
                                 Multiply(Multiply(Patches(grid), Multiply(patch, patch)), 64)));
    ++units;
  }
  // Transform scratch is released before stacked patches are copied/cast and donated.
  estimate = std::max(estimate, Multiply(Multiply(metadata.ImageCount(), maximum_image_patches), patch_bytes));
  for (size_t video = 0; video < metadata.VideoCount(); ++video) {
    const auto grid = metadata.VideoGrid(video);
    const auto first = metadata.Frame(video, 0);
    const auto count = metadata.FrameCount(video);
    const auto groups = static_cast<size_t>(grid[0]);
    units = Add(units, groups);
    rows = Add(rows, Multiply(groups, Patches(grid) / Multiply(merge, merge)));
    const auto patches = Multiply(groups, Patches(grid));
    estimate = Add(estimate, Add(Multiply(count, Multiply(Multiply(
                                                              static_cast<size_t>(first.width), static_cast<size_t>(first.height)),
                                                          16)),
                                 Add(Multiply(patches, patch_bytes), Multiply(Multiply(patches, Multiply(patch, patch)), 64))));
    estimate = Add(estimate, Multiply(count, sizeof(double)));
  }
  if (estimate > host_limit) return estimate;
  if (rows >= input.remaining_rows)
    throw std::invalid_argument("Expanded media rows exceed the remaining session capacity.");
  const auto prompt_bytes = QwenMediaPromptBytes(input.prompt, metadata, merge, host_limit);
  estimate = Add(estimate, Add(Add(prompt_bytes, 1), Multiply(Add(prompt_bytes, 1), 128)));
  const auto& names = config_.model.vision.inputs;
  estimate = Add(estimate, Multiply(units, Add(1024, Multiply(Add(names.pixel_values.size(), names.image_grid_thw.size()), 4))));
  return estimate;
}

std::unique_ptr<const VisionPlan> QwenVisionPreparation::Plan(const Tokenizer& tokenizer,
                                                              const VisionPreparationInput& input, size_t host_limit) const {
  const auto patch = static_cast<size_t>(config_.model.vision.patch_size);
  const auto merge = static_cast<size_t>(config_.model.vision.spatial_merge_size);
  Metadata metadata{input, patch, merge, min_pixels_, max_pixels_, video_min_pixels_, video_max_pixels_, max_video_tokens_};
  auto plan = std::make_unique<QwenPlan>();
  plan->owner = this;
  plan->images = input.images;
  plan->videos = input.videos;
  if (!metadata.ImageCount() && !metadata.VideoCount()) {
    plan->tokens = tokenizer.Encode(input.prompt.c_str());
    return plan;
  }
  for (size_t image = 0; image < metadata.ImageCount(); ++image)
    plan->image_grids.push_back(metadata.ImageGrid(image));
  plan->timestamps.reserve(metadata.VideoCount());
  plan->clips.reserve(metadata.VideoCount());
  for (size_t video = 0; video < metadata.VideoCount(); ++video) {
    const auto grid = metadata.VideoGrid(video);
    plan->timestamps.emplace_back();
    const auto count = metadata.FrameCount(video);
    plan->timestamps.back().reserve(count);
    for (size_t frame = 0; frame < count; ++frame)
      plan->timestamps.back().push_back(metadata.Timestamp(video, frame));
    plan->clips.push_back({grid, plan->timestamps.back()});
  }
  const auto expanded = ExpandQwenMediaPrompt(input.prompt, plan->image_grids, plan->clips, merge, host_limit);
  plan->tokens = tokenizer.Encode(expanded.c_str());
  if (plan->tokens.empty() || plan->tokens.size() >= input.remaining_rows)
    throw std::invalid_argument("Expanded media tokens exceed the remaining session capacity.");
  const auto& inputs = config_.model.vision.inputs;
  const auto add_unit = [&](std::array<int64_t, 3> grid) {
    const auto patches = Patches(grid);
    if (patches > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      throw std::overflow_error("Qwen encoder unit rows exceed int64.");
    plan->units.push_back({patches / Multiply(merge, merge),
                           {{inputs.pixel_values, pixel_type_, {static_cast<int64_t>(patches), pixel_width_}},
                            {inputs.image_grid_thw, Ort::TypeToTensorType<int64_t>, {1, 3}}}});
    plan->positions.push_back(MakeQwenVisionPosition(grid, merge));
  };
  for (const auto& grid : plan->image_grids) add_unit(grid);
  for (const auto& clip : plan->clips)
    for (int64_t group = 0; group < clip.grid[0]; ++group) add_unit({1, clip.grid[1], clip.grid[2]});
  size_t image = 0, video = plan->image_grids.size();
  for (size_t index = 0; index < plan->tokens.size(); ++index) {
    const auto token = plan->tokens[index];
    const bool is_image = token == config_.model.image_token_id;
    if (!is_image && (!video_token_ || token != video_token_)) continue;
    const auto unit = is_image ? image++ : video++;
    if ((is_image && unit >= plan->image_grids.size()) || unit >= plan->units.size())
      throw std::runtime_error("Expanded media has no corresponding encoder unit.");
    const auto rows = plan->units[unit].rows;
    if (rows > plan->tokens.size() - index ||
        !std::all_of(plan->tokens.begin() + index, plan->tokens.begin() + index + rows,
                     [&](int32_t value) { return value == token; }))
      throw std::runtime_error("Expanded media span does not match its resolved geometry.");
    plan->spans.push_back({index, unit, rows, 0, plan->positions[unit]});
    index += rows - 1;
  }
  if (image != plan->image_grids.size() || video != plan->units.size())
    throw std::runtime_error("Media prompt omitted an encoder unit.");
  return plan;
}

std::vector<NamedTensors> QwenVisionPreparation::Materialize(const VisionPlan& source) const {
  const auto* plan = dynamic_cast<const QwenPlan*>(&source);
  if (!plan || plan->owner != this)
    throw std::invalid_argument("Qwen materialization requires its own resolved plan.");
  std::vector<NamedTensors> result;
  result.reserve(plan->units.size());
  const auto& inputs = config_.model.vision.inputs;
  const auto snapshot = [&](const Tensor& pixels, size_t offset, std::array<int64_t, 3> grid) {
    const auto patches = Patches(grid);
    const auto row_bytes = Multiply(static_cast<size_t>(pixel_width_), Ort::SizeOf(pixel_type_));
    const auto shape = pixels.GetShape();
    if (pixels.p_device_ != GetDeviceInterface(DeviceType::CPU) || pixels.GetType() != pixel_type_ ||
        (shape.size() != 2 && shape.size() != 3) || shape.back() != pixel_width_ ||
        Add(offset, patches) > pixels.GetElementCount() / static_cast<size_t>(pixel_width_))
      throw std::runtime_error("Qwen materialization violated its patch layout.");
    auto owned_pixels = std::make_shared<Tensor>(GetDeviceInterface(DeviceType::CPU), pixel_type_);
    owned_pixels->CreateTensor(std::array<int64_t, 2>{static_cast<int64_t>(patches), pixel_width_});
    std::memcpy(owned_pixels->GetMutableRawData(),
                static_cast<const uint8_t*>(pixels.GetRawData()) + Multiply(offset, row_bytes),
                Multiply(patches, row_bytes));
    auto owned_grid = std::make_shared<Tensor>(GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<int64_t>);
    owned_grid->CreateTensor(std::array<int64_t, 2>{1, 3});
    std::copy(grid.begin(), grid.end(), owned_grid->GetMutableData<int64_t>());
    NamedTensors feeds;
    feeds.emplace(inputs.pixel_values, std::move(owned_pixels));
    feeds.emplace(inputs.image_grid_thw, std::move(owned_grid));
    result.push_back(std::move(feeds));
  };
  if (!plan->image_grids.empty()) {
    auto processed = images_.ProcessImages(*plan->images);
    const auto& grids = processed->at(std::string{Config::Defaults::ImageGridThwName});
    const auto& pixels = processed->at(std::string{Config::Defaults::PixelValuesName});
    const auto shape = pixels->GetShape();
    if (grids->GetType() != Ort::TypeToTensorType<int64_t> || grids->GetElementCount() != Multiply(plan->image_grids.size(), 3) ||
        (shape.size() != 2 && shape.size() != 3) || shape.back() != pixel_width_ ||
        (shape.size() == 3 && shape[0] != static_cast<int64_t>(plan->image_grids.size())))
      throw std::runtime_error("Image preprocessing returned an unsupported patch/grid layout.");
    const auto physical = pixels->GetElementCount() / static_cast<size_t>(pixel_width_);
    size_t logical = 0;
    for (size_t image = 0; image < plan->image_grids.size(); ++image) {
      if (!std::equal(plan->image_grids[image].begin(), plan->image_grids[image].end(), grids->GetData<int64_t>() + image * 3))
        throw std::runtime_error("Image preprocessing violated its resolved geometry.");
      logical = Add(logical, Patches(plan->image_grids[image]));
    }
    if (physical != logical && physical % plan->image_grids.size() != 0)
      throw std::runtime_error("Image patch padding is not a uniform per-image stride.");
    size_t offset = 0;
    for (const auto& grid : plan->image_grids) {
      const auto stride = physical == logical ? Patches(grid) : physical / plan->image_grids.size();
      if (Patches(grid) > stride) throw std::runtime_error("Image patches exceed their physical stride.");
      snapshot(*pixels, offset, grid);
      offset = Add(offset, stride);
    }
  }
  for (size_t video = 0; video < plan->clips.size(); ++video) {
    const auto& grid = plan->clips[video].grid;
    const OrtxFramePreProcessOptions options{sizeof(OrtxFramePreProcessOptions),
                                             grid[1] * config_.model.vision.patch_size,
                                             grid[2] * config_.model.vision.patch_size};
    ort_extensions::OrtxObjectPtr<OrtxTensorResult> processed;
    CheckResult(OrtxFramesPreProcess(videos_.get(), plan->videos->clips_[video].get(), &options, processed.ToBeAssigned()));
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
    const auto patches = Patches(grid);
    if (pixel_type != kOrtxFloat || grid_type != kOrtxInt64 || rank != 2 || grid_rank != 2 ||
        grid_shape[0] != 1 || grid_shape[1] != 3 || shape[1] != pixel_width_ ||
        shape[0] != static_cast<int64_t>(Multiply(static_cast<size_t>(grid[0]), patches)) ||
        !std::equal(grid.begin(), grid.end(), static_cast<const int64_t*>(grid_data)))
      throw std::runtime_error("Video preprocessing violated its resolved patch/grid layout.");
    NamedTensors converted;
    EmplaceProcessedTensor(thread_pool_, converted, Config::Defaults::PixelValuesName, pixels.get(),
                           pixel_type_, Ort::Allocator::GetWithDefaultOptions());
    for (size_t group = 0; group < static_cast<size_t>(grid[0]); ++group)
      snapshot(*converted.at(std::string{Config::Defaults::PixelValuesName}),
               Multiply(group, patches), {1, grid[1], grid[2]});
  }
  return result;
}

}  // namespace Generators
