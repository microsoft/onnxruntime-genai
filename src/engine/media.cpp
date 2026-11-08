// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "media.h"
#include "models/decoder_only.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/preprocessing/multi_modal_processor.h"
#include "models/preprocessing/processor.h"

#include <algorithm>
#include <cstring>
#include <limits>

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
  if (!model_->vision_ || !model_->vision_->preparation)
    throw std::runtime_error("Packed vision requires a bounded processor preparation contract.");
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
  model_->vision_->preparation->ValidateTextTokens(tokens);
}

PreparedMediaTurn MediaStore::Prepare(const std::string& prompt, std::span<const int32_t> raw_tokens,
                                      const Images* images, size_t first_row, size_t remaining_rows,
                                      const Videos* videos) {
  const VisionPreparationInput input{prompt, raw_tokens, images, videos, remaining_rows};
  const auto& processor = *model_->vision_->preparation;
  const auto estimate = processor.Estimate(input, host_limit_);
  MakeHostRoom(estimate);
  accounting_->host_bytes += estimate;
  HostReservation reservation{accounting_, estimate};
  const auto plan = processor.Plan(*model_->vision_->processor->tokenizer_, input, host_limit_);
  if (!plan || plan->tokens.empty() || plan->tokens.size() >= remaining_rows)
    throw std::invalid_argument("Expanded media tokens exceed the remaining session capacity.");
  std::vector<std::shared_ptr<MediaItem>> units;
  units.reserve(plan->units.size());
  const auto input_count = model_->vision_->session_info.GetInputNames().size();
  for (const auto& unit : plan->units) {
    if (!unit.rows || unit.rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        unit.inputs.size() != input_count)
      throw std::invalid_argument("Media encoder unit does not match its session contract.");
    for (size_t index = 0; index < unit.inputs.size(); ++index) {
      const auto& feed = unit.inputs[index];
      ValidateInput(feed.name, feed.type, feed.shape);
      for (size_t previous = 0; previous < index; ++previous)
        if (feed.name == unit.inputs[previous].name)
          throw std::invalid_argument("Media encoder plan repeats an input name.");
    }
    const auto feature_bytes = MultiplyBytes(unit.rows, MultiplyBytes(
                                                            static_cast<size_t>(model_->cpu_embedding_->hidden_size_), Ort::SizeOf(model_->cpu_embedding_->type_)));
    if (feature_bytes > device_limit_)
      throw std::invalid_argument("An encoder unit cannot fit the device media budget.");
    auto item = std::make_shared<MediaItem>();
    item->rows = unit.rows;
    units.push_back(std::move(item));
  }
  PreparedMediaTurn result;
  result.tokens = plan->tokens;
  result.media.reserve(plan->spans.size());
  size_t previous_end = 0;
  for (const auto& span : plan->spans) {
    if (span.unit >= units.size() || !span.rows || span.start < previous_end ||
        span.start > result.tokens.size() || span.rows > result.tokens.size() - span.start)
      throw std::invalid_argument("Media plan spans do not match its token or feature ranges.");
    result.media.emplace_back(AddBytes(first_row, span.start), units[span.unit], span.positions,
                              span.rows, span.feature_offset);
    previous_end = AddBytes(span.start, span.rows);
  }
  auto feeds = processor.Materialize(*plan);
  if (feeds.size() != units.size())
    throw std::runtime_error("Media materialization omitted an encoder unit.");
  std::vector<MediaSpan> pending;
  pending.reserve(units.size());
  for (size_t unit = 0; unit < units.size(); ++unit) {
    ValidateInputs(feeds[unit]);
    for (const auto& spec : plan->units[unit].inputs) {
      const auto& tensor = feeds[unit].at(spec.name);
      if (tensor->GetType() != spec.type || tensor->GetShape() != spec.shape)
        throw std::runtime_error("Media materialization violated its resolved feed metadata.");
    }
    auto item = Intern(std::move(feeds[unit]), units[unit]->rows, pending, reservation.bytes);
    pending.emplace_back(0, std::move(item));
  }
  for (size_t span = 0; span < result.media.size(); ++span)
    result.media[span].item = pending[plan->spans[span].unit].item;
  return result;
}

void MediaStore::ValidateInput(const std::string& name, ONNXTensorElementDataType type,
                               std::span<const int64_t> shape) const {
  const auto& info = model_->vision_->session_info;
  if (!info.HasInput(name) || type != info.GetInputDataType(name))
    throw std::invalid_argument("Media input does not match its configured encoder feed.");
  const auto expected = info.GetInputShape(name);
  if (shape.size() != expected.size())
    throw std::invalid_argument("Media input rank does not match its encoder feed.");
  size_t elements = 1;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    if (shape[axis] < 0 || (expected[axis] >= 0 && shape[axis] != expected[axis]) ||
        static_cast<uint64_t>(shape[axis]) > std::numeric_limits<size_t>::max())
      throw std::invalid_argument("Media input shape does not match its encoder feed.");
    elements = MultiplyBytes(elements, static_cast<size_t>(shape[axis]));
  }
  (void)MultiplyBytes(elements, Ort::SizeOf(type));
}

void MediaStore::ValidateInputs(const NamedTensors& inputs) const {
  const auto& info = model_->vision_->session_info;
  if (inputs.size() != info.GetInputNames().size())
    throw std::invalid_argument("Media inputs do not cover the encoder session.");
  for (const auto& [name, tensor] : inputs) {
    if (!tensor || tensor->p_device_ != GetDeviceInterface(DeviceType::CPU))
      throw std::invalid_argument("Media input does not match its configured CPU encoder feed.");
    ValidateInput(name, tensor->GetType(), tensor->GetShape());
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
