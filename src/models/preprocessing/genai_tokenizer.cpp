// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "genai_tokenizer.h"

#include "models/model.h"
#include "models/model_type.h"
#include "models/transducer_state.h"
#include "models/preprocessing/tokenizer_tag_utils.h"
#include "tensor.h"

#include <algorithm>
#include <cmath>
#include <cctype>

namespace Generators {

namespace {
double RoundTimestamp(double value) {
  return std::round(value * 100.0) / 100.0;
}

bool EndsWithSeparator(std::string_view text, std::string_view separator) {
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
    text.remove_suffix(1);
  }
  return !separator.empty() && text.ends_with(separator);
}
}  // namespace

TimestampDecodeState::TimestampDecodeState(const TimestampTokenizerConfig& config)
    : level_{config.level},
      segment_separators_{config.segment_separators},
      segment_gap_threshold_frames_{config.segment_gap_threshold_frames} {
  if (config.sample_rate <= 0 || config.hop_length <= 0 || config.subsampling_factor <= 0) {
    throw std::runtime_error("Timestamp decoding requires positive sample_rate, hop_length, and subsampling_factor");
  }
  seconds_per_frame_ = static_cast<double>(config.hop_length) * config.subsampling_factor / config.sample_rate;
}

void TimestampDecodeState::ClearResult() {
  result_.text.clear();
  result_.words.clear();
  result_.segments.clear();
}

void TimestampDecodeState::CompleteSegment() {
  if (!pending_segment_.active) return;

  if (level_ == Config::TimestampLevel::Segment || level_ == Config::TimestampLevel::All) {
    result_.segments.push_back({pending_segment_.text,
                                pending_segment_.start_frame,
                                pending_segment_.stop_frame,
                                RoundTimestamp(pending_segment_.start_frame * seconds_per_frame_),
                                RoundTimestamp(pending_segment_.stop_frame * seconds_per_frame_)});
  }
  pending_segment_ = {};
}

void TimestampDecodeState::PublishWord(std::string_view word_text, int64_t start_frame, int64_t stop_frame) {
  if (level_ == Config::TimestampLevel::Word || level_ == Config::TimestampLevel::All) {
    result_.words.push_back({std::string{word_text},
                             start_frame,
                             stop_frame,
                             RoundTimestamp(start_frame * seconds_per_frame_),
                             RoundTimestamp(stop_frame * seconds_per_frame_)});
  }

  if (level_ == Config::TimestampLevel::Segment || level_ == Config::TimestampLevel::All) {
    if (pending_segment_.active && segment_gap_threshold_frames_ &&
        start_frame - pending_segment_.stop_frame >= *segment_gap_threshold_frames_) {
      CompleteSegment();
    }

    if (!pending_segment_.active) {
      pending_segment_ = {std::string{word_text}, start_frame, stop_frame, true};
    } else {
      pending_segment_.text.append(word_text);
      pending_segment_.stop_frame = stop_frame;
    }

    const bool ends_segment = std::any_of(segment_separators_.begin(), segment_separators_.end(),
                                          [&word_text](const std::string& separator) {
                                            return EndsWithSeparator(word_text, separator);
                                          });
    if (ends_segment) CompleteSegment();
  }
}

void TimestampDecodeState::ConsumeCompletedWords(const OrtxDetokenizeMetadata& metadata) {
  for (size_t index = 0; index < metadata.word_count; ++index) {
    const auto& word = metadata.words[index];
    if (word.text == nullptr || word.start_token_index < first_pending_token_index_ ||
        word.stop_token_index <= word.start_token_index ||
        word.stop_token_index > first_pending_token_index_ + pending_token_timings_.size()) {
      throw std::runtime_error("Tokenizer returned an invalid completed-word token span");
    }
    const auto& first_timing = pending_token_timings_[word.start_token_index - first_pending_token_index_];
    const auto& last_timing = pending_token_timings_[word.stop_token_index - first_pending_token_index_ - 1];
    PublishWord(word.text, first_timing.start_frame, last_timing.stop_frame);
  }

  if (metadata.first_pending_token_index < first_pending_token_index_ ||
      metadata.first_pending_token_index > first_pending_token_index_ + pending_token_timings_.size()) {
    throw std::runtime_error("Tokenizer returned an invalid pending-token watermark");
  }
  while (first_pending_token_index_ < metadata.first_pending_token_index) {
    pending_token_timings_.pop_front();
    ++first_pending_token_index_;
  }
}

void TimestampDecodeState::Consume(const TokenTiming& token, const OrtxDetokenizeMetadata& metadata) {
  pending_token_timings_.push_back({token.start_frame, token.stop_frame});
  ConsumeCompletedWords(metadata);
}

void TimestampDecodeState::Finalize(const OrtxDetokenizeMetadata& metadata) {
  ConsumeCompletedWords(metadata);
  CompleteSegment();
}

std::vector<int32_t> PadInputs(std::span<std::span<const int32_t>> sequences, int32_t pad_token_id) {
  bool pad_right_{true};

  size_t max_length = 0;
  for (auto& sequence : sequences)
    max_length = std::max(max_length, sequence.size());

  std::vector<int32_t> result(max_length * sequences.size());
  std::span<int32_t> result_span(result);

  // Copy and pad the sequences with pad_token_id
  for (size_t i = 0; i < sequences.size(); i++) {
    auto output_span = result_span.subspan(i * max_length, max_length);
    auto input_span = sequences[i];

    auto pad_count = max_length - input_span.size();
    if (pad_right_) {
      std::copy(input_span.begin(), input_span.end(), output_span.begin());
      std::fill(output_span.end() - pad_count, output_span.end(), pad_token_id);
    } else {
      std::fill(output_span.begin(), output_span.begin() + pad_count, pad_token_id);
      std::copy(input_span.begin(), input_span.end(), output_span.begin() + pad_count);
    }
  }

  return result;
}

TokenizerStream::TokenizerStream(const Tokenizer& tokenizer)
    : tokenizer_{tokenizer.shared_from_this()} {
  CheckResult(OrtxCreate(kOrtxKindDetokenizerCache, cache_.Address()));
}

const std::string& TokenizerStream::Decode(int32_t token) {
  if (tokenizer_->timestamp_config_) {
    if (decode_mode_ == DecodeMode::Timestamps) {
      throw std::runtime_error("Cannot mix Decode and DecodeWithTimestamps before Reset");
    }
    decode_mode_ = DecodeMode::Text;
  }
  const char* string;
  CheckResult(OrtxDetokenizeCached(tokenizer_->tokenizer_, cache_, token, &string));
  chunk_ = string;
  return chunk_;
}

const TimestampDecodeResult& TokenizerStream::DecodeWithTimestamps(const TokenTiming& token) {
  if (decode_mode_ == DecodeMode::Text) {
    throw std::runtime_error("Cannot mix Decode and DecodeWithTimestamps before Reset");
  }
  if (!tokenizer_->timestamp_config_) {
    throw std::runtime_error("Timestamp decoding is not enabled for this tokenizer");
  }
  if (token.start_frame < 0 || token.stop_frame <= token.start_frame) {
    throw std::runtime_error("Token timestamp interval must satisfy 0 <= start_frame < stop_frame");
  }

  decode_mode_ = DecodeMode::Timestamps;
  if (!timestamp_state_) {
    timestamp_state_ = std::make_unique<TimestampDecodeState>(*tokenizer_->timestamp_config_);
  }
  timestamp_state_->ClearResult();

  const char* text;
  OrtxDetokenizeMetadata metadata{};
  CheckResult(OrtxDetokenizeCachedWithMetadata(tokenizer_->tokenizer_, cache_, token.token_id, &text, &metadata));
  timestamp_state_->result_.text = text;
  timestamp_state_->Consume(token, metadata);
  return timestamp_state_->result_;
}

const TimestampDecodeResult& TokenizerStream::FinalizeTimestamps() {
  if (decode_mode_ == DecodeMode::Text) {
    throw std::runtime_error("Cannot finalize timestamps after ordinary Decode before Reset");
  }
  if (!tokenizer_->timestamp_config_) {
    throw std::runtime_error("Timestamp decoding is not enabled for this tokenizer");
  }
  decode_mode_ = DecodeMode::Timestamps;
  if (!timestamp_state_) {
    timestamp_state_ = std::make_unique<TimestampDecodeState>(*tokenizer_->timestamp_config_);
  }
  timestamp_state_->ClearResult();
  OrtxDetokenizeMetadata metadata{};
  CheckResult(OrtxFinalizeDetokenizeCachedWithMetadata(cache_, &metadata));
  timestamp_state_->Finalize(metadata);
  return timestamp_state_->result_;
}

void TokenizerStream::Reset() {
  OrtxDispose(&cache_.p_);
  CheckResult(OrtxCreate(kOrtxKindDetokenizerCache, cache_.Address()));
  chunk_.clear();
  decode_mode_ = DecodeMode::Unset;
  timestamp_state_.reset();
}

Tokenizer::Tokenizer(const Config& config) : bos_token_id_{config.model.bos_token_id},
                                             eos_token_id_{config.model.eos_token_id},
                                             pad_token_id_{config.model.pad_token_id},
                                             bot_token_id_{config.model.bot_token_id},
                                             eot_token_id_{config.model.eot_token_id},
                                             bor_token_id_{config.model.bor_token_id},
                                             eor_token_id_{config.model.eor_token_id} {
  // Default tokenizer options
  const char* keys[] = {"add_special_tokens", "skip_special_tokens"};
  const char* values[] = {"false", "true"};

  // Resolve tokenizer_dir (may be empty, relative, absolute, or a "sha256:" shared-asset reference).
  const fs::path tokenizer_dir = config.ResolvePath(config.model.tokenizer_dir);
  CheckResult(OrtxCreateTokenizerWithOptions(tokenizer_.Address(), tokenizer_dir.string().c_str(), keys, values, 2));

  if (ModelType::IsRNNT(config.model.type) && config.model.timestamp_level != Config::TimestampLevel::Off) {
    timestamp_config_ = TimestampTokenizerConfig{config.model.timestamp_level,
                                                  config.model.segment_separators,
                                                  GetSegmentGapThresholdFrames(config.model),
                                                  config.model.sample_rate,
                                                  config.model.hop_length,
                                                  config.model.subsampling_factor};
  }

  // Resolve any unset bot/eot/bor/eor IDs via model-type fallback strings.
  // Resolve any unset bot/eot/bor/eor IDs via model-type fallback.
  if (!bot_token_id_) bot_token_id_ = ResolveFallbackTokenId(config.model.type, std::string(Config::Defaults::BotTokenIdName), *this);
  if (!eot_token_id_) eot_token_id_ = ResolveFallbackTokenId(config.model.type, std::string(Config::Defaults::EotTokenIdName), *this);
  if (!bor_token_id_) bor_token_id_ = ResolveFallbackTokenId(config.model.type, std::string(Config::Defaults::BorTokenIdName), *this);
  if (!eor_token_id_) eor_token_id_ = ResolveFallbackTokenId(config.model.type, std::string(Config::Defaults::EorTokenIdName), *this);
}

int32_t Tokenizer::GetBotTokenId() const {
  if (!bot_token_id_) throw std::runtime_error("bot_token_id is not defined for this model");
  return *bot_token_id_;
}

int32_t Tokenizer::GetEotTokenId() const {
  if (!eot_token_id_) throw std::runtime_error("eot_token_id is not defined for this model");
  return *eot_token_id_;
}

int32_t Tokenizer::GetBorTokenId() const {
  if (!bor_token_id_) throw std::runtime_error("bor_token_id is not defined for this model");
  return *bor_token_id_;
}

int32_t Tokenizer::GetEorTokenId() const {
  if (!eor_token_id_) throw std::runtime_error("eor_token_id is not defined for this model");
  return *eor_token_id_;
}

std::unique_ptr<TokenizerStream> Tokenizer::CreateStream() const {
  return std::make_unique<TokenizerStream>(*this);
}

void Tokenizer::UpdateOptions(const char* const* keys, const char* const* values, size_t num_options) {
  // Tap into ORT Extensions API
  CheckResult(OrtxUpdateTokenizerOptions(tokenizer_, const_cast<const char**>(keys), const_cast<const char**>(values), num_options));
}

std::vector<int32_t> Tokenizer::Encode(const char* text) const {
  OrtxPtr<OrtxTokenId2DArray> ids;
  CheckResult(OrtxTokenize(tokenizer_, &text, 1, ids.Address()));

  const extTokenId_t* tokens;
  size_t count;
  CheckResult(OrtxTokenId2DArrayGetItem(ids, 0, &tokens, &count));
  return {tokens, tokens + count};
}

std::string Tokenizer::Decode(std::span<const int32_t> tokens) const {
  OrtxPtr<OrtxStringArray> ortx_string_array;
  CheckResult(OrtxDetokenize1D(tokenizer_, reinterpret_cast<const uint32_t*>(tokens.data()), tokens.size(), ortx_string_array.Address()));

  const char* string;
  CheckResult(OrtxStringArrayGetItem(ortx_string_array, 0, &string));
  return string;
}

std::string Tokenizer::ApplyChatTemplate(const char* template_str, const char* messages, const char* tools, bool add_generation_prompt) const {
  OrtxPtr<OrtxTensorResult> templated_text;
  CheckResult(OrtxApplyChatTemplate(tokenizer_, template_str, messages, tools, templated_text.Address(), add_generation_prompt, false /*tokenize*/));

  OrtxPtr<OrtxTensor> tensor;
  CheckResult(OrtxTensorResultGetAt(templated_text, 0, tensor.Address()));

  const char* text_ptr{};
  CheckResult(OrtxGetTensorData(tensor, reinterpret_cast<const void**>(&text_ptr), nullptr, nullptr));

  return text_ptr;
}

std::string Tokenizer::ApplyChatTemplateWithOptions(const char* template_str, const char* messages, const char* tools,
                                                    const char* template_kwargs, bool add_generation_prompt) const {
  OrtxPtr<OrtxTensorResult> templated_text;
  CheckResult(OrtxApplyChatTemplateWithOptions(tokenizer_, template_str, messages, tools, template_kwargs,
                                               templated_text.Address(), add_generation_prompt,
                                               false /*tokenize*/));

  OrtxPtr<OrtxTensor> tensor;
  CheckResult(OrtxTensorResultGetAt(templated_text, 0, tensor.Address()));

  const char* text_ptr{};
  CheckResult(OrtxGetTensorData(tensor, reinterpret_cast<const void**>(&text_ptr), nullptr, nullptr));

  return text_ptr;
}

std::vector<int32_t> Tokenizer::EncodeBatch(std::span<const std::string> strings) const {
  std::vector<std::vector<int32_t>> sequences;
  std::vector<std::span<const int32_t>> span_sequences;
  for (size_t i = 0; i < strings.size(); i++) {
    sequences.emplace_back(Encode(strings[i].c_str()));
    span_sequences.emplace_back(sequences.back());
  }

  return PadInputs(span_sequences, pad_token_id_);
}

std::shared_ptr<Tensor> Tokenizer::EncodeBatch(std::span<const char*> strings) const {
  if (strings.empty()) {
    throw std::runtime_error("EncodeBatch: input strings must not be empty");
  }
  for (size_t i = 0; i < strings.size(); i++) {
    if (strings[i] == nullptr) {
      throw std::runtime_error("EncodeBatch: input string at index " + std::to_string(i) + " must not be null");
    }
  }

  std::vector<std::vector<int32_t>> sequences;
  std::vector<std::span<const int32_t>> span_sequences;
  for (size_t i = 0; i < strings.size(); i++) {
    sequences.emplace_back(Encode(strings[i]));
    span_sequences.emplace_back(sequences.back());
  }

  auto encoded = PadInputs(span_sequences, pad_token_id_);  // TODO: Pad directly into tensor vs copying?

  auto shape = std::array<int64_t, 2>{static_cast<int64_t>(strings.size()), static_cast<int64_t>(encoded.size() / strings.size())};
  auto ort_tensor_ = OrtValue::CreateTensor<int32_t>(Ort::Allocator::GetWithDefaultOptions(), shape);
  auto tensor = std::make_shared<Tensor>(std::move(ort_tensor_));
  std::copy(encoded.begin(), encoded.end(), tensor->GetMutableData<int32_t>());

  return tensor;
}

std::vector<std::string> Tokenizer::DecodeBatch(std::span<const int32_t> sequences, size_t count) const {
  if (sequences.size() % count != 0)
    throw std::runtime_error("DecodeBatch: sequences must be evenly divisible by the count");
  size_t sequence_length = sequences.size() / count;
  std::vector<std::string> strings;
  for (size_t i = 0; i < count; i++)
    strings.emplace_back(Decode(sequences.subspan(sequence_length * i, sequence_length)));
  return strings;
}

int32_t Tokenizer::TokenToTokenId(const char* token) const {
  extTokenId_t token_id;
  CheckResult(OrtxConvertTokenToId(tokenizer_, token, &token_id));
  return token_id;
}

std::shared_ptr<Tokenizer> Model::CreateTokenizer() const {
  return std::make_shared<Tokenizer>(*config_);
}

}  // namespace Generators
