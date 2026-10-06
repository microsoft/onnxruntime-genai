#include "metadata_core_state.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <stdexcept>

namespace Generators {

namespace {
double RoundTimestamp(double value) {
  return std::round(value * 100.0) / 100.0;
}

bool EndsWithSeparator(std::string_view text, std::string_view separator) {
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
    text.remove_suffix(1);
  }
  return !separator.empty() && text.size() >= separator.size() &&
         text.compare(text.size() - separator.size(), separator.size(), separator) == 0;
}
}  // namespace

MetadataCoreState::MetadataCoreState(const MetadataCoreConfig& config) : config_{config} {
  switch (config_.timestamps.level) {
    case Config::TimestampLevel::Off:
      break;
    case Config::TimestampLevel::Word:
    case Config::TimestampLevel::Segment:
    case Config::TimestampLevel::All:
      segment_gap_threshold_frames_ = GetSegmentGapThresholdFrames(
          config_.timestamps.segment_gap_threshold_seconds, config_.timestamps.sample_rate,
          config_.timestamps.hop_length, config_.timestamps.subsampling_factor);
      if (config_.timestamps.sample_rate <= 0 || config_.timestamps.hop_length <= 0 ||
          config_.timestamps.subsampling_factor <= 0) {
        throw std::runtime_error("Timestamp decoding requires positive sample_rate, hop_length, and subsampling_factor");
      }
      seconds_per_frame_ = static_cast<double>(config_.timestamps.hop_length) *
                           config_.timestamps.subsampling_factor / config_.timestamps.sample_rate;
      break;
    default:
      throw std::runtime_error("Invalid timestamp level in metadata configuration");
  }
}

void MetadataCoreState::ValidateInput(const OgaTokenMetadataInput& token) const {
  if (TimestampsEnabled()) {
    if (!token.has_token_acoustic_frame_interval)
      throw std::runtime_error("Enabled timestamps require generation timing metadata");
    const auto& token_acoustic_frame_interval = token.token_acoustic_frame_interval;
    if (token_acoustic_frame_interval.start < 0 || token_acoustic_frame_interval.stop <= token_acoustic_frame_interval.start)
      throw std::runtime_error("Token timestamp interval must satisfy 0 <= start_frame < stop_frame");
  }
}

void MetadataCoreState::BeginResult(const char* text) {
  // Only completed events are per-call; buffered intervals and the unfinished segment survive.
  text_ = text;
  record_texts_.clear();
  word_records_.clear();
  segment_records_.clear();
  result_ = {text_.c_str(), nullptr};
}

void MetadataCoreState::AddRecord(std::vector<OgaTokenMetadataTimestampRecord>& records, std::string_view text,
                                  OgaTokenMetadataAcousticFrameInterval interval) {
  // A deque keeps earlier c_str() pointers valid as later words/segments complete in this call.
  record_texts_.emplace_back(text);
  records.push_back({record_texts_.back().c_str(), interval.start, interval.stop,
                     RoundTimestamp(interval.start * seconds_per_frame_),
                     RoundTimestamp(interval.stop * seconds_per_frame_)});
}

void MetadataCoreState::CompleteSegment() {
  if (!pending_segment_text_) return;
  AddRecord(segment_records_, *pending_segment_text_, pending_segment_interval_);
  pending_segment_text_.reset();
}

void MetadataCoreState::PublishWord(std::string_view text, OgaTokenMetadataAcousticFrameInterval interval) {
  const auto level = config_.timestamps.level;
  if (level == Config::TimestampLevel::Word || level == Config::TimestampLevel::All) {
    AddRecord(word_records_, text, interval);
  }

  if (level == Config::TimestampLevel::Segment || level == Config::TimestampLevel::All) {
    if (pending_segment_text_ && segment_gap_threshold_frames_ &&
        (*segment_gap_threshold_frames_ == 0 ||
         interval.start - pending_segment_interval_.stop >= *segment_gap_threshold_frames_)) {
      CompleteSegment();
    }

    if (!pending_segment_text_) {
      pending_segment_text_.emplace(text);
      pending_segment_interval_ = interval;
    } else {
      pending_segment_text_->append(text);
      pending_segment_interval_.stop = interval.stop;
    }

    const bool ends_segment = std::any_of(config_.timestamps.segment_separators.begin(),
                                          config_.timestamps.segment_separators.end(),
                                          [text](const std::string& separator) {
                                            return EndsWithSeparator(text, separator);
                                          });
    if (ends_segment) CompleteSegment();
  }
}

void MetadataCoreState::ConsumeCompletedWords(const OrtxTimestampMetadata& metadata) {
  if (metadata.word_count != 0 && metadata.words == nullptr) {
    throw std::runtime_error("Tokenizer returned a null completed-word array");
  }
  for (size_t index = 0; index < metadata.word_count; ++index) {
    const auto& word = metadata.words[index];
    if (word.text == nullptr || word.start_token_index < first_pending_token_index_ ||
        word.stop_token_index <= word.start_token_index ||
        word.stop_token_index > first_pending_token_index_ + pending_token_intervals_.size()) {
      throw std::runtime_error("Tokenizer returned an invalid completed-word token span");
    }
    // Extensions groups token indices into words; the first and last token supply acoustic bounds.
    const auto& first = pending_token_intervals_[word.start_token_index - first_pending_token_index_];
    const auto& last = pending_token_intervals_[word.stop_token_index - first_pending_token_index_ - 1];
    PublishWord(word.text, {first.start, last.stop});
  }

  if (metadata.first_pending_token_index < first_pending_token_index_ ||
      metadata.first_pending_token_index > first_pending_token_index_ + pending_token_intervals_.size()) {
    throw std::runtime_error("Tokenizer returned an invalid pending-token watermark");
  }
  while (first_pending_token_index_ < metadata.first_pending_token_index) {
    pending_token_intervals_.pop_front();
    ++first_pending_token_index_;
  }
}

const OgaTokenMetadataOutput& MetadataCoreState::PublishResult() {
  if (TimestampsEnabled()) {
    timestamp_result_ = {word_records_.data(), word_records_.size(), segment_records_.data(), segment_records_.size()};
    result_.timestampMetadata = &timestamp_result_;
  }
  return result_;
}

const OgaTokenMetadataOutput& MetadataCoreState::ProcessDecoded(
    const OgaTokenMetadataInput& token, const char* text, const OrtxMetadata& metadata) {
  BeginResult(text);
  if (TimestampsEnabled()) {
    if (!metadata.timestampMetadata) throw std::runtime_error("Tokenizer metadata is missing timestamp data");
    // One interval enters for each decoded token, even when this call completes no words.
    pending_token_intervals_.push_back(token.token_acoustic_frame_interval);
    ConsumeCompletedWords(*metadata.timestampMetadata);
  }
  return PublishResult();
}

const OgaTokenMetadataOutput& MetadataCoreState::ProcessFinalized(const OrtxMetadata& metadata) {
  BeginResult("");
  if (TimestampsEnabled()) {
    // Finalization contributes no interval, but Extensions may now release trailing words.
    const OrtxTimestampMetadata empty{};
    ConsumeCompletedWords(metadata.timestampMetadata ? *metadata.timestampMetadata : empty);
    CompleteSegment();
  }
  return PublishResult();
}

}  // namespace Generators