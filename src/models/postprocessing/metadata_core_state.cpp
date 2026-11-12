#include "metadata_core_state.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>

namespace Generators {

namespace {
int64_t TimestampMilliseconds(int64_t frame, int64_t samples_per_frame, int sample_rate) {
  constexpr auto max_milliseconds = std::numeric_limits<int64_t>::max();
  int64_t whole_seconds = 0;
  const auto add_seconds = [&](int64_t count, int64_t factor) {
    if (factor != 0 && count > (max_milliseconds / 1000 - whole_seconds) / factor)
      throw std::overflow_error("Timestamp interval exceeds the signed 64-bit millisecond range");
    whole_seconds += count * factor;
  };
  const auto samples_remainder = samples_per_frame % sample_rate;
  const auto fractional_samples = (frame % sample_rate) * samples_remainder;
  add_seconds(frame, samples_per_frame / sample_rate);
  add_seconds(frame / sample_rate, samples_remainder);
  add_seconds(fractional_samples / sample_rate, 1);
  const auto fractional_milliseconds = ((fractional_samples % sample_rate) * 1000 + sample_rate / 2) / sample_rate;
  const auto milliseconds = whole_seconds * 1000;
  if (fractional_milliseconds > max_milliseconds - milliseconds) {
    throw std::overflow_error("Timestamp interval exceeds the signed 64-bit millisecond range");
  }
  return milliseconds + fractional_milliseconds;
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
      segment_gap_threshold_frames_ = config_.timestamps.segment_gap_threshold_frames;
      if (segment_gap_threshold_frames_ && *segment_gap_threshold_frames_ < 0)
        throw std::runtime_error("Timestamp segment gap must not be negative");
      if (config_.samples_per_frame <= 0 || config_.sample_rate <= 0)
        throw std::runtime_error("Timestamp decoding requires positive samples_per_frame and sample_rate");
      break;
    default:
      throw std::runtime_error("Invalid timestamp level in metadata configuration");
  }
}

void MetadataCoreState::ValidateInput(const TokenMetadataInput& token) const {
  if (TimestampsEnabled()) {
    if (!token.token_acoustic_frame_interval)
      throw std::runtime_error("Enabled timestamps require generation timing metadata");
    const auto& token_acoustic_frame_interval = *token.token_acoustic_frame_interval;
    if (token_acoustic_frame_interval.first < 0 || token_acoustic_frame_interval.second <= token_acoustic_frame_interval.first)
      throw std::runtime_error("Token timestamp interval must satisfy 0 <= start_frame < stop_frame");
  }
}

void MetadataCoreState::AddRecord(std::vector<TimestampRecord>& records, std::string_view text,
                                  std::pair<int64_t, int64_t> interval) {
  records.push_back({std::string(text), interval, {TimestampMilliseconds(interval.first, config_.samples_per_frame, config_.sample_rate), TimestampMilliseconds(interval.second, config_.samples_per_frame, config_.sample_rate)}});
}

void MetadataCoreState::CompleteSegment(TimestampResult& result) {
  if (!pending_segment_text_) return;
  AddRecord(result.segments, *pending_segment_text_, pending_segment_interval_);
  pending_segment_text_.reset();
}

void MetadataCoreState::PublishWord(std::string_view text, std::pair<int64_t, int64_t> interval, TimestampResult& result) {
  const auto level = config_.timestamps.level;
  if (level == Config::TimestampLevel::Word || level == Config::TimestampLevel::All) {
    AddRecord(result.words, text, interval);
  }

  if (level == Config::TimestampLevel::Segment || level == Config::TimestampLevel::All) {
    if (pending_segment_text_ && segment_gap_threshold_frames_ &&
        (*segment_gap_threshold_frames_ == 0 ||
         interval.first - pending_segment_interval_.second >= *segment_gap_threshold_frames_)) {
      CompleteSegment(result);
    }

    if (!pending_segment_text_) {
      pending_segment_text_.emplace(text);
      pending_segment_interval_ = interval;
    } else {
      pending_segment_text_->append(text);
      pending_segment_interval_.second = interval.second;
    }

    const bool ends_segment = std::any_of(config_.timestamps.segment_separators.begin(),
                                          config_.timestamps.segment_separators.end(),
                                          [text](const std::string& separator) {
                                            return EndsWithSeparator(text, separator);
                                          });
    if (ends_segment) CompleteSegment(result);
  }
}

void MetadataCoreState::ProcessCompletedWords(TimestampResult& result) {
  for (const auto& word : word_grouping_.CompletedWords()) {
    if (word.start_token_index < first_pending_token_index_ ||
        word.stop_token_index <= word.start_token_index ||
        word.stop_token_index > first_pending_token_index_ + pending_token_intervals_.size()) {
      throw std::runtime_error("Tokenizer returned an invalid completed-word token span");
    }
    // The first and last contributing tokens supply each word's acoustic bounds.
    const auto& first = pending_token_intervals_[word.start_token_index - first_pending_token_index_];
    const auto& last = pending_token_intervals_[word.stop_token_index - first_pending_token_index_ - 1];
    PublishWord(word.text, {first.first, last.second}, result);
  }

  const auto first_pending = word_grouping_.FirstPendingTokenIndex();
  if (first_pending < first_pending_token_index_ ||
      first_pending > first_pending_token_index_ + pending_token_intervals_.size()) {
    throw std::runtime_error("Tokenizer returned an invalid pending-token watermark");
  }
  while (first_pending_token_index_ < first_pending) {
    pending_token_intervals_.pop_front();
    ++first_pending_token_index_;
  }
}

void MetadataCoreState::ProcessDecoded(
    const TokenMetadataInput& token, const OrtxWordPieceInfo& piece, ASRPostProcessorResult& result) {
  if (TimestampsEnabled()) {
    auto& timestamps = result.timestamps.emplace();
    pending_token_intervals_.push_back(*token.token_acoustic_frame_interval);
    word_grouping_.Process(piece, result.text);
    ProcessCompletedWords(timestamps);
  }
}

void MetadataCoreState::ProcessFinalized(ASRPostProcessorResult& result) {
  if (TimestampsEnabled()) {
    auto& timestamps = result.timestamps.emplace();
    word_grouping_.Finalize();
    ProcessCompletedWords(timestamps);
    CompleteSegment(timestamps);
  }
}

}  // namespace Generators