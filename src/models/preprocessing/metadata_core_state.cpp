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

void MetadataCoreState::ValidateInput(const TokenMetadataInput& token) const {
  if (TimestampsEnabled()) {
    if (!token.has_token_acoustic_frame_interval)
      throw std::runtime_error("Enabled timestamps require generation timing metadata");
    const auto& token_acoustic_frame_interval = token.token_acoustic_frame_interval;
    if (token_acoustic_frame_interval.start < 0 || token_acoustic_frame_interval.stop <= token_acoustic_frame_interval.start)
      throw std::runtime_error("Token timestamp interval must satisfy 0 <= start_frame < stop_frame");
  }
}

void MetadataCoreState::AddRecord(std::vector<TimestampRecord>& records, std::string_view text,
                                  OgaTokenMetadataAcousticFrameInterval interval) {
  records.push_back({std::string(text), interval.start, interval.stop,
                     RoundTimestamp(interval.start * seconds_per_frame_),
                     RoundTimestamp(interval.stop * seconds_per_frame_)});
}

void MetadataCoreState::CompleteSegment(TimestampResult& result) {
  if (!pending_segment_text_) return;
  AddRecord(result.segments, *pending_segment_text_, pending_segment_interval_);
  pending_segment_text_.reset();
}

void MetadataCoreState::PublishWord(std::string_view text, OgaTokenMetadataAcousticFrameInterval interval, TimestampResult& result) {
  const auto level = config_.timestamps.level;
  if (level == Config::TimestampLevel::Word || level == Config::TimestampLevel::All) {
    AddRecord(result.words, text, interval);
  }

  if (level == Config::TimestampLevel::Segment || level == Config::TimestampLevel::All) {
    if (pending_segment_text_ && segment_gap_threshold_frames_ &&
        (*segment_gap_threshold_frames_ == 0 ||
         interval.start - pending_segment_interval_.stop >= *segment_gap_threshold_frames_)) {
      CompleteSegment(result);
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
    PublishWord(word.text, {first.start, last.stop}, result);
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
    const TokenMetadataInput& token, const OrtxWordPieceInfo& piece, PostProcessorResult& result) {
  if (TimestampsEnabled()) {
    auto& timestamps = result.timestamps.emplace();
    pending_token_intervals_.push_back(token.token_acoustic_frame_interval);
    word_grouping_.Process(piece, result.text);
    ProcessCompletedWords(timestamps);
  }
}

void MetadataCoreState::ProcessFinalized(PostProcessorResult& result) {
  if (TimestampsEnabled()) {
    auto& timestamps = result.timestamps.emplace();
    word_grouping_.Finalize();
    ProcessCompletedWords(timestamps);
    CompleteSegment(timestamps);
  }
}

}  // namespace Generators