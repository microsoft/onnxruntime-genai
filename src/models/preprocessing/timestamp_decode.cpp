#include "timestamp_decode.h"

#include "models/transducer_state.h"

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

TimestampDecodeState::TimestampDecodeState(const TimestampTokenizerConfig& config)
    : level_{config.level},
      segment_separators_{config.segment_separators},
      segment_gap_threshold_frames_{GetSegmentGapThresholdFrames(config.segment_gap_threshold_seconds,
                                                                 config.sample_rate, config.hop_length,
                                                                 config.subsampling_factor)} {
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
        (*segment_gap_threshold_frames_ == 0 ||
         start_frame - pending_segment_.stop_frame >= *segment_gap_threshold_frames_)) {
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

void TimestampDecodeState::ConsumeCompletedWords(const OrtxTimestampMetadata& metadata) {
  if (metadata.word_count != 0 && metadata.words == nullptr) {
    throw std::runtime_error("Tokenizer returned a null completed-word array");
  }
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

void TimestampDecodeState::Consume(const TokenTiming& token, const OrtxTimestampMetadata& metadata) {
  pending_token_timings_.push_back({token.start_frame, token.stop_frame});
  ConsumeCompletedWords(metadata);
}

void TimestampDecodeState::Finalize(const OrtxTimestampMetadata& metadata) {
  ConsumeCompletedWords(metadata);
  CompleteSegment();
}

}  // namespace Generators