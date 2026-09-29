#pragma once

#include "config.h"
#include "ortx_tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Generators {

struct TokenTiming;

// A completed word or segment with acoustic bounds and derived seconds.
struct TimestampRecord {
  std::string text;
  int64_t start_frame{};
  int64_t stop_frame{};
  double start_time{};
  double stop_time{};
};

// words and segments contain only records completed by the current decode or
// finalize call. A list may be empty or contain multiple records when one
// decoded token spans multiple boundaries; callers retain any desired history.
struct TimestampDecodeResult {
  std::string text;
  std::vector<TimestampRecord> words;
  std::vector<TimestampRecord> segments;
};

// Word or segment text and timing accumulated until a boundary completes it.
struct PendingTimestampSpan {
  std::string text;
  int64_t start_frame{};
  int64_t stop_frame{};
  bool active{false};
};

// Acoustic interval retained while its decoded text awaits a word boundary.
struct BufferedTokenTiming {
  int64_t start_frame{};
  int64_t stop_frame{};
};

// Timestamp options and model timing used to initialize a stream's decoder.
struct TimestampTokenizerConfig {
  Config::TimestampLevel level{Config::TimestampLevel::Off};
  std::vector<std::string> segment_separators;
  std::optional<double> segment_gap_threshold_seconds;
  int sample_rate{};
  int hop_length{};
  int subsampling_factor{};
};

// Accumulates decoded spans and publishes per-call word and segment events.
struct TimestampDecodeState {
  explicit TimestampDecodeState(const TimestampTokenizerConfig& config);

  void ClearResult();
  void Consume(const TokenTiming& token, const OrtxTimestampMetadata& metadata);
  void Finalize(const OrtxTimestampMetadata& metadata);

  Config::TimestampLevel level_{Config::TimestampLevel::Off};
  std::vector<std::string> segment_separators_;
  std::optional<int> segment_gap_threshold_frames_;
  double seconds_per_frame_{};
  std::deque<BufferedTokenTiming> pending_token_timings_;
  size_t first_pending_token_index_{};
  PendingTimestampSpan pending_segment_;
  TimestampDecodeResult result_;

 private:
  void ConsumeCompletedWords(const OrtxTimestampMetadata& metadata);
  void PublishWord(std::string_view text, int64_t start_frame, int64_t stop_frame);
  void CompleteSegment();
};

}  // namespace Generators