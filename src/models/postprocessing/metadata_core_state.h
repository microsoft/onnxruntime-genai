#pragma once

#include "config.h"
#include "generator/token_outputs.h"
#include "ortx_tokenizer.h"
#include "token_word_grouping.h"

#include <cstddef>
#include <deque>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Generators {

struct TokenMetadataInput {
  int32_t token_id;
  std::optional<AcousticFrameInterval> token_acoustic_frame_interval;
};

struct TimestampRecord {
  std::string text;
  int64_t start_frame;
  int64_t stop_frame;
  double start_time;
  double stop_time;
};

struct TimestampResult {
  std::vector<TimestampRecord> words;
  std::vector<TimestampRecord> segments;
};

struct ASRPostProcessorResult {
  std::string text;
  std::optional<TimestampResult> timestamps;
};

// Processor-specific word and segment settings.
struct TimestampPostProcessorConfig {
  Config::TimestampLevel level{Config::TimestampLevel::Off};
  std::vector<std::string> segment_separators;
  std::optional<int> segment_gap_threshold_frames;
};

// Model-derived metadata options copied into each processor.
struct MetadataCoreConfig {
  TimestampPostProcessorConfig timestamps;
  std::optional<double> seconds_per_frame;
};

class MetadataCoreState {
  friend class ASRPostProcessor;

 private:
  explicit MetadataCoreState(const MetadataCoreConfig& config);
  MetadataCoreState(const MetadataCoreState&) = delete;
  MetadataCoreState& operator=(const MetadataCoreState&) = delete;
  bool TimestampsEnabled() const { return config_.timestamps.level != Config::TimestampLevel::Off; }
  void ValidateInput(const TokenMetadataInput& token) const;
  void ProcessDecoded(const TokenMetadataInput& token, const OrtxWordPieceInfo& piece, ASRPostProcessorResult& result);
  void ProcessFinalized(ASRPostProcessorResult& result);
  void ProcessCompletedWords(TimestampResult& result);
  void PublishWord(std::string_view text, AcousticFrameInterval interval, TimestampResult& result);
  void CompleteSegment(TimestampResult& result);
  void AddRecord(std::vector<TimestampRecord>& records, std::string_view text,
                 AcousticFrameInterval interval);

  const MetadataCoreConfig& config_;
  TokenWordGroupingState word_grouping_;
  std::optional<int> segment_gap_threshold_frames_;
  double seconds_per_frame_{};
  // Keep intervals until the word grouper's pending-token watermark releases them.
  std::deque<AcousticFrameInterval> pending_token_intervals_;
  size_t first_pending_token_index_{};
  std::optional<std::string> pending_segment_text_;
  AcousticFrameInterval pending_segment_interval_{};
};

}  // namespace Generators