#pragma once

#include "config.h"
#include "ortx_tokenizer.h"
#include "token_word_grouping.h"
#include "generator/token_outputs.h"

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
  std::optional<TokenSpeakerEvidence> speaker_evidence;
};

struct TimestampRecord {
  std::string text;
  std::optional<int64_t> start_frame;
  std::optional<int64_t> stop_frame;
  std::optional<double> start_time;
  std::optional<double> stop_time;
  std::optional<std::string> speaker_label;
};

struct TimestampResult {
  std::vector<TimestampRecord> words;
  std::vector<TimestampRecord> segments;
};

struct PostProcessorResult {
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
  bool diarization{};
};

class MetadataCoreState {
  friend class TokenPostProcessorInternal;

 private:
  explicit MetadataCoreState(const MetadataCoreConfig& config);
  MetadataCoreState(const MetadataCoreState&) = delete;
  MetadataCoreState& operator=(const MetadataCoreState&) = delete;
  bool TimestampsEnabled() const { return config_.timestamps.level != Config::TimestampLevel::Off; }
  bool AlignmentEnabled() const { return TimestampsEnabled() || config_.diarization; }
  void ValidateInput(const TokenMetadataInput& token) const;
  void ProcessDecoded(const TokenMetadataInput& token, const OrtxWordPieceInfo& piece, PostProcessorResult& result);
  void ProcessFinalized(PostProcessorResult& result);
  void ProcessCompletedWords(TimestampResult& result);
  void PublishWord(std::string_view text, AcousticFrameInterval interval, int speaker, TimestampResult& result);
  void CompleteSegment(TimestampResult& result);
  void AddRecord(std::vector<TimestampRecord>& records, std::string_view text,
                 AcousticFrameInterval interval, int speaker);

  const MetadataCoreConfig& config_;
  TokenWordGroupingState word_grouping_;
  std::optional<int> segment_gap_threshold_frames_;
  double seconds_per_frame_{};
  // Keep intervals until the word grouper's pending-token watermark releases them.
  std::deque<TokenMetadataInput> pending_tokens_;
  std::optional<TokenSpeakerEvidence> last_speaker_evidence_;
  size_t first_pending_token_index_{};
  std::optional<std::string> pending_segment_text_;
  AcousticFrameInterval pending_segment_interval_{};
  int pending_segment_speaker_{-1};
};

}  // namespace Generators