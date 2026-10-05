#pragma once

#include "config.h"
#include "ort_genai_c.h"
#include "ortx_tokenizer.h"

#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Generators {

struct TokenizerStream;

// Stream-specific grouping settings and model timing.
struct TimestampTokenizerConfig {
  Config::TimestampLevel level{Config::TimestampLevel::Off};
  std::vector<std::string> segment_separators;
  std::optional<double> segment_gap_threshold_seconds;
  int sample_rate{};
  int hop_length{};
  int subsampling_factor{};
};

// Configurable metadata options; streams snapshot these when creating a state.
struct MetadataCoreConfig {
  TimestampTokenizerConfig timestamps;
};

void OverlayMetadataCoreConfig(MetadataCoreConfig& config, std::string_view json);

// Owns one stream's decoded metadata and an independent, immutable config copy.
class MetadataCoreState {
 public:
  MetadataCoreState(const MetadataCoreState&) = delete;
  MetadataCoreState& operator=(const MetadataCoreState&) = delete;

  const std::string& Text() const;
  const OrtxMetadata& Metadata() const;
  bool TimestampsEnabled() const { return config_.timestamps.level != Config::TimestampLevel::Off; }

 private:
  friend struct TokenizerStream;
  explicit MetadataCoreState(const MetadataCoreConfig& config);
  void CheckValid() const;
  void ValidateInput(const OgaTokenMetadataInput& token) const;
  const OgaTokenMetadataOutput& ProcessDecoded(const OgaTokenMetadataInput& token, const char* text, const OrtxMetadata& metadata);
  const OgaTokenMetadataOutput& ProcessFinalized(const OrtxMetadata& metadata);
  void BeginResult(const char* text, const OrtxMetadata& metadata);
  void ConsumeCompletedWords(const OrtxTimestampMetadata& metadata);
  void PublishWord(std::string_view text, OgaTokenMetadataAcousticFrameInterval interval);
  void CompleteSegment();
  void AddRecord(std::vector<OgaTokenMetadataTimestampRecord>& records, std::string_view text,
                 OgaTokenMetadataAcousticFrameInterval interval);
  const OgaTokenMetadataOutput& PublishResult();
  void Invalidate();

  const MetadataCoreConfig config_;
  bool valid_{true};
  std::string text_;
  const OrtxMetadata* metadata_{};
  std::optional<int> segment_gap_threshold_frames_;
  double seconds_per_frame_{};
  // Extensions supplies completed-word token indices; keep intervals until its watermark releases them.
  std::deque<OgaTokenMetadataAcousticFrameInterval> pending_token_intervals_;
  size_t first_pending_token_index_{};
  std::optional<std::string> pending_segment_text_;
  OgaTokenMetadataAcousticFrameInterval pending_segment_interval_{};
  // C records borrow text from this per-call storage until the next stream operation.
  std::deque<std::string> record_texts_;
  std::vector<OgaTokenMetadataTimestampRecord> word_records_;
  std::vector<OgaTokenMetadataTimestampRecord> segment_records_;
  OgaTokenMetadataTimestamp timestamp_result_{};
  OgaTokenMetadataOutput result_{};
};

}  // namespace Generators