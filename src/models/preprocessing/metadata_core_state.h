#pragma once

#include "timestamp_decode.h"
#include "models/transducer_state.h"
#include "ort_genai_c.h"

#include <memory>
#include <optional>

namespace Generators {

struct TokenizerStream;

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
  const OgaTokenMetadataOutput& ProcessMetadata();
  const TimestampDecodeResult& ConsumeTimestamps();
  const TimestampDecodeResult& ConsumeFinalTimestamps();
  bool TimestampsEnabled() const { return config_.timestamps.level != Config::TimestampLevel::Off; }

 private:
  friend struct TokenizerStream;
  explicit MetadataCoreState(const MetadataCoreConfig& config);
  void CheckValid() const;
  void CheckCanAdvance() const;
  void ValidateInput(const OgaTokenMetadataInput& token) const;
  void SetDecoded(const OgaTokenMetadataInput& token, const char* text, const OrtxMetadata& metadata);
  void SetFinalized(const OrtxMetadata& metadata);
  void Invalidate();

  enum class Step { Empty, Token, Finalized };
  const MetadataCoreConfig config_;
  bool valid_{true};
  Step step_{Step::Empty};
  std::string text_;
  const OrtxMetadata* metadata_{};
  bool timestamps_pending_{};
  std::optional<TokenTiming> current_token_timing_;
  std::unique_ptr<TimestampDecodeState> timestamp_decode_state_;
  std::vector<OgaTokenMetadataTimestampRecord> word_records_;
  std::vector<OgaTokenMetadataTimestampRecord> segment_records_;
  OgaTokenMetadataTimestamp curr_token_metadata_timestamps_{};
  OgaTokenMetadataOutput curr_token_metadata_output_{};
  bool processed_{};
};

}  // namespace Generators