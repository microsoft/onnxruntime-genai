#include "metadata_core_state.h"

#include <stdexcept>

namespace Generators {

MetadataCoreState::MetadataCoreState(const MetadataCoreConfig& config) : config_{config} {
  switch (config_.timestamps.level) {
    case Config::TimestampLevel::Off:
      break;
    case Config::TimestampLevel::Word:
    case Config::TimestampLevel::Segment:
    case Config::TimestampLevel::All:
      timestamp_decode_state_ = std::make_unique<TimestampDecodeState>(config_.timestamps);
      break;
    default:
      throw std::runtime_error("Invalid timestamp level in metadata configuration");
  }
}

void MetadataCoreState::CheckValid() const {
  if (!valid_) throw std::runtime_error("Metadata state was invalidated by stream reset or destruction");
}

void MetadataCoreState::CheckCanAdvance() const {
  CheckValid();
  if (timestamps_pending_) throw std::runtime_error("Consume pending timestamps before decoding or finalizing again");
}

const std::string& MetadataCoreState::Text() const {
  CheckValid();
  return text_;
}

const OrtxMetadata& MetadataCoreState::Metadata() const {
  CheckValid();
  if (!metadata_) throw std::runtime_error("No metadata has been decoded yet");
  return *metadata_;
}

void MetadataCoreState::SetDecoded(const OgaTokenMetadataInput& token, const char* text, const OrtxMetadata& metadata) {
  current_token_timing_.reset();
  if (TimestampsEnabled() && token.has_token_acoustic_frame_interval)
    current_token_timing_ = TokenTiming{token.token_id, token.token_acoustic_frame_interval.start, token.token_acoustic_frame_interval.stop};
  text_ = text;
  metadata_ = &metadata;
  step_ = Step::Token;
  processed_ = false;
  timestamps_pending_ = TimestampsEnabled();
  if (timestamp_decode_state_) timestamp_decode_state_->ClearResult();
}

void MetadataCoreState::ValidateInput(const OgaTokenMetadataInput& token) const {
  CheckCanAdvance();
  if (TimestampsEnabled()) {
    if (!token.has_token_acoustic_frame_interval)
      throw std::runtime_error("Enabled timestamps require generation timing metadata");
    const auto& token_acoustic_frame_interval = token.token_acoustic_frame_interval;
    if (token_acoustic_frame_interval.start < 0 || token_acoustic_frame_interval.stop <= token_acoustic_frame_interval.start)
      throw std::runtime_error("Token timestamp interval must satisfy 0 <= start_frame < stop_frame");
  }
}

const TimestampDecodeResult& MetadataCoreState::ConsumeTimestamps() {
  CheckValid();
  if (!TimestampsEnabled()) throw std::runtime_error("Timestamps are disabled for this metadata state");
  if (!current_token_timing_) throw std::runtime_error("No generation timing metadata for the current step");
  if (step_ != Step::Token) throw std::runtime_error("Decode a token before consuming its timestamps");
  if (!timestamps_pending_) return timestamp_decode_state_->result_;
  if (!metadata_->timestampMetadata) throw std::runtime_error("Tokenizer metadata is missing timestamp data");
  timestamp_decode_state_->result_.text = text_;
  timestamp_decode_state_->Consume(*current_token_timing_, *metadata_->timestampMetadata);
  timestamps_pending_ = false;
  return timestamp_decode_state_->result_;
}

void MetadataCoreState::SetFinalized(const OrtxMetadata& metadata) {
  current_token_timing_.reset();
  text_.clear();
  metadata_ = &metadata;
  step_ = Step::Finalized;
  processed_ = false;
  timestamps_pending_ = TimestampsEnabled();
  if (timestamp_decode_state_) timestamp_decode_state_->ClearResult();
}

const TimestampDecodeResult& MetadataCoreState::ConsumeFinalTimestamps() {
  CheckValid();
  if (!TimestampsEnabled()) throw std::runtime_error("Timestamps are disabled for this metadata state");
  if (step_ != Step::Finalized) throw std::runtime_error("Finalize metadata before consuming trailing timestamps");
  if (timestamps_pending_) {
    const OrtxTimestampMetadata empty{};
    timestamp_decode_state_->Finalize(metadata_->timestampMetadata ? *metadata_->timestampMetadata : empty);
    timestamps_pending_ = false;
  }
  return timestamp_decode_state_->result_;
}

const OgaTokenMetadataOutput& MetadataCoreState::ProcessMetadata() {
  CheckValid();
  if (step_ == Step::Empty) throw std::runtime_error("Decode or finalize before processing metadata");
  if (processed_) return curr_token_metadata_output_;
  curr_token_metadata_output_ = {text_.c_str(), nullptr};
  if (TimestampsEnabled()) {
    const auto& result = step_ == Step::Finalized ? ConsumeFinalTimestamps() : ConsumeTimestamps();
    const auto populate_views = [](const std::vector<TimestampRecord>& records, std::vector<OgaTokenMetadataTimestampRecord>& views) {
      views.clear();
      views.reserve(records.size());
      for (const auto& record : records) {
        views.push_back({record.text.c_str(), record.start_frame, record.stop_frame, record.start_time, record.stop_time});
      }
    };
    populate_views(result.words, word_records_);
    populate_views(result.segments, segment_records_);
    curr_token_metadata_timestamps_ = {word_records_.data(), word_records_.size(), segment_records_.data(), segment_records_.size()};
    curr_token_metadata_output_.timestampMetadata = &curr_token_metadata_timestamps_;
  }
  processed_ = true;
  return curr_token_metadata_output_;
}

void MetadataCoreState::Invalidate() {
  valid_ = false;
  metadata_ = nullptr;
  timestamp_decode_state_.reset();
  text_.clear();
  curr_token_metadata_output_ = {};
  curr_token_metadata_timestamps_ = {};
  word_records_.clear();
  segment_records_.clear();
}

}  // namespace Generators