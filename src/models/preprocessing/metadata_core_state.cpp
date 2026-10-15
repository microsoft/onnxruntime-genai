#include "metadata_core_state.h"
#include "tokenizer_speech_metadata.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <stdexcept>

namespace Generators {

namespace {
double RoundTimestamp(double value) {
  return std::round(value * 100.0) / 100.0;
}

}  // namespace

MetadataCoreState::MetadataCoreState(const MetadataCoreConfig& config) : config_{config} {
  switch (config_.timestamps.level) {
    case Config::TimestampLevel::Off:
      break;
    case Config::TimestampLevel::Word:
    case Config::TimestampLevel::Segment:
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
  if (config_.diarization && !TimestampsEnabled() &&
      (config_.timestamps.sample_rate <= 0 || config_.timestamps.hop_length <= 0 ||
       config_.timestamps.subsampling_factor <= 0))
    throw std::runtime_error("Diarization decoding requires positive acoustic timing parameters");
}

void MetadataCoreState::ValidateInput(const OgaTokenMetadataInput& token) const {
  if (AlignmentEnabled()) {
    if (!token.has_token_acoustic_frame_interval)
      throw std::runtime_error("Enabled timestamps require generation timing metadata");
    const auto& token_acoustic_frame_interval = token.token_acoustic_frame_interval;
    if (token_acoustic_frame_interval.start < 0 || token_acoustic_frame_interval.stop <= token_acoustic_frame_interval.start)
      throw std::runtime_error("Token timestamp interval must satisfy 0 <= start_frame < stop_frame");
  }
  if (config_.diarization) {
    if (!token.has_speaker_evidence)
      throw std::runtime_error("Enabled diarization requires generation speaker evidence");
    const auto& evidence = token.speaker_evidence;
    ort_extensions::TokenizerDiarization::ValidateSpeakerEvidence(evidence.start_probability_sums, evidence.stop_probability_sums,
                                                                  evidence.start_frame_count, evidence.stop_frame_count);
    if (last_speaker_evidence_) {
      const auto& previous = *last_speaker_evidence_;
      if (evidence.start_frame_count < previous.start_frame_count ||
          evidence.stop_frame_count < previous.stop_frame_count)
        throw std::runtime_error("Speaker evidence must follow token emission order");
      for (size_t speaker = 0; speaker < 8; ++speaker) {
        if (evidence.start_probability_sums[speaker] < previous.start_probability_sums[speaker] ||
            evidence.stop_probability_sums[speaker] < previous.stop_probability_sums[speaker])
          throw std::runtime_error("Speaker probability sums must follow token emission order");
        if (evidence.start_probability_sums[speaker] - previous.start_probability_sums[speaker] >
                static_cast<double>(evidence.start_frame_count - previous.start_frame_count) + 1e-9 ||
            evidence.stop_probability_sums[speaker] - previous.stop_probability_sums[speaker] >
                static_cast<double>(evidence.stop_frame_count - previous.stop_frame_count) + 1e-9)
          throw std::runtime_error("Speaker probability sums exceed the newly evaluated frame count");
      }
    }
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
                                  OgaTokenMetadataAcousticFrameInterval interval, int speaker) {
  // A deque keeps earlier c_str() pointers valid as later words/segments complete in this call.
  record_texts_.emplace_back(text);
  const char* record_text = record_texts_.back().c_str();
  const char* label = nullptr;
  if (speaker >= 0) {
    record_texts_.emplace_back("speaker_" + std::to_string(speaker));
    label = record_texts_.back().c_str();
  }
  records.push_back({record_text, TimestampsEnabled() ? interval.start : 0, TimestampsEnabled() ? interval.stop : 0,
                     RoundTimestamp(interval.start * seconds_per_frame_),
                     RoundTimestamp(interval.stop * seconds_per_frame_), TimestampsEnabled() ? 1 : 0, label});
}

void MetadataCoreState::CompleteSegment() {
  if (!pending_segment_text_) return;
  AddRecord(segment_records_, *pending_segment_text_, pending_segment_interval_, pending_segment_speaker_);
  pending_segment_text_.reset();
}

void MetadataCoreState::PublishWord(std::string_view text, OgaTokenMetadataAcousticFrameInterval interval, int speaker) {
  const auto level = config_.timestamps.level;
  if (level == Config::TimestampLevel::Word) {
    AddRecord(word_records_, text, interval, speaker);
  }

  if (level == Config::TimestampLevel::Segment || (config_.diarization && level == Config::TimestampLevel::Off)) {
    if (ort_extensions::SpeechSegmentStartsNew(pending_segment_text_.has_value(), config_.diarization,
                                               pending_segment_speaker_, speaker, interval.start,
                                               pending_segment_interval_.stop, segment_gap_threshold_frames_.value_or(-1))) {
      CompleteSegment();
    }

    if (!pending_segment_text_) {
      pending_segment_text_.emplace(text);
      pending_segment_interval_ = interval;
      pending_segment_speaker_ = speaker;
    } else {
      pending_segment_text_->append(text);
      pending_segment_interval_.stop = interval.stop;
    }

    if (!config_.diarization) {
      std::vector<const char*> separators;
      for (const auto& separator : config_.timestamps.segment_separators) separators.push_back(separator.c_str());
      if (ort_extensions::SpeechSegmentEnds(text, separators.data(), separators.size())) CompleteSegment();
    }
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
        word.stop_token_index > first_pending_token_index_ + pending_tokens_.size()) {
      throw std::runtime_error("Tokenizer returned an invalid completed-word token span");
    }
    // Extensions groups token indices into words; the first and last token supply acoustic bounds.
    const auto& first = pending_tokens_[word.start_token_index - first_pending_token_index_];
    const auto& last = pending_tokens_[word.stop_token_index - first_pending_token_index_ - 1];
    int speaker = -1;
    if (config_.diarization)
      speaker = ort_extensions::TokenizerDiarization::AggregateWordSpeaker(first.speaker_evidence.start_probability_sums,
                                                                           last.speaker_evidence.stop_probability_sums, first.speaker_evidence.start_frame_count,
                                                                           last.speaker_evidence.stop_frame_count);
    PublishWord(word.text, {first.token_acoustic_frame_interval.start, last.token_acoustic_frame_interval.stop}, speaker);
  }

  if (metadata.first_pending_token_index < first_pending_token_index_ ||
      metadata.first_pending_token_index > first_pending_token_index_ + pending_tokens_.size()) {
    throw std::runtime_error("Tokenizer returned an invalid pending-token watermark");
  }
  while (first_pending_token_index_ < metadata.first_pending_token_index) {
    pending_tokens_.pop_front();
    ++first_pending_token_index_;
  }
}

const OgaTokenMetadataOutput& MetadataCoreState::PublishResult() {
  if (AlignmentEnabled()) {
    timestamp_result_ = {word_records_.data(), word_records_.size(), segment_records_.data(), segment_records_.size()};
    result_.timestampMetadata = &timestamp_result_;
  }
  return result_;
}

const OgaTokenMetadataOutput& MetadataCoreState::ProcessDecoded(
    const OgaTokenMetadataInput& token, const char* text, const OrtxMetadata& metadata) {
  BeginResult(text);
  if (AlignmentEnabled()) {
    if (!metadata.timestampMetadata) throw std::runtime_error("Tokenizer metadata is missing timestamp data");
    // One interval enters for each decoded token, even when this call completes no words.
    pending_tokens_.push_back(token);
    ConsumeCompletedWords(*metadata.timestampMetadata);
    if (config_.diarization) last_speaker_evidence_ = token.speaker_evidence;
  }
  return PublishResult();
}

const OgaTokenMetadataOutput& MetadataCoreState::ProcessFinalized(const OrtxMetadata& metadata) {
  BeginResult("");
  if (AlignmentEnabled()) {
    // Finalization contributes no interval, but Extensions may now release trailing words.
    const OrtxTimestampMetadata empty{};
    ConsumeCompletedWords(metadata.timestampMetadata ? *metadata.timestampMetadata : empty);
    CompleteSegment();
  }
  return PublishResult();
}

}  // namespace Generators