#include "metadata_core_state.h"
#include "token_diarization.h"

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
      segment_gap_threshold_frames_ = config_.timestamps.segment_gap_threshold_frames;
      if (segment_gap_threshold_frames_ && *segment_gap_threshold_frames_ < 0)
        throw std::runtime_error("Timestamp segment gap must not be negative");
      if (!config_.seconds_per_frame || !std::isfinite(*config_.seconds_per_frame) || *config_.seconds_per_frame <= 0)
        throw std::runtime_error("Timestamp decoding requires a positive finite seconds_per_frame timebase");
      seconds_per_frame_ = *config_.seconds_per_frame;
      break;
    default:
      throw std::runtime_error("Invalid timestamp level in metadata configuration");
  }
}

void MetadataCoreState::ValidateInput(const TokenMetadataInput& token) const {
  if (AlignmentEnabled()) {
    if (!token.token_acoustic_frame_interval)
      throw std::runtime_error("Enabled timestamps or diarization require generation timing metadata");
    const auto& token_acoustic_frame_interval = *token.token_acoustic_frame_interval;
    if (token_acoustic_frame_interval.start < 0 || token_acoustic_frame_interval.stop <= token_acoustic_frame_interval.start)
      throw std::runtime_error("Token timestamp interval must satisfy 0 <= start_frame < stop_frame");
  }
  if (config_.diarization) {
    if (!token.speaker_evidence)
      throw std::runtime_error("Enabled diarization requires generation speaker evidence");
    TokenDiarization::ValidateSpeakerEvidence(*token.speaker_evidence,
                                              last_speaker_evidence_ ? &*last_speaker_evidence_ : nullptr);
  }
}

void MetadataCoreState::AddRecord(std::vector<TimestampRecord>& records, std::string_view text,
                                  AcousticFrameInterval interval, int speaker) {
  auto& record = records.emplace_back();
  record.text = text;
  if (TimestampsEnabled()) {
    record.start_frame = interval.start;
    record.stop_frame = interval.stop;
    record.start_time = RoundTimestamp(interval.start * seconds_per_frame_);
    record.stop_time = RoundTimestamp(interval.stop * seconds_per_frame_);
  }
  if (speaker >= 0) record.speaker_label = "speaker_" + std::to_string(speaker);
}

void MetadataCoreState::CompleteSegment(TimestampResult& result) {
  if (!pending_segment_text_) return;
  AddRecord(result.segments, *pending_segment_text_, pending_segment_interval_, pending_segment_speaker_);
  pending_segment_text_.reset();
}

void MetadataCoreState::PublishWord(std::string_view text, AcousticFrameInterval interval, int speaker, TimestampResult& result) {
  const auto level = config_.timestamps.level;
  if (level == Config::TimestampLevel::Word) {
    AddRecord(result.words, text, interval, speaker);
  }

  if (level == Config::TimestampLevel::Segment || (config_.diarization && level == Config::TimestampLevel::Off)) {
    const bool new_turn = config_.diarization
                              ? TokenDiarization::SpeakerChanged(pending_segment_text_.has_value(), pending_segment_speaker_, speaker)
                              : pending_segment_text_ && segment_gap_threshold_frames_ &&
                                    (*segment_gap_threshold_frames_ == 0 ||
                                     interval.start - pending_segment_interval_.stop >= *segment_gap_threshold_frames_);
    if (new_turn) {
      CompleteSegment(result);
    }

    if (!pending_segment_text_) {
      pending_segment_text_.emplace(text);
      pending_segment_interval_ = interval;
      pending_segment_speaker_ = speaker;
    } else {
      pending_segment_text_->append(text);
      pending_segment_interval_.stop = interval.stop;
    }

    const bool ends_segment = !config_.diarization && std::any_of(config_.timestamps.segment_separators.begin(),
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
        word.stop_token_index > first_pending_token_index_ + pending_tokens_.size()) {
      throw std::runtime_error("Tokenizer returned an invalid completed-word token span");
    }
    // The first and last contributing tokens supply each word's acoustic bounds.
    const auto& first = pending_tokens_[word.start_token_index - first_pending_token_index_];
    const auto& last = pending_tokens_[word.stop_token_index - first_pending_token_index_ - 1];
    const int speaker = config_.diarization
                            ? TokenDiarization::AggregateWordSpeaker(*first.speaker_evidence, *last.speaker_evidence)
                            : -1;
    PublishWord(word.text, {first.token_acoustic_frame_interval->start, last.token_acoustic_frame_interval->stop}, speaker, result);
  }

  const auto first_pending = word_grouping_.FirstPendingTokenIndex();
  if (first_pending < first_pending_token_index_ ||
      first_pending > first_pending_token_index_ + pending_tokens_.size()) {
    throw std::runtime_error("Tokenizer returned an invalid pending-token watermark");
  }
  while (first_pending_token_index_ < first_pending) {
    pending_tokens_.pop_front();
    ++first_pending_token_index_;
  }
}

void MetadataCoreState::ProcessDecoded(
    const TokenMetadataInput& token, const OrtxWordPieceInfo& piece, PostProcessorResult& result) {
  if (AlignmentEnabled()) {
    auto& timestamps = result.timestamps.emplace();
    pending_tokens_.push_back(token);
    word_grouping_.Process(piece, result.text);
    ProcessCompletedWords(timestamps);
    if (config_.diarization) last_speaker_evidence_ = token.speaker_evidence;
  }
}

void MetadataCoreState::ProcessFinalized(PostProcessorResult& result) {
  if (AlignmentEnabled()) {
    auto& timestamps = result.timestamps.emplace();
    word_grouping_.Finalize();
    ProcessCompletedWords(timestamps);
    CompleteSegment(timestamps);
  }
}

}  // namespace Generators