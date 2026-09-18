// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "generator/generators.h"
#include "models/utils.h"
#include "ortx_tokenizer.h"

#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace Generators {

struct Config;
struct Tensor;
struct TokenTiming;

struct Tokenizer;

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

struct PendingTimestampSpan {
  std::string text;
  int64_t start_frame{};
  int64_t stop_frame{};
  bool active{false};
};

struct BufferedTokenTiming {
  int64_t start_frame{};
  int64_t stop_frame{};
};

struct TimestampTokenizerConfig {
  Config::TimestampLevel level{Config::TimestampLevel::Off};
  std::vector<std::string> segment_separators;
  std::optional<int> segment_gap_threshold_frames;
  int sample_rate{};
  int hop_length{};
  int subsampling_factor{};
};

struct TimestampDecodeState {
  explicit TimestampDecodeState(const TimestampTokenizerConfig& config);

  void ClearResult();
  void Consume(const TokenTiming& token, const OrtxDetokenizeMetadata& metadata);
  void Finalize(const OrtxDetokenizeMetadata& metadata);

  Config::TimestampLevel level_{Config::TimestampLevel::Off};
  std::vector<std::string> segment_separators_;
  std::optional<int> segment_gap_threshold_frames_;
  double seconds_per_frame_{};
  std::deque<BufferedTokenTiming> pending_token_timings_;
  size_t first_pending_token_index_{};
  PendingTimestampSpan pending_segment_;
  TimestampDecodeResult result_;

 private:
  void ConsumeCompletedWords(const OrtxDetokenizeMetadata& metadata);
  void PublishWord(std::string_view text, int64_t start_frame, int64_t stop_frame);
  void CompleteSegment();
};

struct TokenizerStream : LeakChecked<TokenizerStream> {
  enum class DecodeMode {
    Unset,
    Text,
    Timestamps,
  };

  TokenizerStream(const Tokenizer& tokenizer);

  const std::string& Decode(int32_t token);
  const TimestampDecodeResult& DecodeWithTimestamps(const TokenTiming& token);
  const TimestampDecodeResult& FinalizeTimestamps();
  void Reset();

 private:
  std::shared_ptr<const Tokenizer> tokenizer_;
  OrtxPtr<OrtxObject> cache_;
  std::string chunk_;
  DecodeMode decode_mode_{DecodeMode::Unset};
  // Null unless timestamp decoding is enabled and used; ordinary Decode has no accumulator work.
  std::unique_ptr<TimestampDecodeState> timestamp_state_;
};

// Turn an array of ragged token sequences into a 2D input suitable for batching. Handles padding for the model.
std::vector<int32_t> PadInputs(std::span<std::span<const int32_t>> sequences, int32_t pad_token_id);

struct Tokenizer : std::enable_shared_from_this<Tokenizer>, LeakChecked<Tokenizer>, ExternalRefCounted<Tokenizer> {
  Tokenizer(const Config& config);

  std::unique_ptr<TokenizerStream> CreateStream() const;

  void UpdateOptions(const char* const* keys, const char* const* values, size_t num_options);
  std::vector<int32_t> Encode(const char* text) const;
  std::string Decode(std::span<const int32_t> tokens) const;
  std::string ApplyChatTemplate(const char* template_str, const char* messages, const char* tools, bool add_generation_prompt) const;
  std::string ApplyChatTemplateWithOptions(const char* template_str, const char* messages, const char* tools,
                                           const char* template_kwargs, bool add_generation_prompt) const;

  std::vector<int32_t> EncodeBatch(std::span<const std::string> strings) const;
  std::shared_ptr<Tensor> EncodeBatch(std::span<const char*> strings) const;
  std::vector<std::string> DecodeBatch(std::span<const int32_t> sequences, size_t count) const;

  int32_t TokenToTokenId(const char* token) const;
  int32_t GetBosTokenId() const { return bos_token_id_; }
  const std::vector<int32_t>& GetEosTokenIds() const { return eos_token_id_; }
  int32_t GetPadTokenId() const { return pad_token_id_; }

  int32_t GetBotTokenId() const;
  int32_t GetEotTokenId() const;
  int32_t GetBorTokenId() const;
  int32_t GetEorTokenId() const;

  OrtxPtr<OrtxTokenizer> tokenizer_;

 private:
  friend struct TokenizerStream;

  int32_t bos_token_id_;
  std::vector<int32_t> eos_token_id_;
  int32_t pad_token_id_;
  std::optional<int32_t> bot_token_id_;
  std::optional<int32_t> eot_token_id_;
  std::optional<int32_t> bor_token_id_;
  std::optional<int32_t> eor_token_id_;
  // Empty for timestamp-disabled and unsupported models.
  std::optional<TimestampTokenizerConfig> timestamp_config_;
};

}  // namespace Generators
