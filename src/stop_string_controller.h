// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "stop_string_matcher.h"

namespace Generators {

struct Tokenizer;
struct TokenizerStream;

// Engine-internal, token-aware owner around one tokenizer stream and one StopStringMatcher.
//
// StopStringMatcher works on decoded bytes and knows nothing about tokens. This controller is the
// piece that turns committed generated token IDs into those bytes: it decodes each token exactly
// once through one TokenizerStream and feeds the resulting chunk to the matcher, so cost is linear
// in the generated length.
//
// The Engine never publishes decoded text, only raw tokens, so ObserveToken() immediately drains
// and discards the matcher's safe-output buffer after every call. Without this, StopStringMatcher's
// internal `safe_` buffer would retain the entire turn's decoded bytes for no reason.
//
// It is deliberately independent of Engine and Request so it can be unit tested without a model. A
// caller (Request) owns one controller per in-flight sequence that has stop strings enabled.
//
// The controller tracks every generated token ID observed in the current turn, which is what makes
// RollbackTo() possible: the ORT Extensions detokenizer cache backing a TokenizerStream cannot be
// cloned, so a rollback recreates the stream from scratch and replays the retained tokens through
// it in order.
//
// ObserveToken() is generic per token: callers may invoke it multiple times in one committed step,
// but tokens must be observed in generation order and observation stops as soon as a match is
// reported.
//
// Only tokens the caller actually committed for the current turn may be observed, in generation
// order. Prompt and continuation tokens are never observed, so a stop string can only match text
// generated in this turn.
class StopStringController {
 public:
  // `tokenizer` is retained for the controller's lifetime. `stop_strings` is validated by
  // StopStringMatcher's constructor, which throws std::runtime_error for an empty entry, invalid
  // UTF-8, or a configuration over the documented bounds. Callers should avoid constructing a
  // controller (and therefore a tokenizer stream) at all when there are no stop strings: that is the
  // no-stop fast path.
  StopStringController(std::shared_ptr<const Tokenizer> tokenizer, std::vector<std::string> stop_strings);
  ~StopStringController();

  // Decodes one newly generated token and feeds its bytes to the matcher, then immediately drains
  // and discards the matcher's safe output. Returns the match if this token completed one. Once a
  // match has been reported the matcher is sticky: later calls are still recorded in the token
  // history (so a rollback replay reproduces the same bytes) but cannot change the match.
  const std::optional<StopStringMatch>& ObserveToken(int32_t token);

  bool Matched() const { return matcher_.Matched(); }
  const std::optional<StopStringMatch>& Match() const { return matcher_.Match(); }

  // Generated token IDs observed in the current turn, in order, including the token that completed
  // a match. This is raw model history: a match can start or end inside one token's decoded bytes,
  // so it generally cannot be represented by dropping token IDs. TurnTokens().size() is also this
  // controller's transactional checkpoint: save it before a step, pass it to RollbackTo() to undo.
  const std::vector<int32_t>& TurnTokens() const { return turn_tokens_; }

  const std::vector<std::string>& StopStrings() const { return matcher_.StopStrings(); }

  // Restores a transactional checkpoint expressed as the number of generated tokens that were
  // committed when the transaction began. The tokenizer stream cannot be cloned, so this truncates
  // the staged history, creates a fresh stream, resets the matcher, and replays the retained raw
  // tokens through it. A replay failure (allocation, tokenizer failure) propagates to the caller,
  // which must treat it as a consistency failure rather than continuing with partially rebuilt
  // state -- see the Engine's transaction rollback path.
  void RollbackTo(size_t committed_token_count);

 private:
  void RebuildStream();
  void DecodeIntoMatcher(int32_t token);

  std::shared_ptr<const Tokenizer> tokenizer_;
  std::unique_ptr<TokenizerStream> stream_;
  StopStringMatcher matcher_;
  std::vector<int32_t> turn_tokens_;
};

}  // namespace Generators
