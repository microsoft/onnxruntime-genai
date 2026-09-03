// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Model-free tests for the pure stop-string byte matcher. They pin the semantic contract the rest
// of the stop-string feature is built on: exact UTF-8 byte matching over an incrementally consumed
// stream, deterministic selection between overlapping stop strings, preserved caller indices,
// absolute match offsets, a sticky matched state, and bounded withheld output. No tokenizer, model,
// or ONNX Runtime session is involved.

#include "stop_string_matcher.h"

#include <algorithm>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

namespace Generators::test {
namespace {

using Generators::IsValidUtf8;
using Generators::StopStringMatcher;

std::vector<std::string> Strings(std::initializer_list<const char*> values) {
  return std::vector<std::string>{values.begin(), values.end()};
}

}  // namespace

TEST(StopStringMatcherTest, NoMatchFlushesEverything) {
  StopStringMatcher matcher{Strings({"STOP"})};

  EXPECT_FALSE(matcher.Consume("hello ").has_value());
  EXPECT_FALSE(matcher.Consume("world").has_value());
  EXPECT_FALSE(matcher.Matched());
  EXPECT_EQ(matcher.Flush(), "hello world");
  EXPECT_EQ(matcher.ConsumedBytes(), 11u);
  EXPECT_TRUE(matcher.PendingOutput().empty());
  EXPECT_TRUE(matcher.TakeSafeOutput().empty());
}

TEST(StopStringMatcherTest, NoStopStringsNeverMatches) {
  StopStringMatcher matcher{std::vector<std::string>{}};

  EXPECT_FALSE(matcher.Consume("STOP anything at all").has_value());
  EXPECT_FALSE(matcher.Matched());
  EXPECT_TRUE(matcher.PendingOutput().empty());
  EXPECT_EQ(matcher.TakeSafeOutput(), "STOP anything at all");
}

TEST(StopStringMatcherTest, MatchWithinOneChunk) {
  StopStringMatcher matcher{Strings({"STOP"})};

  const auto match = matcher.Consume("abcSTOPdef");
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->index, 0u);
  EXPECT_EQ(match->start_offset, 3u);
  EXPECT_EQ(match->end_offset, 7u);

  // The marker and everything after it in the same chunk are withheld from the caller.
  EXPECT_EQ(matcher.TakeSafeOutput(), "abc");
  EXPECT_TRUE(matcher.PendingOutput().empty());
  EXPECT_TRUE(matcher.Matched());
  ASSERT_TRUE(matcher.Match().has_value());
  EXPECT_EQ(matcher.Match()->end_offset, 7u);
}

TEST(StopStringMatcherTest, MatchAtStreamStart) {
  StopStringMatcher matcher{Strings({"STOP"})};

  const auto match = matcher.Consume("STOP");
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->start_offset, 0u);
  EXPECT_EQ(match->end_offset, 4u);
  EXPECT_EQ(matcher.Flush(), "");
}

TEST(StopStringMatcherTest, MatchSplitAcrossTwoChunks) {
  StopStringMatcher matcher{Strings({"STOP"})};

  EXPECT_FALSE(matcher.Consume("abcST").has_value());
  EXPECT_EQ(matcher.TakeSafeOutput(), "abc");
  EXPECT_EQ(matcher.PendingOutput(), "ST");

  const auto match = matcher.Consume("OPtrailing");
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->index, 0u);
  EXPECT_EQ(match->start_offset, 3u);
  EXPECT_EQ(match->end_offset, 7u);
  EXPECT_EQ(matcher.ConsumedBytes(), 15u);
  EXPECT_EQ(matcher.Flush(), "");
}

TEST(StopStringMatcherTest, MatchSplitAcrossManyChunks) {
  StopStringMatcher matcher{Strings({"<|end|>"})};

  const std::string stream = "text<|end|>more";
  std::optional<StopStringMatch> match;
  std::string published;
  for (char c : stream) {
    if (!match) {
      match = matcher.Consume(std::string_view{&c, 1});
      published += matcher.TakeSafeOutput();
    }
  }

  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->index, 0u);
  EXPECT_EQ(match->start_offset, 4u);
  EXPECT_EQ(match->end_offset, 11u);
  EXPECT_EQ(published, "text");
}

TEST(StopStringMatcherTest, EmptyChunksAreNoOps) {
  StopStringMatcher matcher{Strings({"STOP"})};

  EXPECT_FALSE(matcher.Consume("").has_value());
  EXPECT_FALSE(matcher.Consume("abST").has_value());
  EXPECT_FALSE(matcher.Consume("").has_value());
  EXPECT_EQ(matcher.PendingOutput(), "ST");
  EXPECT_EQ(matcher.TakeSafeOutput(), "ab");

  const auto match = matcher.Consume("OP");
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->start_offset, 2u);
  EXPECT_EQ(match->end_offset, 6u);
  EXPECT_EQ(matcher.ConsumedBytes(), 6u);
}

// Earliest ending byte wins: "ab" completes before "abc" can.
TEST(StopStringMatcherTest, PrefixOverlapPicksEarliestEnd) {
  StopStringMatcher matcher{Strings({"abc", "ab"})};

  const auto match = matcher.Consume("xxabc");
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->index, 1u);
  EXPECT_EQ(match->start_offset, 2u);
  EXPECT_EQ(match->end_offset, 4u);
  EXPECT_EQ(matcher.TakeSafeOutput(), "xx");
}

// Same ending byte: the earliest start (the longest match) wins regardless of caller order.
TEST(StopStringMatcherTest, SuffixOverlapPicksEarliestStart) {
  StopStringMatcher matcher{Strings({"bc", "abc"})};

  const auto match = matcher.Consume("abc");
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->index, 1u);
  EXPECT_EQ(match->start_offset, 0u);
  EXPECT_EQ(match->end_offset, 3u);
}

// Identical start and end: the lowest caller index wins.
TEST(StopStringMatcherTest, DuplicateStopStringsKeepLowestCallerIndex) {
  StopStringMatcher matcher{Strings({"end", "stop", "end"})};
  ASSERT_EQ(matcher.StopStrings().size(), 3u);

  const auto match = matcher.Consume("the end.");
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->index, 0u);
  EXPECT_EQ(match->start_offset, 4u);
  EXPECT_EQ(match->end_offset, 7u);
}

TEST(StopStringMatcherTest, CallerIndicesSurviveDuplicatesEarlierInTheList) {
  StopStringMatcher matcher{Strings({"aa", "aa", "b"})};

  const auto match = matcher.Consume("zb");
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->index, 2u);
  EXPECT_EQ(match->start_offset, 1u);
  EXPECT_EQ(match->end_offset, 2u);
}

TEST(StopStringMatcherTest, RepeatedPrefixesMatchAtEarliestCompletion) {
  StopStringMatcher matcher{Strings({"aaaa"})};

  EXPECT_FALSE(matcher.Consume("aaa").has_value());
  EXPECT_EQ(matcher.PendingOutput(), "aaa");
  EXPECT_TRUE(matcher.TakeSafeOutput().empty());

  const auto match = matcher.Consume("aa");
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->start_offset, 0u);
  EXPECT_EQ(match->end_offset, 4u);
  EXPECT_TRUE(matcher.Flush().empty());
}

TEST(StopStringMatcherTest, RepeatedPrefixesReleaseUnusableBytes) {
  StopStringMatcher matcher{Strings({"aaaa"})};

  EXPECT_FALSE(matcher.Consume("baaa").has_value());
  EXPECT_EQ(matcher.TakeSafeOutput(), "b");
  EXPECT_EQ(matcher.PendingOutput(), "aaa");

  const auto match = matcher.Consume("a");
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->start_offset, 1u);
  EXPECT_EQ(match->end_offset, 5u);
}

TEST(StopStringMatcherTest, BytesBeforeMatchAndTrailingBytesInOneChunk) {
  StopStringMatcher matcher{Strings({"<stop>"})};

  const auto match = matcher.Consume("before<stop>after<stop>");
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->start_offset, 6u);
  EXPECT_EQ(match->end_offset, 12u);
  EXPECT_EQ(matcher.Flush(), "before");
}

TEST(StopStringMatcherTest, MultiByteStopStringSplitInsideCodePoint) {
  const std::string stop = "\xE6\x97\xA5\xE6\x9C\xAC";  // "日本"
  StopStringMatcher matcher{std::vector<std::string>{stop}};

  EXPECT_FALSE(matcher.Consume("hi \xE6\x97").has_value());
  EXPECT_EQ(matcher.TakeSafeOutput(), "hi ");
  EXPECT_EQ(matcher.PendingOutput(), "\xE6\x97");

  EXPECT_FALSE(matcher.Consume("\xA5\xE6\x9C").has_value());
  const auto match = matcher.Consume("\xAC!");
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->start_offset, 3u);
  EXPECT_EQ(match->end_offset, 9u);
  EXPECT_EQ(matcher.Flush(), "");
}

TEST(StopStringMatcherTest, MultiByteTextIsNotMatchedByASharedLeadByte) {
  StopStringMatcher matcher{std::vector<std::string>{"\xE6\x97\xA5"}};  // "日"

  // "早" (E6 97 A9) shares the first two bytes but must not match.
  EXPECT_FALSE(matcher.Consume("\xE6\x97\xA9\xE6\x97\xA9").has_value());
  EXPECT_EQ(matcher.Flush(), "\xE6\x97\xA9\xE6\x97\xA9");
}

// The pure matcher works on bytes, so a safe prefix may still end inside a code point. Publishing
// UTF-8-safe chunks is the caller's job; this test pins the byte-level behaviour it must handle.
TEST(StopStringMatcherTest, SafeOutputMayEndInsideACodePoint) {
  StopStringMatcher matcher{Strings({"STOP"})};

  EXPECT_FALSE(matcher.Consume("a\xE6\x97").has_value());
  EXPECT_EQ(matcher.TakeSafeOutput(), "a\xE6\x97");
  EXPECT_TRUE(matcher.PendingOutput().empty());
}

TEST(StopStringMatcherTest, MatchedStateIsSticky) {
  StopStringMatcher matcher{Strings({"ab", "cd"})};

  const auto match = matcher.Consume("xxab");
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(matcher.ConsumedBytes(), 4u);

  EXPECT_FALSE(matcher.Consume("cd").has_value());
  EXPECT_FALSE(matcher.Consume("ab more text").has_value());
  ASSERT_TRUE(matcher.Match().has_value());
  EXPECT_EQ(matcher.Match()->index, 0u);
  EXPECT_EQ(matcher.Match()->start_offset, 2u);
  EXPECT_EQ(matcher.Match()->end_offset, 4u);
  EXPECT_EQ(matcher.ConsumedBytes(), 4u);  // Ignored chunks are not counted.
  EXPECT_EQ(matcher.Flush(), "xx");
}

TEST(StopStringMatcherTest, ResetRestartsTheStream) {
  StopStringMatcher matcher{Strings({"STOP"})};

  ASSERT_TRUE(matcher.Consume("abcSTOP").has_value());
  matcher.Reset();

  EXPECT_FALSE(matcher.Matched());
  EXPECT_FALSE(matcher.Match().has_value());
  EXPECT_EQ(matcher.ConsumedBytes(), 0u);
  EXPECT_TRUE(matcher.PendingOutput().empty());
  EXPECT_TRUE(matcher.TakeSafeOutput().empty());
  ASSERT_EQ(matcher.StopStrings().size(), 1u);

  // Offsets restart from zero and the configuration still matches.
  const auto match = matcher.Consume("zSTOP");
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->start_offset, 1u);
  EXPECT_EQ(match->end_offset, 5u);
  EXPECT_EQ(matcher.TakeSafeOutput(), "z");
}

TEST(StopStringMatcherTest, ResetDropsPartialPendingMatch) {
  StopStringMatcher matcher{Strings({"STOP"})};

  EXPECT_FALSE(matcher.Consume("ST").has_value());
  matcher.Reset();
  EXPECT_FALSE(matcher.Consume("OP").has_value());
  EXPECT_FALSE(matcher.Matched());
  EXPECT_EQ(matcher.Flush(), "OP");
}

TEST(StopStringMatcherTest, ConsumeAfterFlushThrowsForAPartialMatchAcrossDistinctBytes) {
  // Pattern "abcd", withheld across the Flush() boundary: "abc" is a genuine possible prefix when
  // Flush() is called, so this also exercises that Flush() correctly ends the stream even though
  // pending_ is nonempty at the time -- the withheld suffix is returned once and then the matcher
  // must refuse to keep matching against it.
  StopStringMatcher matcher{Strings({"abcd"})};

  EXPECT_FALSE(matcher.Consume("abc").has_value());
  EXPECT_EQ(matcher.Flush(), "abc");
  EXPECT_THROW(matcher.Consume("d"), std::runtime_error);
}

TEST(StopStringMatcherTest, ConsumeAfterFlushThrowsForARepeatedCharacterPartialMatch) {
  // Pattern "aaaaa": a uniform-run pending suffix is the case most likely to tempt an
  // implementation into silently treating post-Flush bytes as "just more of the same" instead of
  // rejecting them, since the automaton state alone cannot distinguish a genuinely continued stream
  // from stale state left over from before Flush() ended it.
  StopStringMatcher matcher{Strings({"aaaaa"})};

  EXPECT_FALSE(matcher.Consume("aaa").has_value());
  EXPECT_EQ(matcher.Flush(), "aaa");
  EXPECT_THROW(matcher.Consume("a"), std::runtime_error);
}

TEST(StopStringMatcherTest, FlushIsIdempotentWithoutAnInterveningConsume) {
  StopStringMatcher matcher{Strings({"STOP"})};

  EXPECT_FALSE(matcher.Consume("abc").has_value());
  EXPECT_EQ(matcher.Flush(), "abc");
  // A second Flush() with nothing new to drain is harmless, not a lifecycle error.
  EXPECT_EQ(matcher.Flush(), "");
  EXPECT_EQ(matcher.Flush(), "");
  EXPECT_THROW(matcher.Consume("d"), std::runtime_error);
}

TEST(StopStringMatcherTest, ResetAfterFlushPermitsAFreshStream) {
  StopStringMatcher matcher{Strings({"STOP"})};

  EXPECT_FALSE(matcher.Consume("abc").has_value());
  EXPECT_EQ(matcher.Flush(), "abc");
  EXPECT_THROW(matcher.Consume("d"), std::runtime_error);

  matcher.Reset();
  EXPECT_NO_THROW(matcher.Consume("d"));
  const auto match = matcher.Consume("xSTOP");
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->start_offset, 2u);
  EXPECT_EQ(match->end_offset, 6u);
  EXPECT_EQ(matcher.TakeSafeOutput(), "dx");
}

TEST(StopStringMatcherTest, PendingOutputIsBoundedByLongestStopString) {
  const std::string longest = "abcdef";
  StopStringMatcher matcher{Strings({"zz", "abcdef"})};

  // "Xabcde" repeatedly leaves the longest possible prefix ("abcde") withheld without ever
  // completing a stop string, which is the worst case for pending storage.
  size_t largest_pending = 0;
  for (int i = 0; i < 1000; ++i) {
    ASSERT_FALSE(matcher.Consume("Xabcde").has_value()) << "iteration " << i;
    largest_pending = std::max(largest_pending, matcher.PendingOutput().size());
    EXPECT_EQ(matcher.TakeSafeOutput(), i == 0 ? "X" : "abcdeX");
  }

  EXPECT_EQ(largest_pending, longest.size() - 1);
}

TEST(StopStringMatcherTest, MaxBoundSharedPrefixConsumesDivergenceBytesToCompleteMatch) {
  // 16 configured patterns (the enforced maximum), each sharing a 500-byte common prefix ("a"
  // repeated), diverging only in their final byte. This keeps every pattern's KMP automaton deep
  // (prefix length 500) simultaneously; the completing byte must correctly select the one pattern
  // whose divergence byte matches, not an unrelated one and not a false "no match" from an
  // off-by-one in the per-pattern failure-table walk.
  constexpr size_t kSharedPrefixLength = 500;
  const std::string shared_prefix(kSharedPrefixLength, 'a');
  std::vector<std::string> patterns;
  for (size_t i = 0; i < Generators::kMaxStopStringCount; ++i) {
    patterns.push_back(shared_prefix + static_cast<char>('A' + static_cast<char>(i)));
  }
  StopStringMatcher matcher{patterns};

  // Stream the shared prefix first: every pattern's prefix length grows to kSharedPrefixLength in
  // lockstep, none completing yet.
  for (size_t i = 0; i < kSharedPrefixLength; ++i)
    ASSERT_FALSE(matcher.Consume("a").has_value()) << "prefix byte " << i;
  EXPECT_EQ(matcher.PendingOutput().size(), kSharedPrefixLength);

  // Diverge into pattern index 9's unique final byte ('J'), which must actually complete that
  // pattern -- not merely fail to match, and not match a different index.
  const size_t target_index = 9;
  const std::string divergence_byte(1, static_cast<char>('A' + static_cast<char>(target_index)));
  const auto match = matcher.Consume(divergence_byte);
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->index, target_index);
  EXPECT_EQ(match->start_offset, 0u);
  EXPECT_EQ(match->end_offset, kSharedPrefixLength + 1);
}

TEST(StopStringMatcherTest, MaxBoundSharedPrefixFalseDivergenceRetainsFullPendingDepth) {
  // Same shared-prefix configuration, but the stream diverges into a byte that matches *none* of
  // the configured patterns' final byte. Every pattern's partial match breaks at the same position,
  // so the correct fallback is no match at all and the pending buffer must retain exactly the
  // still-possible shorter prefixes (here: none of the patterns share any prefix shorter than the
  // full shared run with each other beyond the common "a" run itself, so pending collapses back to
  // whatever the shared "a" run's own self-overlap supports).
  constexpr size_t kSharedPrefixLength = 500;
  const std::string shared_prefix(kSharedPrefixLength, 'a');
  std::vector<std::string> patterns;
  for (size_t i = 0; i < Generators::kMaxStopStringCount; ++i) {
    patterns.push_back(shared_prefix + static_cast<char>('A' + static_cast<char>(i)));
  }
  StopStringMatcher matcher{patterns};

  for (size_t i = 0; i < kSharedPrefixLength; ++i)
    ASSERT_FALSE(matcher.Consume("a").has_value());

  // 'Z' matches none of the 16 patterns' final byte ('A'..'P').
  ASSERT_FALSE(matcher.Consume("Z").has_value());
  // No pattern's prefix can still be completed (every pattern requires its own unique final byte
  // right after the shared run, and none of them is "a" followed by "aZ..." or similar), so the
  // withheld pending output collapses to nothing and everything before it is safe.
  EXPECT_EQ(matcher.PendingOutput().size(), 0u);
  EXPECT_EQ(matcher.TakeSafeOutput(), shared_prefix + "Z");
}

TEST(StopStringMatcherTest, RejectsInvalidConfigurations) {
  EXPECT_THROW(StopStringMatcher{Strings({"ok", ""})}, std::runtime_error);
  EXPECT_THROW(StopStringMatcher{Strings({"\xFF"})}, std::runtime_error);
  EXPECT_THROW(StopStringMatcher{Strings({"ok", "\xC3"})}, std::runtime_error);        // Truncated.
  EXPECT_THROW(StopStringMatcher{Strings({"\x80"})}, std::runtime_error);              // Stray continuation.
  EXPECT_THROW(StopStringMatcher{Strings({"\xED\xA0\x80"})}, std::runtime_error);      // Surrogate.
  EXPECT_THROW(StopStringMatcher{Strings({"\xC0\xAF"})}, std::runtime_error);          // Overlong.
  EXPECT_THROW(StopStringMatcher{Strings({"\xF5\x80\x80\x80"})}, std::runtime_error);  // > U+10FFFF.
  EXPECT_NO_THROW(StopStringMatcher{Strings({"\xF4\x8F\xBF\xBF", "\xC2\x80"})});       // U+10FFFF, U+0080.
}

TEST(StopStringMatcherTest, EnforcesConfiguredBounds) {
  std::vector<std::string> at_count_limit(Generators::kMaxStopStringCount, "x");
  EXPECT_NO_THROW(StopStringMatcher{at_count_limit});

  std::vector<std::string> over_count_limit(Generators::kMaxStopStringCount + 1, "x");
  EXPECT_THROW(StopStringMatcher{over_count_limit}, std::runtime_error);

  const size_t per_string = Generators::kMaxStopStringTotalBytes / Generators::kMaxStopStringCount;
  std::vector<std::string> at_byte_limit(Generators::kMaxStopStringCount, std::string(per_string, 'x'));
  EXPECT_NO_THROW(StopStringMatcher{at_byte_limit});

  std::vector<std::string> over_byte_limit = at_byte_limit;
  over_byte_limit.back() += 'x';
  EXPECT_THROW(StopStringMatcher{over_byte_limit}, std::runtime_error);
}

TEST(StopStringMatcherTest, LongStopStringMatchesAcrossChunks) {
  const std::string stop(1024, 'z');
  StopStringMatcher matcher{std::vector<std::string>{stop}};

  EXPECT_FALSE(matcher.Consume("start").has_value());
  for (size_t i = 0; i + 1 < stop.size(); ++i)
    ASSERT_FALSE(matcher.Consume("z").has_value()) << "byte " << i;
  EXPECT_EQ(matcher.PendingOutput().size(), stop.size() - 1);

  const auto match = matcher.Consume("z");
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->start_offset, 5u);
  EXPECT_EQ(match->end_offset, 5u + stop.size());
  EXPECT_EQ(matcher.Flush(), "start");
}

TEST(StopStringMatcherTest, IsValidUtf8AcceptsWellFormedSequences) {
  EXPECT_TRUE(IsValidUtf8(""));
  EXPECT_TRUE(IsValidUtf8("ascii"));
  EXPECT_TRUE(IsValidUtf8("\xC2\x80"));          // U+0080.
  EXPECT_TRUE(IsValidUtf8("\xDF\xBF"));          // U+07FF.
  EXPECT_TRUE(IsValidUtf8("\xE0\xA0\x80"));      // U+0800.
  EXPECT_TRUE(IsValidUtf8("\xEF\xBF\xBF"));      // U+FFFF.
  EXPECT_TRUE(IsValidUtf8("\xF0\x90\x80\x80"));  // U+10000.
  EXPECT_TRUE(IsValidUtf8("\xF4\x8F\xBF\xBF"));  // U+10FFFF.
  EXPECT_TRUE(IsValidUtf8(std::string_view{"a\0b", 3}));
}

TEST(StopStringMatcherTest, IsValidUtf8RejectsMalformedSequences) {
  EXPECT_FALSE(IsValidUtf8("\x80"));              // Stray continuation.
  EXPECT_FALSE(IsValidUtf8("\xC0\xAF"));          // Overlong two-byte.
  EXPECT_FALSE(IsValidUtf8("\xC1\xBF"));          // Overlong two-byte.
  EXPECT_FALSE(IsValidUtf8("\xE0\x80\xAF"));      // Overlong three-byte.
  EXPECT_FALSE(IsValidUtf8("\xF0\x80\x80\xAF"));  // Overlong four-byte.
  EXPECT_FALSE(IsValidUtf8("\xED\xA0\x80"));      // High surrogate.
  EXPECT_FALSE(IsValidUtf8("\xED\xBF\xBF"));      // Low surrogate.
  EXPECT_FALSE(IsValidUtf8("\xF4\x90\x80\x80"));  // > U+10FFFF.
  EXPECT_FALSE(IsValidUtf8("\xF5\x80\x80\x80"));  // Invalid lead.
  EXPECT_FALSE(IsValidUtf8("\xFF"));              // Invalid lead.
  EXPECT_FALSE(IsValidUtf8("\xE6\x97"));          // Truncated.
  EXPECT_FALSE(IsValidUtf8("\xE6\x97\x41"));      // Missing continuation.
}

namespace {

// Independent whole-stream reference for the selection rules: scan end positions left to right and
// take the longest stop string ending at the first position where any of them ends, breaking ties
// by lowest caller index.
std::optional<StopStringMatch> ReferenceMatch(const std::vector<std::string>& stop_strings, std::string_view stream) {
  for (size_t end = 1; end <= stream.size(); ++end) {
    size_t best_length = 0;
    size_t best_index = 0;
    for (size_t i = 0; i < stop_strings.size(); ++i) {
      const std::string& stop_string = stop_strings[i];
      if (stop_string.size() <= end && stop_string.size() > best_length &&
          stream.substr(end - stop_string.size(), stop_string.size()) == stop_string) {
        best_length = stop_string.size();
        best_index = i;
      }
    }
    if (best_length != 0)
      return StopStringMatch{best_index, end - best_length, end};
  }
  return std::nullopt;
}

}  // namespace

// Chunking must not change the outcome: for randomized configurations, streams, and chunk splits,
// the incremental matcher agrees with the whole-stream reference and publishes exactly the bytes
// before the match (or the whole stream when nothing matches).
TEST(StopStringMatcherTest, RandomizedChunkingMatchesWholeStreamReference) {
  std::mt19937 rng{20260828};
  const std::string alphabet = "aab";

  for (int trial = 0; trial < 2000; ++trial) {
    const size_t stop_string_count = 1 + rng() % 3;
    std::vector<std::string> stop_strings;
    for (size_t i = 0; i < stop_string_count; ++i) {
      std::string stop_string;
      const size_t length = 1 + rng() % 4;
      for (size_t k = 0; k < length; ++k)
        stop_string += alphabet[rng() % alphabet.size()];
      stop_strings.push_back(stop_string);
    }

    std::string stream;
    const size_t stream_length = rng() % 24;
    for (size_t i = 0; i < stream_length; ++i)
      stream += alphabet[rng() % alphabet.size()];

    StopStringMatcher matcher{stop_strings};
    std::optional<StopStringMatch> match;
    std::string published;
    for (size_t offset = 0; offset < stream.size() && !match;) {
      const size_t chunk = 1 + rng() % 4;
      const std::string_view bytes = std::string_view{stream}.substr(offset, chunk);
      offset += bytes.size();
      match = matcher.Consume(bytes);
      published += matcher.TakeSafeOutput();
    }
    published += matcher.Flush();

    const auto expected = ReferenceMatch(stop_strings, stream);
    ASSERT_EQ(match.has_value(), expected.has_value()) << "trial " << trial << " stream '" << stream << "'";
    if (expected) {
      EXPECT_EQ(match->index, expected->index) << "trial " << trial << " stream '" << stream << "'";
      EXPECT_EQ(match->start_offset, expected->start_offset) << "trial " << trial << " stream '" << stream << "'";
      EXPECT_EQ(match->end_offset, expected->end_offset) << "trial " << trial << " stream '" << stream << "'";
      EXPECT_EQ(published, stream.substr(0, static_cast<size_t>(expected->start_offset))) << "trial " << trial;
    } else {
      EXPECT_EQ(published, stream) << "trial " << trial;
    }
  }
}

}  // namespace Generators::test
