#include "token_word_grouping.h"

#include "string_utils.h"
#include "ustring.h"

#include <algorithm>

namespace Generators {
namespace {

bool IsWhitespace(char32_t value) {
  return IsSpace(value) || value == U'\v' || value == U'\f';
}

bool IsWhitespace(std::string_view text) {
  const ustring codepoints{text};
  return !codepoints.empty() && std::all_of(codepoints.begin(), codepoints.end(), [](char32_t value) {
    return IsWhitespace(value);
  });
}

bool IsPunctuation(std::string_view text) {
  bool found = false;
  for (const auto value : ustring(text)) {
    if (IsWhitespace(value)) continue;
    if (!IsPunct(value)) return false;
    found = true;
  }
  return found;
}

}  // namespace

void TokenWordGroupingState::Append(PendingText& pending, std::string_view text,
                                    size_t start_token_index, size_t stop_token_index) {
  if (!pending.active) {
    pending.start_token_index = start_token_index;
    pending.active = true;
  }
  pending.text.append(text);
  pending.stop_token_index = stop_token_index;
}

void TokenWordGroupingState::CompleteWord() {
  if (!pending_word_.active || pending_word_.text.empty()) return;
  completed_words_.push_back({std::move(pending_word_.text), pending_word_.start_token_index, pending_word_.stop_token_index});
  pending_word_ = {};
}

void TokenWordGroupingState::ProcessTextPart(std::string_view text, size_t start_token_index,
                                             size_t stop_token_index, bool starts_word) {
  if (IsWhitespace(text)) {
    Append(pending_delimiter_, text, start_token_index, stop_token_index);
    return;
  }
  const bool punctuation = IsPunctuation(text);
  if ((starts_word || pending_delimiter_.active) && !punctuation) CompleteWord();
  if (pending_delimiter_.active) {
    Append(pending_word_, pending_delimiter_.text, pending_delimiter_.start_token_index, pending_delimiter_.stop_token_index);
    pending_delimiter_ = {};
  }
  Append(pending_word_, text, start_token_index, stop_token_index);
}

void TokenWordGroupingState::Process(const OrtxWordPieceInfo& piece, std::string_view decoded_text) {
  completed_words_.clear();
  const size_t token_index = next_token_index_++;
  if (decoded_text.empty()) {
    if (!piece.is_special && !buffered_output_start_token_index_) buffered_output_start_token_index_ = token_index;
    return;
  }
  const size_t output_start_token_index = buffered_output_start_token_index_.value_or(token_index);
  buffered_output_start_token_index_.reset();
  const ustring codepoints{decoded_text};
  size_t offset = 0;
  for (size_t index = 0; index < codepoints.size();) {
    const bool first_part = index == 0;
    const bool whitespace = IsWhitespace(codepoints[index]);
    size_t stop = offset;
    do {
      size_t width = ustring::UTF8Len(decoded_text[stop]);
      if (width > decoded_text.size() - stop || static_cast<unsigned char>(decoded_text[stop]) >= 0xF8) width = 1;
      stop += width;
      ++index;
    } while (index < codepoints.size() && IsWhitespace(codepoints[index]) == whitespace);
    ProcessTextPart(decoded_text.substr(offset, stop - offset), output_start_token_index, token_index + 1, first_part && piece.starts_word);
    offset = stop;
  }
  if (piece.boundary_style == OrtxWordBoundaryStyle_SuffixBpe && piece.ends_word) {
    if (pending_delimiter_.active) {
      Append(pending_word_, pending_delimiter_.text, pending_delimiter_.start_token_index, pending_delimiter_.stop_token_index);
      pending_delimiter_ = {};
    }
    CompleteWord();
  }
}

void TokenWordGroupingState::Finalize() {
  completed_words_.clear();
  buffered_output_start_token_index_.reset();
  if (pending_delimiter_.active) {
    Append(pending_word_, pending_delimiter_.text, pending_delimiter_.start_token_index, pending_delimiter_.stop_token_index);
    pending_delimiter_ = {};
  }
  CompleteWord();
}

size_t TokenWordGroupingState::FirstPendingTokenIndex() const {
  size_t first_pending = next_token_index_;
  if (pending_word_.active) first_pending = std::min(first_pending, pending_word_.start_token_index);
  if (pending_delimiter_.active) first_pending = std::min(first_pending, pending_delimiter_.start_token_index);
  if (buffered_output_start_token_index_) first_pending = std::min(first_pending, *buffered_output_start_token_index_);
  return first_pending;
}

}  // namespace Generators