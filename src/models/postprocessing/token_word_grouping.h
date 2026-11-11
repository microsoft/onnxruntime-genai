#pragma once

#include "ortx_tokenizer.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Generators {

struct CompletedTokenizerWord {
  std::string text;
  size_t start_token_index{};
  size_t stop_token_index{};
};

class TokenWordGroupingState {
 public:
  void Process(const OrtxWordPieceInfo& piece, std::string_view decoded_text);
  void Finalize();
  const std::vector<CompletedTokenizerWord>& CompletedWords() const { return completed_words_; }
  size_t FirstPendingTokenIndex() const;

 private:
  struct PendingText {
    std::string text;
    size_t start_token_index{};
    size_t stop_token_index{};
    bool active{};
  };

  void Append(PendingText& pending, std::string_view text, size_t start_token_index, size_t stop_token_index);
  void CompleteWord();
  void ProcessTextPart(std::string_view text, size_t start_token_index, size_t stop_token_index, bool starts_word);

  size_t next_token_index_{};
  PendingText pending_word_;
  PendingText pending_delimiter_;
  std::optional<size_t> buffered_output_start_token_index_;
  std::vector<CompletedTokenizerWord> completed_words_;
};

}  // namespace Generators