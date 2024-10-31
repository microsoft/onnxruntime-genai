#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <sys/types.h>
#include <regex>

#include "llguidance.h"

#include "logits_processor.h"

namespace Generators {

ConstrainedLogitsProcessor::ConstrainedLogitsProcessor(int vocab_size, uint32_t eos_token,
                                                       const std::string& guidance_type, const std::string& guidance_data,
                                                       std::shared_ptr<Tokenizer> tokenizer)
    : tokenizer_(std::move(tokenizer)), vocab_size_(vocab_size) {
  if (guidance_type.empty() || guidance_data.empty()) {
    throw std::runtime_error("Guidance type and data must be provided");
  }

  if (guidance_type != "json_schema" && guidance_type != "regex" && guidance_type != "grammar") {
    throw std::runtime_error("Unsupported guidance type: " + guidance_type);
  }

  std::vector<uint8_t> tokens;
  std::vector<uint32_t> token_lens;
  for (int i = 0; i < vocab_size; i++) {
    std::vector<int32_t> ids = {i};
    std::string token = tokenizer_->Decode(ids);
    token_lens.push_back(token.size());
    for (char c : token) {
      tokens.push_back(c);
    }
  }

  LlgTokenizeFn tokenizer_fn = [](const void* user_data, const uint8_t* bytes,
                                  size_t bytes_len, uint32_t* output_tokens, size_t output_tokens_len) -> unsigned long {
    std::string input_string = "\x02";
    input_string.reserve(bytes_len);
    for (size_t i = 0; i < bytes_len; i++) {
      input_string.push_back(bytes[i]);
    }
    const Tokenizer* tokenizer = reinterpret_cast<const Tokenizer*>(user_data);
    std::vector<int32_t> output_ids = tokenizer->Encode(input_string.c_str());
    std::string prefix = "\x02";
    std::vector<int32_t> prefix_ids = tokenizer->Encode(prefix.c_str());
    auto prefix_len = prefix_ids.size();

    size_t output_size = std::min(output_tokens_len, output_ids.size() - prefix_len);
    for (size_t i = 0; i < output_size; i++) {
      output_tokens[i] = output_ids[prefix_len + i];
    }
    return output_ids.size();
  };

  LlgTokenizerInit tokenizer_init = {
      .vocab_size = static_cast<uint32_t>(vocab_size),
      .tok_eos = eos_token,
      .token_lens = token_lens.data(),
      .token_bytes = tokens.data(),
      .tokenize_assumes_string = true,
      .tokenize_fn = tokenizer_fn,
      .tokenize_user_data = tokenizer_.get(),
  };

  llg_tokenizer_ = std::unique_ptr<LlgTokenizer, LlgTokenizerDeleter>(llg_new_tokenizer(&tokenizer_init));

  LlgConstraintInit constraint_init;
  llg_constraint_init_set_defaults(&constraint_init, llg_tokenizer_.get());
  LlgConstraint* constraint_ptr;
  if (guidance_type == "json_schema") {
    constraint_ptr = llg_new_constraint_json(&constraint_init, guidance_data.c_str());
  } else if (guidance_type == "regex") {
    constraint_ptr = llg_new_constraint_regex(&constraint_init, guidance_data.c_str());
  } else {
    constraint_ptr = llg_new_constraint(&constraint_init, guidance_data.c_str());
  }
  if (llg_get_error(constraint_ptr) != nullptr) {
    std::string error_message = llg_get_error(constraint_ptr);
    llg_free_constraint(constraint_ptr);
    throw std::runtime_error("Error creating grammar: " + error_message);
  }
  llg_constraint_ = std::unique_ptr<LlgConstraint, LlgConstraintDeleter>(constraint_ptr);
}

std::vector<uint32_t> ConstrainedLogitsProcessor::ComputeMask() {
  // LlgMaskResult mask_result;
  auto error = llg_compute_mask(llg_constraint_.get(), &mask_result_);
  if (error != 0) {
    std::string error_message = llg_get_error(llg_constraint_.get());
    throw std::runtime_error("Error computing mask: " + error_message);
  }

  std::vector<uint32_t> mask;
  mask.reserve((vocab_size_ - 1) / 32 + 1);
  for (int i = 0; i < (vocab_size_ - 1) / 32 + 1; i++) {
    mask.push_back(mask_result_.sample_mask[i]);
  }
  return mask;
}

void ConstrainedLogitsProcessor::CommitTokens(uint32_t token) {
  LlgCommitResult commit_result;
  auto error = llg_commit_token(llg_constraint_.get(), token, &commit_result);
  if (error != 0) {
    std::string error_message = llg_get_error(llg_constraint_.get());
    throw std::runtime_error("Error committing tokens: " + error_message);
  }
}

}  // namespace Generators