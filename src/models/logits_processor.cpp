// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/types.h>

#if USE_GUIDANCE
#include "llguidance.h"
#endif

#include "logits_processor.h"

namespace Generators {

#if USE_GUIDANCE
GuidanceLogitsProcessor::GuidanceLogitsProcessor(int vocab_size, uint32_t eos_token,
                                                 const std::string& guidance_type, const std::string& guidance_data,
                                                 std::shared_ptr<Tokenizer> tokenizer, const std::string& tokenizer_path)
    : vocab_size_(vocab_size), eos_token_(eos_token), tokenizer_(std::move(tokenizer)) {
  if (guidance_type.empty() || guidance_data.empty()) {
    throw std::runtime_error("Guidance type and data must be provided");
  }

  if (guidance_type != "json_schema" && guidance_type != "regex" && guidance_type != "grammar") {
    throw std::runtime_error("Unsupported guidance type: " + guidance_type);
  }

  auto tokenize_fn = (LlgTokenizeFn) + [](const void* user_data, const uint8_t* bytes,
                                          size_t bytes_len, uint32_t* output_tokens, size_t output_tokens_len) -> unsigned long {
    const TokenizeData* tokenize_data = reinterpret_cast<const TokenizeData*>(user_data);
    auto output_ids = tokenize_partial(reinterpret_cast<const Tokenizer*>(tokenize_data->tokenizer), tokenize_data->prefix_len, bytes, bytes_len);
    size_t output_size = std::min(output_tokens_len, output_ids.size());
    for (size_t i = 0; i < output_size; i++) {
      output_tokens[i] = output_ids[i];
    }
    return static_cast<unsigned long>(output_ids.size());
  };

  // TODO reuse the tokenizer between constraints
  fs::path tokenizer_path_fs(tokenizer_path);
  fs::path json_path(tokenizer_path_fs / kDefaultVocabFile);
  std::ifstream json_file(json_path.string());
  std::stringstream json_buffer;
  json_buffer << json_file.rdbuf();
  std::string json_data = json_buffer.str();
  auto prefix_len = tokenizer_->Encode(kTokenizePrefixStr).size();
  tokenize_data_ = {tokenizer_.get(), prefix_len};
  LlgTokenizerInit tokenizer_init = {
      static_cast<uint32_t>(vocab_size_),
      eos_token,
      nullptr,
      nullptr,
      json_data.c_str(),
      false,
      tokenize_fn,
      false,
      &tokenize_data_,
  };

  char error_buf[128];
  llg_tokenizer_ = std::unique_ptr<LlgTokenizer, LlgTokenizerDeleter>(llg_new_tokenizer(&tokenizer_init, error_buf, sizeof(error_buf)));
  if (!llg_tokenizer_) {
    throw std::runtime_error("Error creating llg_tokenizer: " + std::string(error_buf));
  }

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
    auto error = std::runtime_error("Error creating grammar: " + error_message);
    llg_free_constraint(constraint_ptr);
    throw error;
  }
  llg_constraint_ = std::unique_ptr<LlgConstraint, LlgConstraintDeleter>(constraint_ptr);
}

std::vector<uint32_t> GuidanceLogitsProcessor::ComputeMask() {
  LlgMaskResult mask_result;
  auto error = llg_compute_mask(llg_constraint_.get(), &mask_result);
  if (error != 0) {
    std::string error_message = llg_get_error(llg_constraint_.get());
    throw std::runtime_error("Error computing mask: " + error_message);
  }

  std::vector<uint32_t> mask;
  if (mask_result.is_stop) {
    mask = std::vector<uint32_t>((vocab_size_ - 1) / 32 + 1, 0);
    uint32_t eos_mask32 = 1 << (eos_token_ % 32);
    mask[eos_token_ / 32] = eos_mask32;
  } else {
    mask.reserve((vocab_size_ - 1) / 32 + 1);
    for (int i = 0; i < (vocab_size_ - 1) / 32 + 1; i++) {
      mask.push_back(mask_result.sample_mask[i]);
    }
  }
  return mask;
}

void GuidanceLogitsProcessor::CommitTokens(uint32_t token) {
  LlgCommitResult commit_result;
  auto error = llg_commit_token(llg_constraint_.get(), token, &commit_result);
  if (error != 0) {
    std::string error_message = llg_get_error(llg_constraint_.get());
    throw std::runtime_error("Error committing tokens: " + error_message);
  }
}

std::vector<int32_t> GuidanceLogitsProcessor::tokenize_partial(const Tokenizer* tokenizer, const size_t prefix_len,
                                                               const uint8_t* bytes, size_t bytes_len) {
  // add prefix to tokenize for partial tokenization, it will produce ids more stable
  std::string input_string = kTokenizePrefixStr;
  input_string.reserve(bytes_len + 2);
  for (size_t i = 0; i < bytes_len; i++) {
    input_string.push_back(bytes[i]);
  }
  std::vector<int32_t> output_ids = tokenizer->Encode(input_string.c_str());
  return std::vector<int32_t>(output_ids.begin() + prefix_len, output_ids.end());
}
#endif

std::unique_ptr<LogitsProcessor> CreateLogitsProcessor(const LogitsProcessorConfig& config) {
#if USE_GUIDANCE
  if (!config.guidance_type.empty() && !config.guidance_data.empty()) {
    return std::make_unique<GuidanceLogitsProcessor>(config.vocab_size, config.eos_token, config.guidance_type, config.guidance_data, config.tokenizer, config.tokenizer_path);
  }

#endif

  Log("warning", "No supported LogitsProcessor found. e.g. to use guidance, build with use_guidance=true");
  return nullptr;
}
}  // namespace Generators
