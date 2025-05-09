// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/types.h>

#include "generators.h"
#include "models/model.h"
#if USE_GUIDANCE
#include "llguidance.h"
#endif

#include "constrained_logits_processor.h"

namespace Generators {

#if USE_GUIDANCE
GuidanceLogitsProcessor::GuidanceLogitsProcessor(const State& state)
    : params_(state.params_),
      eos_token_(state.params_->config.model.eos_token_id[0]) {
  if (params_->guidance_type.empty() || params_->guidance_type.empty()) {
    throw std::runtime_error("Guidance type and data must be provided together");
  }

  if (params_->guidance_type != "json_schema" && params_->guidance_type != "regex" && params_->guidance_type != "lark_grammar") {
    throw std::runtime_error("Unsupported guidance type: " + std::string(params_->guidance_type) + " (only json_schema, regex and lark_grammar are supported)");
  }

  auto tokenize_fn = (LlgTokenizeFn) + [](const void* user_data, const uint8_t* bytes,
                                          size_t bytes_len, uint32_t* output_tokens, size_t output_tokens_len)
      -> unsigned long {
    const TokenizeData* tokenize_data = reinterpret_cast<const TokenizeData*>(user_data);
    auto output_ids = tokenize_partial(reinterpret_cast<const Tokenizer*>(tokenize_data->tokenizer), tokenize_data->prefix_len, bytes, bytes_len);
    size_t output_size = std::min(output_tokens_len, output_ids.size());
    for (size_t i = 0; i < output_size; i++) {
      output_tokens[i] = output_ids[i];
    }
    return static_cast<unsigned long>(output_ids.size());
  };

  auto tokenizer_path = state.params_->config.config_path.string();
  fs::path tokenizer_path_fs(tokenizer_path);
  fs::path json_path(tokenizer_path_fs / kDefaultVocabFile);
  std::ifstream json_file(json_path.string());
  std::stringstream json_buffer;
  json_buffer << json_file.rdbuf();
  std::string json_data = json_buffer.str();
  tokenizer_ = state.model_.CreateTokenizer();
  auto prefix_len = tokenizer_->Encode(kTokenizePrefixStr).size();
  tokenize_data_ = {tokenizer_.get(), prefix_len};
  LlgTokenizerInit tokenizer_init = {
      static_cast<uint32_t>(params_->config.model.vocab_size),  // vocab_size
      eos_token_,                                               // eos_token
      nullptr,                                                  // token_lens
      nullptr,                                                  // token_bytes
      json_data.c_str(),                                        // tokenizer_json config data
      false,                                                    // tokenize_assumes_string
      tokenize_fn,                                              // tokenize_fn
      false,                                                    // use_approximate_greedy_tokenize_fn
      &tokenize_data_,                                          // user_data
  };

  char error_buf[256];
  llg_tokenizer_ = std::unique_ptr<LlgTokenizer, LlgTokenizerDeleter>(llg_new_tokenizer(&tokenizer_init, error_buf, sizeof(error_buf)));
  if (!llg_tokenizer_) {
    throw std::runtime_error("Error creating llg_tokenizer: " + std::string(error_buf));
  }

  llg_constraints_.resize(params_->search.batch_size);
  for (int i = 0; i < params_->search.batch_size; i++) {
    LlgConstraintInit constraint_init;
    llg_constraint_init_set_defaults(&constraint_init, llg_tokenizer_.get());
    LlgConstraint* constraint_ptr = nullptr;
    if (params_->guidance_type == "json_schema") {
      constraint_ptr = llg_new_constraint_json(&constraint_init, params_->guidance_data.data());
    } else if (params_->guidance_type == "regex") {
      constraint_ptr = llg_new_constraint_regex(&constraint_init, params_->guidance_data.data());
    } else if (params_->guidance_type == "lark_grammar") {
      constraint_ptr = llg_new_constraint_lark(&constraint_init, params_->guidance_data.data());
    }
    if (llg_get_error(constraint_ptr) != nullptr) {
      std::string error_message = llg_get_error(constraint_ptr);
      llg_free_constraint(constraint_ptr);
      throw std::runtime_error("Error creating grammar: " + error_message);
    }
    llg_constraints_[i] = std::unique_ptr<LlgConstraint, LlgConstraintDeleter>(constraint_ptr);
  }

  // Compute the mask asynchronously to avoid blocking the model inference on device
  mask_future_ = std::async(std::launch::async, [&]() {
    return ComputeMask();
  });
}

std::vector<std::vector<uint32_t>> GuidanceLogitsProcessor::ComputeMask() {
  std::vector<std::vector<uint32_t>> masks;
  for (int batch_idx = 0; batch_idx < params_->search.batch_size; batch_idx++) {
    LlgMaskResult mask_result;
    auto error = llg_compute_mask(llg_constraints_[batch_idx].get(), &mask_result);
    if (error != 0) {
      // If the mask computation fails, we need to reset the constraint
      // and try again. LLGuidance needs to be reset for every new prompt.
      ResetWithoutCompute();
      auto retry_error = llg_compute_mask(llg_constraints_[batch_idx].get(), &mask_result);
      if (retry_error != 0) {
        std::string error_message = llg_get_error(llg_constraints_[batch_idx].get());
        throw std::runtime_error("Error computing mask: " + error_message);
      }
    }

    std::vector<uint32_t> mask;
    if (mask_result.is_stop) {
      // when logits processor decides to stop, we mask all tokens except the EOS token
      mask = std::vector<uint32_t>((params_->config.model.vocab_size - 1) / 32 + 1, 0);
      uint32_t eos_mask32 = 1 << (eos_token_ % 32);
      mask[eos_token_ / 32] = eos_mask32;
    } else {
      mask.reserve((params_->config.model.vocab_size - 1) / 32 + 1);
      for (int i = 0; i < (params_->config.model.vocab_size - 1) / 32 + 1; i++) {
        mask.push_back(mask_result.sample_mask[i]);
      }
    }
    masks.push_back(mask);
  }
  return masks;
}

void GuidanceLogitsProcessor::CommitTokens(std::span<int32_t> tokens) {
  for (int i = 0; i < params_->search.batch_size; i++) {
    LlgCommitResult commit_result;
    auto error = llg_commit_token(llg_constraints_[i].get(), static_cast<uint32_t>(tokens[i]), &commit_result);
    if (error != 0) {
      std::string error_message = llg_get_error(llg_constraints_[i].get());
      throw std::runtime_error("Error committing tokens: " + error_message);
    }
  }
  mask_future_ = std::async(std::launch::async, [&]() {
    return ComputeMask();
  });
  masks_.clear();
}

std::vector<std::vector<uint32_t>> GuidanceLogitsProcessor::GetMask() {
  if (masks_.empty()) {
    masks_ = mask_future_.get();
  }
  return masks_;
}

void GuidanceLogitsProcessor::ProcessLogits(DeviceSpan<float> logits) {
  auto masks = GetMask();

  if (params_->p_device->GetType() == DeviceType::CUDA) {
    const size_t words_per_row = params_->config.model.vocab_size / 32;
    const size_t total_words = masks.size() * words_per_row;
    std::vector<uint32_t> flat_masks(total_words);
    uint32_t* dst = flat_masks.data();
    for (const auto& row : masks) {
      std::memcpy(dst, row.data(), words_per_row * sizeof(uint32_t));
      dst += words_per_row;
    }
    auto cuda_logits_mask_ptr_ = params_->p_device->Allocate<uint32_t>(total_words);
    copy(std::span<const uint32_t>{flat_masks}, cuda_logits_mask_ptr_.CpuSpan());
    cuda_logits_mask_ptr_.CopyCpuToDevice();
    params_->p_device->LaunchAddLogitsMask(logits.Span().data(), params_->search.batch_size, params_->config.model.vocab_size, cuda_logits_mask_ptr_.Span().data());
    return;
  }
  size_t vocab_index = 0;

  auto logits_span = logits.CpuSpan();
  for (int index = 0; index < params_->search.batch_size; index++) {
    auto subspan = logits_span.subspan(vocab_index, params_->config.model.vocab_size);
    auto& mask = masks[index];
    for (size_t i = 0; i < params_->config.model.vocab_size; i++) {
      // mask is a 32-bit integer, where each bit corresponds to a token in the vocabulary.
      // If the bit is set, the corresponding token is masked (i.e., its logit is set to the lowest possible value).
      subspan[i] = mask[i / 32] & (1 << (i % 32)) ? subspan[i] : std::numeric_limits<float>::lowest();
    }
    vocab_index += params_->config.model.vocab_size;
  }
}

void GuidanceLogitsProcessor::ResetWithoutCompute() {
  masks_.clear();
  llg_constraints_.clear();
  llg_constraints_.resize(params_->search.batch_size);
  for (int i = 0; i < params_->search.batch_size; i++) {
    LlgConstraintInit constraint_init;
    llg_constraint_init_set_defaults(&constraint_init, llg_tokenizer_.get());
    LlgConstraint* constraint_ptr;
    if (params_->guidance_type == "json_schema") {
      constraint_ptr = llg_new_constraint_json(&constraint_init, params_->guidance_data.data());
    } else if (params_->guidance_type == "regex") {
      constraint_ptr = llg_new_constraint_regex(&constraint_init, params_->guidance_data.data());
    } else if (params_->guidance_type == "lark_grammar") {
      constraint_ptr = llg_new_constraint_lark(&constraint_init, params_->guidance_data.data());
    } else {
      throw std::runtime_error("Unsupported guidance type: " + std::string(params_->guidance_type) + " (only json_schema, regex and lark_grammar are supported)");
    }
    if (llg_get_error(constraint_ptr) != nullptr) {
      std::string error_message = llg_get_error(constraint_ptr);
      llg_free_constraint(constraint_ptr);
      throw std::runtime_error("Error creating grammar: " + error_message);
    }
    llg_constraints_[i] = std::unique_ptr<LlgConstraint, LlgConstraintDeleter>(constraint_ptr);
  }
}

// Reset the masks and llguidance constraints and then recompute the mask
void GuidanceLogitsProcessor::Reset() {
  ResetWithoutCompute();
  mask_future_ = std::async(std::launch::async, [&]() {
    return ComputeMask();
  });
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

std::unique_ptr<ConstrainedLogitsProcessor> CreateGuidanceLogitsProcessor(const State& state) {
  if (!state.params_->guidance_type.empty() && !state.params_->guidance_data.empty()) {
#if USE_GUIDANCE
    return std::make_unique<GuidanceLogitsProcessor>(state);
#endif
    Log("warning", "No supported ConstrainedLogitsProcessor found. e.g. to use guidance, build with use_guidance=true");
  }
  return nullptr;
}
}  // namespace Generators
