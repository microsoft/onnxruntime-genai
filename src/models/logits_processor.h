#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if USE_GUIDANCE
#include <llguidance.h>
#endif

#include "model.h"

namespace Generators {

struct LogitsProcessorConfig {
  int vocab_size;
  uint32_t eos_token;
  std::string guidance_type;
  std::string guidance_data;
  std::shared_ptr<Tokenizer> tokenizer;
  std::string tokenizer_path;
};

struct LogitsProcessor {
  LogitsProcessor() = default;
  virtual ~LogitsProcessor() = default;
  virtual std::vector<uint32_t> ComputeMask() = 0;
  virtual void CommitTokens(uint32_t token) = 0;
};

#if USE_GUIDANCE
struct LlgConstraintDeleter {
  void operator()(LlgConstraint* lc) const {
    llg_free_constraint(lc);
  }
};

struct LlgTokenizerDeleter {
  void operator()(LlgTokenizer* lt) const {
    llg_free_tokenizer(lt);
  }
};

struct GuidanceLogitsProcessor : public LogitsProcessor {
  static constexpr const char* kDefaultVocabFile = "tokenizer.json";
  static constexpr const char* kTokenizePrefixStr = "\x02";

  GuidanceLogitsProcessor(int vocab_size, uint32_t eos_token, const std::string& guidance_type,
                          const std::string& guidance_data, std::shared_ptr<Tokenizer> tokenizer,
                          const std::string& tokenizer_path);
  std::vector<uint32_t> ComputeMask() override;
  void CommitTokens(uint32_t token) override;

  size_t vocab_size_;
  uint32_t eos_token_;
  std::unique_ptr<LlgConstraint, LlgConstraintDeleter> llg_constraint_;
  std::unique_ptr<LlgTokenizer, LlgTokenizerDeleter> llg_tokenizer_;
  std::shared_ptr<Tokenizer> tokenizer_;

  static std::vector<int32_t> tokenize_partial(const Tokenizer* tokenizer, const size_t prefix_len,
                                               const uint8_t* bytes, size_t bytes_len);

  struct TokenizeData {
    Tokenizer* tokenizer;
    size_t prefix_len;
  };
  TokenizeData tokenize_data_;
};
#endif

std::unique_ptr<LogitsProcessor> CreateLogitsProcessor(const LogitsProcessorConfig& config);

}  // namespace Generators