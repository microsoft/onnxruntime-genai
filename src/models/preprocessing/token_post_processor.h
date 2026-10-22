#pragma once

#include "genai_tokenizer.h"
#include "metadata_core_state.h"
#include "post_processor.h"

namespace Generators {

class TokenPostProcessorInternal {
 public:
  TokenPostProcessorInternal(std::shared_ptr<Tokenizer> tokenizer, MetadataCoreConfig config);
  void Process(const TokenOutputs& outputs);
  const std::vector<PostProcessorResult>& GetNextTokenResults() const { return results_; }
  void Finalize();
  void Reset();

 private:
  std::shared_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<TokenizerStream> stream_;
  const MetadataCoreConfig config_;
  std::unique_ptr<MetadataCoreState> metadata_state_;
  std::vector<PostProcessorResult> results_;
};

class TokenPostProcessor : public PostProcessor, public LeakChecked<TokenPostProcessor> {
 public:
  explicit TokenPostProcessor(const Generator& generator);
  void ProcessNextTokens();
  const std::vector<PostProcessorResult>& GetNextTokenResults() const { return internal_.GetNextTokenResults(); }
  void Finalize();
  void Reset();

 private:
  TokenPostProcessorInternal internal_;
};

}  // namespace Generators