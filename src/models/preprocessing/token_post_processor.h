#pragma once

#include "genai_tokenizer.h"
#include "metadata_core_state.h"

namespace Generators {

class TokenPostProcessor : public LeakChecked<TokenPostProcessor> {
 public:
  explicit TokenPostProcessor(const Generator& generator);
  void ProcessNextTokens();
  const std::vector<PostProcessorResult>& GetNextTokenResults() const { return results_; }
  void Finalize();
  void Reset();

 private:
  const Generator& generator_;
  std::shared_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<TokenizerStream> stream_;
  const MetadataCoreConfig config_;
  std::unique_ptr<MetadataCoreState> metadata_state_;
  std::vector<PostProcessorResult> results_;
};

}  // namespace Generators