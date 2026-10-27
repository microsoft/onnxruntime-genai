#pragma once

#include "models/preprocessing/genai_tokenizer.h"
#include "metadata_core_state.h"
#include "post_processor.h"

namespace Generators {

class ASRPostProcessorInternal {
 public:
  ASRPostProcessorInternal(std::shared_ptr<Tokenizer> tokenizer, MetadataCoreConfig config);
  void Process(const TokenOutputs& outputs);
  const std::vector<ASRPostProcessorResult>& GetNextResults() const { return results_; }
  void Finalize();
  void Reset();

 private:
  std::shared_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<TokenizerStream> stream_;
  const MetadataCoreConfig config_;
  std::unique_ptr<MetadataCoreState> metadata_state_;
  std::vector<ASRPostProcessorResult> results_;
};

class ASRPostProcessor : public PostProcessor, public LeakChecked<ASRPostProcessor> {
 public:
  explicit ASRPostProcessor(const Generator& generator);
  void ProcessNext() override;
  const std::vector<ASRPostProcessorResult>& GetNextResults() const { return internal_.GetNextResults(); }
  void Finalize() override;
  void Reset() override;

 private:
  ASRPostProcessorInternal internal_;
};

}  // namespace Generators