#pragma once

#include "genai_tokenizer.h"
#include "metadata_core_state.h"

namespace Generators {

struct PostProcessorTimestamp {
  std::string text;
  int64_t start_frame{};
  int64_t stop_frame{};
  double start_time{};
  double stop_time{};
};

struct PostProcessorResult {
  std::string text;
  bool has_timestamps{};
  std::vector<PostProcessorTimestamp> words;
  std::vector<PostProcessorTimestamp> segments;
};

class TokenPostProcessor : public LeakChecked<TokenPostProcessor> {
 public:
  TokenPostProcessor(const Model& model, const Generator& generator);
  void ProcessNextTokens();
  const std::vector<PostProcessorResult>& GetNextTokenResults() const { return results_; }
  void Finalize();
  void Reset();

 private:
  void CheckGenerator() const;
  void AppendResult(const MetadataOutput& result);
  const Generator& generator_;
  std::weak_ptr<int> generator_lifetime_;
  std::shared_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<TokenizerStream> stream_;
  std::unique_ptr<MetadataCoreState> metadata_state_;
  MetadataCoreConfig config_;
  std::vector<PostProcessorResult> results_;
};

}  // namespace Generators