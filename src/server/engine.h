#pragma once
#include <memory>
#include <new>
#include <string>
#include <vector>
#include <queue>

#include "engine_utils.h"
#include "scheduler.h"
#include "model_runner.h"
#include "../models/model.h"

namespace Generators {

struct CompletionOutput {
  int index;
  std::string text;
  std::vector<int> token_ids;
  float cumulative_logprob;
  float logprobs;
  std::string finish_reason;
  std::string stop_reason;
};

struct RequestOutput {
  std::string request_id;
  std::string prompt;
  std::vector<int> prompt_token_ids;
  std::vector<CompletionOutput> outputs;
  bool finished;
};

class OgaEngine {
 public:
  OgaEngine(const char* config_path);
  ~OgaEngine() = default;

  int64_t AddRequestTest(const char* prompt);
  void AddRequest(std::string request_id, const std::string& inputs, SamplingParams params, float arrival_time = 0);
  // std::vector<std::tuple<int64_t, const char*>> Step();
  void RunEngine();
  std::vector<std::string> Generate(std::vector<const char*> prompts);
  std::vector<RequestOutput> Step();

 private:
  std::vector<const char*> Schedule();

  const int kMaxBatchSize = 16;
  int seq_count_ = 0;
  // std::unique_ptr<OgaModel> model_;
  std::shared_ptr<Model> model_;
  std::unique_ptr<ModelRunner> model_runner_;
  // std::unique_ptr<OgaTokenizer> tokenizer_;
  std::shared_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<Scheduler> scheduler_;
  std::queue<const char*> unscheduled_prompts_;
  int block_size_;
};
}  // namespace Generators