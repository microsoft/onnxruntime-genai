#include <memory>
#include <new>
#include <string>
#include <vector>
#include <queue>

#include "ort_genai.h"
#include "server/engine_utils.h"
#include "server/scheduler.h"
#include "models/model.h"

namespace engine {

struct CompletionOutput {
  int index;
  std::string text;
  std::vector<int> token_ids;
  float cumulative_logprob;
  std::string finish_reason;
  std::string stop_reason;
};

struct RequestOutput {
  int64_t request_id;
  std::string output;
  std::vector<int> prompt_token_ids;
  std::vector<CompletionOutput> outputs;
  bool finished;
};

class OgaEngine {
 public:
  OgaEngine(const char* config_path);
  ~OgaEngine() {}

  int64_t AddRequestTest(const char* prompt);
  void AddRequest(std::string request_id, std::string inputs, SamplingParams params,
                  float arrival_time = 0);
  // std::vector<std::tuple<int64_t, const char*>> Step();
  void RunEngine();
  std::vector<std::string> Generate(std::vector<const char*> prompts);
  std::vector<RequestOutput> Step();

 private:
  std::vector<const char*> Schedule();

  const int kMaxBatchSize = 16;
  int64_t seq_count_ = 0;
  // std::unique_ptr<OgaModel> model_;
  std::shared_ptr<Generators::Model> model_;
  // std::unique_ptr<OgaTokenizer> tokenizer_;
  std::shared_ptr<Generators::Tokenizer> tokenizer_;
  std::unique_ptr<Scheduler> scheduler_;
  std::queue<const char*> unscheduled_prompts_;
};
}  // namespace engine