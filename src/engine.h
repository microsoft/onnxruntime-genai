#include <memory>
#include <vector>
#include <queue>

#include "ort_genai.h"

class OgaEngine {
 public:
  OgaEngine(const char* config_path);
  ~OgaEngine() {}

  int64_t AddRequest(const char* prompt);
  std::vector<std::tuple<int64_t, const char*>> Step();
  void RunEngine();
  std::vector<std::string> Generate(std::vector<const char*> prompts);

 private:
  std::vector<const char*> Schedule();

  const int kMaxBatchSize = 16;
  int64_t request_id_ = 0;
  std::unique_ptr<OgaModel> model_;
  std::unique_ptr<OgaTokenizer> tokenizer_;
  std::queue<const char*> unscheduled_prompts_;
};