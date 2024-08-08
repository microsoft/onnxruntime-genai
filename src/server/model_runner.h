#include <vector>
#include "../models/model.h"
#include "../generators.h"
#include "engine_utils.h"
#include "scheduler.h"

namespace Generators {
class ModelRunner {
 public:
  ModelRunner(const Generators::Model& model);
  std::vector<CompletionSequenceGroupOutput> ExecuteModel(ExecuteModelRequest request);

 private:
  const Generators::Model& model_;
  const SchedulerConfig& scheduler_config_;
  const CacheConfig& cache_config_;


  std::vector<std::vector<int32_t>> RunGenerator(const Generators::GeneratorParams& params);
};
};  // namespace engine