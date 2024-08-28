#pragma once
#include <vector>
#include "../models/model.h"
#include "../generators.h"
#include "../search.h"
#include "engine_utils.h"
#include "scheduler.h"
#include "../models/cache_manager.h"

namespace Generators {
class ModelRunner {
 public:
  ModelRunner(std::shared_ptr<Generators::Model> model,
              const CacheOptions& cache_config);
  std::vector<CompletionSequenceGroupOutput> ExecuteModel(
      const ExecuteModelRequest& request);

 private:
  std::shared_ptr<Generators::Model> model_;
  CacheOptions cache_config_;

  std::vector<int32_t> RunGenerator(const GeneratorParams& params);
};
};  // namespace Generators