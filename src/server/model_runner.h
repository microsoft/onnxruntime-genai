#pragma once
#include <memory>
#include <vector>
#include "../models/model.h"
#include "../generators.h"
#include "../search.h"
#include "engine_utils.h"
#include "scheduler.h"
#include "../models/cache_manager.h"
#include "../tensor.h"

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

  std::shared_ptr<Tensor> block_tables_;
  std::shared_ptr<Tensor> slot_mapping_;
  std::shared_ptr<Tensor> context_lens_;
  std::shared_ptr<Tensor> is_prompt_;
  std::unique_ptr<NamedTensors> named_tensors_;

  std::vector<std::string> key_cache_names_;
  std::vector<std::string> value_cache_names_;
  std::vector<std::unique_ptr<OrtValue>> key_caches_;
  std::vector<std::unique_ptr<OrtValue>> value_caches_;

  std::vector<int32_t> RunGenerator(const GeneratorParams& params, std::vector<int32_t>& seq_lens, bool is_prompt);
};
};  // namespace Generators