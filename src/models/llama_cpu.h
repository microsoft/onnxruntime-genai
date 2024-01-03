#include "llama_common.h"
#include "model.h"
#include "kv_cache.h"
#include "position_ids.h"

namespace Generators {

struct Llama_State : State {

  Llama_State(Llama_Model& model, RoamingArray<int32_t> sequence_lengths, const SearchParams& params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;

private:
  void UpdateInputs(std::span<const int32_t> next_tokens, std::span<const int32_t> next_indices, int current_length);

  Llama_Model* model_;
  const SearchParams& search_params_;
  bool first_run_{true};

  KV_Cache kv_cache_;
  PositionIDs<int64_t> position_ids_;

  // Inputs
  std::unique_ptr<OrtValue> input_ids_;

  std::vector<const char *> input_names_;
  std::vector<OrtValue*> inputs_;

  // Outputs
  std::unique_ptr<OrtValue> logits_;
  std::vector<const char*> output_names_;
  std::vector<OrtValue*> outputs_;
};

}