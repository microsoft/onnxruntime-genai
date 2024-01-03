#include "gpt_common.h"
#include "model.h"
#include "kv_cache.h"
#include "position_ids.h"

namespace Generators {

struct Gpt_State : State {

  Gpt_State(Gpt_Model& model, RoamingArray<int32_t> sequence_lengths, const SearchParams& search_params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;

 private:
  void UpdateInputs(cpu_span<const int32_t> next_tokens, cpu_span<const int32_t> beam_indices, int current_length);

  const SearchParams& search_params_;
  bool first_run_{true};

  Gpt_Model* model_;
  KV_Cache_Combined kv_cache_;
  PositionIDs<int32_t> position_ids_;

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