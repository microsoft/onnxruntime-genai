// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "config.h"
#include "models/onnxruntime_api.h"
#include "mtp_generator_common.h"
#include "smartptrs.h"
#include "speculative_stats.h"

namespace Generators {

struct DecoderOnly_Model;
struct Generator;
struct GeneratorParams;
struct Model;
struct MultiModalLanguageModel;
struct Tensor;

void ValidateGemma4AssistantOptions(const Config::Search& search, const Config::Speculative& speculative,
                                    int max_logits_sequence_length);

struct Gemma4AssistantGenerator : MtpGeneratorBase {
  Gemma4AssistantGenerator(const Model& target_model, const Model& assistant_model,
                           const GeneratorParams& params);

  void AppendTokens(cpu_span<const int32_t> input_ids) override;

 private:
  void RunRound() override;
  int32_t Draft();
  void EmbedToken(int32_t token);
  void CaptureTargetState(int row);
  void UpdateTargetPrediction(int row, bool capture_state = true);
  void SynchronizeTarget();
  void ArgmaxTargetRows(int first_row, int count, int32_t* output);
  int32_t ArgmaxAssistant();
  OrtValue* ResolveTargetOutput(const std::string& name) const;

  const MultiModalLanguageModel& target_model_;
  const DecoderOnly_Model& assistant_model_;
  std::shared_ptr<GeneratorParams> target_params_;
  std::unique_ptr<Generator> target_;
  std::unique_ptr<OrtRunOptions> assistant_run_options_;
  std::unique_ptr<OrtRunOptions> embedding_run_options_;
  std::unique_ptr<OrtValue> embedding_token_;
  std::shared_ptr<Tensor> current_embedding_;
  std::shared_ptr<Tensor> carried_hidden_;
  std::shared_ptr<Tensor> assistant_input_;
  std::shared_ptr<Tensor> assistant_logits_;
  std::shared_ptr<Tensor> assistant_projected_;
  std::string target_embeddings_name_;
  std::string target_hidden_states_name_;
  std::string target_attention_mask_name_;
  // Target present-KV outputs bound to the head, interleaved key/value per shared layer.
  std::vector<std::string> shared_kv_target_names_;
  std::vector<std::string> assistant_input_name_storage_;
  std::vector<std::string> assistant_output_name_storage_;
  std::vector<const char*> assistant_input_names_;
  std::vector<const char*> assistant_output_names_;
  std::vector<const OrtValue*> assistant_inputs_;
  std::vector<OrtValue*> assistant_outputs_;
  std::vector<int32_t> drafts_;
  std::vector<int32_t> verify_argmax_;
  int32_t target_next_{};
  int num_speculative_tokens_{1};
  int vocab_size_{};
  int hidden_size_{};
  size_t length_{};
  // The target has not yet consumed the last committed token; the next round must feed it first.
  bool target_sync_pending_{};
};

}  // namespace Generators
