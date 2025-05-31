// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "extra_inputs.h"

namespace Generators {

struct DiffusionModel : Model {
  DiffusionModel(std::unique_ptr<Config> config, OrtEnv& ort_env);
  DiffusionModel(const DiffusionModel&) = delete;
  DiffusionModel& operator=(const DiffusionModel&) = delete;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  std::unique_ptr<OrtValue> Generate(const ImageGeneratorParams* params) const;

 private:
  std::unique_ptr<OrtValue> CreateLatents(int64_t latent_height, int64_t latent_width, Ort::Allocator* allocator) const;

  //Following constants should be moved to the config
  const int32_t batch_size_ = 1;
  const int32_t max_sequence_length = 77;
  const int32_t hidden_size = 1024;
  const int32_t unet_channels_ = 4;
  const int32_t image_height = 512;
  const int32_t image_width = 512;

  const float init_noise_sigma = 1.0;
  const float vae_scaling_factor = 0.18215f;

  // for SD-turbo, this is 0
  // source: ORT/onnxruntime/python/tools/transformers/models/stable_diffusion/demo_utils.py:
  //     if args.guidance is None:
  //         args.guidance = 0.0 if (is_lcm or is_turbo) else(5.0 if args.version == "xl-1.0" else 7.5)
  const float guidance = 0.0f;

  
  std::unique_ptr<OrtMemoryInfo> p_memory_info_;
  Ort::Allocator& cpu_allocator_;

  std::unique_ptr<OrtSession> p_session_;
  std::unique_ptr<Ort::Allocator> text_encoder_allocator_;

  std::unique_ptr<OrtSession> p_unet_session_;
  std::unique_ptr<Ort::Allocator> unet_allocator_;

  std::unique_ptr<OrtSession> p_vae_session_;
  std::unique_ptr<Ort::Allocator> vae_allocator_;

};

struct DiffusionState : State {
  DiffusionState(const DiffusionModel& model, const GeneratorParams& params);
  DiffusionState(const DiffusionState&) = delete;
  DiffusionState& operator=(const DiffusionState&) = delete;
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;


 private:

  const DiffusionModel& model_;
  ExtraInputs extra_inputs_{*this};  // Model inputs
};
}  // namespace Generators
