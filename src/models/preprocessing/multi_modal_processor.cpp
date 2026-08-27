// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "multi_modal_processor.h"

#include "models/model.h"
#include "models/preprocessing/gemma4_multimodal_processor.h"
#include "models/preprocessing/gemma_image_processor.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/preprocessing/mistral3_image_processor.h"
#include "models/preprocessing/parakeet_processor.h"
#include "models/preprocessing/phi_image_processor.h"
#include "models/preprocessing/phi_multimodal_processor.h"
#include "models/preprocessing/qwen2_5_vl_image_processor.h"
#include "models/preprocessing/videochat_flash_processor.h"
#include "models/preprocessing/whisper_processor.h"

#include <stdexcept>

namespace Generators {

std::shared_ptr<MultiModalProcessor> Model::CreateMultiModalProcessor() const {
  return std::make_shared<MultiModalProcessor>(*config_, session_info_);
}

MultiModalProcessor::MultiModalProcessor(Config& config, const SessionInfo& session_info)
    : tokenizer_{std::make_shared<Tokenizer>(config)},
      processor_factory_{
          {"phi3v", Processor::Create<PhiImageProcessor>},
          {"whisper", Processor::Create<WhisperProcessor>},
          {"parakeet_tdt", Processor::Create<ParakeetTdtProcessor>},
          {"phi4mm", Processor::Create<PhiMultiModalProcessor>},
          {"gemma3", Processor::Create<GemmaImageProcessor>},
          {"gemma4", Processor::Create<Gemma4MultiModalProcessor>},
          {"mistral3", Processor::Create<Mistral3ImageProcessor>},
          {"fara", Processor::Create<QwenImageProcessor>},
          {"qwen2_5_vl", Processor::Create<QwenImageProcessor>},
          {"qwen3_vl", Processor::Create<QwenImageProcessor>},
          {"qwen3_5", Processor::Create<QwenImageProcessor>},
          {"qwen3_5_moe", Processor::Create<QwenImageProcessor>},
          {"videochat_flash_qwen", Processor::Create<VideoChatFlashProcessor>}} {
  auto processor = processor_factory_.find(config.model.type);
  if (processor != processor_factory_.end()) {
    processor_ = processor->second(config, session_info);
  } else {
    throw std::runtime_error("MultiModalProcessor cannot be created. " + config.model.type + " is not a registered multi-modal model type.");
  }
}

std::unique_ptr<NamedTensors> MultiModalProcessor::Process(const std::string& prompt, const Images* images, const Audios* audios) const {
  Payload payload{prompt, {}, images, audios};
  return processor_->Process(*tokenizer_, payload);
}

std::unique_ptr<NamedTensors> MultiModalProcessor::Process(std::span<const char*> prompts, const Images* images, const Audios* audios) const {
  Payload payload{"", prompts, images, audios};
  return processor_->Process(*tokenizer_, payload);
}

}  // namespace Generators
