// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "multi_modal_processor.h"

#include "models/model.h"
#include "models/env_utils.h"
#include "models/preprocessing/gemma4_multimodal_processor.h"
#include "models/preprocessing/gemma_image_processor.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/preprocessing/lfm2_audio_processor.h"
#include "models/preprocessing/lfm2_vl_image_processor.h"
#include "models/preprocessing/mistral3_image_processor.h"
#include "models/preprocessing/parakeet_processor.h"
#include "models/preprocessing/phi_image_processor.h"
#include "models/preprocessing/phi_multimodal_processor.h"
#include "models/preprocessing/qwen2_5_vl_image_processor.h"
#include "models/preprocessing/videochat_flash_processor.h"
#include "models/preprocessing/whisper_processor.h"
#include "models/threadpool.h"

#include <algorithm>
#include <charconv>
#include <stdexcept>
#include <thread>

namespace Generators {

size_t GetPreprocessingWorkerCount(size_t hardware_threads) {
  const auto setting = GetEnv("ORTGENAI_PREPROCESSING_THREADS");
  if (setting.empty())
    return hardware_threads > 1 ? std::min<size_t>(3, hardware_threads / 2) : 0;
  unsigned int workers{};
  const auto result = std::from_chars(setting.data(), setting.data() + setting.size(), workers);
  if (result.ec != std::errc{} || result.ptr != setting.data() + setting.size() || workers > 3)
    throw std::invalid_argument("ORTGENAI_PREPROCESSING_THREADS must be an integer from 0 to 3");
  return workers;
}

std::shared_ptr<ThreadPool> Model::GetOrCreatePreprocessingThreadPool() const {
  std::call_once(preprocessing_thread_pool_once_, [&] {
    const size_t preprocessing_workers = GetPreprocessingWorkerCount(std::thread::hardware_concurrency());
    if (preprocessing_workers != 0)
      preprocessing_thread_pool_ = std::make_shared<ThreadPool>(preprocessing_workers);
  });
  return preprocessing_thread_pool_;
}

std::shared_ptr<MultiModalProcessor> Model::CreateMultiModalProcessor() const {
  if (config_->external_file_sources_ &&
      config_->model.type != "whisper" && config_->model.type != "phi4mm") {
    throw std::runtime_error("File-source preprocessing is unsupported for model type: " + config_->model.type);
  }
  return std::make_shared<MultiModalProcessor>(*config_, session_info_,
                                               config_->model.type == "whisper" ? nullptr : GetOrCreatePreprocessingThreadPool());
}

MultiModalProcessor::MultiModalProcessor(Config& config, const SessionInfo& session_info,
                                         std::shared_ptr<ThreadPool> thread_pool)
    : tokenizer_{std::make_shared<Tokenizer>(config)},
      thread_pool_{std::move(thread_pool)},
      processor_factory_{
          {"phi3v", Processor::Create<PhiImageProcessor>},
          {"whisper", Processor::Create<WhisperProcessor>},
          {"parakeet_tdt", Processor::Create<ParakeetTdtProcessor>},
          {"phi4mm", Processor::Create<PhiMultiModalProcessor>},
          {"gemma3", Processor::Create<GemmaImageProcessor>},
          {"lfm2_audio", Processor::Create<Lfm2AudioProcessor>},
          {"lfm2_vl", Processor::Create<Lfm2VlImageProcessor>},
          {"gemma4", Processor::Create<Gemma4MultiModalProcessor>},
          {"gemma4_unified", Processor::Create<Gemma4MultiModalProcessor>},
          {"mistral3", Processor::Create<Mistral3ImageProcessor>},
          {"fara", Processor::Create<QwenImageProcessor>},
          {"qwen2_5_vl", Processor::Create<QwenImageProcessor>},
          {"qwen3_vl", Processor::Create<QwenImageProcessor>},
          {"qwen3_5", Processor::Create<QwenImageProcessor>},
          {"qwen3_5_moe", Processor::Create<QwenImageProcessor>},
          {"videochat_flash_qwen", Processor::Create<VideoChatFlashProcessor>}} {
  auto processor = processor_factory_.find(config.model.type);
  if (processor != processor_factory_.end()) {
    processor_ = processor->second(config, session_info, thread_pool_.get());
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
