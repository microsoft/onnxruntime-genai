// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ortx_cpp_helper.h"
#include "speech_extractor.h"
#include "utils.h"

namespace Generators {

struct Audios {
  Audios(ort_extensions::OrtxObjectPtr<OrtxRawAudios> audios, size_t num_audios)
      : audios_(std::move(audios)), num_audios_{num_audios} {}

  Audios() = delete;
  Audios(const Audios&) = delete;
  Audios& operator=(const Audios&) = delete;

  ort_extensions::OrtxObjectPtr<OrtxRawAudios> audios_;
  size_t num_audios_{};
};

std::unique_ptr<Audios> LoadAudioImpl(const char* audio_path);

struct AudioProcessor {
  AudioProcessor(Config& config, const SessionInfo& session_info);

  AudioProcessor() = delete;
  AudioProcessor(const AudioProcessor&) = delete;
  AudioProcessor& operator=(const AudioProcessor&) = delete;

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Audios* audios,
                                        const std::string& language, const std::string& task,
                                        int32_t no_timestamps) const;

 private:
  ort_extensions::OrtxObjectPtr<OrtxFeatureExtractor> processor_;
  ONNXTensorElementDataType input_features_type_;
};

}  // namespace Generators
