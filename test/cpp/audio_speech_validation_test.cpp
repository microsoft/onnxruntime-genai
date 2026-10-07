// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string>

#include "models/nemotron_speech.h"
#include "models/parakeet.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/preprocessing/nemotron_streaming_processor.h"
#include "models/whisper.h"

namespace {
std::string GetExceptionMessage(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const std::exception& ex) {
    return ex.what();
  }

  return {};
}
}  // namespace

TEST(AudioSpeechValidationTests, WhisperAudioFeaturesRankValidation) {
  EXPECT_NO_THROW(Generators::ValidateWhisperAudioFeaturesShape({1, 80, 3000}, 3000));

  const std::string rank_error = GetExceptionMessage([] {
    Generators::ValidateWhisperAudioFeaturesShape({1, 3000}, 3000);
  });
  EXPECT_NE(rank_error.find("rank 3"), std::string::npos);
}

TEST(AudioSpeechValidationTests, NemotronMelTensorRankValidation) {
  const std::string rank_error = GetExceptionMessage([] {
    Generators::GetValidatedNemotronMelFrameCount({304, 128}, 128);
  });
  EXPECT_NE(rank_error.find("rank 3"), std::string::npos);
}

TEST(AudioSpeechValidationTests, NemotronMelShapeValidationReturnsFrameCountFromMiddleDimension) {
  EXPECT_EQ(Generators::GetValidatedNemotronMelFrameCount({1, 304, 128}, 128), 304);
}

TEST(AudioSpeechValidationTests, NemotronMelDimensionValidation) {
  const std::string mels_error = GetExceptionMessage([] {
    Generators::GetValidatedNemotronMelFrameCount({1, 128, 304}, 128);
  });
  EXPECT_NE(mels_error.find("expected num_mels"), std::string::npos);
  EXPECT_NE(mels_error.find("304"), std::string::npos);
  EXPECT_NE(mels_error.find("128"), std::string::npos);
}

TEST(AudioSpeechValidationTests, NemotronEncoderOutputRankValidation) {
  EXPECT_NO_THROW(Generators::ValidateNemotronEncoderOutputRank({1, 64, 512}));

  const std::string rank_error = GetExceptionMessage([] {
    Generators::ValidateNemotronEncoderOutputRank({1, 64});
  });
  EXPECT_NE(rank_error.find("rank 3"), std::string::npos);
}

TEST(AudioSpeechValidationTests, NemotronTimestampConfiguration) {
  Generators::Config config;
  config.model.type = "nemotron_speech";
  config.model.timestamp_level = Generators::Config::TimestampLevel::All;
  config.model.sample_rate = 16000;
  config.model.hop_length = 160;
  config.model.subsampling_factor = 8;
  config.model.segment_separators = {".", "!"};
  config.model.segment_gap_threshold_seconds = 1.0;

  Generators::NemotronConfig nemotron_config;
  EXPECT_NO_THROW(nemotron_config.PopulateFromConfig(config));
  EXPECT_EQ(nemotron_config.timestamp_level, Generators::Config::TimestampLevel::All);
  EXPECT_EQ(Generators::GetSegmentGapThresholdFrames(config.model), 13);
}

TEST(AudioSpeechValidationTests, NemotronTimestampGapRoundsToNearestFrame) {
  Generators::Config config;
  config.model.type = "nemotron_speech";
  config.model.timestamp_level = Generators::Config::TimestampLevel::Segment;
  config.model.sample_rate = 16000;
  config.model.hop_length = 160;
  config.model.subsampling_factor = 8;
  Generators::NemotronConfig nemotron_config;

  config.model.segment_gap_threshold_seconds = 0.99;
  nemotron_config.PopulateFromConfig(config);
  EXPECT_EQ(Generators::GetSegmentGapThresholdFrames(config.model), 12);

  config.model.segment_gap_threshold_seconds = 1.0;
  nemotron_config.PopulateFromConfig(config);
  EXPECT_EQ(Generators::GetSegmentGapThresholdFrames(config.model), 13);
}

TEST(AudioSpeechValidationTests, NemotronGlobalFrameUsesAbsoluteSampleOrigin) {
  EXPECT_EQ(Generators::GetNemotronGlobalFrame(0, 3, 160, 8), 3);
  EXPECT_EQ(Generators::GetNemotronGlobalFrame(25600, 3, 160, 8), 23);
  EXPECT_EQ(Generators::GetNemotronGlobalFrame(16000, 3, 160, 8), 15);
  EXPECT_THROW(Generators::GetNemotronGlobalFrame(-1, 0, 160, 8), std::runtime_error);
}

TEST(AudioSpeechValidationTests, NemotronChunkOriginKeyMatchesConsumer) {
  auto& allocator = Generators::GetDeviceInterface(Generators::DeviceType::CPU)->GetAllocator();
  auto features = std::make_shared<Generators::Tensor>(
      OrtValue::CreateTensor<float>(allocator, std::array<int64_t, 1>{1}));
  Generators::NamedTensors tensors;
  tensors.emplace(std::string(Generators::Config::Defaults::AudioFeaturesName), features);
  Generators::AddTimestampChunkOrigin(tensors, 8960);
  ASSERT_EQ(tensors.size(), 2U);
  ASSERT_NE(tensors.find("chunk_start_sample"), tensors.end());

  const auto as_extra_inputs = [](const Generators::NamedTensors& named_tensors) {
    std::vector<Generators::ExtraInput> inputs;
    for (const auto& [name, tensor] : named_tensors) inputs.push_back({name, tensor});
    return inputs;
  };
  EXPECT_EQ(Generators::GetNemotronChunkStartSample(as_extra_inputs(tensors)), 8960);

  auto renamed = tensors;
  auto origin = renamed.at("chunk_start_sample");
  renamed.erase("chunk_start_sample");
  renamed.emplace("renamed_chunk_start_sample", origin);
  const auto error = GetExceptionMessage([&] {
    (void)Generators::GetNemotronChunkStartSample(as_extra_inputs(renamed));
  });
  EXPECT_NE(error.find("missing chunk_start_sample"), std::string::npos);
  renamed.erase("renamed_chunk_start_sample");
  EXPECT_THROW(Generators::GetNemotronChunkStartSample(as_extra_inputs(renamed)), std::runtime_error);

  auto wrong_type = std::make_shared<Generators::Tensor>(
      OrtValue::CreateTensor<float>(allocator, std::array<int64_t, 1>{1}));
  tensors.at("chunk_start_sample") = wrong_type;
  EXPECT_THROW(Generators::GetNemotronChunkStartSample(as_extra_inputs(tensors)), std::runtime_error);

  auto wrong_count = std::make_shared<Generators::Tensor>(
      OrtValue::CreateTensor<int64_t>(allocator, std::array<int64_t, 1>{2}));
  tensors.at("chunk_start_sample") = wrong_count;
  EXPECT_THROW(Generators::GetNemotronChunkStartSample(as_extra_inputs(tensors)), std::runtime_error);

  auto negative = OrtValue::CreateTensor<int64_t>(allocator, std::array<int64_t, 1>{1});
  *negative->GetTensorMutableData<int64_t>() = -1;
  tensors.at("chunk_start_sample") = std::make_shared<Generators::Tensor>(std::move(negative));
  EXPECT_THROW(Generators::GetNemotronChunkStartSample(as_extra_inputs(tensors)), std::runtime_error);
}

TEST(AudioSpeechValidationTests, NemotronProcessorEmitsChunkStartSample) {
  const char* model_override = std::getenv("NEMOTRON_STREAMING_MODEL_PATH");
  const auto model_path = model_override ? fs::path{model_override} : fs::path{MODEL_PATH "nemotron-speech-streaming"};
  if (!fs::exists(model_path))
    GTEST_SKIP() << "Streaming ASR model not found at " << model_path.string();

  auto config = std::make_unique<Generators::Config>(model_path, "");
  config->model.timestamp_level = Generators::Config::TimestampLevel::All;
  const auto chunk_samples = static_cast<size_t>(config->model.chunk_samples);
  ASSERT_GT(chunk_samples, 1U);
  // The downloaded CUDA export uses standard ONNX operators; run this metadata test on CPU too.
  config->model.encoder.session_options->provider_options.clear();
  config->model.encoder.session_options->providers.clear();
  config->model.decoder.session_options.provider_options.clear();
  config->model.decoder.session_options.providers.clear();
  config->model.joiner.session_options->provider_options.clear();
  config->model.joiner.session_options->providers.clear();
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), std::move(config));
  Generators::NemotronStreamingProcessor processor{*model};
  processor.SetOption("use_vad", "false");

  std::vector<float> silence(chunk_samples, 0.0f);
  for (int64_t chunk = 0; chunk < 2; ++chunk) {
    auto inputs = processor.Process(silence.data(), silence.size());
    ASSERT_NE(inputs, nullptr);
    ASSERT_EQ(inputs->size(), 2U);
    EXPECT_NE(inputs->find(std::string(Generators::Config::Defaults::AudioFeaturesName)), inputs->end());
    const auto& origin = inputs->at("chunk_start_sample");
    EXPECT_EQ(origin->GetType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    EXPECT_EQ(origin->GetElementCount(), 1U);
    EXPECT_EQ(*origin->GetData<int64_t>(), chunk * static_cast<int64_t>(chunk_samples));
  }

  processor.Process(silence.data(), chunk_samples / 2);
  auto tail = processor.Flush();
  ASSERT_NE(tail, nullptr);
  ASSERT_EQ(tail->size(), 2U);
  const auto& features = tail->at(std::string(Generators::Config::Defaults::AudioFeaturesName));
  const auto& origin = tail->at("chunk_start_sample");
  EXPECT_EQ(*origin->GetData<int64_t>(), 2 * static_cast<int64_t>(chunk_samples));

  auto params = std::make_shared<Generators::GeneratorParams>(*model);
  auto& nemotron_model = dynamic_cast<Generators::NemotronSpeechModel&>(*model);
  Generators::NemotronSpeechState state{nemotron_model, *params};
  const std::string audio_name{Generators::Config::Defaults::AudioFeaturesName};
  const auto missing_origin = GetExceptionMessage([&] {
    state.SetExtraInputs({{audio_name, features}});
  });
  EXPECT_NE(missing_origin.find("chunk_start_sample"), std::string::npos);
  auto invalid_value = OrtValue::CreateTensor<int64_t>(model->allocator_cpu_, std::array<int64_t, 1>{1});
  *invalid_value->GetTensorMutableData<int64_t>() = -1;
  auto invalid_origin = std::make_shared<Generators::Tensor>(std::move(invalid_value));
  EXPECT_THROW(state.SetExtraInputs({{audio_name, features}, {"chunk_start_sample", invalid_origin}}),
               std::runtime_error);
  EXPECT_NO_THROW(state.SetExtraInputs({{audio_name, features}, {"chunk_start_sample", origin}}));

  auto renamed = *tail;
  renamed.erase("chunk_start_sample");
  renamed.emplace("renamed_chunk_start_sample", origin);
  Generators::Generator mismatched_generator{*model, *params};
  const auto mismatch = GetExceptionMessage([&] { mismatched_generator.SetInputs(renamed); });
  EXPECT_NE(mismatch.find("chunk_start_sample"), std::string::npos);

  Generators::Generator generator{*model, *params};
  generator.SetInputs(*tail);
  ASSERT_FALSE(generator.IsDone());
  generator.GenerateNextToken();
  for (const auto& token : generator.GetNextTokensWithMetadata()) {
    EXPECT_EQ(token.has_token_acoustic_frame_interval, 1);
    EXPECT_GE(token.token_acoustic_frame_interval.start,
              2 * static_cast<int64_t>(chunk_samples) /
                  (model->config_->model.hop_length * model->config_->model.subsampling_factor));
  }

  Generators::NemotronStreamingProcessor vad_processor{*model};
  vad_processor.SetOption("silence_duration_ms", "1");
  vad_processor.SetOption("prefix_padding_ms", "1");
  EXPECT_EQ(vad_processor.Process(silence.data(), silence.size()), nullptr);
  vad_processor.SetOption("use_vad", "false");
  auto after_silence = vad_processor.Process(silence.data(), silence.size());
  ASSERT_NE(after_silence, nullptr);
  EXPECT_EQ(*after_silence->at("chunk_start_sample")->GetData<int64_t>(),
            static_cast<int64_t>(chunk_samples));
}

TEST(AudioSpeechValidationTests, NemotronTimestampsRejectMissingFrameDurationParameters) {
  Generators::Config config;
  config.model.type = "nemotron_speech";
  config.model.timestamp_level = Generators::Config::TimestampLevel::Word;
  config.model.segment_gap_threshold_seconds = 1.0;

  Generators::NemotronConfig nemotron_config;
  EXPECT_THROW(nemotron_config.PopulateFromConfig(config), std::runtime_error);
}

TEST(AudioSpeechValidationTests, NemotronTimestampsRejectOversizedSegmentGap) {
  Generators::Config config;
  config.model.type = "nemotron_speech";
  config.model.timestamp_level = Generators::Config::TimestampLevel::Segment;
  config.model.sample_rate = 16000;
  config.model.hop_length = 160;
  config.model.subsampling_factor = 8;
  config.model.segment_gap_threshold_seconds = 1e20;

  Generators::NemotronConfig nemotron_config;
  EXPECT_THROW(nemotron_config.PopulateFromConfig(config), std::runtime_error);
}

TEST(AudioSpeechValidationTests, ZeroAndSubFrameGapsSplitEachWord) {
  for (double threshold : {0.0, 0.01}) {
    EXPECT_EQ(Generators::GetSegmentGapThresholdFrames(threshold, 100, 10, 1), 0);
  }
}

class MetadataCoreStateTests : public testing::Test {
 protected:
  struct TestTransducerState : Generators::TransducerState {
    using TransducerState::last_token_intervals_;
    using TransducerState::TransducerState;
    void SetTimestampsEnabled(bool enabled) { timestamps_enabled_ = enabled; }
    void SetExtraInputs(const std::vector<Generators::ExtraInput>& inputs) override {
      received_inputs = inputs;
    }
    std::vector<Generators::ExtraInput> received_inputs;
    Generators::DeviceSpan<float> Run(int, Generators::DeviceSpan<int32_t>&, Generators::DeviceSpan<int32_t>) override {
      throw std::runtime_error("Synthetic metadata test does not run inference");
    }
    void StepToken() override {}
    void SetStep(const std::vector<int32_t>& tokens, int64_t token_frame_position) {
      last_tokens_ = tokens;
      last_token_intervals_.clear();
      for (size_t index = 0; index < tokens.size(); ++index)
        last_token_intervals_.push_back({token_frame_position, token_frame_position + 1});
    }
  };

  void SetUp() override {
    Generators::Config config{fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"}, ""};
    config.model.type = "nemotron_speech";
    config.model.timestamp_level = Generators::Config::TimestampLevel::All;
    config.model.sample_rate = 100;
    config.model.hop_length = 10;
    config.model.subsampling_factor = 1;
    config.model.segment_separators = {"."};
    tokenizer = std::make_shared<Generators::Tokenizer>(config);
    tokens = tokenizer->Encode("Hello world.");
    ASSERT_GT(tokens.size(), 1U);
    model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
    model->config_->model.timestamp_level = Generators::Config::TimestampLevel::All;
    params = std::make_shared<Generators::GeneratorParams>(*model);
    generator = std::make_unique<Generators::Generator>(*model, *params);
    auto transducer = std::make_unique<TestTransducerState>(*params, *model);
    source = transducer.get();
    source->SetTimestampsEnabled(true);
    generator->state_ = std::move(transducer);
    source->SetStep(tokens, 4);
  }

  std::shared_ptr<Generators::Tokenizer> tokenizer;
  std::vector<int32_t> tokens;
  std::shared_ptr<Generators::Model> model;
  std::shared_ptr<Generators::GeneratorParams> params;
  std::unique_ptr<Generators::Generator> generator;
  TestTransducerState* source{};
  const OgaTokenMetadataAcousticFrameInterval interval{4, 5};
};

TEST_F(MetadataCoreStateTests, ChunkStartSampleSurvivesGeneratorInputHandoff) {
  auto features = std::make_shared<Generators::Tensor>(
      OrtValue::CreateTensor<float>(model->allocator_cpu_, std::array<int64_t, 1>{1}));
  auto origin_value = OrtValue::CreateTensor<int64_t>(model->allocator_cpu_, std::array<int64_t, 1>{1});
  *origin_value->GetTensorMutableData<int64_t>() = 25600;
  auto origin = std::make_shared<Generators::Tensor>(std::move(origin_value));
  Generators::NamedTensors inputs;
  inputs.emplace(std::string(Generators::Config::Defaults::AudioFeaturesName), features);
  inputs.emplace("chunk_start_sample", origin);
  model->config_->model.type = "nemotron_speech";

  generator->SetInputs(inputs);
  ASSERT_EQ(source->received_inputs.size(), 2U);
  const auto origin_input = std::find_if(source->received_inputs.begin(), source->received_inputs.end(),
                                        [](const Generators::ExtraInput& input) {
                                          return input.name == "chunk_start_sample";
                                        });
  ASSERT_NE(origin_input, source->received_inputs.end());
  EXPECT_EQ(origin_input->tensor, origin);
  EXPECT_EQ(*origin_input->tensor->GetData<int64_t>(), 25600);
}

TEST_F(MetadataCoreStateTests, DecodeProcessesMetadataOncePerStep) {
  auto stream = tokenizer->CreateStream();
  auto plain = tokenizer->CreateStream();
  std::string transcript;
  std::string words;
  std::string segments;

  for (size_t index = 0; index < tokens.size(); ++index) {
    const auto& expected = stream->DecodeWithMetadata({tokens[index], 1, interval});
    EXPECT_STREQ(expected.text, plain->Decode(tokens[index]).c_str());
    transcript += expected.text;
    ASSERT_NE(expected.timestampMetadata, nullptr);
    for (size_t word_index = 0; word_index < expected.timestampMetadata->word_count; ++word_index) {
      const auto& word = expected.timestampMetadata->words[word_index];
      words += word.text;
      EXPECT_EQ(word.start_frame, 4);
      EXPECT_EQ(word.stop_frame, 5);
      EXPECT_DOUBLE_EQ(word.start_time, 0.4);
      EXPECT_DOUBLE_EQ(word.stop_time, 0.5);
    }
    for (size_t segment_index = 0; segment_index < expected.timestampMetadata->segment_count; ++segment_index)
      segments += expected.timestampMetadata->segments[segment_index].text;
  }

  const auto& final = stream->FinalizeMetadata();
  for (size_t index = 0; index < final.timestampMetadata->word_count; ++index) {
    const auto& word = final.timestampMetadata->words[index];
    words += word.text;
    EXPECT_EQ(word.start_frame, 4);
    EXPECT_EQ(word.stop_frame, 5);
  }
  for (size_t index = 0; index < final.timestampMetadata->segment_count; ++index)
    segments += final.timestampMetadata->segments[index].text;
  EXPECT_EQ(transcript, "Hello world.");
  EXPECT_EQ(words, transcript);
  EXPECT_EQ(segments, transcript);
  const auto& repeated = stream->FinalizeMetadata();
  EXPECT_EQ(repeated.timestampMetadata->word_count, 0U);
  EXPECT_EQ(repeated.timestampMetadata->segment_count, 0U);
}

TEST_F(MetadataCoreStateTests, ZeroAndSubFrameGapsSplitWordsEvenOnSharedFrames) {
  for (const double threshold : {0.0, 0.01}) {
    Generators::Config config{fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"}, ""};
    config.model.type = "nemotron_speech";
    config.model.timestamp_level = Generators::Config::TimestampLevel::All;
    config.model.sample_rate = 100;
    config.model.hop_length = 10;
    config.model.subsampling_factor = 1;
    config.model.segment_separators.clear();
    config.model.segment_gap_threshold_seconds = threshold;
    auto stream = std::make_shared<Generators::Tokenizer>(config)->CreateStream();
    std::vector<std::string> words;
    std::vector<std::string> segments;
    const auto collect = [&](const OgaTokenMetadataOutput& result) {
      ASSERT_NE(result.timestampMetadata, nullptr);
      for (size_t i = 0; i < result.timestampMetadata->word_count; ++i)
        words.emplace_back(result.timestampMetadata->words[i].text);
      for (size_t i = 0; i < result.timestampMetadata->segment_count; ++i) {
        const auto& segment = result.timestampMetadata->segments[i];
        segments.emplace_back(segment.text);
        EXPECT_EQ(segment.start_frame, 4);
        EXPECT_EQ(segment.stop_frame, 5);
      }
    };
    for (const auto token : tokens)
      collect(stream->DecodeWithMetadata({token, 1, interval}));
    collect(stream->FinalizeMetadata());
    ASSERT_GE(words.size(), 2U);
    EXPECT_EQ(segments, words);
  }
}

TEST_F(MetadataCoreStateTests, FrameGapCompletesPriorSegmentBeforeNextWord) {
  Generators::Config config{fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"}, ""};
  config.model.type = "nemotron_speech";
  config.model.timestamp_level = Generators::Config::TimestampLevel::All;
  config.model.sample_rate = 100;
  config.model.hop_length = 10;
  config.model.subsampling_factor = 1;
  config.model.segment_separators.clear();
  config.model.segment_gap_threshold_seconds = 0.3;  // Three frames at 100 Hz / 10 samples per frame.
  auto stream = std::make_shared<Generators::Tokenizer>(config)->CreateStream();
  std::vector<std::string> segments;
  std::vector<std::pair<int64_t, int64_t>> bounds;
  size_t completed_words = 0;
  for (size_t index = 0; index < tokens.size(); ++index) {
    const int64_t frame = index == 0 ? 0 : 8;
    const auto& result = stream->DecodeWithMetadata({tokens[index], 1, {frame, frame + 1}});
    completed_words += result.timestampMetadata->word_count;
    for (size_t i = 0; i < result.timestampMetadata->segment_count; ++i) {
      const auto& segment = result.timestampMetadata->segments[i];
      segments.emplace_back(segment.text);
      bounds.emplace_back(segment.start_frame, segment.stop_frame);
    }
  }
  const auto& trailing = stream->FinalizeMetadata();
  completed_words += trailing.timestampMetadata->word_count;
  EXPECT_EQ(completed_words, 2U);
  for (size_t i = 0; i < trailing.timestampMetadata->segment_count; ++i) {
    const auto& segment = trailing.timestampMetadata->segments[i];
    segments.emplace_back(segment.text);
    bounds.emplace_back(segment.start_frame, segment.stop_frame);
  }
  ASSERT_EQ(segments.size(), 2U);
  EXPECT_EQ(bounds[0], (std::pair<int64_t, int64_t>{0, 1}));
  EXPECT_EQ(bounds[1], (std::pair<int64_t, int64_t>{8, 9}));
  EXPECT_EQ(segments[0] + segments[1], "Hello world.");
}

TEST_F(MetadataCoreStateTests, RepeatedTokenIdsKeepPositionSpecificIntervals) {
  source->SetStep({tokens[0], tokens[0]}, 4);
  source->last_token_intervals_[1] = {9, 12};
  const auto emitted = generator->GetNextTokensWithMetadata();
  ASSERT_EQ(emitted.size(), 2U);
  EXPECT_EQ(emitted[0].token_id, emitted[1].token_id);
  EXPECT_EQ(emitted[0].token_acoustic_frame_interval.start, 4);
  EXPECT_EQ(emitted[0].token_acoustic_frame_interval.stop, 5);
  EXPECT_EQ(emitted[1].token_acoustic_frame_interval.start, 9);
  EXPECT_EQ(emitted[1].token_acoustic_frame_interval.stop, 12);
}

TEST_F(MetadataCoreStateTests, DisabledConsumersDoNotBlockOrEnableProducers) {
  Generators::Config text_config{fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"}, ""};
  auto disabled_tokenizer = std::make_shared<Generators::Tokenizer>(text_config);
  auto stream = disabled_tokenizer->CreateStream();
  auto plain = tokenizer->CreateStream();
  auto enabled = tokenizer->CreateStream();
  for (size_t index = 0; index < tokens.size(); ++index) {
    const auto& result = stream->DecodeWithMetadata({tokens[index], 0, {}});
    EXPECT_EQ(result.text, plain->Decode(tokens[index]));
    EXPECT_EQ(result.timestampMetadata, nullptr);
    EXPECT_NE(enabled->DecodeWithMetadata({tokens[index], 1, interval}).timestampMetadata, nullptr);
  }
  EXPECT_EQ(stream->FinalizeMetadata().timestampMetadata, nullptr);
  EXPECT_NE(enabled->FinalizeMetadata().timestampMetadata, nullptr);
}

TEST_F(MetadataCoreStateTests, ResetRebuildsMetadataFromTokenizerConfig) {
  auto stream = tokenizer->CreateStream();
  auto other = tokenizer->CreateStream();
  EXPECT_THROW(stream->DecodeWithMetadata({tokens[0], 0, {}}), std::runtime_error);
  const auto& first = stream->DecodeWithMetadata({tokens[0], 1, interval});
  EXPECT_NE(first.timestampMetadata, nullptr);
  EXPECT_THROW(stream->Decode(tokens[0]), std::runtime_error);
  stream->Reset();
  EXPECT_THROW(stream->DecodeWithMetadata({tokens[0], 0, {}}), std::runtime_error);
  const auto& restored = stream->DecodeWithMetadata({tokens[0], 1, interval});
  EXPECT_NE(restored.timestampMetadata, nullptr);
  const auto& final = stream->FinalizeMetadata();
  ASSERT_EQ(final.timestampMetadata->word_count, 1U);
  EXPECT_DOUBLE_EQ(final.timestampMetadata->words[0].start_time, 0.4);
  EXPECT_NE(other->FinalizeMetadata().timestampMetadata, nullptr);
  stream->Reset();
  EXPECT_NO_THROW(stream->Decode(tokens[0]));
  EXPECT_THROW(stream->FinalizeMetadata(), std::runtime_error);
  stream->Reset();
  EXPECT_NE(stream->FinalizeMetadata().timestampMetadata, nullptr);
}

TEST_F(MetadataCoreStateTests, MissingModelTimingRejectsTimestamps) {
  Generators::Config config{fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"}, ""};
  config.model.type = "nemotron_speech";
  config.model.timestamp_level = Generators::Config::TimestampLevel::All;
  EXPECT_THROW({ Generators::Tokenizer invalid_tokenizer{config}; }, std::runtime_error);
}

TEST_F(MetadataCoreStateTests, DisabledTokenizerProducesTextWithoutTimestamps) {
  Generators::Config text_config{fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"}, ""};
  auto text_tokenizer = std::make_shared<Generators::Tokenizer>(text_config);
  auto text_stream = text_tokenizer->CreateStream();
  EXPECT_EQ(text_stream->DecodeWithMetadata({tokens[0], 1, interval}).timestampMetadata, nullptr);
  EXPECT_NO_THROW(text_stream->DecodeWithMetadata({tokens[1], 1, interval}));
}

TEST_F(MetadataCoreStateTests, MetadataDefaultsAndReset) {
  auto stream = tokenizer->CreateStream();
  EXPECT_NO_THROW(stream->DecodeWithMetadata({tokens[0], 1, interval}));
  EXPECT_NO_THROW(stream->FinalizeMetadata());
  stream->Reset();
  EXPECT_NO_THROW(stream->DecodeWithMetadata({tokens[0], 1, interval}));
  stream->Reset();
  EXPECT_NO_THROW(stream->Decode(tokens[0]));
  EXPECT_THROW(stream->DecodeWithMetadata({tokens[1], 1, interval}), std::runtime_error);
  stream->Reset();
  EXPECT_NE(stream->FinalizeMetadata().timestampMetadata, nullptr);
  EXPECT_THROW(stream->Decode(tokens[0]), std::runtime_error);
  Generators::Config text_config{fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"}, ""};
  auto disabled_tokenizer = std::make_shared<Generators::Tokenizer>(text_config);
  auto disabled_stream = disabled_tokenizer->CreateStream();
  EXPECT_EQ(disabled_stream->FinalizeMetadata().timestampMetadata, nullptr);
}

TEST_F(MetadataCoreStateTests, GeneratorOmitsTimingWhenDisabled) {
  model->config_->model.timestamp_level = Generators::Config::TimestampLevel::Off;
  source->SetTimestampsEnabled(false);
  source->last_token_intervals_.pop_back();
  const auto records = generator->GetNextTokensWithMetadata();
  ASSERT_EQ(records.size(), tokens.size());
  auto plain = tokenizer->CreateStream();
  Generators::Config text_config{fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"}, ""};
  auto disabled_tokenizer = std::make_shared<Generators::Tokenizer>(text_config);
  auto disabled = disabled_tokenizer->CreateStream();
  auto enabled = tokenizer->CreateStream();
  for (size_t index = 0; index < records.size(); ++index) {
    const auto& token = records[index];
    EXPECT_EQ(token.token_id, tokens[index]);
    EXPECT_EQ(token.has_token_acoustic_frame_interval, 0);
    const auto& result = disabled->DecodeWithMetadata(token);
    EXPECT_EQ(result.text, plain->Decode(tokens[index]));
    EXPECT_EQ(result.timestampMetadata, nullptr);
    EXPECT_THROW(enabled->DecodeWithMetadata(token), std::runtime_error);
  }
  EXPECT_EQ(enabled->FinalizeMetadata().timestampMetadata->word_count, 0U);
  for (const auto level : {Generators::Config::TimestampLevel::Word,
                           Generators::Config::TimestampLevel::Segment,
                           Generators::Config::TimestampLevel::All}) {
    model->config_->model.timestamp_level = level;
    source->SetTimestampsEnabled(true);
    EXPECT_THROW(generator->GetNextTokensWithMetadata(), std::runtime_error);
  }
}

TEST_F(MetadataCoreStateTests, GeneratedMetadataPreservesMultiFrameIntervals) {
  auto stream = tokenizer->CreateStream();
  for (auto& timing : source->last_token_intervals_) timing.stop = 9;
  const auto emitted = generator->GetNextTokensWithMetadata();
  ASSERT_EQ(emitted.size(), tokens.size());
  std::string transcript;
  std::string words;
  for (size_t index = 0; index < tokens.size(); ++index) {
    const auto token = emitted[index];
    EXPECT_EQ(token.token_id, tokens[index]);
    ASSERT_EQ(token.has_token_acoustic_frame_interval, 1);
    EXPECT_EQ(token.token_acoustic_frame_interval.start, 4);
    EXPECT_EQ(token.token_acoustic_frame_interval.stop, 9);
    const auto& result = stream->DecodeWithMetadata(token);
    transcript += result.text;
    ASSERT_NE(result.timestampMetadata, nullptr);
    for (size_t word_index = 0; word_index < result.timestampMetadata->word_count; ++word_index) {
      words += result.timestampMetadata->words[word_index].text;
      EXPECT_EQ(result.timestampMetadata->words[word_index].start_frame, 4);
      EXPECT_EQ(result.timestampMetadata->words[word_index].stop_frame, 9);
    }
  }
  source->SetStep({}, 99);
  EXPECT_TRUE(generator->GetNextTokensWithMetadata().empty());
  const auto& trailing = stream->FinalizeMetadata();
  for (size_t index = 0; index < trailing.timestampMetadata->word_count; ++index) {
    const auto& word = trailing.timestampMetadata->words[index];
    words += word.text;
    EXPECT_EQ(word.start_frame, 4);
    EXPECT_EQ(word.stop_frame, 9);
    EXPECT_DOUBLE_EQ(word.start_time, 0.4);
    EXPECT_DOUBLE_EQ(word.stop_time, 0.9);
  }
  EXPECT_EQ(transcript, "Hello world.");
  EXPECT_EQ(words, transcript);
  EXPECT_EQ(stream->FinalizeMetadata().timestampMetadata->word_count, 0U);
  EXPECT_EQ(stream->FinalizeMetadata().timestampMetadata->segment_count, 0U);
}

TEST_F(MetadataCoreStateTests, RetainsGenerationTimingAfterGeneratorAdvances) {
  auto stream = tokenizer->CreateStream();
  const auto token = generator->GetNextTokensWithMetadata()[0];
  stream->DecodeWithMetadata(token);
  source->SetStep({}, 99);
  EXPECT_TRUE(generator->GetNextTokensWithMetadata().empty());
  const auto& result = stream->FinalizeMetadata();
  ASSERT_EQ(result.timestampMetadata->word_count, 1U);
  EXPECT_EQ(result.timestampMetadata->words[0].start_frame, 4);
  EXPECT_EQ(result.timestampMetadata->words[0].stop_frame, 5);
}

TEST_F(MetadataCoreStateTests, EmptyFinalizationAndInputValidation) {
  const OgaTokenMetadataAcousticFrameInterval negative_interval{-1, 0};
  const OgaTokenMetadataAcousticFrameInterval empty_interval{3, 3};
  const OgaTokenMetadataAcousticFrameInterval reversed_interval{4, 2};
  const OgaTokenMetadataAcousticFrameInterval valid_interval{0, 3};
  auto stream = tokenizer->CreateStream();
  const auto& empty = stream->FinalizeMetadata();
  ASSERT_NE(empty.timestampMetadata, nullptr);
  EXPECT_EQ(empty.timestampMetadata->word_count, 0U);
  EXPECT_THROW(stream->DecodeWithMetadata({tokens[0], 1, negative_interval}), std::runtime_error);
  EXPECT_THROW(stream->DecodeWithMetadata({tokens[0], 1, empty_interval}), std::runtime_error);
  EXPECT_THROW(stream->DecodeWithMetadata({tokens[0], 1, reversed_interval}), std::runtime_error);
  EXPECT_THROW(stream->DecodeWithMetadata({tokens[0], 0, {}}), std::runtime_error);
  source->last_token_intervals_.pop_back();
  EXPECT_THROW(generator->GetNextTokensWithMetadata(), std::runtime_error);
  source->last_token_intervals_.clear();
  EXPECT_THROW(generator->GetNextTokensWithMetadata(), std::runtime_error);
  const OgaTokenMetadataInput token{tokens[0], 0, {}};
  EXPECT_THROW(stream->DecodeWithMetadata(token), std::runtime_error);
  EXPECT_NO_THROW(stream->DecodeWithMetadata({tokens[0], 1, valid_interval}));
  const auto& final = stream->FinalizeMetadata();
  ASSERT_EQ(final.timestampMetadata->word_count, 1U);
  EXPECT_EQ(final.timestampMetadata->words[0].start_frame, 0);
  EXPECT_EQ(final.timestampMetadata->words[0].stop_frame, 3);
}

TEST(AudioSpeechValidationTests, ParakeetEncoderChannelDimensionValidation) {
  EXPECT_NO_THROW(Generators::ValidateParakeetEncoderOutputShape({1, 512, 64}, 512));

  const std::string dim_error = GetExceptionMessage([] {
    Generators::ValidateParakeetEncoderOutputShape({1, 511, 64}, 512);
  });
  EXPECT_NE(dim_error.find("hidden_dim"), std::string::npos);
}

TEST(AudioSpeechValidationTests, ParakeetDecoderDimensionValidation) {
  EXPECT_NO_THROW(Generators::ValidateParakeetDecoderOutputShape({1, 1024, 1}, 1024));

  const std::string shape_error = GetExceptionMessage([] {
    Generators::ValidateParakeetDecoderOutputShape({1, 1024, 2}, 1024);
  });
  EXPECT_NE(shape_error.find("must have shape"), std::string::npos);
}

TEST(AudioSpeechValidationTests, Rank1TensorsRejected) {
  EXPECT_THROW(Generators::ValidateWhisperAudioFeaturesShape({3000}, 3000), std::runtime_error);
  EXPECT_THROW(Generators::GetValidatedNemotronMelFrameCount({3000}, 80), std::runtime_error);
  EXPECT_THROW(Generators::ValidateParakeetEncoderOutputShape({512}, 512), std::runtime_error);
  EXPECT_THROW(Generators::ValidateParakeetDecoderOutputShape({1024}, 1024), std::runtime_error);
}

TEST(AudioSpeechValidationTests, DimensionMismatchesCaught) {
  EXPECT_THROW(Generators::ValidateWhisperAudioFeaturesShape({1, 80, 2999}, 3000), std::runtime_error);
  EXPECT_THROW(Generators::GetValidatedNemotronMelFrameCount({2, 304, 128}, 128), std::runtime_error);
  EXPECT_THROW(Generators::ValidateParakeetEncoderOutputShape({1, 256, 64}, 512), std::runtime_error);
  EXPECT_THROW(Generators::ValidateParakeetDecoderOutputShape({2, 1024, 1}, 1024), std::runtime_error);
}
