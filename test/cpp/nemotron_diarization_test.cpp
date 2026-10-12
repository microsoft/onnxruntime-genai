// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <gtest/gtest.h>
#include <cstdlib>
#include "models/nemotron_diarization.h"
#include "models/nemotron_speech.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/preprocessing/nemotron_streaming_processor.h"
#include "tokenizer_speech_metadata.h"

TEST(NemotronDiarization, ConfigurationRequiresAlignedGeometry) {
  Generators::Config config;
  auto& m = config.model;
  m.type = "nemotron_speech";
  m.diarization.filename = "diarization.onnx";
  m.sample_rate = 16000;
  m.hop_length = 160;
  m.subsampling_factor = 8;
  m.num_mels = 128;
  m.fft_size = 512;
  m.win_length = 400;
  m.preemph = 0.97f;
  m.log_eps = 5.96046448e-8f;
  m.chunk_samples = 8960;
  EXPECT_NO_THROW(Generators::ValidateTimestampConfiguration(m));
  m.diarization.right_context = -1;
  EXPECT_THROW(Generators::ValidateTimestampConfiguration(m), std::runtime_error);
  m.diarization.right_context = 4;
  m.chunk_samples = 8961;
  EXPECT_THROW(Generators::ValidateTimestampConfiguration(m), std::runtime_error);
  m.chunk_samples = 8960;
  m.hop_length = 80;
  EXPECT_THROW(Generators::ValidateTimestampConfiguration(m), std::runtime_error);
}

TEST(NemotronDiarization, MetadataConfigurationsAndTurnBoundaries) {
  for (const bool diarization : {false, true}) {
    for (const auto level : {Generators::Config::TimestampLevel::Off,
                             Generators::Config::TimestampLevel::Word,
                             Generators::Config::TimestampLevel::Segment}) {
      Generators::Config config{fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"}, ""};
      config.model.type = "nemotron_speech";
      config.model.timestamp_level = level;
      config.model.sample_rate = 100;
      config.model.hop_length = 10;
      config.model.subsampling_factor = 1;
      config.model.segment_gap_threshold_seconds = 0;
      if (diarization) config.model.diarization.filename = "unused.onnx";
      auto tokenizer = std::make_shared<Generators::Tokenizer>(config);
      auto stream = tokenizer->CreateStream();
      const auto tokens = tokenizer->Encode("Hello world.");
      std::string transcript, records;
      size_t count = 0;
      const auto collect = [&](const OgaTokenMetadataOutputV2& result) {
        transcript += result.text;
        if (!diarization && level == Generators::Config::TimestampLevel::Off) {
          EXPECT_EQ(result.timestampMetadata, nullptr);
          return;
        }
        ASSERT_NE(result.timestampMetadata, nullptr);
        const auto& output = *result.timestampMetadata;
        const bool word = level == Generators::Config::TimestampLevel::Word;
        const auto* data = word ? output.words : output.segments;
        const size_t size = word ? output.word_count : output.segment_count;
        for (size_t i = 0; i < size; ++i) {
          records += data[i].text;
          ++count;
          EXPECT_EQ(data[i].has_timestamp, level == Generators::Config::TimestampLevel::Off ? 0 : 1);
          if (diarization)
            EXPECT_STREQ(data[i].speaker_label, "speaker_2");
          else
            EXPECT_EQ(data[i].speaker_label, nullptr);
        }
      };
      for (size_t i = 0; i < tokens.size(); ++i) {
        OgaTokenMetadataInputV2 token{tokens[i], 1, {static_cast<int64_t>(i), static_cast<int64_t>(i + 1)}, 0, {}};
        if (diarization) {
          token.has_speaker_evidence = 1;
          token.speaker_evidence.start_frame_count = static_cast<int64_t>(i);
          token.speaker_evidence.stop_frame_count = static_cast<int64_t>(i + 1);
          token.speaker_evidence.start_probability_sums[2] = static_cast<double>(i);
          token.speaker_evidence.stop_probability_sums[2] = static_cast<double>(i + 1);
        }
        collect(stream->DecodeWithMetadata(token));
      }
      collect(stream->FinalizeMetadata());
      EXPECT_EQ(transcript, "Hello world.");
      if (diarization || level != Generators::Config::TimestampLevel::Off) EXPECT_EQ(records, transcript);
      if (diarization && level != Generators::Config::TimestampLevel::Word) EXPECT_EQ(count, 1U);
      const auto& repeated = stream->FinalizeMetadata();
      if (repeated.timestampMetadata) {
        EXPECT_EQ(repeated.timestampMetadata->word_count, 0U);
        EXPECT_EQ(repeated.timestampMetadata->segment_count, 0U);
      }
      stream->Reset();
      if (diarization) EXPECT_THROW(stream->DecodeWithMetadata({tokens[0], 1, {0, 1}}), std::runtime_error);
    }
  }
}

TEST(NemotronDiarization, RejectsInvalidProbabilityWindows) {
  const double zeros[8]{};
  const double invalid[8]{2};
  EXPECT_THROW(ort_extensions::AggregateWordSpeaker(zeros, invalid, 0, 1), std::runtime_error);
  EXPECT_THROW(ort_extensions::AggregateWordSpeaker(zeros, zeros, 0, 0), std::runtime_error);
  EXPECT_EQ(ort_extensions::AggregateWordSpeaker(zeros, zeros, 0, 1), 0);
}

TEST(NemotronDiarization, SpeakerChangesCompleteTurnsAndInvalidInputConsumesNothing) {
  Generators::Config config{fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"}, ""};
  config.model.type = "nemotron_speech";
  config.model.diarization.filename = "unused.onnx";
  config.model.sample_rate = 100;
  config.model.hop_length = 10;
  config.model.subsampling_factor = 1;
  auto tokenizer = std::make_shared<Generators::Tokenizer>(config);
  auto stream = tokenizer->CreateStream();
  const auto tokens = tokenizer->Encode("Hello world.");
  const auto boundary = tokenizer->Encode("Hello").size();
  std::array<double, 8> sums{};
  std::vector<std::string> texts, speakers;
  const auto collect = [&](const OgaTokenMetadataOutputV2& result) {
    ASSERT_NE(result.timestampMetadata, nullptr);
    for (size_t i = 0; i < result.timestampMetadata->segment_count; ++i) {
      const auto& record = result.timestampMetadata->segments[i];
      texts.emplace_back(record.text);
      speakers.emplace_back(record.speaker_label);
      EXPECT_EQ(record.has_timestamp, 0);
    }
  };
  for (size_t i = 0; i < tokens.size(); ++i) {
    OgaTokenMetadataInputV2 token{tokens[i], 1, {static_cast<int64_t>(i), static_cast<int64_t>(i + 1)}, 1, {}};
    std::copy(sums.begin(), sums.end(), token.speaker_evidence.start_probability_sums);
    token.speaker_evidence.start_frame_count = static_cast<int64_t>(i);
    token.speaker_evidence.stop_frame_count = static_cast<int64_t>(i + 1);
    sums[i < boundary ? 0 : 1] += 1;
    std::copy(sums.begin(), sums.end(), token.speaker_evidence.stop_probability_sums);
    if (i == 0) {
      auto invalid = token;
      invalid.speaker_evidence.stop_frame_count = 0;
      EXPECT_THROW(stream->DecodeWithMetadata(invalid), std::runtime_error);
    }
    collect(stream->DecodeWithMetadata(token));
  }
  collect(stream->FinalizeMetadata());
  ASSERT_EQ(texts.size(), 2U);
  EXPECT_EQ(texts[0] + texts[1], "Hello world.");
  EXPECT_EQ(speakers, (std::vector<std::string>{"speaker_0", "speaker_1"}));
}

TEST(NemotronDiarization, StreamingGraphStateAndLookahead) {
  const char* asr = std::getenv("NEMOTRON_STREAMING_MODEL_PATH");
  const char* diarization = std::getenv("NEMOTRON_DIARIZATION_MODEL_PATH");
  if (!asr || !diarization) GTEST_SKIP() << "Set both Nemotron model paths for ONNX integration tests";
  auto config = std::make_unique<Generators::Config>(fs::path{asr}, "");
  config->model.encoder.session_options->provider_options.clear();
  config->model.encoder.session_options->providers.clear();
  config->model.decoder.session_options.provider_options.clear();
  config->model.decoder.session_options.providers.clear();
  config->model.joiner.session_options->provider_options.clear();
  config->model.joiner.session_options->providers.clear();
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), std::move(config));
  auto& speech = dynamic_cast<Generators::NemotronSpeechModel&>(*model);
  model->config_->model.diarization.filename = "diarization/model.onnx";
  speech.nemotron_config_.diarization_enabled = true;
  speech.diarization_session_options_ = OrtSessionOptions::Create();
  speech.diarization_device_ = model->CreateSessionOptionsFromConfig(
      model->config_->model.decoder.session_options, *speech.diarization_session_options_, true, true);
  speech.session_diarization_ = OrtSession::Create(Generators::GetOrtEnv(),
                                                   (fs::path{diarization} / "model.onnx").c_str(), speech.diarization_session_options_.get());
  auto params = Generators::CreateGeneratorParams(*model);
  Generators::NemotronDiarizationState state{speech, *params};
  auto features = OrtValue::CreateTensor<float>(model->allocator_cpu_, std::array<int64_t, 3>{1, 128, 88});
  std::fill_n(features->GetTensorMutableData<float>(), 128 * 88, -10.0f);
  for (int step = 0; step < 80; ++step) {
    state.Process(*features, 4);
    auto first = state.Evidence(0);
    auto last = state.Evidence(6);
    EXPECT_EQ(last.stop_frame_count, (step + 1) * 56);
    EXPECT_EQ(first.start_frame_count, step == 0 ? 0 : step * 56 - 8);
    EXPECT_NO_THROW(ort_extensions::AggregateWordSpeaker(first.start_probability_sums, last.stop_probability_sums,
                                                         first.start_frame_count, last.stop_frame_count));
  }
  EXPECT_TRUE(*state.GetInput("past_is_compressed")->GetTensorData<bool>());
  EXPECT_EQ(*state.GetInput("past_num_cache_frames")->GetTensorData<int64_t>(), 264);
  state.Reset();
  state.Process(*features, 4);
  EXPECT_EQ(state.Evidence(0).start_frame_count, 0);
  Generators::NemotronStreamingProcessor processor{*model};
  processor.SetOption("use_vad", "false");
  const size_t chunk = static_cast<size_t>(speech.nemotron_config_.chunk_samples);
  std::vector<float> audio(chunk, 0.0f);
  EXPECT_EQ(processor.Process(audio.data(), audio.size()), nullptr);
  auto input = processor.Process(audio.data(), audio.size());
  ASSERT_NE(input, nullptr);
  EXPECT_EQ(input->at("diarization_features")->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape(),
            (std::vector<int64_t>{1, 128, 88}));
  EXPECT_EQ(*input->at("chunk_start_sample")->GetData<int64_t>(), 0);
  auto tail = processor.Flush();
  ASSERT_NE(tail, nullptr);
  EXPECT_EQ(*tail->at("chunk_start_sample")->GetData<int64_t>(), static_cast<int64_t>(chunk));
  EXPECT_EQ(*tail->at("diarization_lookahead")->GetData<int64_t>(), 0);
  EXPECT_EQ(processor.Flush(), nullptr);
  if (!model->config_->model.vad.filename.empty()) {
    Generators::NemotronStreamingProcessor gated{*model};
    gated.SetOption("silence_duration_ms", "1");
    gated.SetOption("prefix_padding_ms", "1");
    std::vector<float> two_chunks(chunk * 2, 0.0f);
    EXPECT_EQ(gated.Process(two_chunks.data(), two_chunks.size()), nullptr);
    gated.SetOption("use_vad", "false");
    auto resumed = gated.Process(audio.data(), audio.size());
    ASSERT_NE(resumed, nullptr);
    EXPECT_EQ(*resumed->at("chunk_start_sample")->GetData<int64_t>(), static_cast<int64_t>(chunk));
  }
}
