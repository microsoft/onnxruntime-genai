// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <gtest/gtest.h>
#include <cstdlib>
#include <limits>
#include "models/nemotron_diarization.h"
#include "models/nemotron_speech.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/preprocessing/nemotron_streaming_processor.h"
#include "models/preprocessing/token_post_processor.h"
#include "models/preprocessing/token_diarization.h"

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
  m.hop_length = 161;
  EXPECT_THROW(Generators::ValidateTimestampConfiguration(m), std::runtime_error);
}

TEST(NemotronDiarization, ConfigurationUsesDeclaredGeometry) {
  Generators::Config config;
  auto& m = config.model;
  m.type = "nemotron_speech";
  m.sample_rate = 8000;
  m.hop_length = 80;
  m.subsampling_factor = 4;
  m.chunk_samples = 4480;
  m.diarization.filename = "compatible-export.onnx";
  auto& diar = m.diarization;
  diar.sample_rate = 8000;
  diar.hop_length = 40;
  diar.subsampling_factor = 4;
  diar.num_mels = 64;
  diar.fft_size = 256;
  diar.win_length = 200;
  diar.preemph = 0.95f;
  diar.log_eps = 1e-7f;
  diar.right_context = 300;  // Look-ahead is not bounded by FIFO capacity.
  EXPECT_NO_THROW(Generators::ValidateTimestampConfiguration(m));
  EXPECT_EQ(diar.SamplesPerEncoderFrame(), 160);
  diar.sample_rate = 16000;
  EXPECT_THROW(Generators::ValidateTimestampConfiguration(m), std::runtime_error);
  diar.sample_rate = 8000;
  diar.hop_length = 30;
  EXPECT_THROW(Generators::ValidateTimestampConfiguration(m), std::runtime_error);
  diar.hop_length = 40;
  diar.log_eps = 0;
  EXPECT_THROW(Generators::ValidateTimestampConfiguration(m), std::runtime_error);
}

namespace {
struct DiarizationTestState : Generators::TransducerState {
  using TransducerState::TransducerState;
  std::vector<Generators::TokenSpeakerEvidence> evidence;
  mutable size_t speaker_evidence_reads{};
  void StepToken() override {}
  Generators::DeviceSpan<float> Run(int, Generators::DeviceSpan<int32_t>&,
                                    Generators::DeviceSpan<int32_t>) override { return {}; }
  std::span<const Generators::TokenSpeakerEvidence> GetStepSpeakerEvidence() const override {
    ++speaker_evidence_reads;
    return evidence;
  }
  void SetStep(int32_t token, int64_t frame, std::optional<Generators::TokenSpeakerEvidence> speaker) {
    last_tokens_ = {token};
    last_token_intervals_ = {{frame, frame + 1}};
    evidence.clear();
    if (speaker) evidence.push_back(*speaker);
  }
  void EmptyStep() {
    last_tokens_.clear();
    last_token_intervals_.clear();
    evidence.clear();
  }
  void AppendStep(int32_t token, int64_t frame, const Generators::TokenSpeakerEvidence& speaker) {
    last_tokens_.push_back(token);
    last_token_intervals_.push_back({frame, frame + 1});
    evidence.push_back(speaker);
  }
};

struct DiarizationTestPipeline {
  std::shared_ptr<Generators::Model> model;
  std::shared_ptr<Generators::GeneratorParams> params;
  std::unique_ptr<Generators::Generator> generator;
  DiarizationTestState* state{};
  std::unique_ptr<Generators::TokenPostProcessor> processor;

  DiarizationTestPipeline(bool diarization, Generators::Config::TimestampLevel level) {
    model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
    auto& config = model->config_->model;
    config.timestamp_level = level;
    config.sample_rate = 16000;
    config.hop_length = 160;
    config.subsampling_factor = 8;
    config.chunk_samples = 8960;
    config.segment_gap_threshold_seconds = 0;
    if (diarization) config.diarization.filename = "unused.onnx";
    params = std::make_shared<Generators::GeneratorParams>(*model);
    generator = std::make_unique<Generators::Generator>(*model, *params);
    auto source = std::make_unique<DiarizationTestState>(*params, *model);
    state = source.get();
    generator->state_ = std::move(source);
    config.type = "nemotron_speech";
    processor = std::make_unique<Generators::TokenPostProcessor>(*generator);
  }
};
}  // namespace

TEST(NemotronDiarization, MetadataConfigurationsAndTurnBoundaries) {
  for (const bool diarization : {false, true}) {
    for (const auto level : {Generators::Config::TimestampLevel::Off,
                             Generators::Config::TimestampLevel::Word,
                             Generators::Config::TimestampLevel::Segment}) {
      DiarizationTestPipeline pipeline{diarization, level};
      auto tokenizer = pipeline.model->CreateTokenizer();
      const auto tokens = tokenizer->Encode("Hello world.");
      std::string transcript, records;
      size_t count = 0;
      const auto collect = [&](const Generators::PostProcessorResult& result) {
        transcript += result.text;
        if (!diarization && level == Generators::Config::TimestampLevel::Off) {
          EXPECT_FALSE(result.timestamps);
          return;
        }
        ASSERT_TRUE(result.timestamps);
        const auto& output = *result.timestamps;
        const bool word = level == Generators::Config::TimestampLevel::Word;
        const auto& data = word ? output.words : output.segments;
        for (const auto& record : data) {
          records += record.text;
          ++count;
          EXPECT_EQ(record.start_time.has_value(), level != Generators::Config::TimestampLevel::Off);
          if (diarization)
            EXPECT_EQ(record.speaker_label, "speaker_2");
          else
            EXPECT_FALSE(record.speaker_label);
        }
      };
      for (size_t i = 0; i < tokens.size(); ++i) {
        std::optional<Generators::TokenSpeakerEvidence> evidence;
        if (diarization) {
          evidence.emplace();
          evidence->start_frame_count = static_cast<int64_t>(i);
          evidence->stop_frame_count = static_cast<int64_t>(i + 1);
          evidence->start_probability_sums[2] = static_cast<double>(i);
          evidence->stop_probability_sums[2] = static_cast<double>(i + 1);
        }
        pipeline.state->SetStep(tokens[i], static_cast<int64_t>(i), evidence);
        pipeline.processor->ProcessNextTokens();
        for (const auto& result : pipeline.processor->GetNextTokenResults()) collect(result);
      }
      pipeline.processor->Finalize();
      for (const auto& result : pipeline.processor->GetNextTokenResults()) collect(result);
      EXPECT_EQ(transcript, "Hello world.");
      if (diarization || level != Generators::Config::TimestampLevel::Off) EXPECT_EQ(records, transcript);
      if (diarization && level != Generators::Config::TimestampLevel::Word) EXPECT_EQ(count, 1U);
      EXPECT_EQ(pipeline.state->speaker_evidence_reads, diarization ? tokens.size() : 0U);
      pipeline.processor->Finalize();
      const auto& repeated = pipeline.processor->GetNextTokenResults()[0];
      if (repeated.timestamps) {
        EXPECT_TRUE(repeated.timestamps->words.empty());
        EXPECT_TRUE(repeated.timestamps->segments.empty());
      }
      pipeline.processor->Reset();
      pipeline.state->SetStep(tokens[0], 0, std::nullopt);
      if (diarization) EXPECT_THROW(pipeline.processor->ProcessNextTokens(), std::runtime_error);
    }
  }
}

TEST(NemotronDiarization, InternalProcessorAlignsBatchAndIncrementalSpeakerEvidence) {
  using namespace Generators;
  DiarizationTestPipeline pipeline{true, Config::TimestampLevel::Word};
  auto tokenizer = pipeline.model->CreateTokenizer();
  const auto tokens = tokenizer->Encode("Hello world.");
  const auto first_word_tokens = tokenizer->Encode("Hello").size();
  std::vector<AcousticFrameInterval> intervals;
  std::vector<TokenSpeakerEvidence> evidence;
  TokenSpeakerEvidence snapshot{};
  for (size_t index = 0; index < tokens.size(); ++index) {
    intervals.push_back({static_cast<int64_t>(index), static_cast<int64_t>(index + 1)});
    std::copy(std::begin(snapshot.stop_probability_sums), std::end(snapshot.stop_probability_sums),
              snapshot.start_probability_sums);
    snapshot.start_frame_count = snapshot.stop_frame_count;
    ++snapshot.stop_frame_count;
    ++snapshot.stop_probability_sums[index < first_word_tokens ? 0 : 1];
    evidence.push_back(snapshot);
  }
  for (const auto level : {Config::TimestampLevel::Off, Config::TimestampLevel::Word, Config::TimestampLevel::Segment}) {
    MetadataCoreConfig config;
    config.timestamps.level = level;
    config.timestamps.segment_separators = {"."};
    config.timestamps.segment_gap_threshold_frames = 0;
    if (level != Config::TimestampLevel::Off) config.seconds_per_frame = 0.1;
    config.diarization = true;
    const auto process = [&](size_t batch_size) {
      TokenPostProcessorInternal processor{tokenizer, config};
      const auto token_span = std::span<const int32_t>{tokens};
      const auto interval_span = std::span<const AcousticFrameInterval>{intervals};
      const auto evidence_span = std::span<const TokenSpeakerEvidence>{evidence};
      EXPECT_THROW(processor.Process(TokenOutputs{token_span, interval_span}), std::runtime_error);
      EXPECT_THROW(processor.Process(TokenOutputs{token_span, interval_span, evidence_span.first(evidence.size() - 1)}),
                   std::runtime_error);
      std::string text;
      std::vector<TimestampRecord> records;
      const auto collect = [&] {
        for (const auto& result : processor.GetNextTokenResults()) {
          text += result.text;
          EXPECT_TRUE(result.timestamps);
          if (!result.timestamps) continue;
          const auto& completed = level == Config::TimestampLevel::Word
                                      ? result.timestamps->words
                                      : result.timestamps->segments;
          records.insert(records.end(), completed.begin(), completed.end());
        }
      };
      for (size_t offset = 0; offset < tokens.size(); offset += batch_size) {
        const auto count = std::min(batch_size, tokens.size() - offset);
        processor.Process(TokenOutputs{token_span.subspan(offset, count), interval_span.subspan(offset, count),
                                       evidence_span.subspan(offset, count)});
        collect();
      }
      processor.Finalize();
      collect();
      EXPECT_EQ(text, "Hello world.");
      EXPECT_EQ(records.size(), 2U);
      processor.Reset();
      EXPECT_TRUE(processor.GetNextTokenResults().empty());
      return records;
    };
    const auto batch = process(tokens.size());
    const auto incremental = process(1);
    ASSERT_EQ(batch.size(), incremental.size());
    for (size_t index = 0; index < batch.size(); ++index) {
      EXPECT_EQ(batch[index].text, incremental[index].text);
      EXPECT_EQ(batch[index].speaker_label, "speaker_" + std::to_string(index));
      EXPECT_EQ(batch[index].speaker_label, incremental[index].speaker_label);
      EXPECT_EQ(batch[index].start_frame, incremental[index].start_frame);
      EXPECT_EQ(batch[index].stop_frame, incremental[index].stop_frame);
      EXPECT_EQ(batch[index].start_time, incremental[index].start_time);
      EXPECT_EQ(batch[index].stop_time, incremental[index].stop_time);
      EXPECT_EQ(batch[index].start_time.has_value(), level != Config::TimestampLevel::Off);
    }
  }
}

TEST(NemotronDiarization, RejectsInvalidProbabilityWindows) {
  Generators::TokenSpeakerEvidence evidence{};
  EXPECT_THROW(Generators::TokenDiarization::ValidateSpeakerEvidence(evidence), std::runtime_error);
  evidence.stop_frame_count = 1;
  EXPECT_EQ(Generators::TokenDiarization::AggregateWordSpeaker(evidence, evidence), 0);
  evidence.stop_probability_sums[0] = 2;
  EXPECT_THROW(Generators::TokenDiarization::ValidateSpeakerEvidence(evidence), std::runtime_error);
}

TEST(NemotronDiarization, AggregationUsesEachWordsFrameWindow) {
  Generators::TokenSpeakerEvidence first{};
  first.stop_probability_sums[0] = 0.9;
  first.stop_probability_sums[1] = 0.1;
  first.stop_frame_count = 1;
  Generators::TokenSpeakerEvidence last{};
  std::copy(std::begin(first.stop_probability_sums), std::end(first.stop_probability_sums),
            last.start_probability_sums);
  last.start_frame_count = 1;
  last.stop_probability_sums[0] = 1.1;
  last.stop_probability_sums[1] = 1.9;
  last.stop_frame_count = 3;
  EXPECT_EQ(Generators::TokenDiarization::AggregateWordSpeaker(first, first), 0);
  EXPECT_EQ(Generators::TokenDiarization::AggregateWordSpeaker(last, last), 1);
  EXPECT_EQ(Generators::TokenDiarization::AggregateWordSpeaker(first, last), 1);
  last.stop_probability_sums[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(Generators::TokenDiarization::ValidateSpeakerEvidence(last), std::runtime_error);
}

TEST(NemotronDiarization, OnePrivateGeneratorOutputCarriesMultipleSequentialEmissions) {
  DiarizationTestPipeline pipeline{true, Generators::Config::TimestampLevel::Word};
  const auto tokens = pipeline.model->CreateTokenizer()->Encode("Hello world.");
  const auto first_word_tokens = pipeline.model->CreateTokenizer()->Encode("Hello").size();
  Generators::TokenSpeakerEvidence evidence{};
  pipeline.state->EmptyStep();
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::copy(std::begin(evidence.stop_probability_sums), std::end(evidence.stop_probability_sums),
              evidence.start_probability_sums);
    evidence.start_frame_count = static_cast<int64_t>(i);
    evidence.stop_frame_count = static_cast<int64_t>(i + 1);
    ++evidence.stop_probability_sums[i < first_word_tokens ? 0 : 1];
    pipeline.state->AppendStep(tokens[i], static_cast<int64_t>(i), evidence);
  }
  // The private getter validates all field counts before any decoding occurs.
  const auto saved = pipeline.state->evidence.back();
  pipeline.state->evidence.pop_back();
  EXPECT_THROW(pipeline.processor->ProcessNextTokens(), std::runtime_error);
  pipeline.state->evidence.push_back(saved);
  pipeline.processor->ProcessNextTokens();
  ASSERT_EQ(pipeline.processor->GetNextTokenResults().size(), tokens.size());
  std::string text;
  std::vector<std::string> speakers;
  for (const auto& result : pipeline.processor->GetNextTokenResults()) {
    text += result.text;
    for (const auto& word : result.timestamps->words) speakers.push_back(*word.speaker_label);
  }
  EXPECT_EQ(text, "Hello world.");
  pipeline.processor->Finalize();
  for (const auto& result : pipeline.processor->GetNextTokenResults())
    for (const auto& word : result.timestamps->words) speakers.push_back(*word.speaker_label);
  EXPECT_EQ(speakers, (std::vector<std::string>{"speaker_0", "speaker_1"}));
}

TEST(NemotronDiarization, SpeakerChangesCompleteTurnsAndInvalidInputConsumesNothing) {
  DiarizationTestPipeline pipeline{true, Generators::Config::TimestampLevel::Off};
  auto tokenizer = pipeline.model->CreateTokenizer();
  const auto tokens = tokenizer->Encode("Hello world.");
  const auto boundary = tokenizer->Encode("Hello").size();
  std::array<double, 8> sums{};
  std::vector<std::string> texts, speakers;
  const auto collect = [&]() {
    for (const auto& result : pipeline.processor->GetNextTokenResults()) {
      ASSERT_TRUE(result.timestamps);
      for (const auto& record : result.timestamps->segments) {
        texts.emplace_back(record.text);
        speakers.emplace_back(*record.speaker_label);
        EXPECT_FALSE(record.start_time);
      }
    }
  };
  for (size_t i = 0; i < tokens.size(); ++i) {
    Generators::TokenSpeakerEvidence evidence{};
    std::copy(sums.begin(), sums.end(), evidence.start_probability_sums);
    evidence.start_frame_count = static_cast<int64_t>(i);
    evidence.stop_frame_count = static_cast<int64_t>(i + 1);
    sums[i < boundary ? 0 : 1] += 1;
    std::copy(sums.begin(), sums.end(), evidence.stop_probability_sums);
    if (i == 0) {
      auto invalid = evidence;
      invalid.stop_frame_count = 0;
      pipeline.state->SetStep(tokens[i], static_cast<int64_t>(i), invalid);
      EXPECT_THROW(pipeline.processor->ProcessNextTokens(), std::runtime_error);
    }
    pipeline.state->SetStep(tokens[i], static_cast<int64_t>(i), evidence);
    pipeline.processor->ProcessNextTokens();
    collect();
  }
  pipeline.processor->Finalize();
  collect();
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
  speech.diarization_session_info_.Add(*speech.session_diarization_);
  auto mismatched = model->config_->model;
  mismatched.diarization.num_mels = 64;
  EXPECT_THROW(Generators::ValidateNemotronDiarizationGraph(mismatched, speech.diarization_session_info_),
               std::runtime_error);
  auto params = Generators::CreateGeneratorParams(*model);
  Generators::NemotronDiarizationState state{speech, *params};
  Generators::DeviceSpan<int32_t> no_tokens;
  EXPECT_THROW(state.Run(0, no_tokens), std::runtime_error);
  EXPECT_THROW(state.GetTokenSpeakerEvidence(0), std::runtime_error);
  auto features = OrtValue::CreateTensor<float>(model->allocator_cpu_, std::array<int64_t, 3>{1, 128, 88});
  std::fill_n(features->GetTensorMutableData<float>(), 128 * 88, -10.0f);
  EXPECT_THROW(state.SetFeaturesInput(*features, -1), std::runtime_error);
  EXPECT_THROW(state.SetFeaturesInput(*features, 5), std::runtime_error);
  for (int step = 0; step < 80; ++step) {
    state.SetFeaturesInput(*features, 4);
    state.Run(0, no_tokens);
    auto first = state.GetTokenSpeakerEvidence(0);
    auto last = state.GetTokenSpeakerEvidence(6);
    EXPECT_EQ(last.stop_frame_count, (step + 1) * 56);
    EXPECT_EQ(first.start_frame_count, step == 0 ? 0 : step * 56 - 8);
    EXPECT_NO_THROW(Generators::TokenDiarization::AggregateWordSpeaker(first, last));
  }
  EXPECT_TRUE(*state.GetInput("past_is_compressed")->GetTensorData<bool>());
  EXPECT_EQ(*state.GetInput("past_num_cache_frames")->GetTensorData<int64_t>(), 264);
  state.Reset();
  state.SetFeaturesInput(*features, 4);
  state.Run(0, no_tokens);
  EXPECT_EQ(state.GetTokenSpeakerEvidence(0).start_frame_count, 0);
  EXPECT_THROW(state.Run(0, no_tokens), std::runtime_error);
  EXPECT_THROW(state.GetTokenSpeakerEvidence(7), std::runtime_error);

  // A different aligned ASR stride changes boundary selection, not ONNX output resolution.
  const auto original_hop = model->config_->model.hop_length;
  const auto original_subsampling = model->config_->model.subsampling_factor;
  model->config_->model.hop_length = 80;
  model->config_->model.subsampling_factor = 4;
  Generators::NemotronDiarizationState finer_asr{speech, *params};
  finer_asr.SetFeaturesInput(*features, 4);
  finer_asr.Run(0, no_tokens);
  EXPECT_EQ(finer_asr.GetTokenSpeakerEvidence(6).start_frame_count, 10);
  EXPECT_EQ(finer_asr.GetTokenSpeakerEvidence(6).stop_frame_count, 14);
  finer_asr.SetFeaturesInput(*features, 4);
  finer_asr.Run(0, no_tokens);
  EXPECT_EQ(finer_asr.GetTokenSpeakerEvidence(0).start_frame_count, 54);
  model->config_->model.hop_length = original_hop;
  model->config_->model.subsampling_factor = original_subsampling;

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

TEST(NemotronDiarization, EvidenceOrderingSurvivesReleasedWordsAndReset) {
  DiarizationTestPipeline pipeline{true, Generators::Config::TimestampLevel::Word};
  Generators::TokenSpeakerEvidence evidence{};
  evidence.start_frame_count = 4;
  evidence.stop_frame_count = 5;
  evidence.start_probability_sums[0] = 2;
  evidence.stop_probability_sums[0] = 3;
  const auto process = [&](const Generators::TokenSpeakerEvidence& value) {
    pipeline.state->SetStep(42, 4, value);
    pipeline.processor->ProcessNextTokens();
  };
  const auto finish = [&]() {
    pipeline.processor->Finalize();
    EXPECT_EQ(pipeline.processor->GetNextTokenResults()[0].timestamps->words.size(), 1U);
  };
  process(evidence);
  finish();

  // A fully released word must not erase the stream's previous snapshot.
  auto backwards_count = evidence;
  backwards_count.start_frame_count = 3;
  backwards_count.stop_frame_count = 4;
  EXPECT_THROW(process(backwards_count), std::runtime_error);
  auto backwards_sum = evidence;
  backwards_sum.start_probability_sums[0] = 1;
  backwards_sum.stop_probability_sums[0] = 2;
  EXPECT_THROW(process(backwards_sum), std::runtime_error);
  auto changed_same_frame = evidence;
  changed_same_frame.start_probability_sums[0] = 2.5;
  changed_same_frame.stop_probability_sums[0] = 3.5;
  EXPECT_THROW(process(changed_same_frame), std::runtime_error);

  // Multiple tokens from one acoustic frame can share identical snapshots.
  EXPECT_NO_THROW(process(evidence));
  finish();

  // Frames with no tokens can advance counts and sums before the next token.
  pipeline.state->EmptyStep();
  pipeline.processor->ProcessNextTokens();
  EXPECT_TRUE(pipeline.processor->GetNextTokenResults().empty());
  evidence.start_frame_count = 9;
  evidence.stop_frame_count = 10;
  evidence.start_probability_sums[0] = 4;
  evidence.stop_probability_sums[0] = 5;
  EXPECT_NO_THROW(process(evidence));
  finish();

  pipeline.processor->Reset();
  EXPECT_NO_THROW(process(backwards_count));
}
