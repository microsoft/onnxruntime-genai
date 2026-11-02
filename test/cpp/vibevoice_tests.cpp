// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// VibeVoice text to speech through the public API, with the tiny deterministic model of
// test/python/create/create_dummy_vibevoice_models.py (test/models/vibevoice).
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define OGA_USE_SPAN 1
#include "ort_genai.h"

#include <gtest/gtest.h>

#ifndef MODEL_PATH
#define MODEL_PATH "../../test/models/"
#endif

namespace {

constexpr const char* kModel = MODEL_PATH "vibevoice";

struct Voice {
  int64_t tts_prompt;
  float negative;
};
constexpr Voice kAlpha{3, 0.05f};
constexpr Voice kBeta{5, -0.05f};

// What the tiny graphs produce for `text`: four samples per frame (see the generator script). The
// text is fed in windows of 5 tokens, each followed by up to 6 speech frames; EOS fires once all text
// is fed and the TTS LM holds more than 40 positions.
std::vector<float> TinySpeech(const std::string& text, Voice voice) {
  std::vector<int> ids(text.begin(), text.end());
  ids.push_back('\n');
  const size_t max_frames = 32 + 4 * ids.size();
  int64_t length = voice.tts_prompt;
  float negative = voice.negative;
  float condition = 0.0f;
  std::vector<float> samples;
  for (size_t begin = 0;; begin += 5) {
    if (begin < ids.size()) {
      const auto end = std::min(begin + 5, ids.size());
      condition = 0.001f * static_cast<float>(ids[end - 1]) + 0.1f;
      length += static_cast<int64_t>(end - begin);
    }
    for (int i = 0; i < 6; ++i) {
      const float latent = negative + 2.0f * (condition - negative);
      samples.insert(samples.end(), 4, latent / 0.5f - 0.1f);
      condition = 0.5f * latent;
      ++length;
      if ((begin + 5 >= ids.size() && length > 40) || samples.size() / 4 >= max_frames)
        return samples;
      negative = condition;
    }
  }
}

std::unique_ptr<OgaModel> LoadModel(const char* overlay = nullptr) {
  auto config = OgaConfig::Create(kModel);
  if (overlay)
    config->Overlay(overlay);
  return OgaModel::Create(*config);
}

std::vector<int64_t> TextIds(const OgaMultiModalProcessor& processor, const char* text) {
  auto inputs = processor.ProcessImagesAndAudios(text);
  auto ids = inputs->Get(OgaInputName_TextIds);
  const auto shape = ids->Shape();
  EXPECT_EQ(ids->Type(), OgaElementType_int64);
  EXPECT_EQ(shape.size(), 2u);
  const auto* data = static_cast<const int64_t*>(ids->Data());
  return std::vector<int64_t>(data, data + shape[1]);
}

struct Speech {
  std::vector<float> samples;
  size_t frames{};
};

// Synthesizes `text` with one generator and checks that every step adds one frame.
Speech Synthesize(const OgaModel& model, const OgaMultiModalProcessor& processor, const char* text,
                  const char* voice = nullptr) {
  auto params = OgaGeneratorParams::Create(model);
  auto generator = OgaGenerator::Create(model, *params);
  if (voice)
    generator->SetRuntimeOption(OgaRuntimeOption_Voice, voice);
  auto inputs = processor.ProcessImagesAndAudios(text);
  generator->SetInputs(*inputs);
  size_t steps = 0;
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
    ++steps;
  }
  // One frame per step; empty text takes one step without a frame.
  EXPECT_TRUE(generator->TokenCount() == steps || (generator->TokenCount() == 0 && steps == 1));
  auto audio = generator->GetOutput(OgaOutputName_Audio);
  EXPECT_EQ(audio->Type(), OgaElementType_float32);
  const auto shape = audio->Shape();
  EXPECT_EQ(shape.size(), 2u);
  EXPECT_EQ(shape[0], 1);
  const auto* data = static_cast<const float*>(audio->Data());
  return {std::vector<float>(data, data + shape[1]), generator->TokenCount()};
}

void ExpectSpeech(const std::vector<float>& actual, const std::vector<float>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    // The float16 graphs leave a small error in every sample.
    EXPECT_NEAR(actual[i], expected[i], 0.02f) << "sample " << i;
  }
}

std::vector<int64_t> Bytes(const std::string& text) {
  return std::vector<int64_t>(text.begin(), text.end());
}

}  // namespace

TEST(VibeVoiceTests, ListsVoicesDefaultFirst) {
  auto model = LoadModel();
  EXPECT_STREQ(model->GetType(), "vibevoice");
  auto voices = model->GetVoices();
  ASSERT_EQ(voices->Count(), 2u);
  EXPECT_STREQ(voices->Get(0), "Alpha");
  EXPECT_STREQ(voices->Get(1), "Beta");
  EXPECT_STREQ(model->GetVoiceProperty("Alpha", OgaVoiceProperty_Language), "en");
  EXPECT_STREQ(model->GetVoiceProperty("Beta", OgaVoiceProperty_Language), "en-in");
  EXPECT_STREQ(model->GetVoiceProperty("Beta", OgaVoiceProperty_Gender), "male");
  EXPECT_THROW(model->GetVoiceProperty("Nobody", OgaVoiceProperty_Language), std::runtime_error);
  EXPECT_THROW(model->GetVoiceProperty("Alpha", "age"), std::runtime_error);

  // The default voice comes first, also when it is not first in the config.
  auto beta = LoadModel(R"({"model": {"default_voice": "Beta"}})");
  voices = beta->GetVoices();
  ASSERT_EQ(voices->Count(), 2u);
  EXPECT_STREQ(voices->Get(0), "Beta");
  EXPECT_STREQ(voices->Get(1), "Alpha");

  // Without default_voice the first voice in the config is the default.
  auto first = LoadModel(R"({"model": {"default_voice": ""}})");
  EXPECT_STREQ(first->GetVoices()->Get(0), "Alpha");
}

TEST(VibeVoiceTests, ProcessorTokenizesLiteralText) {
  auto model = LoadModel();
  auto processor = OgaMultiModalProcessor::Create(*model);
  EXPECT_EQ(TextIds(*processor, "  It\xE2\x80\x99s \xE2\x80\x9Cok\xE2\x80\x9D \n"), Bytes("It's \"ok\"\n"));
  // Text that looks like a special token is read as plain text.
  EXPECT_EQ(TextIds(*processor, "<|endoftext|>"), Bytes("<|endoftext|>\n"));
  EXPECT_TRUE(TextIds(*processor, " \t\n").empty());
  EXPECT_EQ(TextIds(*processor, "Hello world.").size(), 13u);
}

TEST(VibeVoiceTests, SynthesizesTextWithTheDefaultVoice) {
  auto model = LoadModel();
  auto processor = OgaMultiModalProcessor::Create(*model);
  const auto speech = Synthesize(*model, *processor, "Hello world.");
  const auto expected = TinySpeech("Hello world.", kAlpha);
  ExpectSpeech(speech.samples, expected);
  EXPECT_EQ(speech.frames, expected.size() / 4);

  const auto again = Synthesize(*model, *processor, "Hello world.");
  EXPECT_EQ(again.samples, speech.samples) << "fixed search.random_seed";
  EXPECT_TRUE(Synthesize(*model, *processor, "  \n ").samples.empty());
}

TEST(VibeVoiceTests, AppliesTheVoiceOption) {
  auto model = LoadModel();
  auto processor = OgaMultiModalProcessor::Create(*model);
  ExpectSpeech(Synthesize(*model, *processor, "Hello world.", "Beta").samples, TinySpeech("Hello world.", kBeta));
  ExpectSpeech(Synthesize(*model, *processor, "Hello world.", "Alpha").samples, TinySpeech("Hello world.", kAlpha));

  auto params = OgaGeneratorParams::Create(*model);
  auto generator = OgaGenerator::Create(*model, *params);
  EXPECT_THROW(generator->SetRuntimeOption(OgaRuntimeOption_Voice, "Nobody"), std::runtime_error);
}

TEST(VibeVoiceTests, GeneratesOneFrameAndOnePlaceholderPerStep) {
  auto model = LoadModel();
  auto processor = OgaMultiModalProcessor::Create(*model);
  auto params = OgaGeneratorParams::Create(*model);
  auto generator = OgaGenerator::Create(*model, *params);
  EXPECT_TRUE(generator->IsDone());
  auto inputs = processor->ProcessImagesAndAudios("Hi.");
  generator->SetInputs(*inputs);
  EXPECT_FALSE(generator->IsDone());
  EXPECT_THROW(generator->GetOutput(OgaOutputName_Audio), std::runtime_error);
  generator->GenerateNextToken();
  EXPECT_EQ(generator->TokenCount(), 1u);
  EXPECT_EQ(generator->GetNextTokens().size(), 1u);
  while (!generator->IsDone())
    generator->GenerateNextToken();
  const auto expected = TinySpeech("Hi.", kAlpha);
  EXPECT_EQ(generator->TokenCount(), expected.size() / 4);
  EXPECT_EQ(generator->GetOutput(OgaOutputName_Audio)->Shape()[1], static_cast<int64_t>(expected.size()));

  // The same generator speaks a second chunk; "audio" is replaced.
  inputs = processor->ProcessImagesAndAudios("Hello world.");
  generator->SetInputs(*inputs);
  while (!generator->IsDone())
    generator->GenerateNextToken();
  auto audio = generator->GetOutput(OgaOutputName_Audio);
  const auto* data = static_cast<const float*>(audio->Data());
  ExpectSpeech(std::vector<float>(data, data + audio->Shape()[1]), TinySpeech("Hello world.", kAlpha));
}

TEST(VibeVoiceTests, StopsBetweenFramesAndOnTermination) {
  auto model = LoadModel();
  auto processor = OgaMultiModalProcessor::Create(*model);
  auto params = OgaGeneratorParams::Create(*model);
  auto generator = OgaGenerator::Create(*model, *params);
  auto inputs = processor->ProcessImagesAndAudios("This sentence is long enough to need many frames.");
  generator->SetInputs(*inputs);
  for (int i = 0; i < 3; ++i)
    generator->GenerateNextToken();
  EXPECT_FALSE(generator->IsDone());
  EXPECT_THROW(generator->GetOutput(OgaOutputName_Audio), std::runtime_error);

  generator->SetRuntimeOption("terminate_session", "1");
  EXPECT_TRUE(generator->IsSessionTerminated());
  EXPECT_THROW(generator->GenerateNextToken(), std::runtime_error);
}

TEST(VibeVoiceTests, FeedsAllTextWithNarrowWindows) {
  // 50 tokens in windows of one token need 49 * 6 frames before the last token is fed; the tiny EOS
  // then fires at the next frame. A budget of 32 + 4 frames per token would cut off the text.
  auto model = LoadModel(R"({"model": {"vibevoice": {"text_window_size": 1, "speech_window_size": 6}}})");
  auto processor = OgaMultiModalProcessor::Create(*model);
  const auto speech = Synthesize(*model, *processor, std::string(49, 'a').c_str());
  EXPECT_EQ(speech.frames, 49u * 6 + 1);
  EXPECT_EQ(speech.samples.size(), speech.frames * 4);
}

TEST(VibeVoiceTests, SetInputsRejectsTextThatDoesNotFitTheContext) {
  auto model = LoadModel(R"({"model": {"vibevoice": {"text_window_size": 1, "speech_window_size": 64}}})");
  auto processor = OgaMultiModalProcessor::Create(*model);
  auto params = OgaGeneratorParams::Create(*model);
  auto generator = OgaGenerator::Create(*model, *params);
  auto too_long = processor->ProcessImagesAndAudios(std::string(199, 'a').c_str());
  EXPECT_THROW(generator->SetInputs(*too_long), std::runtime_error);

  // The rejected text is dropped, so the same generator speaks shorter text; reused, it rejects again.
  EXPECT_TRUE(generator->IsDone());
  auto inputs = processor->ProcessImagesAndAudios("Hi.");
  for (int round = 0; round < 2; ++round) {
    generator->SetInputs(*inputs);
    while (!generator->IsDone())
      generator->GenerateNextToken();
    EXPECT_GT(generator->GetOutput(OgaOutputName_Audio)->Shape()[1], 0);
    EXPECT_THROW(generator->SetInputs(*too_long), std::runtime_error);
    EXPECT_TRUE(generator->IsDone());
  }

  // The voice prompt counts too: Beta has 5 positions, Alpha 3.
  auto small = LoadModel(R"({"model": {"context_length": 96}})");
  processor = OgaMultiModalProcessor::Create(*small);
  params = OgaGeneratorParams::Create(*small);
  const std::string text(11, 'a');  // 12 tokens and up to 32 + 48 frames: 95 positions with Alpha, 97 with Beta
  generator = OgaGenerator::Create(*small, *params);
  inputs = processor->ProcessImagesAndAudios(text.c_str());
  generator->SetInputs(*inputs);
  generator = OgaGenerator::Create(*small, *params);
  generator->SetRuntimeOption(OgaRuntimeOption_Voice, "Beta");
  inputs = processor->ProcessImagesAndAudios(text.c_str());
  EXPECT_THROW(generator->SetInputs(*inputs), std::runtime_error);

  // After a voice change, waiting text that no longer fits is checked again at the next step and dropped.
  generator = OgaGenerator::Create(*small, *params);
  generator->SetInputs(*inputs);
  generator->SetRuntimeOption(OgaRuntimeOption_Voice, "Beta");
  EXPECT_THROW(generator->GenerateNextToken(), std::runtime_error);
  EXPECT_TRUE(generator->IsDone());
  auto hi = processor->ProcessImagesAndAudios("Hi.");
  generator->SetInputs(*hi);
  while (!generator->IsDone())
    generator->GenerateNextToken();
  auto audio = generator->GetOutput(OgaOutputName_Audio);
  const auto* data = static_cast<const float*>(audio->Data());
  ExpectSpeech(std::vector<float>(data, data + audio->Shape()[1]), TinySpeech("Hi.", kBeta));

  // Text replaced before the first step is not checked again: only the latest text waits.
  generator = OgaGenerator::Create(*small, *params);
  generator->SetInputs(*inputs);
  generator->SetInputs(*hi);
  generator->SetRuntimeOption(OgaRuntimeOption_Voice, "Beta");
  auto ok = processor->ProcessImagesAndAudios("Ok.");
  generator->SetInputs(*ok);
  while (!generator->IsDone())
    generator->GenerateNextToken();
  audio = generator->GetOutput(OgaOutputName_Audio);
  data = static_cast<const float*>(audio->Data());
  ExpectSpeech(std::vector<float>(data, data + audio->Shape()[1]), TinySpeech("Ok.", kBeta));
}

TEST(VibeVoiceTests, GeneratorsShareOneModelAcrossThreads) {
  auto model = LoadModel();
  const std::vector<std::pair<std::string, const char*>> requests = {
      {"Hello world.", "Alpha"}, {"A second sentence.", "Beta"}, {"Hello world.", "Beta"}, {"Short.", "Alpha"}};
  std::vector<Speech> results(requests.size());
  std::vector<std::thread> threads;
  for (size_t i = 0; i < requests.size(); ++i) {
    threads.emplace_back([&, i] {
      // A processor per thread: processor calls are not documented as thread safe.
      auto processor = OgaMultiModalProcessor::Create(*model);
      for (int repeat = 0; repeat < 3; ++repeat)
        results[i] = Synthesize(*model, *processor, requests[i].first.c_str(), requests[i].second);
    });
  }
  for (auto& thread : threads)
    thread.join();
  for (size_t i = 0; i < requests.size(); ++i) {
    SCOPED_TRACE(requests[i].first + " / " + requests[i].second);
    ExpectSpeech(results[i].samples, TinySpeech(requests[i].first, std::strcmp(requests[i].second, "Beta") == 0 ? kBeta : kAlpha));
  }
}

TEST(VibeVoiceTests, RejectsInvalidPackages) {
  const char* invalid[] = {
      R"({"model": {"vibevoice": {"cfg_scale": 30}}})",
      R"({"model": {"vibevoice": {"diffusion": {"timesteps": [70000], "sigmas": [1.0, 0.0]}}}})",
      R"({"model": {"vibevoice": {"diffusion": {"timesteps": [1, 2], "sigmas": [1.0, 2.0, 0.0]}}}})",
      R"({"model": {"vibevoice": {"vocoder": {"filename": "../vocoder.onnx"}}}})",
      R"({"model": {"default_voice": "Nobody"}})",
      R"({"model": {"voices": [{"name": "A", "file": "voices/alpha.onnx", "language": "en"},
                               {"name": "A", "file": "voices/beta.onnx", "language": "en"}], "default_voice": ""}})",
      // A missing file, a graph with inputs, and caches with 2 KV heads instead of 1.
      R"({"model": {"voices": [{"name": "A", "file": "voices/missing.onnx", "language": "en"}], "default_voice": ""}})",
      R"({"model": {"voices": [{"name": "A", "file": "acoustic_connector.onnx", "language": "en"}], "default_voice": ""}})",
      R"({"model": {"voices": [{"name": "A", "file": "voices/two_heads.onnx", "language": "en"}], "default_voice": ""}})",
  };
  for (const char* overlay : invalid) {
    SCOPED_TRACE(overlay);
    EXPECT_THROW(LoadModel(overlay), std::runtime_error);
  }
}
