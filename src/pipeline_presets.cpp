// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Portions of this file consist of AI generated content.
#include "generators.h"
#include "pipeline_presets.h"

namespace Generators {

namespace {

Config::Pipeline MakeAutoregressiveDecoder() {
  Config::Pipeline p;
  p.flow.push_back({/*run*/ "decoder", /*when*/ "step"});
  // Default KV cache and position-id strategy ("auto") are introspected at load time.
  return p;
}

Config::Pipeline MakeVisionLanguage() {
  Config::Pipeline p;
  p.flow.push_back({/*run*/ "vision", /*when*/ "init", /*loop*/ "batched"});
  p.flow.push_back({/*run*/ "embedding", /*when*/ "init"});
  p.flow.push_back({/*run*/ "decoder", /*when*/ "step"});
  p.dataflow.push_back({/*from*/ "vision.image_features", /*to*/ "embedding.image_features"});
  p.dataflow.push_back({/*from*/ "embedding.inputs_embeds", /*to*/ "decoder.inputs_embeds"});
  return p;
}

Config::Pipeline MakeEncoderDecoder() {
  Config::Pipeline p;
  Config::Pipeline::FlowStep encoder_step{/*run*/ "encoder", /*when*/ "init"};
  p.flow.push_back(encoder_step);
  Config::Pipeline::FlowStep decoder_step{/*run*/ "decoder", /*when*/ "step"};
  decoder_step.cross_attention_from = "encoder";
  p.flow.push_back(decoder_step);
  Config::Pipeline::State::CrossCache cross_cache;
  cross_cache.source = "encoder";
  cross_cache.frozen = true;
  p.state.cross_cache = cross_cache;
  return p;
}

Config::Pipeline MakeSpeechLanguage() {
  Config::Pipeline p;
  p.flow.push_back({/*run*/ "speech", /*when*/ "init", /*loop*/ "batched"});
  p.flow.push_back({/*run*/ "embedding", /*when*/ "init"});
  p.flow.push_back({/*run*/ "decoder", /*when*/ "step"});
  p.dataflow.push_back({/*from*/ "speech.audio_features", /*to*/ "embedding.audio_features"});
  p.dataflow.push_back({/*from*/ "embedding.inputs_embeds", /*to*/ "decoder.inputs_embeds"});
  return p;
}

}  // namespace

const Config::Pipeline* GetPipelinePreset(std::string_view name) {
  static const Config::Pipeline autoregressive_decoder = MakeAutoregressiveDecoder();
  static const Config::Pipeline vision_language = MakeVisionLanguage();
  static const Config::Pipeline encoder_decoder = MakeEncoderDecoder();
  static const Config::Pipeline speech_language = MakeSpeechLanguage();

  if (name == "autoregressive-decoder") return &autoregressive_decoder;
  if (name == "vision-language") return &vision_language;
  if (name == "encoder-decoder") return &encoder_decoder;
  if (name == "speech-language") return &speech_language;
  return nullptr;
}

}  // namespace Generators
