// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "vibevoice.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>

#include "../ort_genai_c.h"

namespace Generators {

namespace {

constexpr auto kInt64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
constexpr auto kFloat16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;

// A voice preset is a few MB of cached prompt. The limit stops a package from exhausting memory.
constexpr uint64_t kMaxVoiceTotalBytes = 512ULL * 1024 * 1024;

// The model makes 7.5 speech frames per second, and English speech needs about 2 frames per text
// token. The frame budget stops an utterance whose EOS classifier never fires, but always leaves
// room to feed all text.
constexpr size_t kMinFrameBudget = 32;
constexpr size_t kFramesPerToken = 4;

constexpr int kMaxWindowSize = 64;
constexpr size_t kMaxDiffusionSteps = 100;
constexpr double kMaxFloat16 = 65504.0;  // Timesteps go to the diffusion head as float16.

[[noreturn]] void Invalid(const std::string& message) {
  throw std::runtime_error("VibeVoice: " + message);
}

struct TensorSpec {
  const char* name;
  ONNXTensorElementDataType type;
};

// The shape of the graph input or output `name`, or nullopt when the graph has no such tensor.
std::optional<std::vector<int64_t>> FindTensor(const OrtSession& session, bool input, std::string_view name,
                                               ONNXTensorElementDataType type, const char* label) {
  const auto count = input ? session.GetInputCount() : session.GetOutputCount();
  for (size_t i = 0; i < count; ++i) {
    if ((input ? session.GetInputName(i) : session.GetOutputName(i)) != name)
      continue;
    const auto type_info = input ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
    const auto& tensor = type_info->GetTensorTypeAndShapeInfo();
    if (tensor.GetElementType() != type)
      Invalid(std::string{"graph "} + label + " " + (input ? "input '" : "output '") + std::string{name} +
              "' must be " + TypeToString(type));
    return tensor.GetShape();
  }
  return std::nullopt;
}

std::vector<int64_t> RequireTensor(VibeVoiceGraph& graph, bool input, const std::string& name,
                                   ONNXTensorElementDataType type, const char* label) {
  auto shape = FindTensor(*graph.session, input, name, type, label);
  if (!shape)
    Invalid(std::string{"graph "} + label + " needs " + (input ? "input '" : "output '") + name + "' of type " +
            TypeToString(type));
  (input ? graph.input_names : graph.output_names).push_back(name);
  return *shape;
}

int64_t StaticDim(const std::vector<int64_t>& shape, size_t index) {
  return index < shape.size() && shape[index] > 0 ? shape[index] : 0;
}

// Agrees two sizes that each may be unknown (0).
int64_t Agree(int64_t first, int64_t second, const char* what) {
  if (first > 0 && second > 0 && first != second)
    Invalid(std::string{"the graphs disagree on the "} + what);
  return first > 0 ? first : second;
}

std::vector<uint16_t> HalfData(const OrtValue& value) {
  const auto info = value.GetTensorTypeAndShapeInfo();
  if (info->GetElementType() != kFloat16)
    Invalid("graph outputs must be float16");
  const auto* data = static_cast<const uint16_t*>(value.GetTensorRawData());
  return std::vector<uint16_t>(data, data + info->GetElementCount());
}

// The last position of a float16 [..., width] output.
std::vector<uint16_t> LastRow(const OrtValue& value) {
  const auto shape = value.GetTensorTypeAndShapeInfo()->GetShape();
  auto data = HalfData(value);
  const auto width = shape.empty() ? 0 : static_cast<size_t>(std::max<int64_t>(shape.back(), 0));
  if (width == 0 || data.size() < width)
    Invalid("a hidden state is empty");
  data.erase(data.begin(), data.end() - static_cast<std::ptrdiff_t>(width));
  return data;
}

// Standard normal samples from a Mersenne Twister with the Box-Muller transform, which gives the same
// values on every platform (std::normal_distribution does not).
std::vector<float> GaussianNoise(std::mt19937& rng, size_t count) {
  constexpr double kTwoPi = 6.283185307179586;
  std::vector<float> noise(count);
  for (size_t i = 0; i < count; i += 2) {
    const double u1 = (static_cast<double>(rng()) + 0.5) / 4294967296.0;
    const double u2 = (static_cast<double>(rng()) + 0.5) / 4294967296.0;
    const double radius = std::sqrt(-2.0 * std::log(u1));
    noise[i] = static_cast<float>(radius * std::cos(kTwoPi * u2));
    if (i + 1 < count)
      noise[i + 1] = static_cast<float>(radius * std::sin(kTwoPi * u2));
  }
  return noise;
}

// The token count of a text input. Throws for any other input.
size_t TextTokens(const ExtraInput& input) {
  if (input.name != OgaInputName_TextIds)
    throw std::runtime_error("VibeVoice takes only the input '" OgaInputName_TextIds "', not '" + input.name + "'");
  const auto info = input.tensor && input.tensor->ort_tensor_ ? input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo() : nullptr;
  if (!info || (info->GetElementType() != kInt64 && info->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) ||
      info->GetShape().size() != 2 || info->GetShape()[0] != 1)
    throw std::runtime_error("VibeVoice input '" OgaInputName_TextIds "' must be an int64 or int32 tensor of shape [1, length]");
  return static_cast<size_t>(info->GetShape()[1]);
}

}  // namespace

std::vector<float> SampleDpmSolver(std::span<const double> sigmas, const std::vector<float>& noise,
                                   const VelocityPredictor& predict) {
  if (sigmas.size() < 2)
    throw std::runtime_error("The diffusion schedule needs at least two sigmas");
  const size_t steps = sigmas.size() - 1;

  std::vector<double> sample(noise.begin(), noise.end());
  std::vector<double> previous_x0;
  std::vector<float> input(noise.size());
  for (size_t step = 0; step < steps; ++step) {
    std::transform(sample.begin(), sample.end(), input.begin(), [](double v) { return static_cast<float>(v); });
    const auto velocity = predict(step, input);
    if (velocity.size() != sample.size())
      throw std::runtime_error("The diffusion model output has the wrong size");

    // With alpha = 1 / sqrt(1 + sigma^2), the noise level is sigma * alpha and lambda = log(alpha / (sigma * alpha)).
    const double s0 = sigmas[step];
    const double alpha0 = 1.0 / std::sqrt(1.0 + s0 * s0);
    std::vector<double> x0(sample.size());
    for (size_t i = 0; i < sample.size(); ++i)
      x0[i] = alpha0 * sample[i] - s0 * alpha0 * static_cast<double>(velocity[i]);

    const double s1 = sigmas[step + 1];
    if (s1 == 0.0) {
      sample = x0;
      previous_x0 = std::move(x0);
      continue;
    }

    const double alpha1 = 1.0 / std::sqrt(1.0 + s1 * s1);
    const double h = std::log(s0) - std::log(s1);
    const double ratio = (s1 * alpha1) / (s0 * alpha0);
    const double phi = alpha1 * std::expm1(-h);
    const bool first_order = step == 0 || step + 1 == steps || previous_x0.empty();
    const double r0 = first_order ? 1.0 : (std::log(sigmas[step - 1]) - std::log(s0)) / h;
    for (size_t i = 0; i < sample.size(); ++i) {
      sample[i] = ratio * sample[i] - phi * x0[i];
      if (!first_order)
        sample[i] -= 0.5 * phi * (x0[i] - previous_x0[i]) / r0;
    }
    previous_x0 = std::move(x0);
  }
  std::vector<float> result(sample.size());
  std::transform(sample.begin(), sample.end(), result.begin(), [](double v) { return static_cast<float>(v); });
  return result;
}

void ValidateVibeVoiceConfig(const Config& config) {
  const auto& m = config.model;
  const auto& v = m.vibevoice;
  for (const auto* graph : {&v.text_lm, &v.tts_lm, &v.diffusion_head, &v.acoustic_connector, &v.vocoder}) {
    if (graph->filename.empty())
      Invalid("model.vibevoice needs the filename of the text_lm, tts_lm, diffusion_head, acoustic_connector and vocoder graphs");
  }
  if (!std::isfinite(v.speech_scaling_factor) || v.speech_scaling_factor == 0.0f || !std::isfinite(v.speech_bias_factor))
    Invalid("speech_scaling_factor must be a finite non-zero number and speech_bias_factor a finite number");
  if (!(v.cfg_scale >= 0.0f && v.cfg_scale <= 20.0f))
    Invalid("cfg_scale must be from 0 to 20");
  if (v.text_window_size < 1 || v.text_window_size > kMaxWindowSize || v.speech_window_size < 1 ||
      v.speech_window_size > kMaxWindowSize)
    Invalid("text_window_size and speech_window_size must be from 1 to " + std::to_string(kMaxWindowSize));

  const auto& timesteps = v.diffusion_timesteps;
  const auto& sigmas = v.diffusion_sigmas;
  if (timesteps.empty() || timesteps.size() > kMaxDiffusionSteps || sigmas.size() != timesteps.size() + 1)
    Invalid("diffusion needs 1 to " + std::to_string(kMaxDiffusionSteps) + " timesteps and one sigma more than timesteps");
  for (double timestep : timesteps) {
    if (!(timestep >= 0.0 && timestep <= kMaxFloat16))
      Invalid("diffusion timesteps must be from 0 to 65504 (the float16 range)");
  }
  // The solver divides by every sigma but the last and needs a strictly decreasing noise level.
  for (size_t i = 0; i + 1 < sigmas.size(); ++i) {
    if (!std::isfinite(sigmas[i]) || !std::isfinite(sigmas[i + 1]) || sigmas[i] <= 0.0 || sigmas[i + 1] < 0.0 ||
        sigmas[i + 1] >= sigmas[i])
      Invalid("diffusion sigmas must be finite, decrease strictly, be > 0 and end at a value >= 0");
  }

  if (m.sample_rate != 0 && m.sample_rate != 24000)
    Invalid("the vocoder makes 24000 Hz audio, but sample_rate is " + std::to_string(m.sample_rate));
  if (m.voices.empty())
    Invalid("model.voices must name at least one voice preset");
  std::set<std::string, std::less<>> names;
  for (const auto& voice : m.voices) {
    if (voice.name.empty() || voice.file.empty() || voice.language.empty() || !names.insert(voice.name).second)
      Invalid("every voice needs a unique name, a file and a language (voice '" + voice.name + "')");
  }
  if (!m.default_voice.empty() && names.count(m.default_voice) == 0)
    Invalid("default_voice '" + m.default_voice + "' is not in model.voices");
}

VibeVoiceModel::VibeVoiceModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)}, memory_info_{OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)} {
  ValidateVibeVoiceConfig(*config_);

  // Each graph must have exactly these inputs and outputs, plus a KV cache for the language models.
  auto load = [&](VibeVoiceGraph& graph, const char* label, const std::string& filename,
                  std::initializer_list<TensorSpec> inputs, std::initializer_list<TensorSpec> outputs, bool kv_cache) {
    graph.session = CreateSession(ort_env, filename, session_options_.get());
    for (const auto& spec : inputs)
      RequireTensor(graph, true, spec.name, spec.type, label);
    for (const auto& spec : outputs)
      RequireTensor(graph, false, spec.name, spec.type, label);
    graph.head_outputs = outputs.size();
    while (kv_cache && FindTensor(*graph.session, true, "past_key_" + std::to_string(graph.layers), kFloat16, label)) {
      for (const char* kind : {"key", "value"}) {
        RequireTensor(graph, true, std::string{"past_"} + kind + "_" + std::to_string(graph.layers), kFloat16, label);
        RequireTensor(graph, false, std::string{"present_"} + kind + "_" + std::to_string(graph.layers), kFloat16, label);
      }
      ++graph.layers;
    }
    if (kv_cache && graph.layers == 0)
      Invalid(std::string{"graph "} + label + " needs a KV cache (past_key_0)");
    if (graph.session->GetInputCount() != graph.input_names.size())
      Invalid(std::string{"graph "} + label + " has unexpected inputs");
    for (const auto& name : graph.input_names)
      graph.inputs.push_back(name.c_str());
    for (const auto& name : graph.output_names)
      graph.outputs.push_back(name.c_str());
  };

  const auto& v = config_->model.vibevoice;
  load(text_lm_, "text_lm", v.text_lm.filename, {{"input_ids", kInt64}}, {{"hidden_states", kFloat16}}, true);
  load(tts_lm_, "tts_lm", v.tts_lm.filename, {{"inputs_embeds", kFloat16}, {"tts_text_mask", kInt64}},
       {{"hidden_states", kFloat16}, {"eos_logits", kFloat16}}, true);
  load(diffusion_head_, "diffusion_head", v.diffusion_head.filename,
       {{"noisy_latent", kFloat16}, {"timestep", kFloat16}, {"condition", kFloat16}}, {{"predicted_noise", kFloat16}}, false);
  load(acoustic_connector_, "acoustic_connector", v.acoustic_connector.filename, {{"latent", kFloat16}},
       {{"embedding", kFloat16}}, false);
  load(vocoder_, "vocoder", v.vocoder.filename, {{"latents", kFloat16}}, {{"audio", kFloat16}}, false);

  auto shape = [](const VibeVoiceGraph& graph, const char* name) {
    return *FindTensor(*graph.session, true, name, kFloat16, "");
  };
  hidden_size_ = Agree(StaticDim(shape(tts_lm_, "inputs_embeds"), 2), StaticDim(shape(diffusion_head_, "condition"), 1),
                       "hidden size");
  latent_size_ = Agree(StaticDim(shape(diffusion_head_, "noisy_latent"), 1),
                       StaticDim(shape(acoustic_connector_, "latent"), 2), "latent size");
  if (latent_size_ == 0)
    Invalid("graph diffusion_head needs a static latent size");
  LoadVoices(ort_env);
}

void VibeVoiceModel::LoadVoices(OrtEnv& ort_env) {
  const auto text_kv = *FindTensor(*text_lm_.session, true, "past_key_0", kFloat16, "text_lm");
  const auto tts_kv = *FindTensor(*tts_lm_.session, true, "past_key_0", kFloat16, "tts_lm");
  const auto kv_heads = Agree(StaticDim(text_kv, 1), StaticDim(tts_kv, 1), "KV heads");
  const auto head_size = Agree(StaticDim(text_kv, 3), StaticDim(tts_kv, 3), "KV head size");
  auto matches = [](int64_t actual, int64_t expected) { return expected <= 0 || actual == expected; };

  // The outputs that a voice preset graph must have: the caches in layer order, then the hidden state.
  std::vector<std::string> names;
  for (const auto& [prefix, layers] : {std::pair{"lm", text_lm_.layers}, std::pair{"tts_lm", tts_lm_.layers},
                                       std::pair{"neg_tts_lm", tts_lm_.layers}}) {
    for (size_t layer = 0; layer < layers; ++layer) {
      for (const char* kind : {"key", "value"})
        names.push_back(std::string{prefix} + "_" + kind + "_" + std::to_string(layer));
    }
  }
  names.push_back("neg_tts_lm_hidden");
  std::vector<const char*> outputs;
  for (const auto& name : names)
    outputs.push_back(name.c_str());

  // A preset is data: it runs once on the CPU, whatever the providers of the model graphs are.
  const auto options = OrtSessionOptions::Create();
  uint64_t total_bytes = 0;
  for (const auto& voice : config_->model.voices) {
    auto fail = [&](const std::string& message) { Invalid("voice preset " + voice.file + ": " + message); };
    const auto session = CreateSession(ort_env, voice.file, options.get());
    if (session->GetInputCount() != 0)
      fail("must be an ONNX graph without inputs");
    for (const auto& name : names) {
      if (!FindTensor(*session, false, name, kFloat16, "voice preset"))
        fail("needs the float16 output '" + name + "'");
    }
    auto values = session->Run(nullptr, nullptr, nullptr, 0, outputs.data(), outputs.size());
    for (const auto& value : values) {
      total_bytes += value->GetTensorTypeAndShapeInfo()->GetElementCount() * sizeof(uint16_t);
      if (total_bytes > kMaxVoiceTotalBytes)
        Invalid("the voice presets are larger than " + std::to_string(kMaxVoiceTotalBytes) + " bytes");
    }

    // A cache is [1, kv_heads, length, head_size] per layer, with one length for all layers.
    size_t next = 0;
    auto take_cache = [&](size_t layers, int64_t& length) {
      std::vector<std::unique_ptr<OrtValue>> cache;
      for (size_t i = 0; i < 2 * layers; ++i, ++next) {
        const auto s = values[next]->GetTensorTypeAndShapeInfo()->GetShape();
        if (s.size() != 4 || s[0] != 1 || !matches(s[1], kv_heads) || s[2] < 1 || !matches(s[3], head_size) ||
            (i > 0 && s[2] != length))
          fail("output '" + names[next] + "' must have shape [1, " + std::to_string(kv_heads) + ", length, " +
               std::to_string(head_size) + "] with one length for all layers");
        length = s[2];
        cache.push_back(std::move(values[next]));
      }
      return cache;
    };

    VibeVoicePreset preset;
    preset.text_lm_kv = take_cache(text_lm_.layers, preset.text_lm_length);
    preset.tts_lm_kv = take_cache(tts_lm_.layers, preset.tts_lm_length);
    preset.negative_tts_lm_kv = take_cache(tts_lm_.layers, preset.negative_tts_lm_length);
    const auto hidden = values[next]->GetTensorTypeAndShapeInfo()->GetShape();
    if (hidden.size() != 3 || hidden[0] != 1 || hidden[1] < 1 || hidden[2] < 1 || !matches(hidden[2], hidden_size_))
      fail("output 'neg_tts_lm_hidden' must have shape [1, length, " + std::to_string(hidden_size_) + "]");
    const auto data = HalfData(*values[next]);
    preset.negative_condition.assign(data.end() - hidden[2], data.end());
    voices_.emplace(voice.name, std::move(preset));
  }
}

const VibeVoicePreset& VibeVoiceModel::GetVoice(std::string_view name) const {
  const auto& m = config_->model;
  if (name.empty())
    name = m.default_voice.empty() ? std::string_view{m.voices.front().name} : std::string_view{m.default_voice};
  const auto it = voices_.find(name);
  if (it == voices_.end())
    throw std::runtime_error("Unknown voice '" + std::string{name} + "'");
  return it->second;
}

std::unique_ptr<State> VibeVoiceModel::CreateState(DeviceSpan<int32_t> /*sequence_lengths*/,
                                                   const GeneratorParams& params) const {
  return std::make_unique<VibeVoiceState>(*this, params);
}

VibeVoiceState::VibeVoiceState(const VibeVoiceModel& model, const GeneratorParams& params)
    : StepState{params, model},
      model_{model},
      config_{model.config_->model.vibevoice},
      voice_{&model.GetVoice({})},
      rng_{params.search.random_seed >= 0 ? static_cast<uint32_t>(params.search.random_seed) : std::random_device{}()} {
  chunk_done_ = true;  // Idle until SetInputs supplies text.
  if (model.config_->model.decoder.run_options.has_value())
    State::SetRunOptions(*model.config_->model.decoder.run_options);
}

DeviceSpan<float> VibeVoiceState::Run(int /*total_length*/, DeviceSpan<int32_t>& /*next_tokens*/,
                                      DeviceSpan<int32_t> /*next_indices*/) {
  return {};  // StepToken drives execution.
}

void VibeVoiceState::SetRunOption(const char* key, const char* value) {
  if (key && std::strcmp(key, OgaRuntimeOption_Voice) == 0) {
    voice_ = &model_.GetVoice(value ? value : "");
    return;
  }
  State::SetRunOption(key, value);
}

void VibeVoiceState::ValidateExtraInputs(std::span<const ExtraInput> extra_inputs) {
  for (const auto& input : extra_inputs)
    FrameBudget(TextTokens(input), *voice_);
}

void VibeVoiceState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  if (extra_inputs.empty())
    return;
  // The latest text replaces any text that waits. It is checked again because the voice may have
  // changed since SetInputs; text that does not fit is dropped.
  try {
    FrameBudget(TextTokens(extra_inputs.back()), *voice_);
  } catch (...) {
    pending_text_.reset();
    chunk_done_ = true;
    throw;
  }
  pending_text_ = extra_inputs.back().tensor;
  audio_.reset();
  chunk_done_ = false;
}

std::unique_ptr<OrtValue> VibeVoiceState::HalfView(std::span<const uint16_t> data, std::span<const int64_t> shape) const {
  // ORT does not write to inputs, so a view of const data is safe.
  return OrtValue::CreateTensor(*model_.memory_info_, const_cast<uint16_t*>(data.data()), data.size_bytes(), shape, kFloat16);
}

std::unique_ptr<OrtValue> VibeVoiceState::Int64View(std::span<int64_t> data) const {
  const std::array<int64_t, 2> shape{1, static_cast<int64_t>(data.size())};
  return OrtValue::CreateTensor<int64_t>(*model_.memory_info_, data, shape);
}

std::vector<std::unique_ptr<OrtValue>> VibeVoiceState::RunGraph(const VibeVoiceGraph& graph,
                                                                std::vector<const OrtValue*> inputs, KvCache* cache) {
  if (cache)
    inputs.insert(inputs.end(), cache->past.begin(), cache->past.end());
  auto outputs = graph.session->Run(run_options_.get(), graph.inputs.data(), inputs.data(), inputs.size(),
                                    graph.outputs.data(), graph.outputs.size());
  if (cache) {
    cache->owned.assign(std::make_move_iterator(outputs.begin() + static_cast<std::ptrdiff_t>(graph.head_outputs)),
                        std::make_move_iterator(outputs.end()));
    cache->past.assign(cache->owned.size(), nullptr);
    std::transform(cache->owned.begin(), cache->owned.end(), cache->past.begin(), [](auto& v) { return v.get(); });
    outputs.resize(graph.head_outputs);
  }
  return outputs;
}

size_t VibeVoiceState::FrameBudget(size_t tokens, const VibeVoicePreset& voice) const {
  // The last text window is fed after (windows - 1) speech windows.
  const size_t windows = (tokens + config_.text_window_size - 1) / config_.text_window_size;
  const size_t feed_frames = windows == 0 ? 0 : (windows - 1) * config_.speech_window_size;
  const size_t frames = std::max(kMinFrameBudget + kFramesPerToken * tokens, feed_frames + kMinFrameBudget);
  const auto prompt = static_cast<size_t>(
      std::max({voice.text_lm_length, voice.tts_lm_length, voice.negative_tts_lm_length}));
  const auto context = static_cast<size_t>(model_.config_->model.context_length);
  if (prompt + tokens + frames > context)
    throw std::runtime_error("VibeVoice: text of " + std::to_string(tokens) + " tokens may need " +
                             std::to_string(frames) + " speech frames, which with the voice prompt of " +
                             std::to_string(prompt) + " positions exceeds the context length " +
                             std::to_string(context) + ". Split the text into shorter chunks.");
  return frames;
}

void VibeVoiceState::StartUtterance() {
  // Stay done if the text is rejected, so a failed utterance does not look like a running one.
  chunk_done_ = true;
  const auto text = std::move(pending_text_);
  const auto& value = *text->ort_tensor_;
  const auto count = value.GetTensorTypeAndShapeInfo()->GetElementCount();
  if (value.GetTensorTypeAndShapeInfo()->GetElementType() == kInt64)
    text_ids_.assign(value.GetTensorData<int64_t>(), value.GetTensorData<int64_t>() + count);
  else
    text_ids_.assign(value.GetTensorData<int32_t>(), value.GetTensorData<int32_t>() + count);

  max_frames_ = FrameBudget(text_ids_.size(), *voice_);

  auto prompt_cache = [](const std::vector<std::unique_ptr<OrtValue>>& kv) {
    KvCache cache;
    for (const auto& value : kv)
      cache.past.push_back(value.get());
    return cache;
  };
  text_cache_ = prompt_cache(voice_->text_lm_kv);
  tts_cache_ = prompt_cache(voice_->tts_lm_kv);
  negative_cache_ = prompt_cache(voice_->negative_tts_lm_kv);
  condition_.clear();
  negative_condition_ = voice_->negative_condition;
  latents_.clear();
  all_tokens_.clear();
  text_fed_ = 0;
  window_frames_ = 0;
  frames_ = 0;
  if (text_ids_.empty()) {
    audio_ = OrtValue::CreateTensor<float>(model_.allocator_cpu_, std::array<int64_t, 2>{1, 0});
    return;
  }
  chunk_done_ = false;
}

void VibeVoiceState::FeedText() {
  // The text LM encodes the window, and the TTS LM reads it as text (tts_text_mask 1).
  const auto end = std::min(text_fed_ + config_.text_window_size, text_ids_.size());
  std::vector<int64_t> window(text_ids_.begin() + static_cast<std::ptrdiff_t>(text_fed_),
                              text_ids_.begin() + static_cast<std::ptrdiff_t>(end));
  const auto ids = Int64View(window);
  const auto hidden = std::move(RunGraph(model_.text_lm_, {ids.get()}, &text_cache_).front());
  std::vector<int64_t> mask(window.size(), 1);
  const auto mask_value = Int64View(mask);
  condition_ = LastRow(*RunGraph(model_.tts_lm_, {hidden.get(), mask_value.get()}, &tts_cache_).front());
  text_fed_ = end;
}

std::vector<float> VibeVoiceState::SampleLatent() {
  const auto latent_size = static_cast<size_t>(model_.latent_size_);
  const std::array<int64_t, 2> latent_shape{1, model_.latent_size_};
  const std::array<int64_t, 1> timestep_shape{1};
  const std::array<int64_t, 2> condition_shape{1, static_cast<int64_t>(condition_.size())};
  auto predict = [&](size_t step, const std::vector<float>& sample) {
    std::vector<uint16_t> noisy(sample.size());
    ConvertFloat32ToFloat16(sample.data(), noisy.data(), sample.size());
    const std::array<uint16_t, 1> timestep{FastFloat32ToFloat16(static_cast<float>(config_.diffusion_timesteps[step]))};

    // The diffusion head has batch size 1, so the guided and the unguided prediction are two runs.
    auto run = [&](const std::vector<uint16_t>& condition) {
      const auto a = HalfView(noisy, latent_shape);
      const auto b = HalfView(timestep, timestep_shape);
      const auto c = HalfView(condition, condition_shape);
      auto data = HalfData(*RunGraph(model_.diffusion_head_, {a.get(), b.get(), c.get()}).front());
      if (data.size() != latent_size)
        Invalid("the diffusion head output has the wrong size");
      return data;
    };
    const auto positive = run(condition_);
    const auto negative = run(negative_condition_);
    std::vector<float> velocity(latent_size);
    for (size_t i = 0; i < latent_size; ++i) {
      const float unguided = Float16ToFloat32(negative[i]);
      velocity[i] = unguided + config_.cfg_scale * (Float16ToFloat32(positive[i]) - unguided);
    }
    return velocity;
  };
  return SampleDpmSolver(config_.diffusion_sigmas, GaussianNoise(rng_, latent_size), predict);
}

void VibeVoiceState::StepToken() {
  last_tokens_.clear();
  ThrowErrorIfSessionTerminated(session_terminated_);
  if (pending_text_)
    StartUtterance();
  if (chunk_done_)
    return;

  if (window_frames_ == 0 && text_fed_ < text_ids_.size())
    FeedText();
  const bool all_text_fed = text_fed_ >= text_ids_.size();
  const auto latent = SampleLatent();
  latents_.insert(latents_.end(), latent.begin(), latent.end());
  ++frames_;
  window_frames_ = (window_frames_ + 1) % config_.speech_window_size;
  // The sequence records each frame as one placeholder token, so TokenCount() counts frames.
  last_tokens_.push_back(model_.config_->model.audio_token_id);
  all_tokens_.push_back(model_.config_->model.audio_token_id);

  // Feed the latent back to both TTS LMs through the acoustic connector (tts_text_mask 0 marks speech).
  std::vector<uint16_t> latent_half(latent.size());
  ConvertFloat32ToFloat16(latent.data(), latent_half.data(), latent.size());
  const auto latent_value = HalfView(latent_half, std::array<int64_t, 3>{1, 1, model_.latent_size_});
  const auto embedding = std::move(RunGraph(model_.acoustic_connector_, {latent_value.get()}).front());
  std::array<int64_t, 1> speech_mask{0};
  const auto mask_value = Int64View(speech_mask);
  auto outputs = RunGraph(model_.tts_lm_, {embedding.get(), mask_value.get()}, &tts_cache_);
  condition_ = LastRow(*outputs[0]);
  const auto eos = HalfData(*outputs[1]);
  // VibeVoice stops when sigmoid(eos_logit) > 0.5. Ignoring EOS until all text is fed keeps a
  // float16 rounding difference from cutting off the text.
  if ((all_text_fed && !eos.empty() && Float16ToFloat32(eos.front()) > 0.0f) || frames_ >= max_frames_) {
    FinishUtterance();
    return;
  }
  negative_condition_ = LastRow(*RunGraph(model_.tts_lm_, {embedding.get(), mask_value.get()}, &negative_cache_).front());
}

void VibeVoiceState::FinishUtterance() {
  // Undo the latent scaling and decode all frames at once.
  std::vector<uint16_t> scaled(latents_.size());
  for (size_t i = 0; i < latents_.size(); ++i)
    scaled[i] = FastFloat32ToFloat16(latents_[i] / config_.speech_scaling_factor - config_.speech_bias_factor);
  const auto latents = HalfView(scaled, std::array<int64_t, 3>{1, static_cast<int64_t>(frames_), model_.latent_size_});
  const auto audio = HalfData(*RunGraph(model_.vocoder_, {latents.get()}).front());
  audio_ = OrtValue::CreateTensor<float>(model_.allocator_cpu_, std::array<int64_t, 2>{1, static_cast<int64_t>(audio.size())});
  ConvertFloat16ToFloat32(audio.data(), audio_->GetTensorMutableData<float>(), audio.size());

  text_cache_ = {};
  tts_cache_ = {};
  negative_cache_ = {};
  latents_.clear();
  chunk_done_ = true;
}

OrtValue* VibeVoiceState::GetOutput(const char* name) {
  if (std::strcmp(name, OgaOutputName_Audio) == 0) {
    ThrowErrorIfSessionTerminated(session_terminated_);
    return audio_.get();
  }
  return State::GetOutput(name);
}

}  // namespace Generators
