// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Opt-in interoperability probe requiring the native packed-op ORT prerequisite.
// Build/run instructions are in test/README.md. This is not a checkpoint test.
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define OGA_USE_SPAN 1
#include "ort_genai.h"

namespace {
void Check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

float Bfloat(float value) {
  auto bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return std::bit_cast<float>(bits & 0xffff0000);
}

float Read(OgaTensor& tensor, size_t index) {
  if (tensor.Type() == OgaElementType_bfloat16)
    return std::bit_cast<float>(uint32_t(static_cast<const uint16_t*>(tensor.Data())[index]) << 16);
  Check(tensor.Type() == OgaElementType_float32, "Unexpected tensor dtype");
  return static_cast<const float*>(tensor.Data())[index];
}

void Probe(const std::string& directory, bool nvfp4) {
  auto config = OgaConfig::Create(directory.c_str());
  auto model = OgaModel::Create(*config);
  Check(std::strcmp(model->GetDeviceType(), "CUDA") == 0, "Not CUDA");
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 12);
  auto generator = OgaGenerator::Create(*model, *params);
  std::array<int32_t, 3> ids{1, 18, 3};
  std::array<uint16_t, 2> features{0x4080, 0x4080};  // Actual BF16 4.0.
  auto ids_tensor = OgaTensor::Create(ids.data(), std::array<int64_t, 2>{1, 3});
  auto features_tensor = OgaTensor::Create(features.data(), std::array<int64_t, 2>{1, 2}, OgaElementType_bfloat16);
  auto inputs = OgaNamedTensors::Create();
  inputs->Set("input_ids", *ids_tensor);
  inputs->Set("image_features", *features_tensor);
  generator->SetInputs(*inputs);
  // The prepared-feature copy is owned by GenAI; decode requires no new media.
  features.fill(0);
  std::vector<float> rows{1, 5, 5};
  int state_total = 14;
  bool discriminated_old_order = false;
  bool discriminated_rounded_scale = false;
  for (int step = 0; step < 4; ++step) {
    for (const auto [layer, scale] : {std::pair{0, 1}, std::pair{3, 3}}) {
      for (int kind = 0; kind < 2; ++kind) {
        auto tensor = generator->GetOutput(("present." + std::to_string(layer) +
                                            (kind ? ".ssm_state" : ".conv_state"))
                                               .c_str());
        Check(tensor->Type() == OgaElementType_float32, "State dtype");
        const auto shape = kind ? std::vector<int64_t>{1, 2, 2, 2}
                                : std::vector<int64_t>{1, 2, 3};
        Check(tensor->Shape() == shape, "State shape");
        for (int i = 0; i < (kind ? 8 : 6); ++i)
          Check(Read(*tensor, i) == (scale + kind) * state_total, "Carried state mismatch");
      }
    }
    for (int kind = 0; kind < 2; ++kind) {
      auto kv = generator->GetOutput(kind ? "present.2.value" : "present.2.key");
      Check(kv->Type() == OgaElementType_float32, "KV dtype");
      Check(kv->Shape() == std::vector<int64_t>({1, 1, int64_t(rows.size()), 1}), "KV shape");
      for (size_t i = 0; i < rows.size(); ++i)
        Check(Read(*kv, i) == (kind + 1) * rows[i], "Carried KV mismatch");
    }
    auto logits = generator->GetOutput("logits");
    auto native = generator->GetOutput("native_logits");
    Check(logits->Shape() == std::vector<int64_t>({1, 1, 16}), "Logit shape");
    Check(native->Type() == OgaElementType_bfloat16, "Native output dtype");
    for (int i = 0; i < 16; ++i) {
      const float bias = i == 0 ? -1.25f : i == 1 ? 0.375f
                                       : i == 2   ? 0.625f
                                                  : 0;
      const float code = i == 0 ? (nvfp4 ? .5f : 1.f) : i == 1 ? (nvfp4 ? 1.f : 9.f)
                                                    : i == 2   ? (nvfp4 ? -1.f : -9.f)
                                                               : 0;
      // Independent scalar source-order oracle: native local weights, FP32 dot,
      // global epilogue (A16 only), native output, then native bias addition.
      const float local = nvfp4 ? 1.625f : 0.0999755859375f;  // Exact FP16 .1.
      const float weight = Bfloat(local * code);
      float dot = 0;
      for (int k = 0; k < (nvfp4 ? 16 : 32); ++k) dot += state_total * weight;
      const float expected = Bfloat(Bfloat(dot * (nvfp4 ? .333f : 1.f)) + bias);
      Check(Read(*logits, i) == expected && Read(*native, i) == expected,
            "Full native logit mismatch at step " + std::to_string(step) + " column " + std::to_string(i));
      const float old_weight = nvfp4 ? Bfloat(local * code * .333f) : local * code;
      const float old = Bfloat(Bfloat((nvfp4 ? 16 : 32) * state_total * old_weight) + bias);
      discriminated_old_order |= old != expected;
      const float early_scale = Bfloat(Bfloat(32 * state_total * Bfloat(Bfloat(local) * code)) + bias);
      discriminated_rounded_scale |= early_scale != expected;
    }
    generator->GenerateNextToken();
    Check(generator->GetSequenceCount(0) == size_t(4 + step), "Sequence length");
    Check(generator->GetSequence(0).back() == 1, "Native selected token");
    if (step < 3) {
      generator->GetLogits();
      rows.push_back(float(1 + rows.size()));
      state_total += int(rows.back()) + int(rows.size());
    }
  }
  Check(discriminated_old_order, "Fixture failed to discriminate old rounding");
  if (!nvfp4) Check(discriminated_rounded_scale, "Fixture failed to discriminate prematurely rounded Q8 scales");
  std::cout << (nvfp4 ? "NVFP4A16" : "Q8_0")
            << ": PASS BF16 prepared features + native bias, prefill + 3 decode, full logits/all sparse states; old ordering discriminated\n";
}
}  // namespace

int main(int argc, char** argv) {
  Check(argc == 2, "Usage: native_quantized_cache_probe <cache-dispatch-root>");
  for (const bool nvfp4 : {false, true})
    Probe(std::string{argv[1]} + (nvfp4 ? "/native-nvfp4-bf16" : "/native-q8-bf16"), nvfp4);
  std::ifstream maps{"/proc/self/maps"};
  for (std::string line; std::getline(maps, line);)
    if (line.find("libonnxruntime") != std::string::npos) std::cout << line << '\n';
  OgaShutdown();
}
