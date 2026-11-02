// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "vibevoice_processor.h"

#include <array>

#include "genai_tokenizer.h"
#include "../../ort_genai_c.h"

namespace Generators {

std::string PrepareVibeVoiceText(std::string_view text) {
  std::string prepared;
  prepared.reserve(text.size() + 1);
  for (size_t i = 0; i < text.size(); ++i) {
    // U+2018/U+2019 become ' and U+201C/U+201D become " (UTF-8 E2 80 98/99/9C/9D).
    if (i + 2 < text.size() && text[i] == '\xE2' && text[i + 1] == '\x80') {
      const auto third = static_cast<unsigned char>(text[i + 2]);
      if (third == 0x98 || third == 0x99 || third == 0x9C || third == 0x9D) {
        prepared += third <= 0x99 ? '\'' : '"';
        i += 2;
        continue;
      }
    }
    // The tokenizer reads a C string, so a NUL would end the text early.
    prepared += text[i] == '\0' ? ' ' : text[i];
  }

  constexpr std::string_view kSpace = " \t\r\n\f\v";
  const auto first = prepared.find_first_not_of(kSpace);
  if (first == std::string::npos)
    return {};
  return prepared.substr(first, prepared.find_last_not_of(kSpace) + 1 - first) + "\n";
}

VibeVoiceProcessor::VibeVoiceProcessor(Config& /*config*/, const SessionInfo& /*session_info*/) {}

void VibeVoiceProcessor::ConfigureTokenizer(Tokenizer& tokenizer) const {
  const char* keys[] = {"literal"};
  const char* values[] = {"true"};
  tokenizer.UpdateOptions(keys, values, 1);
}

std::unique_ptr<NamedTensors> VibeVoiceProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  if (payload.images || payload.audios)
    throw std::runtime_error("VibeVoice reads text only; pass no images or audios");
  if (payload.prompts.size() > 1)
    throw std::runtime_error("VibeVoice processes one text at a time");
  if (!payload.prompts.empty() && payload.prompts[0] == nullptr)
    throw std::runtime_error("VibeVoice text must not be null");

  const auto text = PrepareVibeVoiceText(payload.prompts.empty() ? std::string_view{payload.prompt}
                                                                 : std::string_view{payload.prompts[0]});
  const auto ids = text.empty() ? std::vector<int32_t>{} : tokenizer.Encode(text.c_str());
  auto value = OrtValue::CreateTensor<int64_t>(Ort::Allocator::GetWithDefaultOptions(),
                                               std::array<int64_t, 2>{1, static_cast<int64_t>(ids.size())});
  std::copy(ids.begin(), ids.end(), value->GetTensorMutableData<int64_t>());

  auto named_tensors = std::make_unique<NamedTensors>();
  named_tensors->emplace(OgaInputName_TextIds, std::make_shared<Tensor>(std::move(value)));
  return named_tensors;
}

}  // namespace Generators
