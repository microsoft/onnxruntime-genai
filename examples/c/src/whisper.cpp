// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <string>
#include <fstream>
#include <memory>

#include "ort_genai.h"

bool FileExists(const char* path) {
  return static_cast<bool>(std::ifstream(path));
}

std::string trim(const std::string& str) {
  const size_t first = str.find_first_not_of(' ');
  if (std::string::npos == first) {
    return str;
  }
  const size_t last = str.find_last_not_of(' ');
  return str.substr(first, (last - first + 1));
}

// C++ API Example

void CXX_API(const char* model_path, int32_t num_beams) {
  std::cout << "Creating model..." << std::endl;
  auto model = OgaModel::Create(model_path);
  std::cout << "Creating multimodal processor..." << std::endl;
  auto processor = OgaMultiModalProcessor::Create(*model);
  std::cout << "Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);

  while (true) {
    std::string audio_paths_str;
    std::cout << "Audio Paths (comma separated):" << std::endl;
    std::getline(std::cin, audio_paths_str);
    std::unique_ptr<OgaAudios> audios;
    std::vector<std::string> audio_paths;
    for (size_t start = 0, end = 0; end < audio_paths_str.size(); start = end + 1) {
      end = audio_paths_str.find(',', start);
      audio_paths.push_back(trim(audio_paths_str.substr(start, end - start)));
    }
    if (audio_paths.empty()) {
      throw std::runtime_error("No audio file provided.");
    } else {
      std::cout << "Loading audios..." << std::endl;
      for (const auto& audio_path : audio_paths) {
        if (!FileExists(audio_path.c_str())) {
          throw std::runtime_error(std::string("Audio file not found: ") + audio_path);
        }
      }
      std::vector<const char*> audio_paths_c;
      for (const auto& audio_path : audio_paths) audio_paths_c.push_back(audio_path.c_str());
      audios = OgaAudios::Load(audio_paths_c);
    }

    std::cout << "Processing audio..." << std::endl;
    auto mel = processor->ProcessAudios(audios.get());
    const std::vector<const char*> prompt_tokens = {"<|startoftranscript|>", "<|en|>", "<|transcribe|>",
                                                    "<|notimestamps|>"};
    auto input_ids = OgaSequences::Create();
    const size_t batch_size = audio_paths.size();
    for (size_t i = 0; i < batch_size; ++i) {
      for (const auto& token : prompt_tokens) {
        input_ids->Append(tokenizer->ToTokenId(token), i);
      }
    }

    std::cout << "Generating response..." << std::endl;
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 256);
    params->SetSearchOptionBool("do_sample", false);
    params->SetSearchOption("num_beams", num_beams);
    params->SetSearchOption("num_return_sequences", num_beams);
    params->SetInputs(*mel);
    params->SetInputSequences(*input_ids);

    auto generator = OgaGenerator::Create(*model, *params);

    while (!generator->IsDone()) {
      generator->ComputeLogits();
      generator->GenerateNextToken();
    }

    for (size_t i = 0; i < static_cast<size_t>(num_beams * batch_size); ++i) {
      std::cout << "Transcription:" << std::endl;
      std::cout << "    batch " << i / num_beams << ", beam " << i % num_beams << ":";
      const auto num_tokens = generator->GetSequenceCount(i);
      const auto tokens = generator->GetSequenceData(i);
      std::cout << processor->Decode(tokens, num_tokens) << std::endl;
    }

    for (int i = 0; i < 3; ++i)
      std::cout << std::endl;
  }
}

// C API Example

void CheckResult(OgaResult* result) {
  if (result) {
    std::string string = OgaResultGetError(result);
    OgaDestroyResult(result);
    throw std::runtime_error(string);
  }
}

void C_API(const char* model_path, int32_t num_beams) {
  OgaModel* model;
  std::cout << "Creating model..." << std::endl;
  CheckResult(OgaCreateModel(model_path, &model));

  OgaMultiModalProcessor* processor;
  std::cout << "Creating multimodal processor..." << std::endl;
  CheckResult(OgaCreateMultiModalProcessor(model, &processor));
  OgaTokenizer* tokenizer;
  std::cout << "Creating tokenizer..." << std::endl;
  CheckResult(OgaCreateTokenizer(model, &tokenizer));

  while (true) {
    std::string audio_paths_str;
    std::cout << "Audio Paths (comma separated):" << std::endl;
    std::getline(std::cin, audio_paths_str);
    OgaAudios* audios = nullptr;
    std::vector<std::string> audio_paths;
    for (size_t start = 0, end = 0; end < audio_paths_str.size(); start = end + 1) {
      end = audio_paths_str.find(',', start);
      audio_paths.push_back(trim(audio_paths_str.substr(start, end - start)));
    }
    if (audio_paths.empty()) {
      throw std::runtime_error("No audio file provided.");
    } else {
      std::cout << "Loading audios..." << std::endl;
      for (const auto& audio_path : audio_paths) {
        if (!FileExists(audio_path.c_str())) {
          throw std::runtime_error(std::string("Audio file not found: ") + audio_path);
        }
        std::vector<const char*> audio_paths_c;
        for (const auto& audio_path : audio_paths) audio_paths_c.push_back(audio_path.c_str());
        OgaStringArray* audio_paths_string_array;
        CheckResult(OgaCreateStringArrayFromStrings(audio_paths_c.data(), audio_paths_c.size(), &audio_paths_string_array));
        CheckResult(OgaLoadAudios(audio_paths_string_array, &audios));
        OgaDestroyStringArray(audio_paths_string_array);
      }
    }

    std::cout << "Processing audio..." << std::endl;
    OgaNamedTensors* mel;
    CheckResult(OgaProcessorProcessAudios(processor, audios, &mel));
    const std::vector<const char*> prompt_tokens = {"<|startoftranscript|>", "<|en|>", "<|transcribe|>",
                                                    "<|notimestamps|>"};
    OgaSequences* input_ids;
    CheckResult(OgaCreateSequences(&input_ids));
    const size_t batch_size = audio_paths.size();
    for (size_t i = 0; i < batch_size; ++i) {
      for (const auto& token : prompt_tokens) {
        int32_t token_id;
        CheckResult(OgaTokenizerToTokenId(tokenizer, token, &token_id));
        CheckResult(OgaAppendTokenToSequence(token_id, input_ids, i));
      }
    }

    std::cout << "Generating response..." << std::endl;
    OgaGeneratorParams* params;
    CheckResult(OgaCreateGeneratorParams(model, &params));
    CheckResult(OgaGeneratorParamsSetSearchNumber(params, "max_length", 256));
    CheckResult(OgaGeneratorParamsSetSearchBool(params, "do_sample", false));
    CheckResult(OgaGeneratorParamsSetSearchNumber(params, "num_beams", num_beams));
    CheckResult(OgaGeneratorParamsSetSearchNumber(params, "num_return_sequences", num_beams));
    CheckResult(OgaGeneratorParamsSetInputs(params, mel));
    CheckResult(OgaGeneratorParamsSetInputSequences(params, input_ids));

    OgaGenerator* generator;
    CheckResult(OgaCreateGenerator(model, params, &generator));

    while (!OgaGenerator_IsDone(generator)) {
      CheckResult(OgaGenerator_ComputeLogits(generator));
      CheckResult(OgaGenerator_GenerateNextToken(generator));
    }

    for (size_t i = 0; i < static_cast<size_t>(num_beams * batch_size); ++i) {
      std::cout << "Transcription:" << std::endl;
      std::cout << "    batch " << i / num_beams << ", beam " << i % num_beams << ":";
      const int32_t num_tokens = OgaGenerator_GetSequenceCount(generator, i);
      const int32_t* tokens = OgaGenerator_GetSequenceData(generator, i);

      const char* str;
      CheckResult(OgaProcessorDecode(processor, tokens, num_tokens, &str));
      std::cout << str << std::endl;
    }

    for (int i = 0; i < 3; ++i)
      std::cout << std::endl;

    OgaDestroyGenerator(generator);
    OgaDestroyGeneratorParams(params);
    OgaDestroySequences(input_ids);
    OgaDestroyNamedTensors(mel);
    OgaDestroyAudios(audios);
  }

  OgaDestroyTokenizer(tokenizer);
  OgaDestroyMultiModalProcessor(processor);
  OgaDestroyModel(model);
}

static void print_usage(int /*argc*/, char** argv) {
  std::cerr << "usage: " << argv[0] << " <model_path>"
            << "<num_beams>" << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    print_usage(argc, argv);
    return -1;
  }

  std::cout << "---------------" << std::endl;
  std::cout << "Hello, Whisper!" << std::endl;
  std::cout << "---------------" << std::endl;

#ifdef USE_CXX
  std::cout << "C++ API" << std::endl;
  CXX_API(argv[1], std::stoi(argv[2]));
#else
  std::cout << "C API" << std::endl;
  C_API(argv[1], std::stoi(argv[2]));
#endif

  return 0;
}