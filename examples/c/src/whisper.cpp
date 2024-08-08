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

// C++ API Example

void CXX_API(const char* model_path, int32_t num_beams) {
  std::cout << "Creating model..." << std::endl;
  auto model = OgaModel::Create(model_path);
  std::cout << "Creating multimodal processor..." << std::endl;
  auto processor = OgaMultiModalProcessor::Create(*model);

  while (true) {
    std::string audio_path;
    std::cout << "Audio Path:" << std::endl;
    std::getline(std::cin, audio_path);
    std::cout << "Loading audio..." << std::endl;
    if (audio_path.empty()) {
      throw std::runtime_error("Audio file not provided.");
    } else if (!FileExists(audio_path.c_str())) {
      throw std::runtime_error(std::string("Audio file not found: ") + audio_path);
    }
    std::unique_ptr<OgaAudios> audios = OgaAudios::Load(audio_path.c_str());

    std::cout << "Processing audio..." << std::endl;
    auto input_tensors = processor->ProcessAudios(audios.get(), "english", "transcribe", 1);

    std::cout << "Generating response..." << std::endl;
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 256);
    params->SetSearchOption("num_beams", num_beams);
    params->SetSearchOption("num_return_sequences", 4);
    params->SetInputs(*input_tensors);

    auto generator = OgaGenerator::Create(*model, *params);

    while (!generator->IsDone()) {
      generator->ComputeLogits();
      generator->GenerateNextToken();
    }

    std::cout << "Transcription:" << std::endl;
    for (size_t beam = 0; beam < static_cast<size_t>(num_beams); ++beam) {
      std::cout << "    Beam " << beam << ":";
      const auto num_tokens = generator->GetSequenceCount(beam);
      const auto tokens = generator->GetSequenceData(beam);
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

  while (true) {
    std::string audio_path;
    std::cout << "Audio Path:" << std::endl;
    std::getline(std::cin, audio_path);
    std::cout << "Loading audio..." << std::endl;
    if (audio_path.empty()) {
      throw std::runtime_error("Audio file not provided.");
    } else if (!FileExists(audio_path.c_str())) {
      throw std::runtime_error(std::string("Audio file not found: ") + audio_path);
    }
    OgaAudios* audios = nullptr;
    CheckResult(OgaLoadAudio(audio_path.c_str(), &audios));

    std::cout << "Processing audio..." << std::endl;
    OgaNamedTensors* input_tensors;
    CheckResult(OgaProcessorProcessAudios(processor, audios, &input_tensors));

    std::cout << "Generating response..." << std::endl;
    OgaGeneratorParams* params;
    CheckResult(OgaCreateGeneratorParams(model, &params));
    CheckResult(OgaGeneratorParamsSetSearchNumber(params, "max_length", 256));
    CheckResult(OgaGeneratorParamsSetSearchNumber(params, "num_beams", num_beams));
    CheckResult(OgaGeneratorParamsSetSearchNumber(params, "num_return_sequences", num_beams));
    CheckResult(OgaGeneratorParamsSetInputs(params, input_tensors));
    const std::array<int32_t, 3> input_ids = {50258, 50259, 50359};
    CheckResult(OgaGeneratorParamsSetInputIDs(params, input_ids.data(), input_ids.size(), input_ids.size(), 1));

    OgaGenerator* generator;
    CheckResult(OgaCreateGenerator(model, params, &generator));

    while (!OgaGenerator_IsDone(generator)) {
      CheckResult(OgaGenerator_ComputeLogits(generator));
      CheckResult(OgaGenerator_GenerateNextToken(generator));
    }

    std::cout << "Transcription:" << std::endl;
    for (size_t beam = 0; beam < static_cast<size_t>(num_beams); ++beam) {
      std::cout << "    Beam " << beam << ":";
      const int32_t num_tokens = OgaGenerator_GetSequenceCount(generator, beam);
      const int32_t* tokens = OgaGenerator_GetSequenceData(generator, beam);

      const char* str;
      CheckResult(OgaProcessorDecode(processor, tokens, num_tokens, &str));
      std::cout << str << std::endl;
    }

    for (int i = 0; i < 3; ++i)
      std::cout << std::endl;

    OgaDestroyGeneratorParams(params);
    OgaDestroyNamedTensors(input_tensors);
    OgaDestroyAudios(audios);
  }

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