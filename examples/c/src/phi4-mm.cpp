// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <string>
#include <fstream>
#include <memory>
#include "common.h"
#include "ort_genai.h"

// C++ API Example

void CXX_API(const char* model_path, const char* execution_provider) {
  std::cout << "Creating config..." << std::endl;
  auto config = OgaConfig::Create(model_path);

  std::string provider(execution_provider);
  append_provider(*config, provider);

  std::cout << "Creating model..." << std::endl;
  auto model = OgaModel::Create(*config);

  std::cout << "Creating multimodal processor..." << std::endl;
  auto processor = OgaMultiModalProcessor::Create(*model);

  auto tokenizer_stream = OgaTokenizerStream::Create(*processor);

  while (true) {
    // Get images
    std::string image_paths_str;
    std::cout << "Image Path (comma separated; leave empty if no image):" << std::endl;
    std::getline(std::cin, image_paths_str);
    std::unique_ptr<OgaImages> images;
    std::vector<std::string> image_paths;
    for (size_t start = 0, end = 0; end < image_paths_str.size(); start = end + 1) {
      end = image_paths_str.find(',', start);
      image_paths.push_back(trim(image_paths_str.substr(start, end - start)));
    }
    if (image_paths.empty()) {
      std::cout << "No image provided" << std::endl;
    } else {
      std::cout << "Loading images..." << std::endl;
      for (const auto& image_path : image_paths) {
        if (!FileExists(image_path.c_str())) {
          throw std::runtime_error(std::string("Image file not found: ") + image_path);
        }
      }
      std::vector<const char*> image_paths_c;
      for (const auto& image_path : image_paths) image_paths_c.push_back(image_path.c_str());
      images = OgaImages::Load(image_paths_c);
    }

    // Get audios
    std::string audio_paths_str;
    std::cout << "Audio Path (comma separated; leave empty if no audio):" << std::endl;
    std::getline(std::cin, audio_paths_str);
    std::unique_ptr<OgaAudios> audios;
    std::vector<std::string> audio_paths;
    for (size_t start = 0, end = 0; end < audio_paths_str.size(); start = end + 1) {
      end = audio_paths_str.find(',', start);
      audio_paths.push_back(trim(audio_paths_str.substr(start, end - start)));
    }
    if (audio_paths.empty()) {
      std::cout << "No audio provided" << std::endl;
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

    std::string text;
    std::cout << "Prompt: " << std::endl;
    std::getline(std::cin, text);
    std::string prompt = "<|user|>\n";
    if (images) {
      for (size_t i = 0; i < image_paths.size(); ++i)
        prompt += "<|image_" + std::to_string(i + 1) + "|>\n";
    }
    if (audios) {
      for (size_t i = 0; i < audio_paths.size(); ++i)
        prompt += "<|audio_" + std::to_string(i + 1) + "|>\n";
    }
    prompt += text + "<|end|>\n<|assistant|>\n";

    std::cout << "Processing images, audios, and prompt..." << std::endl;
    auto input_tensors = processor->ProcessImagesAndAudios(prompt.c_str(), images.get(), audios.get());

    std::cout << "Generating response..." << std::endl;
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 7680);
    params->SetInputs(*input_tensors);

    auto generator = OgaGenerator::Create(*model, *params);

    while (!generator->IsDone()) {
      generator->GenerateNextToken();

      const auto num_tokens = generator->GetSequenceCount(0);
      const auto new_token = generator->GetSequenceData(0)[num_tokens - 1];
      std::cout << tokenizer_stream->Decode(new_token) << std::flush;
    }

    for (int i = 0; i < 3; ++i)
      std::cout << std::endl;
  }
}

int main(int argc, char** argv) {
  std::string model_path, ep;
  if (!parse_args(argc, argv, model_path, ep)) {
    return -1;
  }

  std::cout << "--------------------" << std::endl;
  std::cout << "Hello, Phi-4-Multimodal!" << std::endl;
  std::cout << "--------------------" << std::endl;
  CXX_API(model_path.c_str(), ep.c_str());

  return 0;
}