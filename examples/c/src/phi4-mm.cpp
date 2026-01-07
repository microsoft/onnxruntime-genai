// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <string>
#include <fstream>
#include <memory>
#include "common.h"
#include "ort_genai.h"

// C++ API Example

void CXX_API(
  GeneratorParamsArgs& generator_params_args,
  const std::string& model_path,
  const std::string& ep,
  const std::string& system_prompt,
  bool verbose,
  bool interactive
) {
  if (verbose) std::cout << "Creating config..." << std::endl;
  std::unordered_map<std::string, std::string> ep_options;
  auto config = GetConfig(model_path, ep, ep_options, generator_params_args);

  if (verbose) std::cout << "Creating model..." << std::endl;
  auto model = OgaModel::Create(*config);

  if (verbose) std::cout << "Creating multimodal processor..." << std::endl;
  auto processor = OgaMultiModalProcessor::Create(*model);

  if (verbose) std::cout << "Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);
  auto stream = OgaTokenizerStream::Create(*processor);

  while (true) {
    // Get images
    std::string image_paths_str;
    std::cout << "Image Path (comma separated; leave empty if no image):" << std::endl;
    std::getline(std::cin, image_paths_str);
    std::unique_ptr<OgaImages> images;
    std::vector<std::string> image_paths;
    for (size_t start = 0, end = 0; end < image_paths_str.size(); start = end + 1) {
      end = image_paths_str.find(',', start);
      image_paths.push_back(Trim(image_paths_str.substr(start, end - start)));
    }
    if (image_paths.empty()) {
      std::cout << "No image provided" << std::endl;
    } else {
      std::cout << "Loading images..." << std::endl;
      for (const auto& image_path : image_paths) {
        std::filesystem::path p(image_path);
        if (!std::filesystem::exists(p)) {
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
      audio_paths.push_back(Trim(audio_paths_str.substr(start, end - start)));
    }
    if (audio_paths.empty()) {
      std::cout << "No audio provided" << std::endl;
    } else {
      std::cout << "Loading audios..." << std::endl;
      for (const auto& audio_path : audio_paths) {
        std::filesystem::path p(audio_path);
        if (!std::filesystem::exists(p)) {
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

    // Construct messages string with special tokens for ApplyChatTemplate
    std::string content;
    for (size_t i = 0; i < image_paths.size(); ++i)
      content += "<|image_" + std::to_string(i + 1) + "|>\\n";
    for (size_t i = 0; i < audio_paths.size(); ++i)
      content += "<|audio_" + std::to_string(i + 1) + "|>\\n";
    content += text;

    const std::string messages = R"([{"role": "user", "content": ")" + content + R"("}])";

    std::string prompt = std::string(tokenizer->ApplyChatTemplate("", messages.c_str(), "", true));

    std::cout << "Processing images, audios, and prompt..." << std::endl;
    auto input_tensors = processor->ProcessImagesAndAudios(prompt.c_str(), images.get(), audios.get());

    std::cout << "Generating response..." << std::endl;
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 7680);

    auto generator = OgaGenerator::Create(*model, *params);
    generator->SetInputs(*input_tensors);

    while (true) {
      generator->GenerateNextToken();

      if (generator->IsDone()) {
        break;
      }

      const auto num_tokens = generator->GetSequenceCount(0);
      const auto new_token = generator->GetSequenceData(0)[num_tokens - 1];
      std::cout << stream->Decode(new_token) << std::flush;
    }

    std::cout << "\n\n" << std::endl;
  }
}

int main(int argc, char** argv) {
  // Get command-line args
  GeneratorParamsArgs generator_params_args;
  GuidanceArgs guidance_args;
  std::string model_path, ep = "follow_config", system_prompt = "You are a helpful AI assistant.";
  bool verbose = false, interactive = true, rewind = false;
  if (!ParseArgs(argc, argv, generator_params_args, guidance_args, model_path, ep, system_prompt, verbose, interactive, rewind)) {
    return -1;
  }

  // Responsible for cleaning up the library during shutdown
  OgaHandle handle;

  std::cout << "--------------------" << std::endl;
  std::cout << "Hello, Phi-4-Multimodal!" << std::endl;
  std::cout << "--------------------" << std::endl;
  
  std::cout << "Model path: " << model_path << std::endl;
  std::cout << "Execution provider: " << ep << std::endl;
  std::cout << "System prompt: " << system_prompt << std::endl;
  std::cout << "Verbose: " << verbose << std::endl;
  std::cout << "Interactive: " << interactive << std::endl;
  std::cout << "--------------------------" << std::endl;
  std::cout << std::endl;

  try {
    CXX_API(generator_params_args, model_path, ep, system_prompt, verbose, interactive);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}