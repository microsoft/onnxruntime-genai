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

void CXX_API(const char* model_path, const char* execution_provider) {
  std::cout << "Creating config..." << std::endl;
  auto config = OgaConfig::Create(model_path);

  config->ClearProviders();
  std::string provider(execution_provider);
  if (provider.compare("cpu") != 0) {
    config->AppendProvider(execution_provider);
    if (provider.compare("cuda") == 0) {
      config->SetProviderOption(execution_provider, "enable_cuda_graph", "0");
    }
  }

  std::cout << "Creating model..." << std::endl;
  auto model = OgaModel::Create(*config);

  std::cout << "Creating multimodal processor..." << std::endl;
  auto processor = OgaMultiModalProcessor::Create(*model);

  auto tokenizer_stream = OgaTokenizerStream::Create(*processor);

  while (true) {
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

    std::string text;
    std::cout << "Prompt: " << std::endl;
    std::getline(std::cin, text);
    std::string prompt = "<|user|>\n";
    if (images) {
      for (size_t i = 0; i < image_paths.size(); ++i)
        prompt += "<|image_" + std::to_string(i + 1) + "|>\n";
    }
    prompt += text + "<|end|>\n<|assistant|>\n";

    std::cout << "Processing images and prompt..." << std::endl;
    auto input_tensors = processor->ProcessImages(prompt.c_str(), images.get());

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

// C API Example

void CheckResult(OgaResult* result) {
  if (result) {
    std::string string = OgaResultGetError(result);
    OgaDestroyResult(result);
    throw std::runtime_error(string);
  }
}

void C_API(const char* model_path, const char* execution_provider) {
  OgaConfig* config;
  std::cout << "Creating config..." << std::endl;
  CheckResult(OgaCreateConfig(model_path, &config));

  CheckResult(OgaConfigClearProviders(config));
  if (strcmp(execution_provider, "cpu") != 0) {
    CheckResult(OgaConfigAppendProvider(config, execution_provider));
    if (strcmp(execution_provider, "cuda") == 0) {
      CheckResult(OgaConfigSetProviderOption(config, execution_provider, "enable_cuda_graph", "0"));
    }
  }

  OgaModel* model;
  std::cout << "Creating model..." << std::endl;
  CheckResult(OgaCreateModelFromConfig(config, &model));

  OgaMultiModalProcessor* processor;
  std::cout << "Creating multimodal processor..." << std::endl;
  CheckResult(OgaCreateMultiModalProcessor(model, &processor));

  OgaTokenizerStream* tokenizer_stream;
  CheckResult(OgaCreateTokenizerStreamFromProcessor(processor, &tokenizer_stream));

  while (true) {
    std::string image_paths_str;
    std::cout << "Image Path (comma separated; leave empty if no image):" << std::endl;
    std::getline(std::cin, image_paths_str);
    OgaImages* images = nullptr;
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
      OgaStringArray* image_paths_string_array;
      CheckResult(OgaCreateStringArrayFromStrings(image_paths_c.data(), image_paths_c.size(), &image_paths_string_array));
      CheckResult(OgaLoadImages(image_paths_string_array, &images));
      OgaDestroyStringArray(image_paths_string_array);
    }

    std::string text;
    std::cout << "Prompt: " << std::endl;
    std::getline(std::cin, text);
    std::string prompt = "<|user|>\n";
    if (images) {
      for (size_t i = 0; i < image_paths.size(); ++i)
        prompt += "<|image_" + std::to_string(i + 1) + "|>\n";
    }
    prompt += text + "<|end|>\n<|assistant|>\n";

    std::cout << "Processing images and prompt..." << std::endl;
    OgaNamedTensors* input_tensors;
    CheckResult(OgaProcessorProcessImages(processor, prompt.c_str(), images, &input_tensors));

    std::cout << "Generating response..." << std::endl;
    OgaGeneratorParams* params;
    CheckResult(OgaCreateGeneratorParams(model, &params));
    CheckResult(OgaGeneratorParamsSetSearchNumber(params, "max_length", 7680));
    CheckResult(OgaGeneratorParamsSetInputs(params, input_tensors));

    OgaGenerator* generator;
    CheckResult(OgaCreateGenerator(model, params, &generator));

    while (!OgaGenerator_IsDone(generator)) {
      CheckResult(OgaGenerator_GenerateNextToken(generator));

      const int32_t num_tokens = OgaGenerator_GetSequenceCount(generator, 0);
      int32_t new_token = OgaGenerator_GetSequenceData(generator, 0)[num_tokens - 1];
      const char* new_token_string;
      CheckResult(OgaTokenizerStreamDecode(tokenizer_stream, new_token, &new_token_string));
      std::cout << new_token_string << std::flush;
    }

    for (int i = 0; i < 3; ++i)
      std::cout << std::endl;

    OgaDestroyGeneratorParams(params);
    OgaDestroyNamedTensors(input_tensors);
    OgaDestroyImages(images);
  }

  OgaDestroyTokenizerStream(tokenizer_stream);
  OgaDestroyMultiModalProcessor(processor);
  OgaDestroyModel(model);
}

static void print_usage(int /*argc*/, char** argv) {
  std::cerr << "usage: " << argv[0] << std::endl;
  std::cerr << "model_path = " << argv[1] << std::endl;
  std::cerr << "execution_provider = " << argv[2] << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    print_usage(argc, argv);
    return -1;
  }

  std::cout << "--------------------" << std::endl;
  std::cout << "Hello, Phi-3-Vision!" << std::endl;
  std::cout << "--------------------" << std::endl;

#ifdef USE_CXX
  std::cout << "C++ API" << std::endl;
  CXX_API(argv[1], argv[2]);
#else
  std::cout << "C API" << std::endl;
  C_API(argv[1], argv[2]);
#endif

  return 0;
}