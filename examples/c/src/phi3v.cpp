#include <iostream>
#include <string>
#include <fstream>

#include "ort_genai_c.h"

// C API Example

void CheckResult(OgaResult* result) {
  if (result) {
    std::string string = OgaResultGetError(result);
    OgaDestroyResult(result);
    throw std::runtime_error(string);
  }
}

bool FileExists(const char* path) {
  return static_cast<bool>(std::ifstream(path));
}

void C_API(const char* model_path) {
  OgaModel* model;
  std::cout << "Creating model..." << std::endl;
  CheckResult(OgaCreateModel(model_path, &model));

  OgaMultiModalProcessor* processor;
  std::cout << "Creating multimodal processor..." << std::endl;
  CheckResult(OgaCreateMultiModalProcessor(model, &processor));

  OgaTokenizerStream* tokenizer_stream;
  CheckResult(OgaCreateTokenizerStreamFromProcessor(processor, &tokenizer_stream));

  while (true) {
    std::string image_path;
    std::cout << "Image Path (leave empty if no image):" << std::endl;
    std::getline(std::cin, image_path);
    OgaImages* images = nullptr;
    if (image_path.empty()) {
      std::cout << "No image provided" << std::endl;
    } else {
      std::cout << "Loading image..." << std::endl;
      if (!FileExists(image_path.c_str())) {
        throw std::runtime_error(std::string("Image file not found: ") + image_path);
      }
      CheckResult(OgaLoadImage(image_path.c_str(), &images));
    }

    std::string text;
    std::cout << "Prompt: " << std::endl;
    std::getline(std::cin, text);
    const std::string prompt = "<|user|>\n<|image_1|>\n" + text + "<|end|>\n<|assistant|>\n";

    std::cout << "Processing image and prompt..." << std::endl;
    OgaNamedTensors* input_tensors;
    CheckResult(OgaProcessorProcessImages(processor, prompt.c_str(), images, &input_tensors));

    std::cout << "Generating response..." << std::endl;
    OgaGeneratorParams* params;
    CheckResult(OgaCreateGeneratorParams(model, &params));
    CheckResult(OgaGeneratorParamsSetSearchNumber(params, "max_length", 3072));
    CheckResult(OgaGeneratorParamsSetInputs(params, input_tensors));

    OgaGenerator* generator;
    CheckResult(OgaCreateGenerator(model, params, &generator));

    while (!OgaGenerator_IsDone(generator)) {
      CheckResult(OgaGenerator_ComputeLogits(generator));
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
  std::cerr << "usage: " << argv[0] << " <model_path>" << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    print_usage(argc, argv);
    return -1;
  }

  std::cout << "--------------------" << std::endl;
  std::cout << "Hello, Phi-3-Vision!" << std::endl;
  std::cout << "--------------------" << std::endl;

  std::cout << "C API" << std::endl;
  C_API(argv[1]);

  return 0;
}