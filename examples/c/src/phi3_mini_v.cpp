#include <iostream>

#include "ort_genai_c.h"

// C API Example

void CheckResult(OgaResult* result) {
  if (result) {
    std::string string = OgaResultGetError(result);
    OgaDestroyResult(result);
    throw std::runtime_error(string);
  }
}

void C_API(const char* model_path, const char* image_path, const char* prompt) {
  OgaModel* model;
  std::cout << "Creating model..." << std::endl;
  CheckResult(OgaCreateModel(model_path, &model));

  OgaMultiModalProcessor* processor;
  std::cout << "Creating multimodal processor..." << std::endl;
  CheckResult(OgaCreateMultiModalProcessor(model, &processor));

  // Example prompt:
  // "<|user|>\n<|image_1|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n";
  std::cout << "Prompt: " << std::endl
            << prompt << std::endl;

  OgaImages* images;
  CheckResult(OgaLoadImage(image_path, &images));

  OgaNamedTensors* input_tensors;
  CheckResult(OgaProcessorProcessImages(processor, prompt, images, &input_tensors));

  OgaGeneratorParams* params;
  CheckResult(OgaCreateGeneratorParams(model, &params));
  CheckResult(OgaGeneratorParamsSetSearchNumber(params, "max_length", 3072));
  CheckResult(OgaGeneratorParamsSetInputs(params, input_tensors));

  OgaSequences* output_sequences;
  CheckResult(OgaGenerate(model, params, &output_sequences));

  size_t sequence_length = OgaSequencesGetSequenceCount(output_sequences, 0);
  const int32_t* sequence = OgaSequencesGetSequenceData(output_sequences, 0);

  const char* out_string;
  CheckResult(OgaProcessorDecode(processor, sequence, sequence_length, &out_string));

  std::cout << "Output: " << std::endl
            << out_string << std::endl;

  OgaDestroyString(out_string);
  OgaDestroySequences(output_sequences);
  OgaDestroyGeneratorParams(params);
  OgaDestroyNamedTensors(input_tensors);
  OgaDestroyImages(images);
  OgaDestroyMultiModalProcessor(processor);
  OgaDestroyModel(model);
}

static void print_usage(int /*argc*/, char** argv) {
  std::cerr << "usage: " << argv[0] << " <model_path> <image_path> <prompt>" << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    print_usage(argc, argv);
    return -1;
  }

  std::cout << "--------------------" << std::endl;
  std::cout << "Hello, Phi-3-Mini-V!" << std::endl;
  std::cout << "--------------------" << std::endl;

  std::cout << "C API" << std::endl;
  C_API(argv[1], argv[2], argv[3]);

  return 0;
}