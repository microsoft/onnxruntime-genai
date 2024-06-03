#include <iostream>
#include <span>
#include "ort_genai.h"

// C++ API Example

void CXX_API(const char* model_path) {
  std::cout << "Creating model..." << std::endl;
  auto model = OgaModel::Create(model_path);
  std::cout << "Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);

  const char* prompt = "def is_prime(num):";
  std::cout << "Prompt: " << std::endl
            << prompt << std::endl;

  auto sequences = OgaSequences::Create();
  tokenizer->Encode(prompt, *sequences);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 200);
  params->SetInputSequences(*sequences);

  auto output_sequences = model->Generate(*params);
  const auto output_sequence_length = output_sequences->SequenceCount(0);
  const auto* output_sequence_data = output_sequences->SequenceData(0);
  auto out_string = tokenizer->Decode(output_sequence_data, output_sequence_length);

  std::cout << "Output: " << std::endl
            << out_string << std::endl;
}

// C API Example

void CheckResult(OgaResult* result) {
  if (result) {
    std::string string = OgaResultGetError(result);
    OgaDestroyResult(result);
    throw std::runtime_error(string);
  }
}

void C_API(const char* model_path) {
  OgaModel* model;
  std::cout << "Creating model..." << std::endl;
  OgaCreateModel(model_path, &model);

  OgaTokenizer* tokenizer;
  std::cout << "Creating tokenizer..." << std::endl;
  CheckResult(OgaCreateTokenizer(model, &tokenizer));

  const char* prompt = "def is_prime(num):";
  std::cout << "Prompt: " << std::endl
            << prompt << std::endl;

  OgaSequences* sequences;
  CheckResult(OgaCreateSequences(&sequences));
  CheckResult(OgaTokenizerEncode(tokenizer, prompt, sequences));

  OgaGeneratorParams* params;
  CheckResult(OgaCreateGeneratorParams(model, &params));
  CheckResult(OgaGeneratorParamsSetSearchNumber(params, "max_length", 200));
  CheckResult(OgaGeneratorParamsSetInputSequences(params, sequences));

  OgaSequences* output_sequences;
  CheckResult(OgaGenerate(model, params, &output_sequences));

  size_t sequence_length = OgaSequencesGetSequenceCount(output_sequences, 0);
  const int32_t* sequence = OgaSequencesGetSequenceData(output_sequences, 0);

  const char* out_string;
  CheckResult(OgaTokenizerDecode(tokenizer, sequence, sequence_length, &out_string));

  std::cout << "Output: " << std::endl
            << out_string << std::endl;

  OgaDestroyString(out_string);
  OgaDestroySequences(output_sequences);
  OgaDestroyGeneratorParams(params);
  OgaDestroySequences(sequences);
  OgaDestroyTokenizer(tokenizer);
  OgaDestroyModel(model);
}

static void print_usage(int /*argc*/, char** argv) {
  std::cerr << "usage: " << argv[0] << " model_path" << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    print_usage(argc, argv);
    return -1;
  }

  // Responsible for cleaning up the library during shutdown
  OgaHandle handle;

  std::cout << "-------------" << std::endl;
  std::cout << "Hello, Phi-3!" << std::endl;
  std::cout << "-------------" << std::endl;

#ifdef USE_CXX
  std::cout << "C++ API" << std::endl;
  CXX_API(argv[1]);
#else
  std::cout << "C API" << std::endl;
  C_API(argv[1]);
#endif

  return 0;
}