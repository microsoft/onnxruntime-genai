#include <iostream>
#include <span>
#include "ort_genai.h"

// C++ API Example

void CXX_API(const char* model_path) {
  auto model = OgaModel::Create(model_path);
  auto tokenizer = OgaTokenizer::Create(*model);

  const char* prompt = "def is_prime(num):";
  std::cout << "Prompt: " << std::endl << prompt << std::endl;

  auto sequences = OgaSequences::Create();
  tokenizer->Encode(prompt, *sequences);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 200);
  params->SetInputSequences(*sequences);

  auto output_sequences = model->Generate(*params);
  auto out_string = tokenizer->Decode(output_sequences->Get(0));

  std::cout << "Output: " << std::endl << out_string << std::endl;
}

// C API Example

void CheckResult(OgaResult* result) {
  if (result) {
    std::string string=OgaResultGetError(result);
    OgaDestroyResult(result);
    throw std::runtime_error(string);
  }
}

void C_API(const char* model_path) {
  OgaModel* model;
  OgaCreateModel(model_path, &model);

  OgaTokenizer* tokenizer;
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


  std::cout << "-------------" << std::endl;
  std::cout << "Hello, Phi-2!" << std::endl;
  std::cout << "-------------" << std::endl;

  std::cout << "C++ API" << std::endl;
  CXX_API(argv[1]);

  std::cout << "C API" << std::endl;
  C_API(argv[1]);

  return 0;
}