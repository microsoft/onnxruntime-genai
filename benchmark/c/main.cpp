#include <iostream>

#include "ort_genai.h"

constexpr const char* kModelPath = "C:/Users/edch/models/phi3-3.8b-int4-cpu";

int main() {
  auto model = OgaModel::Create(kModelPath);

  auto tokenizer = OgaTokenizer::Create(*model);

  auto sequences = OgaSequences::Create();

  const char* prompt = "My perfect Sunday is ";
  tokenizer->Encode(prompt, *sequences);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 200);
  params->SetInputSequences(*sequences);

  auto generator = OgaGenerator::Create(*model, *params);
  while(!generator->IsDone()) {
    generator->ComputeLogits();
    generator->GenerateNextToken();
  }

  auto output_sequence = generator->GetSequence(0);
  auto out_string = tokenizer->Decode(output_sequence);

  std::cout << "output: " << out_string << "\n";

  return 0;
}
