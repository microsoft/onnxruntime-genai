#include <iostream>
#include "ort_genai_c.h"

int main() {
  std::cout << "-------------" << std::endl;
  std::cout << "Hello, Phi-2!" << std::endl;
  std::cout << "-------------" << std::endl;

  auto model = OgaModel::Create("phi-2");
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

  return 0;
}
