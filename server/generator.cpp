// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator.h"

namespace Generators::Server {

Generator::Generator(const std::string& model_path)
    : model(OgaModel::Create(model_path.c_str())), tokenizer(OgaTokenizer::Create(*model)) {}

std::string Generator::Generate(const std::string& prompt, int max_length) {
  auto sequences = OgaSequences::Create();
  tokenizer->Encode(prompt.c_str(), *sequences);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetInputSequences(*sequences);

  auto output_sequences = model->Generate(*params);
  return std::string(tokenizer->Decode(output_sequences->Get(0)));
}

}  // namespace Generators::Server