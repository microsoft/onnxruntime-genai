// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_genai.h"

#include <string>
#include <memory>

namespace Generators::Server {

class Generator {
 private:
  std::unique_ptr<OgaModel> model;
  std::unique_ptr<OgaTokenizer> tokenizer;

 public:
  Generator(const std::string& model_path);

  std::string Generate(const std::string& prompt, int max_length);
};

}  // namespace Generators::Server