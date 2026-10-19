// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// C++ API smoke test of a qwen38_flash_graph package: the model is used through the unmodified public C++ API
// (ort_genai.h): OgaModel / OgaTokenizer / OgaGenerator. Built by build_smoke_cpp.sh next to this file.
//
// usage: smoke_cpp <package dir> <provider .so> [new tokens]
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "ort_genai.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: smoke_cpp <package dir> <provider .so> [new tokens]" << std::endl;
    return 2;
  }
  const int new_tokens = argc > 3 ? std::atoi(argv[3]) : 64;
  try {
    OgaRegisterExecutionProviderLibrary("CUDAExecutionProvider", argv[2]);
    auto model = OgaModel::Create(argv[1]);
    auto tokenizer = OgaTokenizer::Create(*model);
    auto stream = OgaTokenizerStream::Create(*tokenizer);

    const char* messages = R"([{"role": "user", "content": "Name the three primary colors in one short sentence."}])";
    auto prompt = tokenizer->ApplyChatTemplate(nullptr, messages, nullptr, true);
    auto sequences = OgaSequences::Create();
    tokenizer->Encode(prompt, *sequences);

    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", static_cast<double>(sequences->SequenceCount(0)) + new_tokens);
    params->SetSearchOptionBool("do_sample", false);
    auto generator = OgaGenerator::Create(*model, *params);
    generator->AppendTokenSequences(*sequences);

    int count = 0;
    const auto start = std::chrono::steady_clock::now();
    auto first = start;
    while (!generator->IsDone()) {
      generator->GenerateNextToken();
      if (count++ == 0) first = std::chrono::steady_clock::now();
      const auto next = generator->GetNextTokens();
      std::cout << stream->Decode(next[0]) << std::flush;
    }
    const auto end = std::chrono::steady_clock::now();
    std::cout << std::endl;
    const double decode_s = std::chrono::duration<double>(end - first).count();
    std::cout << count << " tokens, first token after " << std::chrono::duration<double>(first - start).count()
              << " s, decode " << (count - 1) / decode_s << " tokens/s" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << std::endl;
    return 1;
  }
}
