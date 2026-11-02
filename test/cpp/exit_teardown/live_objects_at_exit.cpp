// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// A host may destroy genai objects from an atexit handler that it registered before its first genai
// call. Such a handler runs after genai's own process-exit teardown, so the process exits cleanly
// only if genai keeps its globals while objects are alive. The test passes when the process exits
// with code 0.
#include <cstdio>
#include <cstdlib>

#include "ort_genai_c.h"

namespace {

OgaModel* g_model = nullptr;
OgaTokenizer* g_tokenizer = nullptr;

void DestroyObjects() {
  OgaDestroyTokenizer(g_tokenizer);
  OgaDestroyModel(g_model);
  std::puts("destroyed the model after genai's process-exit teardown");
}

bool Check(OgaResult* result) {
  if (!result)
    return true;
  std::fprintf(stderr, "%s\n", OgaResultGetError(result));
  OgaDestroyResult(result);
  return false;
}

}  // namespace

int main() {
  std::atexit(DestroyObjects);
  if (!Check(OgaCreateModel(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32", &g_model)) ||
      !Check(OgaCreateTokenizer(g_model, &g_tokenizer)))
    return 1;
  return 0;
}
