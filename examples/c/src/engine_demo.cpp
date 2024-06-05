#include <iostream>
#include <vector>

#include "engine.h"

void test_batch(OgaEngine& engine, std::vector<const char*>& prompts) {
  std::vector<std::string> outputs = engine.Generate(prompts);

  std::cout << "Output: " << outputs.back() << std::endl;
}

void test_tream(OgaEngine& engine, const char* prompt) {
}

int main(int argc, char** argv) {
  std::cout << "Hello！" << std::endl;

  OgaEngine engine("/home/yingxiong/projects/onnxruntime-genai/models/llama7b_fp16");

  int data_size = 128;
  auto test_prompt = "def is_prime(num):";
  std::vector<const char*> prompts(data_size, test_prompt);

  test_batch(engine, prompts);

  return 0;
}