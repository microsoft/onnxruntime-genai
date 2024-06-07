#include <iostream>
#include <chrono>

#include <vector>
#include "ort_genai.h"
#include "engine.h"

void test_single(OgaEngine& engine, std::vector<const char*>& prompts) {
  std::vector<std::string> outputs;
  for (const char* prompt : prompts) {
    outputs = engine.Generate({prompt});
  }

  std::cout << "Complete batch size: " << outputs.size() << std::endl;
  std::cout << "Output: " << outputs.back() << std::endl;
}

void test_batch(OgaEngine& engine, std::vector<const char*>& prompts) {
  std::vector<std::string> outputs = engine.Generate(prompts);

  std::cout << "Output: " << outputs.back() << std::endl;
  std::cout << "Complete batch size: " << outputs.size() << std::endl;
}

void test_stream(OgaEngine& engine, const char* prompt) {
}

void test() {
  const char* config_path = "/home/yingxiong/projects/onnxruntime-genai/models/llama7b_fp16";
  std::cout << "Creating model..." << std::endl;
  auto model = OgaModel::Create(config_path);
  std::cout << "Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);

  int data_size = 130;
  auto test_prompt = "def is_prime(num):";
  std::vector<const char*> prompts(data_size, test_prompt);

  auto sequences = OgaSequences::Create();
  tokenizer->EncodeBatch(prompts, *sequences);
  std::cout << "Encoded batch size: " << sequences->Count() << std::endl;
}

int main(int argc, char** argv) {
  std::cout << "Hello！" << std::endl;

  OgaHandle handle;

  OgaEngine engine("/home/yingxiong/projects/onnxruntime-genai/models/llama7b_fp16");

  int data_size = 128;
  auto test_prompt = "def is_prime(num):";
  std::vector<const char*> prompts(data_size, test_prompt);

  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  auto t1 = high_resolution_clock::now();
  test_single(engine, prompts);
  auto t2 = high_resolution_clock::now();
  auto ms_int = duration_cast<milliseconds>(t2 - t1);
  std::cout << "single batch time: " << ms_int.count() << "ms\n";

  auto t3 = high_resolution_clock::now();
  test_batch(engine, prompts);
  auto t4 = high_resolution_clock::now();
  auto ms_int2 = duration_cast<milliseconds>(t4 - t3);
  std::cout << "dynamic batch time: " << ms_int2.count() << "ms\n";

  // test();

  return 0;
}