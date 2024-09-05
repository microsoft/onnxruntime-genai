#include <string>
#include <exception>
#include <generators.h>

#include <server/engine.h>

int main(int argc, char** argv) {
  // Generators::SetLogBool("enabled", true);
  // Generators::SetLogBool("model_input_values", true);
  // Generators::SetLogBool("model_output_values", true);
  // Generators::SetLogBool("model_logits", true);
  // Generators::SetLogBool("generate_next_token", true);
  // Generators::SetLogBool("append_next_tokens", true);
  try {
    auto engine = Generators::OgaEngine("/home/yingxiong/llama2-7b-page-genai-rot");
    auto params = Generators::SamplingParams();
    params.max_tokens = 32;
    // engine.AddRequest("1", "Hello, world!Hello, world!Hello, world!", params, 1.2f);
    // engine.AddRequest("2", "Three biggest countries in the world:", params, 1.5f);
    for (int i = 0; i < 100; i++) {
      engine.AddRequest(std::to_string(i + 1), "What is ONNX Runtime:", params, 1.5f);
    }

    for (int i = 0; i < 64; i++) {
      engine.Step();
    }
  } catch (...) {
    std::exception_ptr ex = std::current_exception();

    if (ex) {
      try {
        std::rethrow_exception(ex);
      } catch (const std::exception& e) {
        std::cout << "Caught exception: '" << e.what() << "'\n";
      }
    }
  }
}
