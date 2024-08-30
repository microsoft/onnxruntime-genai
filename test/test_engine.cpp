#include <exception>
#include <generators.h>

#include <server/engine.h>

int main(int argc, char** argv) {
  Generators::SetLogBool("enabled", true);
  Generators::SetLogBool("model_input_values", true);
  Generators::SetLogBool("model_output_values", true);
  Generators::SetLogBool("model_logits", true);
  Generators::SetLogBool("append_next_tokens", true);
  try {
    auto engine = Generators::OgaEngine("/home/yingxiong/llama2-7b-page-genai");
    auto params = Generators::SamplingParams();
    engine.AddRequest("1", "Hello, world!", params, 1.2f);
    engine.AddRequest("2", "Thanks, world!", params, 1.5f);
    engine.Step();
    engine.Step();
    engine.Step();
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
