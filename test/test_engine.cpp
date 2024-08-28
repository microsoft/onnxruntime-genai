#include <generators.h>
#include <server/engine.h>

int main(int argc, char** argv) {
  Generators::SetLogBool("enabled", true);
  Generators::SetLogBool("model_input_values", true);
  Generators::SetLogBool("model_logits", true);
  auto engine = Generators::OgaEngine("/raid/yingxiong/phi3.5-mini-genai/");
  auto params = Generators::SamplingParams();
  engine.AddRequest("1", "Hello, world!", params, 1.2f);
  engine.Step();
}