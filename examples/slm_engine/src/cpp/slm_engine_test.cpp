#include "gtest/gtest.h"

#include "slm_engine.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <memory>
#include <string>

using namespace std;

namespace microsoft {
namespace slm_engine {
namespace testing {

// Define the path to the model file
// This should be set in the environment variable MODEL_FILE_PATH
const char* MODEL_FILE_PATH = std::getenv("MODEL_FILE_PATH");

// Define the path to the model root directory
// Al the model directories are expected to be under this directory
const char* MODEL_ROOT_DIR = std::getenv("MODEL_ROOT_DIR");

struct ModelInfo {
  std::string model_path;
  std::string model_family;
};

const ModelInfo MODELS[] = {
    {"Llama-3.2-1B-Instruct-ONNX/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/", "llama"},
    {"Phi-4-mini-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/", "phi3"},
};

TEST(SLMEngineTest, TestModelFamily) {
  ASSERT_TRUE(MODEL_ROOT_DIR != nullptr) << "MODEL_ROOT_DIR is not set";

  for (const auto& model : MODELS) {
    std::string model_path = std::string(MODEL_ROOT_DIR) + "/" + model.model_path;
    std::string model_family = SLMEngine::GetModelFamily(model_path);
    ASSERT_EQ(model_family, model.model_family);
  }
}

// A few Test Prompts and their expected answers
struct TestPrompt {
  std::string system_prompt;
  std::string user_prompt;
  std::string expected_answer;
};

const TestPrompt TEST_PROMPTS[] = {
    {"You are a shool teacher who's very specific. Just provide the answer and nothing else.", "What is 2 * 20 - 10?", "Expected answer 1"},

    {"You are a travel assistant. Mention just the answer and do not explain anything.",
     "What is the airport code for Los Angeles?",
     "Expected answer 2"},

    {"You are a travel assistant. Provided a detailed answer as good as you can.",
     "Which state is San Diego in?", ""},

    {"Briefly answer any question you are asked. Do not provide any explanation.",
     "Which is the capital of France?",
     ""},

    {"You are a travel assistant. Just provide the answer and nothing else.",
     "Which state is San Diego in?", ""}};

TEST(SLMEngineTest, LoadUnloadModel) {
  ASSERT_TRUE(MODEL_FILE_PATH != nullptr) << "MODEL_FILE_PATH is not set";

  std::cout << "Initial Memory Usage: "
            << microsoft::slm_engine::SLMEngine::GetMemoryUsage() << " MB"
            << std::endl;

  for (int i = 0; i < 5; i++) {
    // Test loading a model
    std::cout << "Before Engine Create Memory Usage: "
              << microsoft::slm_engine::SLMEngine::GetMemoryUsage() << " MB"
              << std::endl;
    auto slm_engine = microsoft::slm_engine::SLMEngine::CreateEngine(
        MODEL_FILE_PATH, false);

    std::cout << "After Loading Model Memory Usage: "
              << microsoft::slm_engine::SLMEngine::GetMemoryUsage() << " MB"
              << std::endl;

    ASSERT_NE(slm_engine, nullptr);

    // Reset the engine to free up resources
    slm_engine.reset();

    std::cout << "After delete engine Memory Usage: "
              << microsoft::slm_engine::SLMEngine::GetMemoryUsage() << " MB"
              << std::endl;
  }
}

TEST(SLMEngineTest, TestGeneration) {
  ASSERT_TRUE(MODEL_FILE_PATH != nullptr) << "MODEL_FILE_PATH is not set";

  auto slm_engine = microsoft::slm_engine::SLMEngine::CreateEngine(
      MODEL_FILE_PATH, false);
  ASSERT_NE(slm_engine, nullptr);

  std::cout << "After Loading Model Memory Usage: "
            << microsoft::slm_engine::SLMEngine::GetMemoryUsage() << " MB"
            << std::endl;

  for (const auto& test_prompt : TEST_PROMPTS) {
    SLMEngine::RuntimePerf kpi;
    SLMEngine::GenerationOptions generator_options;

    string response;
    slm_engine->generate(test_prompt.system_prompt + test_prompt.user_prompt, generator_options, response, kpi);
    cout << "Response: " << response << std::endl;
    cout << "TTFT: " << kpi.TimeToFirstToken << " TPS: " << kpi.TokenRate
         << " Memory Usage: " << kpi.CurrentMemoryUsed << " MB" << endl;
  }
}

}  // namespace testing
}  // namespace slm_engine
}  // namespace microsoft
