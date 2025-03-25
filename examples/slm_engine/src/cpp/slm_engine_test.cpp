#include "gtest/gtest.h"

#include "slm_engine.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <filesystem>
#include <vector>

using namespace std;

namespace microsoft {
namespace slm_engine {
namespace testing {

// Define the path to the model file
// This should be set in the environment variable MODEL_FILE_PATH
const char* MODEL_FILE_PATH = getenv("MODEL_FILE_PATH");

// Define the path to the model root directory
// Al the model directories are expected to be under this directory
const char* MODEL_ROOT_DIR = getenv("MODEL_ROOT_DIR");

struct ModelInfo {
  string model_path;
  string model_family;
};

const ModelInfo MODELS[] = {
    {"Llama-3.2-1B-Instruct-ONNX/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/", "llama"},
    {"Phi-4-mini-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/", "phi3"},
};

void trim_left_inplace(std::string& s) {
  s.erase(
      s.begin(), std::find_if_not(s.begin(), s.end(),
                                  [](unsigned char c) { return std::isspace(c); }));
}

void trim_right_inplace(std::string& s) {
  s.erase(
      std::find_if_not(s.rbegin(), s.rend(),
                       [](unsigned char c) { return std::isspace(c); })
          .base(),
      s.end());
}

void trim_inplace(std::string& s) {
  trim_left_inplace(s);
  trim_right_inplace(s);
}

std::string trim(const std::string& s) {
  std::string trimmed = s;
  trim_inplace(trimmed);
  return trimmed;
}

TEST(SLMEngineTest, TestModelFamily) {
  ASSERT_TRUE(MODEL_ROOT_DIR != nullptr) << "MODEL_ROOT_DIR is not set";

  for (const auto& model : MODELS) {
    string model_path = string(MODEL_ROOT_DIR) + "/" + model.model_path;
    string model_family = SLMEngine::GetModelFamily(model_path);
    ASSERT_EQ(model_family, model.model_family);
  }
}

TEST(SLMEngineTest, LoadUnloadModel) {
  ASSERT_TRUE(MODEL_FILE_PATH != nullptr) << "MODEL_FILE_PATH is not set";

  cout << "Initial Memory Usage: "
       << microsoft::slm_engine::SLMEngine::GetMemoryUsage() << " MB"
       << endl;

  for (int i = 0; i < 5; i++) {
    // Test loading a model
    cout << "Before Engine Create Memory Usage: "
         << microsoft::slm_engine::SLMEngine::GetMemoryUsage() << " MB"
         << endl;
    auto slm_engine = microsoft::slm_engine::SLMEngine::CreateEngine(
        MODEL_FILE_PATH, false);

    cout << "After Loading Model Memory Usage: "
         << microsoft::slm_engine::SLMEngine::GetMemoryUsage() << " MB"
         << endl;

    ASSERT_NE(slm_engine, nullptr);

    // Reset the engine to free up resources
    slm_engine.reset();

    cout << "After delete engine Memory Usage: "
         << microsoft::slm_engine::SLMEngine::GetMemoryUsage() << " MB"
         << endl;
  }
}

// A few Test Prompts and their expected answers
struct TestPrompt {
  string system_prompt;
  string user_prompt;
  string expected_answer;
};

const char* SYS_PROMPT =
    "You are a helpful assistant. "
    "Analyze the sentiment of the statement and just answer in one word what it is. "
    "You can chose from one of Positive, Neutral, or Negative. "
    "Use a new line after the answer.";

const TestPrompt TEST_PROMPTS[] = {
    {SYS_PROMPT,
     "I love the new design of your website!\nThe sentiment of the text is: ", "Positive"},
    {SYS_PROMPT,
     "The customer service was terrible and unhelpful.\nThe sentiment of the text is: ",
     "Negative"},
    {SYS_PROMPT,
     "I'm not sure how I feel about the new update.\nThe sentiment of the text is: ", "Neutral"},
    {SYS_PROMPT,
     "The product quality has improved significantly.\nThe sentiment of the text is: ",
     "Positive"},
    {SYS_PROMPT,
     "I had a bad experience with the delivery service.\nThe sentiment of the text is: ",
     "Negative"},
    {SYS_PROMPT,
     "The instructions were clear and easy to follow.\nThe sentiment of the text is: ",
     "Positive"},
    {SYS_PROMPT,
     "I'm disappointed with the recent changes.\nThe sentiment of the text is: ",
     "Negative"},
    {SYS_PROMPT,
     "The event was well-organized and enjoyable.\nThe sentiment of the text is: ",
     "Positive"},
    {SYS_PROMPT,
     "I have mixed feelings about the new policy.\nThe sentiment of the text is: ",
     "Neutral"},
    {SYS_PROMPT,
     "The food was delicious and the service was excellent.\nThe sentiment of the text is: ",
     "Positive"}};

TEST(SLMEngineTest, TestGeneration) {
  ASSERT_TRUE(MODEL_FILE_PATH != nullptr) << "MODEL_FILE_PATH is not set";

  auto slm_engine = microsoft::slm_engine::SLMEngine::CreateEngine(
      MODEL_FILE_PATH, false);
  ASSERT_NE(slm_engine, nullptr);

  cout << "After Loading Model Memory Usage: "
       << microsoft::slm_engine::SLMEngine::GetMemoryUsage() << " MB"
       << endl;

  for (const auto& test_prompt : TEST_PROMPTS) {
    SLMEngine::GenerationOptions generator_options;
    generator_options.MaxGeneratedTokens = 250;
    generator_options.Temperature = 0.000000001f;

    string response;
    SLMEngine::RuntimePerf kpi;

    cout << "Question: " << test_prompt.user_prompt << endl;
    slm_engine->generate(
        test_prompt.system_prompt + test_prompt.user_prompt,
        generator_options, response, kpi);

    string stop_token("\n");
    // We need to remove the stop token(s) from the response
    auto stop_token_pos = response.find(stop_token);
    if (stop_token_pos != std::string::npos) {
      response = response.substr(0, stop_token_pos);
    }

    trim_inplace(response);

    cout << "Response: " << response << endl;
    cout << "Expected: " << test_prompt.expected_answer << endl;
    cout << "TTFT: " << kpi.TimeToFirstToken << " TPS: " << kpi.TokenRate
         << " Memory Usage: " << kpi.CurrentMemoryUsed << " MB "
         << "Generated Tokens: " << kpi.GeneratedTokenCount
         << " Prompt Tokens: " << kpi.PromptTokenCount << endl
         << endl;

    EXPECT_STREQ(response.c_str(), test_prompt.expected_answer.c_str())
        << "Test failed for prompt: " << test_prompt.user_prompt
        << " with response: " << response;
  }
}

const char* TEST_INPUT_FILE = getenv("TEST_INPUT_FILE");

TEST(SLMEngineTest, CaptureMemoryUsage) {
  // This test captures the memory usage at various stages of the SLM  Engine lifecycle
  // Produces a JSON file with performance metrics that can be used for analysis
  ASSERT_TRUE(MODEL_FILE_PATH != nullptr) << "MODEL_FILE_PATH is not set";
  ASSERT_TRUE(TEST_INPUT_FILE != nullptr) << "TEST_INPUT_FILE is not set";

  nlohmann::json overall_status_json;
  overall_status_json["model_path"] = MODEL_FILE_PATH;
  overall_status_json["test_input_file"] = TEST_INPUT_FILE;

  string ort_version, oga_version, slm_version;

  SLMEngine::GetVersion(ort_version, oga_version, slm_version);
  overall_status_json["ort_version"] = ort_version;
  overall_status_json["oga_version"] = oga_version;
  overall_status_json["slm_version"] = slm_version;

  overall_status_json["memory_before_run"] = SLMEngine::GetMemoryUsage();

  auto slm_engine = microsoft::slm_engine::SLMEngine::CreateEngine(
      MODEL_FILE_PATH, false);
  ASSERT_NE(slm_engine, nullptr) << "Failed to create SLMEngine";
  overall_status_json["memory_after_load"] = SLMEngine::GetMemoryUsage();

  // Now start the run
  ASSERT_TRUE(filesystem::exists(TEST_INPUT_FILE))
      << "Input file doesn't exist: " << TEST_INPUT_FILE;

  ifstream file(TEST_INPUT_FILE);
  ASSERT_TRUE(file.is_open()) << "Failed to open test input file";

  auto per_prompt_stats_json_array = nlohmann::json::array();
  string line;
  while (getline(file, line)) {
    try {
      auto jsonObject = nlohmann::json::parse(line);  // Parse the JSON object
      cout << "Question: " << jsonObject["messages"][1]["content"] << endl;

      auto per_prompt_stats_json = nlohmann::json::object();
      per_prompt_stats_json["memory_before_generate"] = SLMEngine::GetMemoryUsage();

      SLMEngine::RuntimePerf kpi;
      SLMEngine::GenerationOptions generator_options;
      string response;

      slm_engine->generate(line, generator_options, response, kpi);

      // Capture the stats
      per_prompt_stats_json["ttft"] = kpi.TimeToFirstToken;
      per_prompt_stats_json["tok_rate"] = kpi.TokenRate;
      per_prompt_stats_json["memory_usage"] = kpi.CurrentMemoryUsed;
      per_prompt_stats_json["total_time"] = kpi.TotalTime;
      per_prompt_stats_json["prompt_toks"] = kpi.PromptTokenCount;
      per_prompt_stats_json["generated_toks"] = kpi.GeneratedTokenCount;

      per_prompt_stats_json["memory_after_generate"] = SLMEngine::GetMemoryUsage();

      cout << "Response: " << response << endl;
      cout << "TTFT: " << kpi.TimeToFirstToken << " TPS: " << kpi.TokenRate
           << " Memory Usage: " << kpi.CurrentMemoryUsed << " MB" << endl
           << endl;

      per_prompt_stats_json_array.push_back(per_prompt_stats_json);

    } catch (const nlohmann::json::parse_error& e) {
      FAIL() << "Failed to parse JSON: " << e.what();
    }
  }

  // Destroy the engine
  slm_engine.reset();
  overall_status_json["memory_after_unload"] = SLMEngine::GetMemoryUsage();

  // At the end - capture the memory usage
  overall_status_json["memory_after_run"] = SLMEngine::GetMemoryUsage();

  nlohmann::json test_output_json;
  test_output_json["overall_stats"] = overall_status_json;
  test_output_json["per_prompt_stats"] = per_prompt_stats_json_array;

  // Write the JSON object to a file
  ofstream output_file("test_output.json");
  if (output_file.is_open()) {
    output_file << test_output_json.dump(4);  // Pretty print with 4 spaces
    output_file.close();
  } else {
    FAIL() << "Failed to open output file for writing";
  }
}
}  // namespace testing
}  // namespace slm_engine
}  // namespace microsoft
