#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

#define MAGENTA_BOLD "\033[35;1m"
#define MAGENTA "\033[35m"
#define RED_BOLD "\033[31;1m"
#define RED "\033[31m"
#define BLUE_BOLD "\033[34;1m"
#define BLUE "\033[34m"
#define GREEN_BOLD "\033[32;1m"
#define GREEN "\033[32m"
#define CLEAR "\033[0m"

#include "slm_engine.h"

using namespace std;

/// @brief Reading from the input JSONL file, get the LLM response and write to
/// the output
/// @param model_path Path to the ONNX Quantized GenAI model
/// @param test_data_file JSONL file containing the question set to ask SLM
/// @param output_file Path to the JSONL file to save the SLM response and stats
/// @return 0 if successful, -1 otherwise
int run_test(const string& model_path,
             const std::vector<std::string>& adapter_files,
             const string& test_data_file, const string& output_file,
             bool verbose, int wait_between_requests) {
  // Make sure that the files exist
  if (!filesystem::exists(model_path)) {
    cout << "Error! Model path doesn't exist: " << model_path << "\n";
    return -1;
  }

  // Make sure that the files exist
  if (!filesystem::exists(test_data_file)) {
    cout << "Error! Test Data file doesn't exist: " << test_data_file
         << "\n";
    return -1;
  }

  cout << "Model: " << model_path << "\n"
       << "Test File: " << test_data_file << "\n";

  std::unique_ptr<microsoft::slm_engine::SLMEngine> slm_engine;

  if (adapter_files.empty()) {
    slm_engine = microsoft::slm_engine::SLMEngine::Create(
        model_path.c_str(), verbose);
  } else {
    vector<microsoft::slm_engine::SLMEngine::LoRAAdapter> adapters;

    for (const auto& adapter_file : adapter_files) {
      // Check if the adapter path exists
      if (!std::filesystem::exists(adapter_file)) {
        cout << RED << "Adapter path does not exist: " << adapter_file
             << CLEAR << endl;
        return -1;
      }

      // Get the filename from the path
      std::filesystem::path adapter_name(adapter_file);
      microsoft::slm_engine::SLMEngine::LoRAAdapter adapter(
          adapter_name.stem().string(),
          adapter_file);
      adapters.push_back(adapter);

      cout << "Adapter: " << adapter_name.stem().string() << "\n";
      cout << "Adapter Path: " << adapter_file << "\n";
    }

    microsoft::slm_engine::SLMEngine::Status status;
    slm_engine = microsoft::slm_engine::SLMEngine::Create(
        model_path.c_str(), adapters, verbose, status);
  }

  if (!slm_engine) {
    cout << "Cannot create engine!\n";
    return -1;
  }

  ofstream output(output_file);
  string line;
  ifstream test_data(test_data_file);
  while (getline(test_data, line)) {
    if (line.empty()) {
      continue;
    }

    auto response = slm_engine->complete(line.c_str());
    json output_json = json::parse(response);

    if (!verbose) {
      cout << BLUE << "Question: " << output_json["question"]
           << CLEAR << endl;
      cout << GREEN << "Answer: " << output_json["choices"][0]["message"]["content"]
           << CLEAR << endl;
    }
    // Save to the file
    output << output_json.dump() << endl;

    cout << "Prompt Tokens: "
         << output_json["kpi"]["prompt_toks"] << " "
         << "TTFT: " << MAGENTA_BOLD
         << output_json["kpi"]["ttft"].template get<float>() /
                1000.0f
         << " sec " << CLEAR << "Generated: "
         << output_json["kpi"]["generated_toks"] << " "
         << "Token Rate: " << MAGENTA_BOLD
         << output_json["kpi"]["tok_rate"] << CLEAR << " "
         << "Time: "
         << output_json["kpi"]["total_time"]
                    .template get<float>() /
                1000.0f
         << " sec "
         << "Memory: " << MAGENTA_BOLD
         << output_json["kpi"]["memory_usage"] << CLEAR << " MB"
         << "\n";
    flush(cout);
    if (wait_between_requests > 0) {
      cout << "Waiting for " << wait_between_requests << " ms\n";
      this_thread::sleep_for(chrono::milliseconds(wait_between_requests));
    }
  }
  return 0;
}

/// @brief Program entry point
int main(int argc, char** argv) {
  argparse::ArgumentParser program("slm_runner", "1.0",
                                   argparse ::default_arguments::none);
  string model_path;
  program.add_argument("-m", "--model_path")
      .required()
      .help("Path to the model file")
      .store_into(model_path);

  program.add_argument("-a", "--adapters")
      .help(
          "List of LoRA adapter files to process. "
          "We will use the filename part as the name of the LoRA adpater")
      .nargs(argparse::nargs_pattern::any);

  string test_data_file;
  program.add_argument("-t", "--test_data_file")
      .required()
      .help("Path to the test data file (JSONL)")
      .store_into(test_data_file);

  string output_file;
  program.add_argument("-o", "--output_file")
      .required()
      .help("Path to the output file (JSONL)")
      .store_into(output_file);

  int wait_between_requests = 0;
  program.add_argument("-w", "--wait_between_requests")
      .help("Wait time between requests in milliseconds")
      .store_into(wait_between_requests);

  program.add_argument("-v", "--verbose")
      .default_value(false)
      .implicit_value(true)
      .help(
          "If provided, more debugging information printed on standard "
          "output");

  std::string slm_ver, oga_ver, ort_ver;
  microsoft::slm_engine::SLMEngine::GetVersion(slm_ver, oga_ver, ort_ver);

  cout << "SLM Runner Version: "
       << slm_ver << "\nORT GenAI Version: " << oga_ver << "\nORT Version: " << ort_ver
       << endl;
  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(-1);
  }

  bool verbose = false;
  if (program["--verbose"] == true) {
    verbose = true;
  }

  std::vector<std::string> adapters =
      program.get<std::vector<std::string>>("adapters");
  // Responsible for cleaning up the library during shutdown
  // OgaHandle handle;

  run_test(model_path, adapters, test_data_file, output_file, verbose,
           wait_between_requests);

  OgaShutdown();
}