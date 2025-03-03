#include "input_decoder.h"

#include <gtest/gtest.h>

#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define MAGENTA_BOLD "\033[35;1m"
#define MAGENTA "\033[35m"
#define RED_BOLD "\033[31;1m"
#define RED "\033[31m"
#define BLUE_BOLD "\033[34;1m"
#define BLUE "\033[34m"
#define GREEN_BOLD "\033[32;1m"
#define GREEN "\033[32m"
#define CLEAR "\033[0m"

using namespace std;

/// @brief Reading from the input JSONL file, get the LLM response and write to
/// the output
/// @param model_path Path to the ONNX Quantized GenAI model
/// @param test_data_file JSONL file containing the question set to ask SLM
/// @param output_file Path to the JSONL file to save the SLM response and stats
/// @return 0 if successful, -1 otherwise
int run_test(const string& test_data_file) {
  // Make sure that the files exist

  // Make sure that the files exist
  if (!filesystem::exists(test_data_file)) {
    cout << "Error! Test Data file doesn't exist: " << test_data_file
         << "\n";
    return -1;
  }

  auto open_ai_decoder =
      microsoft::slm_engine::InputDecoder::CreateDecoder("openai");
  string line;
  ifstream test_data(test_data_file);
  while (getline(test_data, line)) {
    if (line.empty()) {
      continue;
    }
    // call the decoder
    microsoft::slm_engine::InputDecoder::InputParams input_params;
    auto status = open_ai_decoder->decode(line, input_params);
    if (status) {
      cout << BLUE << input_params.get_messages() << CLEAR << endl;
    } else {
      cout << MAGENTA_BOLD << "Error in decoding\n"
           << CLEAR;
    }
  }
  return 0;
}

/// @brief Program entry point
int main(int argc, char** argv) {
  argparse::ArgumentParser program("slm_runner", "1.0",
                                   argparse ::default_arguments::none);
  string test_data_file;
  program.add_argument("-t", "--test_data_file")
      .required()
      .help("Path to the test data file (JSONL)")
      .store_into(test_data_file);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(-1);
  }

  return run_test(test_data_file);
}