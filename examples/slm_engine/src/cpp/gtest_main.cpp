
#include "gtest/gtest.h"
#include <argparse/argparse.hpp>

#include "httplib.h"
#include "ort_genai.h"

using namespace std;

extern const char* MODEL_FILE_PATH;
extern const char* ADAPTER_ROOT_DIR;

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  argparse::ArgumentParser program("slm_engine_test", "1.0",
                                   argparse ::default_arguments::none);
  string model_path;
  program.add_argument("-m", "--model_path")
      .help("Path to the model file")
      .store_into(model_path);

  string adapter_root_path;
  program.add_argument("-m", "--adapter_root_path")
      .help("Path to the LoRA adapter root directory")
      .store_into(adapter_root_path);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(-1);
  }

  if (!model_path.empty()) {
    cout << "Setting Model path: " << model_path << endl;
    MODEL_FILE_PATH = model_path.c_str();
  }

  if (!adapter_root_path.empty()) {
    cout << "Setting Adapter path: " << adapter_root_path << endl;
    ADAPTER_ROOT_DIR = adapter_root_path.c_str();
  }

  auto status = RUN_ALL_TESTS();

  OgaShutdown();
  return status;
}
