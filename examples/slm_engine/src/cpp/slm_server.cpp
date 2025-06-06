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

#include "httplib.h"
#include "slm_engine.h"

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

int run_server(const string& model_path,
               int port_number, bool verbose) {
  // Create the SLM
  auto slm_engine = microsoft::slm_engine::SLMEngine::Create(
      model_path.c_str(), verbose);
  if (!slm_engine) {
    cout << "Cannot create engine!\n";
    return -1;
  }

  httplib::Server svr;

  svr.Get("/", [&](const httplib::Request& req, httplib::Response& res) {
    json response_body;
    response_body["status"] = "success";

    std::string slm_ver, oga_ver, ort_ver;
    microsoft::slm_engine::SLMEngine::GetVersion(slm_ver, oga_ver, ort_ver);
    json engine_state = {
        {"model", std::filesystem::path(model_path).filename().string()},
        {"engine_version", {"slm_version", slm_ver, "oga_version", oga_ver, "ort_version", ort_ver}}};
    response_body["engine_state"] = engine_state;
    json get_response;
    get_response["response"] = response_body;
    res.status = 200;
    res.set_content(get_response.dump(), "application/json");
  });

  // POST /completions endpoint
  svr.Post("/completions", [&](const httplib::Request& req,
                               httplib::Response& res) {
    auto response = slm_engine->complete(req.body.c_str());
    // Get the KPIs
    json output_json = json::parse(response);
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

    res.status = 200;
    res.set_content(response, "application/json");
  });

  cout << MAGENTA_BOLD << "Starting server on port: " << port_number << CLEAR << endl;
  svr.listen("0.0.0.0", port_number);
  return 0;
}

/// @brief Program entry point
int main(int argc, char** argv) {
  argparse::ArgumentParser program("slm_server", "1.0",
                                   argparse ::default_arguments::none);
  string model_path;
  program.add_argument("-m", "--model_path")
      .required()
      .help("Path to the model file")
      .store_into(model_path);

  int port_number = 8080;
  program.add_argument("-p", "--port_number")
      .help("HTTP Port Number to use (default 8080)")
      .store_into(port_number);

  program.add_argument("-v", "--verbose")
      .default_value(false)
      .implicit_value(true)
      .help(
          "If provided, more debugging information printed on standard "
          "output");

  string slm_ver, oga_ver, ort_ver;
  microsoft::slm_engine::SLMEngine::GetVersion(slm_ver, oga_ver, ort_ver);
  cout << "SLM Runner Version: " << slm_ver << "\n"
       << "ORT GenAI Version: " << oga_ver << "\n"
       << "ORT Version: " << ort_ver
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

  run_server(model_path, port_number, verbose);
  OgaShutdown();
}