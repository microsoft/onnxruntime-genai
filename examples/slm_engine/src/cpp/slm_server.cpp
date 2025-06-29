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

// Function to clean qwen model response formatting
std::string cleanQwenResponse(const std::string& input) {
  std::string result = input;

  // Remove <think>\n\n</think>\n\n pattern
  std::regex think_pattern(R"(<think>[\s\S]*?</think>\s*\n*)");
  result = std::regex_replace(result, think_pattern, "");

  // Replace markdown json code blocks ```json ``` with just the content
  std::regex json_pattern(R"(```json\s*\n([\s\S]*?)\n\s*```)");
  result = std::regex_replace(result, json_pattern, "$1");

  return result;
}

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
        {"engine_version", {"slm_version", slm_ver, "oga_version", oga_ver, "ort_version", ort_ver}},
        {"capabilities", {"text_completion", "function_calling"}}};
    response_body["engine_state"] = engine_state;
    json get_response;
    get_response["response"] = response_body;
    res.status = 200;
    res.set_content(get_response.dump(), "application/json");
  });

  // POST /completions endpoint
  svr.Post("/completions", [&](const httplib::Request& req,
                               httplib::Response& res) {
    try {
      // Parse the request body to check for function calling
      json request_json = json::parse(req.body);
      bool has_tools = request_json.contains("tools") && !request_json["tools"].empty();

      json response_json;

      if (has_tools) {
        // Handle function calling request
        cout << GREEN_BOLD << "Processing function calling request..." << CLEAR << endl;

        // Use the enhanced complete method that handles function calling internally
        auto response = slm_engine->complete(req.body.c_str());
        json output_json = json::parse(response);

        cout << RED_BOLD << "Response from SLM Engine: "
             << output_json.dump(2) << CLEAR << endl;

        // Check if this is a function call response
        // Function calls are detected by checking if the answer starts with '[' and contains JSON
        bool is_function_call = false;
        json function_calls_array;

        if (output_json.contains("response") &&
            output_json["response"].contains("answer")) {
          std::string answer = output_json["response"]["answer"];

          // Clean qwen model specific formatting
          answer = cleanQwenResponse(answer);

          // Trim whitespace
          answer.erase(0, answer.find_first_not_of(" \t\n\r"));
          answer.erase(answer.find_last_not_of(" \t\n\r") + 1);

          // Update the cleaned answer back to the response
          output_json["response"]["answer"] = answer;

          // Check if answer starts with '[' and ends with ']' (JSON array format)
          if (answer.length() > 0 && answer[0] == '[' && answer.back() == ']') {
            try {
              function_calls_array = json::parse(answer);
              if (function_calls_array.is_array() && !function_calls_array.empty()) {
                // Verify that each element has 'name' and 'arguments' fields
                bool valid_function_calls = true;
                for (const auto& call : function_calls_array) {
                  if (!call.contains("name") || !call.contains("arguments")) {
                    valid_function_calls = false;
                    break;
                  }
                }
                if (valid_function_calls) {
                  is_function_call = true;
                }
              }
            } catch (const json::exception& e) {
              // Not valid JSON, treat as regular text response
              is_function_call = false;
            }
          }
        }

        if (is_function_call) {
          // Function call(s) detected - always use function_calls array format
          cout << BLUE_BOLD << "Function call" << (function_calls_array.size() > 1 ? "s" : "")
               << " detected (" << function_calls_array.size() << " call"
               << (function_calls_array.size() > 1 ? "s" : "") << "):" << CLEAR << endl;

          for (size_t i = 0; i < function_calls_array.size(); ++i) {
            cout << BLUE_BOLD << "  " << (i + 1) << ". "
                 << function_calls_array[i]["name"] << CLEAR << endl;
          }

          // Ensure the response has the unified function_calls array format
          if (!output_json["response"].contains("function_calls")) {
            // Convert function_calls_array to the proper format with string arguments
            json unified_function_calls = json::array();
            for (const auto& call : function_calls_array) {
              unified_function_calls.push_back({{"name", call["name"]},
                                                {"arguments", call["arguments"].dump()}});
            }
            output_json["response"]["function_calls"] = unified_function_calls;
          }

          // Print KPIs for function calling
          if (output_json.contains("kpi")) {
            cout << "Prompt Tokens: " << output_json["kpi"]["prompt_toks"] << " "
                 << "TTFT: " << MAGENTA_BOLD
                 << output_json["kpi"]["ttft"].template get<float>() / 1000.0f
                 << " sec " << CLEAR << "Generated: "
                 << output_json["kpi"]["generated_toks"] << " "
                 << "Token Rate: " << MAGENTA_BOLD
                 << output_json["kpi"]["tok_rate"] << CLEAR << " "
                 << "Time: " << output_json["kpi"]["total_time"].template get<float>() / 1000.0f
                 << " sec " << "Memory: " << MAGENTA_BOLD
                 << output_json["kpi"]["memory_usage"] << CLEAR << " MB"
                 << " [FUNCTION_CALL" << (function_calls_array.size() > 1 ? "S" : "") << "]" << endl;
          }
        } else {
          // Regular text response with tools available
          cout << "Text response generated (no function call)" << endl;

          // Print KPIs for regular generation
          if (output_json.contains("kpi")) {
            cout << "Prompt Tokens: " << output_json["kpi"]["prompt_toks"] << " "
                 << "TTFT: " << MAGENTA_BOLD
                 << output_json["kpi"]["ttft"].template get<float>() / 1000.0f
                 << " sec " << CLEAR << "Generated: "
                 << output_json["kpi"]["generated_toks"] << " "
                 << "Token Rate: " << MAGENTA_BOLD
                 << output_json["kpi"]["tok_rate"] << CLEAR << " "
                 << "Time: " << output_json["kpi"]["total_time"].template get<float>() / 1000.0f
                 << " sec " << "Memory: " << MAGENTA_BOLD
                 << output_json["kpi"]["memory_usage"] << CLEAR << " MB" << endl;
          }
        }

        res.status = 200;
        res.set_content(output_json.dump(), "application/json");

      } else {
        // Handle regular completion request (no tools)
        auto response = slm_engine->complete(req.body.c_str());
        json output_json = json::parse(response);

        // Clean qwen model specific formatting for regular responses too
        if (output_json.contains("response") &&
            output_json["response"].contains("answer")) {
          std::string answer = output_json["response"]["answer"];
          answer = cleanQwenResponse(answer);
          output_json["response"]["answer"] = answer;
        }

        // Print KPIs for regular completion
        cout << "Prompt Tokens: "
             << output_json["kpi"]["prompt_toks"] << " "
             << "TTFT: " << MAGENTA_BOLD
             << output_json["kpi"]["ttft"].template get<float>() / 1000.0f
             << " sec " << CLEAR << "Generated: "
             << output_json["kpi"]["generated_toks"] << " "
             << "Token Rate: " << MAGENTA_BOLD
             << output_json["kpi"]["tok_rate"] << CLEAR << " "
             << "Time: " << output_json["kpi"]["total_time"].template get<float>() / 1000.0f
             << " sec " << "Memory: " << MAGENTA_BOLD
             << output_json["kpi"]["memory_usage"] << CLEAR << " MB" << "\n";
        flush(cout);

        res.status = 200;
        res.set_content(output_json.dump(), "application/json");
      }

    } catch (const std::exception& e) {
      // Handle JSON parsing errors or other exceptions
      json error_response;
      error_response["status"] = "error";
      error_response["message"] = std::string("Request processing error: ") + e.what();

      cout << RED_BOLD << "Error processing request: " << e.what() << CLEAR << endl;

      res.status = 400;
      res.set_content(error_response.dump(), "application/json");
    }
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