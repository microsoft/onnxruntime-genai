
#include "slm_engine.h"

#include "onnxruntime_cxx_api.h"

#include <stdio.h>
#include <string.h>
#if !defined(_WIN32)
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#else
#include <windows.h>
#include <psapi.h>
#endif

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

using namespace std;
using json = nlohmann::json;

#define MAGENTA "\033[35;1m"
#define RED "\033[31;1m"
#define BLUE "\033[34;1m"
#define GREEN "\033[32;1m"
#define CLEAR "\033[0m"

namespace microsoft {
namespace slm_engine {

std::unique_ptr<SLMEngine> SLMEngine::CreateEngine(
    const char* model_path, bool verbose) {
  // Convert the model name to the SupportedModelType
  auto model_type = StringToModelType(GetModelFamily(model_path));
  if (model_type == SupportedModelType::UNKNOWN) {
    cout << RED << "Error! Cannot detect the model type for model: " << model_path << CLEAR
         << endl;
    return nullptr;
  }

  auto new_obj = std::unique_ptr<SLMEngine>(new SLMEngine(verbose));
  if (!new_obj->load_model(model_path, model_type)) {
    cout << RED << "Error creating the SLM Engine" << CLEAR << endl;
    return nullptr;
  }

  return std::move(new_obj);
}

void SLMEngine::GetVersion(std::string& slm_version, std::string& ortga_version,
                           std::string& ort_version) {
  // SW_VERSION_NUMBER is defined in the CMakeLists.txt file
  slm_version = std::string(SW_VERSION_NUMBER);
  ortga_version = std::string(ORT_GENAI_VERSION);
  ort_version = Ort::GetVersionString();
}

std::string SLMEngine::GetModelFamily(const std::string& model_path) {
  // Open the config.json file
  std::ifstream config_file(model_path + "/config.json");
  if (!config_file.is_open()) {
    std::cout << RED << "Error opening config.json file" << CLEAR << std::endl;
    return "";
  }
  // Parse the JSON file
  json config_json;
  config_file >> config_json;
  config_file.close();

  // Check if the "model_type" field exists
  if (config_json.find("model_type") == config_json.end()) {
    std::cout << RED << "Error: model_type field not found in config.json" << CLEAR << std::endl;
    return "";
  }
  // Get the value of the "model_type" field
  std::string model_type = config_json["model_type"];

  return model_type;
}

SLMEngine::~SLMEngine() {
  m_onnx_model.reset();
  m_tokenizer.reset();
  m_tokenizer_stream.reset();
  m_input_decoder.reset();
}

void SLMEngine::generate(
    const std::string& formatted_prompt,
    const GenerationOptions& generation_options,
    std::string& response_str,
    RuntimePerf& kpi) {
  // Use a scoped mutex to ensure only one thread can call complete at a time
  std::lock_guard<std::mutex> lock(m_mutex);

  auto api_start = std::chrono::steady_clock::now();

  auto sequences = OgaSequences::Create();
  m_tokenizer->Encode(formatted_prompt.c_str(), *sequences);

  auto start = std::chrono::steady_clock::now();

  auto params = OgaGeneratorParams::Create(*m_onnx_model);
  params->SetSearchOption("max_length", generation_options.MaxGeneratedTokens);
  params->SetSearchOption("temperature", generation_options.Temperature);
  params->SetSearchOption("top_k", generation_options.TopK);
  params->SetSearchOption("top_p", generation_options.TopP);

  auto generator = OgaGenerator::Create(*m_onnx_model, *params);
  generator->AppendTokenSequences(*sequences);
  bool is_first_token = true;
  auto time_count = 0;

  auto initial_prompt_token_count = generator->GetSequenceCount(0);

  std::ostringstream response;
  bool stop_token_found = false;
  while (!generator->IsDone()) {
    generator->GenerateNextToken();

    const auto num_tokens = generator->GetSequenceCount(0);
    const auto new_token = generator->GetSequenceData(0)[num_tokens - 1];
    auto end = std::chrono::steady_clock::now();
    if (is_first_token) {
      is_first_token = false;
      auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                start)
              .count();
      kpi.PromptTokenCount = initial_prompt_token_count;
      kpi.TimeToFirstToken = elapsed;
    } else {
      time_count += std::chrono::duration_cast<std::chrono::milliseconds>(
                        end - start)
                        .count();
    }

    auto next_string_piece = m_tokenizer_stream->Decode(new_token);
    // TODO: Use the actual token for the end or line below
    if (strncmp(next_string_piece, "</s>", 10) != 0) {
      // We received end of the text - so will exclude
      response << next_string_piece;
    }

    // Print next output string if the generation continues
    if (m_verbose) {
      cout << next_string_piece;
      flush(cout);
    }
    start = std::chrono::steady_clock::now();
  }
  if (m_verbose) {
    cout << CLEAR << endl;
  }

  response_str = response.str();
  m_llm_output_dbg_stream << response_str << endl;

  kpi.GeneratedTokenCount =
      generator->GetSequenceCount(0) - initial_prompt_token_count;
  kpi.TokenRate = kpi.GeneratedTokenCount / (time_count / 1000.0f);

  kpi.TotalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - api_start)
                      .count();

  // // Get the current memory
  kpi.CurrentMemoryUsed = GetMemoryUsage();
}

std::string SLMEngine::complete(const char* user_prompt) {
  // Use a scoped mutex to ensure only one thread can call complete at a time

  auto api_start = std::chrono::steady_clock::now();

  InputDecoder::InputParams input_parameters;
  // Decode the user prompt
  if (!m_input_decoder->decode(user_prompt, input_parameters)) {
    cout << RED << "Error decoding input message: " << user_prompt << CLEAR
         << endl;
    json output_json;
    output_json["status"] = "error";
    output_json["message"] =
        "Error decoding input message: " + string(user_prompt);

    json response_message;
    response_message["response"] = output_json;

    return response_message.dump();
  }

  auto formatted_prompt = format_input(input_parameters);
  m_llm_input_dbg_stream << formatted_prompt << endl;

  if (m_verbose) {
    cout << BLUE << "User: " << input_parameters.UserPrompt << endl;
    cout << GREEN;
  }

  RuntimePerf kpi;
  std::string response;
  bool stop_token_found = false;

  GenerationOptions generator_options;
  generator_options.MaxGeneratedTokens = input_parameters.MaxGeneratedTokens;
  generator_options.Temperature = input_parameters.Temperature;
  generator_options.TopK = input_parameters.TopK;
  generator_options.TopP = input_parameters.TopP;

  generate(formatted_prompt, generator_options, response, kpi);

  m_llm_output_dbg_stream << response << endl;
  // We need to remove the stop token from the response
  for (const auto& stop_token : input_parameters.StopTokens) {
    auto stop_token_pos = response.find(stop_token);
    if (stop_token_pos != std::string::npos) {
      response = response.substr(0, stop_token_pos);
      break;
    }
  }

  auto choices_str = R"(
    [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "logprobs": null,
            "finish_reason": "stop"
        }
    ]
    )";

  json choices = json::parse(choices_str);
  choices[0]["message"]["content"] = response;

  json output_json;
  output_json["status"] = "success";
  output_json["question"] = input_parameters.UserPrompt;
  output_json["choices"] = choices;
  output_json["llm_input"] = formatted_prompt;

  json kpi_json;
  kpi_json["prompt_toks"] = kpi.PromptTokenCount;
  kpi_json["ttft"] = kpi.TimeToFirstToken;
  kpi_json["generated_toks"] = kpi.GeneratedTokenCount;
  kpi_json["tok_rate"] = kpi.TokenRate;
  kpi_json["total_time"] = kpi.TotalTime;
  kpi_json["memory_usage"] = kpi.CurrentMemoryUsed;

  output_json["kpi"] = kpi_json;

  json response_message;
  response_message["response"] = output_json;
  return output_json.dump();
}

// Use a Dictionary to store various types of prompt formatting
// LLama3.2 and Phi3 have different prompt formats
// Llama3.2 format described here:
// https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md
//

// Following is a dictionary that stores the prompt format for different models
const auto PromptFormatTable = R"(
[
   {
      "llm_type": "phi",
      "prompt_format": {
         "system": { "prefix": "<|system|>\n", "suffix": "<|end|>\n" },
         "user": { "prefix": "<|user|>\n", "suffix": "<|end|>\n" },
         "assistant": { "prefix": "<|assistant|>\n", "suffix": "<|end|>\n" }
      }
   },
   {     
      "llm_type": "llama",
      "prompt_format": {
         "system": { "prefix": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", "suffix": "<|eot_id|>" },
         "user": { "prefix": "<|start_header_id|>user<|end_header_id|>\n\n", "suffix": "<|eot_id|>" },
         "assistant": { "prefix": "<|start_header_id|>assistant<|end_header_id|>\n\n", "suffix": "<|eot_id|>" }
      }
   },
   {
      "llm_type": "custom",
      "prompt_format": {
         "system": { "prefix": "", "suffix": "" },
         "user": { "prefix": "", "suffix": "" },
         "assistant": { "prefix": "", "suffix": "" }
      }
   }
]
)";

// Define a function to parse the JSON dictionary to the c++ data structure
bool SLMEngine::parse_prompt_format_dict(
    SupportedModelType model_type, const std::string& json_dict,
    PromptFormatDictionary& prompt_format_dict) {
  auto j = json::parse(json_dict);
  for (const auto& llm_type : j) {
    if (llm_type["llm_type"] != ModelTypeToString(model_type)) {
      continue;
    }

    prompt_format_dict.llm_type = llm_type["llm_type"];

    for (const auto& role : llm_type["prompt_format"].items()) {
      PromptFormat pf;
      pf.prefix = role.value()["prefix"];
      pf.suffix = role.value()["suffix"];
      prompt_format_dict
          .prompt_format[InputDecoder::InputParams::ToRole(role.key())] =
          pf;
    }
    return true;
  }
  return false;
}

bool SLMEngine::load_model(const char* model_path,
                           SupportedModelType model_type) {
  if (m_verbose) {
    cout << RED << "Memory Usage Before Model Load: " << GetMemoryUsage() << " MB"
         << CLEAR << endl;
  }

  m_onnx_model = OgaModel::Create(model_path);
  m_tokenizer = OgaTokenizer::Create(*m_onnx_model);
  m_tokenizer_stream = OgaTokenizerStream::Create(*m_tokenizer);

  m_input_decoder = InputDecoder::CreateDecoder("openai");
  if (m_input_decoder == nullptr) {
    cout << "Error!" << endl;
    return false;
  }

  if (!parse_prompt_format_dict(model_type, PromptFormatTable,
                                m_prompt_format)) {
    cout << "Error parsing the prompt format dictionary" << endl;
    return false;
  }

  if (m_verbose) {
    string slm_ver, oga_ver, ort_ver;
    GetVersion(slm_ver, oga_ver, ort_ver);

    cout << "Loaded Model: " << model_path << endl;
    cout << "Model Type: " << ModelTypeToString(model_type) << endl;
    cout << "Prompt Format: " << m_prompt_format.llm_type << endl;
    cout << "SLM Engine Initialized" << endl;
    cout << "SLM VERSION: " << slm_ver << endl;
    cout << "ORT GenAI VERSION: " << oga_ver << endl;
    cout << "ORT VERSION: " << ort_ver << endl;
  }
  m_llm_input_dbg_stream.open("slm-input-records.jsonl");
  m_llm_output_dbg_stream.open("slm-output-records.jsonl");

  return true;
}

// Now define a function to format the input
std::string SLMEngine::format_input(
    const InputDecoder::InputParams& input_params) {
  ostringstream ss_output;
  bool no_assistant_messages = true;
  for (const auto& msg : input_params.Messages) {
    switch (msg.first) {
      case InputDecoder::InputParams::Role::SYSTEM:
        ss_output << m_prompt_format.prompt_format.at(msg.first).prefix
                  << msg.second
                  << m_prompt_format.prompt_format.at(msg.first).suffix;
        break;
      case InputDecoder::InputParams::Role::USER:
        ss_output << m_prompt_format.prompt_format.at(msg.first).prefix
                  << msg.second
                  << m_prompt_format.prompt_format.at(msg.first).suffix;
        // Each time we get a user message we reset the flag
        no_assistant_messages = true;
        break;
      case InputDecoder::InputParams::Role::ASSISTANT:
        ss_output << m_prompt_format.prompt_format.at(msg.first).prefix
                  << msg.second;
        // if there are more messages then add the assistant suffix
        if (msg != input_params.Messages.back()) {
          ss_output
              << m_prompt_format.prompt_format.at(msg.first).suffix;
        }
        no_assistant_messages = false;
        break;
    }
  }

  if (no_assistant_messages) {
    ss_output << m_prompt_format.prompt_format
                     .at(InputDecoder::InputParams::Role::ASSISTANT)
                     .prefix;
  }

  return ss_output.str();
}

uint32_t SLMEngine::GetMemoryUsage() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS_EX pmc;
  if (GetProcessMemoryInfo(
          GetCurrentProcess(),
          (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
    return pmc.WorkingSetSize / (1024 * 1024);
  }
  return 0;
#else
#if defined(__ANDROID__)
  // Read the /proc/self/status file to get the memory usage
  std::ifstream status_file("/proc/self/status");
  std::string line;
  while (std::getline(status_file, line)) {
    if (line.find("VmRSS") != std::string::npos) {
      // remove the non-numeric characters
      line.erase(std::remove_if(
                     line.begin(), line.end(),
                     [](unsigned char c) { return !std::isdigit(c); }),
                 line.end());

      // Convert to MB
      auto memory = std::stoul(line) / 1024;
      return memory;
    }
  }
  return 0;
#else
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  auto current_memory = usage.ru_maxrss;

#if defined(__linux__)
  current_memory = current_memory / 1024;
#elif defined(__aarch64__) && defined(__APPLE__)
  current_memory = current_memory / (1024 * 1024);
#endif
  return current_memory;
#endif
#endif
}

}  // namespace slm_engine
}  // namespace microsoft
