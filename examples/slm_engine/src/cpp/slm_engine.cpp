#include "slm_engine.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
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

#include <nlohmann/json.hpp>

#include "onnxruntime_cxx_api.h"

using namespace std;
using json = nlohmann::json;

#define MAGENTA "\033[35;1m"
#define RED "\033[31;1m"
#define BLUE "\033[34;1m"
#define GREEN "\033[32;1m"
#define CLEAR "\033[0m"

// Function calling instructions to be added to system prompts
const std::string function_calling_instructions = R"(

In addition to plain text responses, you can chose to call one or more of the provided functions.

Use the following rule to decide when to call a function:
  * if the response can be generated from your internal knowledge (e.g., as in the case of queries like "What is the capital of Poland?"), do so
  * if you need external information that can be obtained by calling one or more of the provided functions, generate a function calls

If you decide to call functions:
  * prefix function calls with <|tool|> marker and end with <|/tool|> marker
  * all function calls should be generated in a single JSON list formatted as [{"name": [function name], "arguments": [function arguments as JSON]}, ...]
  * follow the provided JSON schema. Do not hallucinate arguments or values. Do not blindly copy values from the provided samples
  * respect the argument type formatting. E.g., if the type is number and format is float, write value 7 as 7.0
  * make sure you pick the right functions that match the user intent

Available functions as JSON spec:
)";

namespace microsoft {
namespace slm_engine {

SLMEngine::SupportedModelType SLMEngine::StringToModelType(const std::string& model_type) {
  if (strncasecmp(model_type.c_str(), "phi", 3) == 0) {
    return SLMEngine::SupportedModelType::PHI;
  } else if (strncasecmp(model_type.c_str(), "llama", 5) == 0) {
    return SLMEngine::SupportedModelType::Llama;
  } else if (strncasecmp(model_type.c_str(), "qwen", 4) == 0) {
    return SLMEngine::SupportedModelType::Qwen;
  } else if (strncasecmp(model_type.c_str(), "custom", 6) == 0) {
    return SLMEngine::SupportedModelType::CUSTOM;
  }
  return SLMEngine::SupportedModelType::UNKNOWN;
}

std::string SLMEngine::ModelTypeToString(SLMEngine::SupportedModelType model_type) {
  switch (model_type) {
    case SLMEngine::SupportedModelType::PHI:
      return "phi";
    case SLMEngine::SupportedModelType::Llama:
      return "llama";
    case SLMEngine::SupportedModelType::Qwen:
      return "qwen";
    case SLMEngine::SupportedModelType::CUSTOM:
      return "custom";
    case SLMEngine::SupportedModelType::UNKNOWN:
    default:
      return "unknown";
  }
}

std::unique_ptr<SLMEngine> SLMEngine::Create(
    const char* model_path, bool verbose) {
  auto new_obj = std::unique_ptr<SLMEngine>(new SLMEngine(verbose));
  if (!new_obj->load_model(model_path)) {
    cout << RED << "Error creating the SLM Engine" << CLEAR << endl;
    return nullptr;
  }
  return std::move(new_obj);
}

std::unique_ptr<SLMEngine> SLMEngine::Create(
    const char* model_path,
    const std::vector<LoRAAdapter> adapters,
    bool verbose, Status& status_msg) {
  // Load the model
  auto new_obj = std::unique_ptr<SLMEngine>(new SLMEngine(verbose));
  if (!new_obj->load_model(model_path)) {
    cout << RED << "Error creating the SLM Engine" << CLEAR << endl;
    status_msg.code = false;
    status_msg.message = "Failed to load model: " + std::string(model_path);
    return nullptr;
  }

  new_obj->m_adapters = OgaAdapters::Create(*new_obj->m_onnx_model.get());
  if (!new_obj->m_adapters) {
    status_msg.code = false;
    status_msg.message = "Failed to create adapters";
    return nullptr;
  }

  // Create the adapters
  for (const auto& adapter : adapters) {
    if (adapter.name.empty() || adapter.adapter_path.empty()) {
      status_msg.code = false;
      status_msg.message = "Adapter name or path is empty";
      return nullptr;
    }

    // Check if the adapter path exists
    std::ifstream file_check(adapter.adapter_path);
    if (!file_check.good()) {
      status_msg.code = false;
      status_msg.message = "Adapter path does not exist: " + adapter.adapter_path;
      return nullptr;
    }
    file_check.close();

    // Load the adapter
    new_obj->m_adapters->LoadAdapter(adapter.adapter_path.c_str(),
                                     adapter.name.c_str());
  }

  new_obj->m_adapters_list = adapters;

  return std::move(new_obj);
}

std::vector<SLMEngine::LoRAAdapter> SLMEngine::get_adapter_list() {
  std::vector<SLMEngine::LoRAAdapter> adapter_list;
  for (const auto& adapter : m_adapters_list) {
    adapter_list.emplace_back(adapter.name, adapter.adapter_path);
  }
  return adapter_list;
}

void SLMEngine::GetVersion(std::string& slm_version, std::string& ortga_version,
                           std::string& ort_version) {
// SW_VERSION_NUMBER is defined in the CMakeLists.txt file
#ifdef SW_VERSION_NUMBER
  slm_version = std::string(SW_VERSION_NUMBER);
#else
  slm_version = "unknown";
#endif

#ifdef ORT_GENAI_VERSION
  ortga_version = std::string(ORT_GENAI_VERSION);
#else
  ortga_version = "unknown";
#endif

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

std::string SLMEngine::format_prompt(
    const std::string& system_prompt,
    const std::string& user_prompt) {
  std::stringstream ss_output;
  ss_output << m_prompt_format.prompt_format.at(InputDecoder::InputParams::Role::SYSTEM).prefix
            << system_prompt
            << m_prompt_format.prompt_format.at(InputDecoder::InputParams::Role::SYSTEM).suffix;
  ss_output << m_prompt_format.prompt_format.at(InputDecoder::InputParams::Role::USER).prefix
            << user_prompt
            << m_prompt_format.prompt_format.at(InputDecoder::InputParams::Role::USER).suffix;
  ss_output << m_prompt_format.prompt_format.at(InputDecoder::InputParams::Role::ASSISTANT).prefix;

  return ss_output.str();
}

SLMEngine::~SLMEngine() {
  m_onnx_model.reset();
  m_tokenizer.reset();
  m_tokenizer_stream.reset();
  m_input_decoder.reset();
}

std::unique_ptr<OgaGenerator> SLMEngine::create_generator(
    const std::string& formatted_prompt,
    const GenerationOptions& generation_options,
    uint32_t& time_to_prefill) {
  auto generator_params = OgaGeneratorParams::Create(*m_onnx_model);
  if (!generator_params) {
    return nullptr;
  }

  generator_params->SetSearchOption("max_length", generation_options.MaxGeneratedTokens);
  generator_params->SetSearchOption("temperature", generation_options.Temperature);
  generator_params->SetSearchOption("top_p", generation_options.TopP);
  generator_params->SetSearchOption("top_k", generation_options.TopK);

  auto mem_before = GetMemoryUsage();

  // Create the generator
  auto generator = OgaGenerator::Create(*m_onnx_model, *generator_params);
  if (!generator) {
    return nullptr;
  }

  auto sequences = OgaSequences::Create();

  auto start = std::chrono::steady_clock::now();
  m_tokenizer->Encode(formatted_prompt.c_str(), *sequences);
  auto time_to_encode =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start)
          .count();

  start = std::chrono::steady_clock::now();

  generator->AppendTokenSequences(*sequences);
  time_to_prefill =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start)
          .count();

  auto mem_after = GetMemoryUsage();

  if (m_verbose) {
    cout << BLUE << "Time to encode: " << time_to_encode
         << " ms Initial Tokens: " << generator->GetSequenceCount(0)
         << " Time to append: " << time_to_prefill << " ms" << CLEAR << endl;

    cout << BLUE << "Memory used: " << mem_after - mem_before << " bytes" << CLEAR << endl;
  }

  return std::move(generator);
}

SLMEngine::Status SLMEngine::generate(
    const std::string& adapter_name,
    const std::string& formatted_prompt,
    const GenerationOptions& generation_options,
    std::string& response_str,
    RuntimePerf& kpi) {
  // Verify that the adapter is a valid one
  if (!m_adapters) {
    return Status{false, "Adapter not found: " + adapter_name};
  }
  auto api_start = std::chrono::steady_clock::now();

  uint32_t time_to_prefill;
  auto generator = create_generator(
      formatted_prompt, generation_options, time_to_prefill);

  if (!generator) {
    return Status{false, "Failed to create generator"};
  }

  // Set the adapter
  generator->SetActiveAdapter(*(m_adapters.get()), adapter_name.c_str());

  // Add the time_to_prefill to the KPI
  kpi.TimeToFirstToken = time_to_prefill;

  // Delegate to generate
  auto status = generate(generator.get(), nullptr, response_str, kpi);
  kpi.TotalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - api_start)
                      .count();
  return status;
}

SLMEngine::Status SLMEngine::generate(
    const std::string& formatted_prompt,
    const GenerationOptions& generation_options,
    std::string& response_str,
    RuntimePerf& kpi) {
  auto api_start = std::chrono::steady_clock::now();

  uint32_t time_to_prefill;
  auto generator = create_generator(
      formatted_prompt, generation_options, time_to_prefill);

  if (!generator) {
    cout << RED << "Error creating the generator" << CLEAR << endl;
    return Status{false, "Error creating the generator"};
  }

  kpi.TimeToFirstToken = time_to_prefill;
  generate(generator.get(), nullptr, response_str, kpi);
  kpi.TotalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - api_start)
                      .count();
  return Status{true, "Generation successful"};
}

SLMEngine ::Status SLMEngine::generate(
    OgaGenerator* generator,
    std::function<bool(const std::string&, OgaTensor* logits)> generation_callback,
    std::string& response_str,
    RuntimePerf& kpi) {
  std::lock_guard<std::mutex> lock(m_mutex);

  auto start = std::chrono::steady_clock::now();
  bool is_first_token = true;
  auto time_count = 0;

  auto initial_prompt_token_count = generator->GetSequenceCount(0);

  int count = 0;
  uint32_t total_generation_time = 0;
  std::ostringstream response;
  while (!generator->IsDone()) {
    auto gen_start = std::chrono::steady_clock::now();
    generator->GenerateNextToken();
    auto gen_end = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start)
            .count();
    total_generation_time += elapsed;
    count++;
    // cout << BLUE << "Generation time: " << elapsed << " us" << CLEAR
    //      << endl;

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
      kpi.TimeToFirstToken += elapsed;
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
    } else {
      cout << RED << "Got </s>!!!" << CLEAR << endl;
    }

    // Print next output string if the generation continues
    if (m_verbose) {
      cout << next_string_piece;
      flush(cout);
    }

    // Call the generation callback if provided
    if (generation_callback) {
      auto logits = generator->GetLogits();
      if (logits) {
        if (!generation_callback(next_string_piece, logits.get())) {
          cout << RED << "Sopping generation due to callback request." << endl;
          break;
        }
      }
    }

    // Reset the start time for the next token
    start = std::chrono::steady_clock::now();
  }

  // Find out the generation time
  uint32_t avg_generation_time =
      static_cast<float>(total_generation_time) / static_cast<float>(count);
  kpi.GenerationTimePerToken = avg_generation_time;

  if (m_verbose) {
    cout << CLEAR << endl;
  }

  response_str = response.str();
  m_llm_output_dbg_stream << response_str << endl;

  kpi.GeneratedTokenCount =
      generator->GetSequenceCount(0) - initial_prompt_token_count;
  kpi.TokenRate = kpi.GeneratedTokenCount / (time_count / 1000.0f);

  // // Get the current memory
  kpi.CurrentMemoryUsed = GetMemoryUsage();
  return Status({true, "Generation successful"});
}

std::string SLMEngine::complete(const char* user_prompt) {
  InputDecoder::InputParams input_parameters;
  // Decode the user prompt
  if (!m_input_decoder->decode(user_prompt, input_parameters)) {
    cout << RED << "❌ Error decoding input message: " << user_prompt << CLEAR << endl;
    json output_json;
    output_json["status"] = "error";
    output_json["message"] = "Error decoding input message: " + string(user_prompt);
    return output_json.dump();
  }

  // cout<< BLUE << input_parameters  << endl;

  // Check if tools are provided for function calling
  bool use_function_calling = input_parameters.HasTools && !input_parameters.ToolsJson.empty();

  if (m_verbose) {
    cout << "Input Parameters processed successfully" << endl;
  }

  cout << "Input Parameters has tools: " << input_parameters.HasTools << endl;

  // Format prompt with tools if function calling is enabled
  std::string formatted_prompt;
  if (use_function_calling) {
    formatted_prompt = format_input_with_tools(input_parameters);
  } else {
    formatted_prompt = format_input(input_parameters);
  }

  m_llm_input_dbg_stream << formatted_prompt << endl;

  if (m_verbose) {
    cout << BLUE << "User: " << input_parameters.UserPrompt << endl;
    if (use_function_calling) {
      cout << BLUE << "🔧 Function calling mode enabled with tools" << endl;
    }
    cout << GREEN;
  }

  RuntimePerf kpi;
  std::string response;
  FunctionCallResult function_result;

  GenerationOptions generator_options;
  generator_options.MaxGeneratedTokens = input_parameters.MaxGeneratedTokens;
  generator_options.Temperature = input_parameters.Temperature;
  generator_options.TopP = input_parameters.TopP;

  SLMEngine::Status status;
  auto api_start = std::chrono::steady_clock::now();

  if (use_function_calling) {
    cout << BLUE << "🔧 Function calling mode enabled" << CLEAR << endl;
    // Setup function calling options
    FunctionCallOptions function_options;
    function_options.tools = parse_tools_from_json(input_parameters.ToolsJson);

    if (m_verbose) {
      cout << BLUE << "   Tools available: " << function_options.tools.size() << CLEAR << endl;
      for (const auto& tool : function_options.tools) {
        cout << BLUE << "   - " << tool.name << ": " << tool.description << CLEAR << endl;
      }
    }

    if (m_verbose) {
      cout << RED << "Function result status: " << "FUNCTION_CALL"
           << ", calls: " << function_options.tools.size() << CLEAR << endl;
    }

    if (input_parameters.LoRAAdapterName.empty()) {
      status = generate_with_functions(formatted_prompt, generator_options,
                                       function_options, response, function_result, kpi);
    } else {
      status = generate_with_functions(input_parameters.LoRAAdapterName, formatted_prompt,
                                       generator_options, function_options, response, function_result, kpi);
    }
  } else {
    // Regular generation without function calling
    if (input_parameters.LoRAAdapterName.empty()) {
      status = generate(formatted_prompt, generator_options, response, kpi);
    } else {
      status = generate(input_parameters.LoRAAdapterName, formatted_prompt,
                        generator_options, response, kpi);
    }
  }

  kpi.TotalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - api_start)
                      .count();

  m_llm_output_dbg_stream << response << endl;

  // Remove stop tokens from response
  for (const auto& stop_token : input_parameters.StopTokens) {
    auto stop_token_pos = response.find(stop_token);
    if (stop_token_pos != std::string::npos) {
      response = response.substr(0, stop_token_pos);
      break;
    }
  }

  json output_json;
  if (!status.code) {
    output_json["status"] = "error";
    output_json["message"] = status.message;
    return output_json.dump();
  }

  output_json["status"] = "success";
  output_json["question"] = input_parameters.UserPrompt;
  output_json["llm_input"] = formatted_prompt;

  // Handle function calling response
  if (use_function_calling && function_result.is_function_call) {
    // Function call(s) detected - format answer as JSON array string like in your examples
    json function_calls_json = json::array();
    for (const auto& call : function_result.function_calls) {
      json call_json;
      call_json["name"] = call.function_name;

      // Parse the parameters_json string into a JSON object for arguments
      try {
        call_json["arguments"] = json::parse(call.parameters_json);
      } catch (const json::exception& e) {
        // If parsing fails, store as raw string
        call_json["arguments"] = call.parameters_json;
      }

      function_calls_json.push_back(call_json);
    }

    json response_data = {
        {"answer", function_calls_json.dump()}  // Store as JSON string like in your examples
    };

    cout << GREEN << "Function call detected" << CLEAR << endl;

    // Always add the structured function_calls array (unified format)
    json function_calls_array = json::array();
    for (const auto& call : function_result.function_calls) {
      json structured_call;
      structured_call["name"] = call.function_name;
      structured_call["arguments"] = call.parameters_json;  // Keep as string format as requested

      function_calls_array.push_back(structured_call);
    }
    response_data["function_calls"] = function_calls_array;

    output_json["response"] = response_data;

    if (m_verbose) {
      if (function_result.function_calls.size() == 1) {
        cout << GREEN << "📞 Function call response: " << function_result.function_calls[0].function_name << CLEAR << endl;
      } else {
        cout << GREEN << "📞 Multiple function calls response (" << function_result.function_calls.size() << " calls):" << CLEAR << endl;
        for (size_t i = 0; i < function_result.function_calls.size(); ++i) {
          cout << GREEN << "   " << (i + 1) << ". " << function_result.function_calls[i].function_name << CLEAR << endl;
        }
      }
    }
  } else {
    // Regular text response
    json response_data = {
        {"answer", use_function_calling ? function_result.text_response : response}};
    output_json["response"] = response_data;

    if (m_verbose && use_function_calling) {
      cout << GREEN << "💬  Response  with tools available" << CLEAR << endl;
    }
  }

  json kpi_json;
  kpi_json["prompt_toks"] = kpi.PromptTokenCount;
  kpi_json["ttft"] = kpi.TimeToFirstToken;
  kpi_json["generated_toks"] = kpi.GeneratedTokenCount;
  kpi_json["tok_rate"] = kpi.TokenRate;
  kpi_json["total_time"] = kpi.TotalTime;
  kpi_json["memory_usage"] = kpi.CurrentMemoryUsed;

  output_json["kpi"] = kpi_json;

  // Return the output_json directly (not wrapped in "response")
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
         "system": { "prefix": "<|system|>", "suffix": "<|end|>" },
         "user": { "prefix": "<|user|>", "suffix": "<|end|>" },
         "assistant": { "prefix": "<|assistant|>", "suffix": "<|end|>" },
         "tool": { "prefix": "<|tool|>", "suffix": "<|/tool|>" }
      }
   },
   {     
      "llm_type": "llama",
      "prompt_format": {
         "system": { "prefix": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", "suffix": "<|eot_id|>" },
         "user": { "prefix": "<|start_header_id|>user<|end_header_id|>\n\n", "suffix": "<|eot_id|>" },
         "assistant": { "prefix": "<|start_header_id|>assistant<|end_header_id|>\n\n", "suffix": "<|eot_id|>" },
         "tool": { "prefix": "<|tool|>", "suffix": "<|/tool|>" }
      }
   },
   {     
      "llm_type": "qwen",
      "prompt_format": {
         "system": { "prefix": "<|im_start|>system\n", "suffix": "<|im_end|>" },
         "user": { "prefix": "<|im_start|>user\n/no_think ", "suffix": "<|im_end|>" },
         "assistant": { "prefix": "<|im_start|>assistant\n", "suffix": "<|im_end|>" },
         "tool": { "prefix": "<|tool|>", "suffix": "<|/tool|>" }
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

bool SLMEngine::load_model(const char* model_path) {
  if (m_verbose) {
    cout << RED << "Memory Usage Before Model Load: " << GetMemoryUsage() << " MB"
         << CLEAR << endl;
  }

  m_model_path = model_path;
  m_model_type = GetModelFamily(m_model_path);

  // Convert the model name to the SupportedModelType
  auto model_type = StringToModelType(m_model_type);
  if (model_type == SupportedModelType::UNKNOWN) {
    cout << RED << "Error! Cannot detect the model type for model: " << model_path << CLEAR
         << endl;
    return false;
  }

  m_onnx_model = OgaModel::Create(model_path);
  m_tokenizer = OgaTokenizer::Create(*m_onnx_model);
  m_tokenizer_stream = OgaTokenizerStream::Create(*m_tokenizer);
  // m_generator_params = OgaGeneratorParams::Create(*m_onnx_model);
  // m_sequences = OgaSequences::Create();

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
      case InputDecoder::InputParams::Role::TOOL:
        ss_output << m_prompt_format.prompt_format.at(msg.first).prefix
                  << msg.second
                  << m_prompt_format.prompt_format.at(msg.first).suffix;
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

// Format input with tools for function calling (Phi format)
std::string SLMEngine::format_input_with_tools(
    const InputDecoder::InputParams& input_params) {
  ostringstream ss_output;
  bool no_assistant_messages = true;

  for (const auto& msg : input_params.Messages) {
    switch (msg.first) {
      case InputDecoder::InputParams::Role::SYSTEM:
        ss_output << m_prompt_format.prompt_format.at(msg.first).prefix
                  << msg.second;

        // Add function calling instructions and tools information to system message if available
        if (input_params.HasTools && !input_params.ToolsJson.empty()) {
          // Add function calling instructions
          ss_output << function_calling_instructions;

          // Add the actual tools JSON spec
          ss_output << input_params.ToolsJson;
        }

        ss_output << m_prompt_format.prompt_format.at(msg.first).suffix;
        break;

      case InputDecoder::InputParams::Role::USER:
        ss_output << m_prompt_format.prompt_format.at(msg.first).prefix
                  << msg.second
                  << m_prompt_format.prompt_format.at(msg.first).suffix;
        no_assistant_messages = true;
        break;

      case InputDecoder::InputParams::Role::TOOL:
        ss_output << m_prompt_format.prompt_format.at(msg.first).prefix
                  << msg.second
                  << m_prompt_format.prompt_format.at(msg.first).suffix;
        break;

      case InputDecoder::InputParams::Role::ASSISTANT:
        ss_output << m_prompt_format.prompt_format.at(msg.first).prefix
                  << msg.second;
        if (msg != input_params.Messages.back()) {
          ss_output << m_prompt_format.prompt_format.at(msg.first).suffix;
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

bool SLMEngine::create_lark_grammar(const std::vector<FunctionTool>& tools,
                                    std::string& prompt_tool_input,
                                    std::string& grammar_input) {
  if (tools.empty()) {
    return false;
  }

  prompt_tool_input = create_prompt_tool_input(tools);

  if (tools.size() == 1) {
    // Single tool case
    std::string tool_schema = convert_tool_to_grammar_input(tools[0]);
    grammar_input =
        "start: TEXT | fun_call\n"
        "TEXT: /[^{](.|\\n)*/\n"
        " fun_call: <|tool_call|> %json " +
        tool_schema;
  } else {
    // Multiple tools case
    std::string anyof_schema = "{\"anyOf\": [";
    for (size_t i = 0; i < tools.size(); ++i) {
      if (i > 0) anyof_schema += ",";
      anyof_schema += convert_tool_to_grammar_input(tools[i]);
    }
    anyof_schema += "]}";

    grammar_input =
        "start: TEXT | fun_call\n"
        "TEXT: /[^{](.|\\n)*/\n"
        " fun_call: <|tool_call|> %json " +
        anyof_schema;
  }

  return true;
}

std::string SLMEngine::convert_tool_to_grammar_input(const FunctionTool& tool) {
  json param_props = json::object();
  json required_params = json::array();

  for (const auto& [param_name, param_info] : tool.parameters) {
    param_props[param_name] = {
        {"type", param_info.type},
        {"description", param_info.description}};
    required_params.push_back(param_name);
  }

  json output_schema = {
      {"description", tool.description},
      {"type", "object"},
      {"required", {"name", "parameters"}},
      {"additionalProperties", false},
      {"properties", {{"name", {{"const", tool.name}}}, {"parameters", {{"type", "object"}, {"properties", param_props}, {"required", required_params}, {"additionalProperties", false}}}}}};

  if (param_props.empty()) {
    output_schema["required"] = json::array({"name"});
  }

  return output_schema.dump();
}

std::string SLMEngine::create_prompt_tool_input(const std::vector<FunctionTool>& tools) {
  json tools_json = json::array();

  for (const auto& tool : tools) {
    json tool_json = {
        {"name", tool.name},
        {"description", tool.description},
        {"parameters", json::object()}};

    for (const auto& [param_name, param_info] : tool.parameters) {
      tool_json["parameters"][param_name] = {
          {"description", param_info.description},
          {"type", param_info.type}};
      if (!param_info.default_value.empty()) {
        tool_json["parameters"][param_name]["default"] = param_info.default_value;
      }
    }

    tools_json.push_back(tool_json);
  }

  return tools_json.dump();
}

bool SLMEngine::parse_function_call(const std::string& generated_text,
                                    FunctionCallResult& function_result) {
  // Look for multiple function call patterns: <|tool_call|>{...}
  std::regex function_call_regex(R"(<\|tool_call\|>\s*(\{.*?\}))");
  std::sregex_iterator iter(generated_text.begin(), generated_text.end(), function_call_regex);
  std::sregex_iterator end;

  std::vector<FunctionCall> detected_calls;
  size_t first_call_pos = std::string::npos;

  for (auto it = iter; it != end; ++it) {
    std::smatch match = *it;
    try {
      std::string json_str = match[1].str();
      json function_call = json::parse(json_str);

      if (function_call.contains("name") && function_call.contains("parameters")) {
        std::string name = function_call["name"];
        std::string params = function_call["parameters"].dump();
        detected_calls.emplace_back(name, params);

        // Record position of first function call for text extraction
        if (first_call_pos == std::string::npos) {
          first_call_pos = match.position();
        }

        if (m_verbose) {
          cout << GREEN << "📞 Detected function call: " << name << CLEAR << endl;
        }
      }
    } catch (const json::exception& e) {
      if (m_verbose) {
        cout << RED << "Error parsing function call JSON: " << e.what() << CLEAR << endl;
      }
    }
  }

  if (!detected_calls.empty()) {
    function_result.is_function_call = true;
    function_result.function_calls = std::move(detected_calls);

    // Extract text before first function call as text response
    if (first_call_pos > 0) {
      function_result.text_response = generated_text.substr(0, first_call_pos);
      // Trim whitespace
      function_result.text_response.erase(
          function_result.text_response.find_last_not_of(" \n\r\t") + 1);
    }

    if (m_verbose) {
      cout << GREEN << "🔧 Total function calls detected: " << detected_calls.size() << CLEAR << endl;
    }

    return true;
  }

  // No function call found, treat as regular text
  function_result.is_function_call = false;
  function_result.text_response = generated_text;
  return false;
}

std::unique_ptr<OgaGenerator> SLMEngine::create_function_generator(
    const std::string& formatted_prompt,
    const GenerationOptions& generation_options,
    const FunctionCallOptions& function_options,
    uint32_t& time_to_prefill) {
  auto generator_params = OgaGeneratorParams::Create(*m_onnx_model);
  if (!generator_params) {
    return nullptr;
  }

  generator_params->SetSearchOption("max_length", generation_options.MaxGeneratedTokens);
  generator_params->SetSearchOption("temperature", generation_options.Temperature);
  generator_params->SetSearchOption("top_k", generation_options.TopK);
  generator_params->SetSearchOption("top_p", generation_options.TopP);

  auto mem_before = GetMemoryUsage();

  // Create the generator
  auto generator = OgaGenerator::Create(*m_onnx_model, *generator_params);
  if (!generator) {
    return nullptr;
  }

  auto sequences = OgaSequences::Create();

  auto start = std::chrono::steady_clock::now();
  m_tokenizer->Encode(formatted_prompt.c_str(), *sequences);
  auto time_to_encode =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start)
          .count();

  start = std::chrono::steady_clock::now();

  generator->AppendTokenSequences(*sequences);
  time_to_prefill =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start)
          .count();

  auto mem_after = GetMemoryUsage();

  if (m_verbose) {
    cout << BLUE << "Time to encode: " << time_to_encode
         << " ms Initial Tokens: " << generator->GetSequenceCount(0)
         << " Time to append: " << time_to_prefill << " ms" << CLEAR << endl;

    cout << BLUE << "Memory used: " << mem_after - mem_before << " bytes" << CLEAR << endl;
  }

  return std::move(generator);
}

SLMEngine::Status SLMEngine::generate_with_functions(
    const std::string& formatted_prompt,
    const GenerationOptions& generation_options,
    const FunctionCallOptions& function_options,
    std::string& response_str,
    FunctionCallResult& function_result,
    RuntimePerf& kpi) {
  auto api_start = std::chrono::steady_clock::now();

  uint32_t time_to_prefill;
  auto generator = create_function_generator(
      formatted_prompt, generation_options, function_options, time_to_prefill);

  if (!generator) {
    cout << RED << "Error creating the function generator" << CLEAR << endl;
    return Status{false, "Error creating the function generator"};
  }

  kpi.TimeToFirstToken = time_to_prefill;

  // Generate response
  auto status = generate(generator.get(), nullptr, response_str, kpi);

  if (status.code) {
    // Parse function call from response
    parse_function_call(response_str, function_result);

    if (m_verbose && function_result.is_function_call) {
      cout << MAGENTA << "Function Call Detected: " << function_result.function_name()
           << " with parameters: " << function_result.parameters_json() << CLEAR << endl;
    }
  }

  kpi.TotalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - api_start)
                      .count();
  return status;
}

SLMEngine::Status SLMEngine::generate_with_functions(
    const std::string& adapter_name,
    const std::string& formatted_prompt,
    const GenerationOptions& generation_options,
    const FunctionCallOptions& function_options,
    std::string& response_str,
    FunctionCallResult& function_result,
    RuntimePerf& kpi) {
  // Verify that the adapter is a valid one
  if (!m_adapters) {
    return Status{false, "Adapter not found: " + adapter_name};
  }

  auto api_start = std::chrono::steady_clock::now();

  uint32_t time_to_prefill;
  auto generator = create_function_generator(
      formatted_prompt, generation_options, function_options, time_to_prefill);

  if (!generator) {
    return Status{false, "Failed to create function generator"};
  }

  // Set the adapter
  generator->SetActiveAdapter(*(m_adapters.get()), adapter_name.c_str());

  // Add the time_to_prefill to the KPI
  kpi.TimeToFirstToken = time_to_prefill;

  // Generate response
  auto status = generate(generator.get(), nullptr, response_str, kpi);

  if (status.code) {
    // Parse function call from response
    parse_function_call(response_str, function_result);

    if (m_verbose && function_result.is_function_call) {
      cout << MAGENTA << "Function Call Detected: " << function_result.function_name()
           << " with parameters: " << function_result.parameters_json() << CLEAR << endl;
    }
  }

  kpi.TotalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - api_start)
                      .count();
  return status;
}

std::vector<SLMEngine::FunctionTool> SLMEngine::parse_tools_from_json(const std::string& tools_json) {
  std::vector<FunctionTool> tools;

  try {
    json tools_array = json::parse(tools_json);

    for (const auto& tool_json : tools_array) {
      if (tool_json.contains("name") && tool_json.contains("description")) {
        FunctionTool tool(tool_json["name"], tool_json["description"]);

        if (tool_json.contains("parameters")) {
          const auto& params = tool_json["parameters"];
          for (auto it = params.begin(); it != params.end(); ++it) {
            std::string param_name = it.key();
            const auto& param_info = it.value();

            FunctionParameter param;
            if (param_info.contains("description")) {
              param.description = param_info["description"];
            }
            if (param_info.contains("type")) {
              param.type = param_info["type"];
            }
            if (param_info.contains("default")) {
              param.default_value = param_info["default"];
            }

            tool.parameters[param_name] = param;
          }
        }

        tools.push_back(tool);
      }
    }
  } catch (const json::exception& e) {
    if (m_verbose) {
      cout << RED << "Error parsing tools JSON: " << e.what() << CLEAR << endl;
    }
  }

  return tools;
}

}  // namespace slm_engine
}  // namespace microsoft
