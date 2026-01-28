// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <optional>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include "ort_genai.h"

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Duration = std::chrono::duration<double>;

// `Timing` is a utility class for measuring performance metrics.
class Timing {
 public:
  Timing(const Timing&) = delete;
  Timing& operator=(const Timing&) = delete;
  Timing() = default;
  ~Timing() = default;

  void RecordStartTimestamp();
  void RecordFirstTokenTimestamp();
  void RecordEndTimestamp();
  void Log(const int prompt_tokens_length, const int new_tokens_length);

 private:
  TimePoint start_timestamp_;
  TimePoint first_token_timestamp_;
  TimePoint end_timestamp_;
};

/**
 * @brief Trim user-provided filepath
 *
 * @param str Filepath to trim
 *
 * @return Trimmed filepath
 */
std::string Trim(const std::string& str);

/**
 * @brief A class for defining a tool in a JSON schema compatible way
 */
struct ToolSchema {
  std::string description;
  std::string type;
  nlohmann::ordered_json properties;
  std::vector<std::string> required;
  bool additionalProperties;
};

/**
 * @brief Convert ToolSchema to JSON
 *
 * @param j JSON object
 * @param tool ToolSchema object
 *
 * @return None
 */
void to_json(nlohmann::ordered_json& j, const ToolSchema& tool);

/**
 * @brief Convert JSON to ToolSchema
 *
 * @param j JSON object
 * @param tool ToolSchema object
 *
 * @return None
 */
void from_json(const nlohmann::ordered_json& j, ToolSchema& tool);

/**
 * @brief A class for defining a JSON schema for guidance
 */
struct JsonSchema {
  nlohmann::ordered_json xGuidance;
  std::string type;
  std::unordered_map<std::string, std::vector<ToolSchema>> items;
  int minItems;
};

/**
 * @brief Convert JsonSchema to JSON
 *
 * @param j JSON object
 * @param schema JsonSchema object
 *
 * @return None
 */
void to_json(nlohmann::ordered_json& j, const JsonSchema& schema);

/**
 * @brief Convert JSON to JsonSchema
 *
 * @param j JSON object
 * @param schema JsonSchema object
 *
 * @return None
 */
void from_json(const nlohmann::ordered_json& j, JsonSchema& schema);

/**
 * @brief A class for defining a function in an OpenAI-compatible way
 */
struct FunctionDefinition {
  std::string name;
  std::string description;
  nlohmann::ordered_json parameters;
};

/**
 * @brief Convert FunctionDefinition to JSON
 *
 * @param j JSON object
 * @param func FunctionDefinition object
 *
 * @return None
 */
void to_json(nlohmann::ordered_json& j, const FunctionDefinition& func);

/**
 * @brief Convert JSON to FunctionDefinition
 *
 * @param j JSON object
 * @param func FunctionDefinition object
 *
 * @return None
 */
void from_json(const nlohmann::ordered_json& j, FunctionDefinition& func);

/**
 * @brief A class for defining a tool in an OpenAI-compatible way
 */
struct Tool {
  std::string type;
  FunctionDefinition function;
};

/**
 * @brief Convert Tool to JSON
 *
 * @param j JSON object
 * @param t Tool object
 *
 * @return None
 */
void to_json(nlohmann::ordered_json& j, const Tool& t);

/**
 * @brief Convert JSON to Tool
 *
 * @param j JSON object
 * @param t Tool object
 *
 * @return None
 */
void from_json(const nlohmann::ordered_json& j, Tool& t);

/**
 * @brief A class for holding parsed values for generator params
 */
struct GeneratorParamsArgs {
  int batch_size = 1;
  int chunk_size = 0;
  std::optional<bool> do_sample;
  std::optional<int> min_length;
  std::optional<int> max_length;
  int num_beams = 1;
  int num_return_sequences = 1;
  std::optional<double> repetition_penalty;
  std::optional<double> temperature;
  std::optional<int> top_k;
  std::optional<double> top_p;
};

/**
 * @brief Convert GeneratorParamsArgs to JSON
 *
 * @param j JSON object
 * @param a Args object
 *
 * @return None
 */
void to_json(nlohmann::ordered_json& j, const GeneratorParamsArgs& a);

/**
 * @brief Convert JSON to GeneratorParamsArgs
 *
 * @param j JSON object
 * @param a Args object
 *
 * @return None
 */
void from_json(const nlohmann::ordered_json& j, GeneratorParamsArgs& a);

/**
 * @brief A class for holding parsed values for guidance
 */
struct GuidanceArgs {
  std::string response_format = "";
  std::string tools_file = "";
  bool text_output = false;
  bool tool_output = false;
  std::string tool_call_start = "";
  std::string tool_call_end = "";
};

/**
 * @brief Parse command-line arguments from user
 *
 * @param argc Number of command-line arguments provided
 * @param argv Contents of command-line arguments provided
 * @param generator_params_args Struct to hold args for generation params
 * @param guidance_args Struct to hold args for guidance
 * @param model_path Path to model folder containing GenAI config
 * @param ep Name of execution provider to set
 * @param system_prompt System prompt to use for the model
 * @param user_prompt User prompt to use for the model
 * @param verbose Use verbose logging
 * @param interactive Run in interactive mode
 * @param rewind Rewind to the system prompt after each generation
 *
 * @return true if command-line arguments can be parsed, else false
 */
bool ParseArgs(int argc, char** argv, GeneratorParamsArgs& generator_params_args, GuidanceArgs& guidance_args, std::string& model_path, std::string& ep, std::string& system_prompt, std::string& user_prompt, bool& verbose, bool& interactive, bool& rewind);

/**
 * @brief Set log options inside ORT GenAI
 *
 * @param inputs Dump inputs to the model in the console
 * @param outputs Dump outputs to the model in the console
 *
 * @return None
 */
void SetLogger(bool inputs = true, bool outputs = true);

/**
 * @brief Get OgaConfig object and set EP-specific and search-specific options inside it
 *
 * @param path Path to model folder containing GenAI config
 * @param ep Name of execution provider to set
 * @param ep_options Map of EP-specific option names and their values
 * @param search_options Map of search-specific option names and their values
 *
 * @return ORT GenAI config object with all options set
 */
std::unique_ptr<OgaConfig> GetConfig(const std::string& path, const std::string& ep, const std::unordered_map<std::string, std::string>& ep_options, GeneratorParamsArgs& search_options);

/**
 * @brief Set search options for a generator's params during decoding
 *
 * @param generatorParams Generator params object to set on
 * @param args Arguments provided by user
 * @param verbose Use verbose logging
 *
 * @return None
 */
void SetSearchOptions(OgaGeneratorParams& generatorParams, GeneratorParamsArgs& args, bool verbose);

/**
 * @brief Apply the chat template with various fallback options
 *
 * @param model_path Path to folder containing model
 * @param tokenizer Tokenizer object to use
 * @param messages String-encoded list of messages
 * @param add_generation_prompt Add tokens to indicate the start of the AI's response
 * @param tools String-encoded list of tools
 *
 * @return Prompt to encode
 */
std::string ApplyChatTemplate(const std::string& model_path, OgaTokenizer& tokenizer, const std::string& messages, bool add_generation_prompt, const std::string& tools = "");

/**
 * @brief Convert a list of tools to a list of tool schemas
 *
 * @param tools List of OpenAI-compatible tools
 *
 * @return List of JSON schema compatible tools
 */
std::vector<ToolSchema> ToolsToSchemas(std::vector<Tool>& tools);

/**
 * @brief Create a JSON schema from a list of tools
 *
 * @param tools List of OpenAI-compatible tools
 * @param tool_output Output can have a tool call
 *
 * @return JSON schema as a JSON-compatible string
 */
std::string GetJsonSchema(std::vector<Tool>& tools, bool tool_output);

/**
 * @brief Create a LARK grammar from a list of tools
 *
 * @param tools List of OpenAI-compatible tools
 * @param text_output Output can have text
 * @param tool_output Output can have a tool call
 * @param tool_call_start String representation of tool call starting token
 * @param tool_call_end String representation of tool call ending token
 *
 * @return LARK grammar as a string
 */
std::string GetLarkGrammar(std::vector<Tool>& tools, bool text_output, bool tool_output, const std::string& tool_call_start, const std::string& tool_call_end);

/**
 * @brief Convert a JSON-deserialized object of tools to a list of Tool objects
 *
 * @param tool_defs JSON-deserialized object containing OpenAI-compatible tool definitions
 *
 * @return List of Tool objects
 */
std::vector<Tool> ToTool(std::vector<nlohmann::ordered_json>& tool_defs);

/**
 * @brief Create a grammar to use with LLGuidance
 *
 * @param response_format Type of format requested
 * @param filepath Path to file containing OpenAI-compatible tool definitions
 * @param tools_str JSON-serialized string containing OpenAI-compatible tool definitions
 * @param tools List of OpenAI-compatible tools defined in memory
 * @param text_output Output can have text
 * @param tool_output Output can have a tool call
 * @param tool_call_start String representation of tool call starting token
 * @param tool_call_end String representation of tool call ending token
 *
 * @return (grammar type, grammar data, tools) as a tuple of strings
 */
std::tuple<std::string, std::string, std::string> GetGuidance(const std::string& response_format = "", const std::string& filepath = "", const std::string& tools_str = "", std::vector<nlohmann::ordered_json>* tools = nullptr, bool text_output = true, bool tool_output = false, const std::string& tool_call_start = "", const std::string& tool_call_end = "");
