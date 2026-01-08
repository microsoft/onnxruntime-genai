// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>

#include "common.h"
#include "getopt.h"

void Timing::RecordStartTimestamp() {
  assert(start_timestamp_.time_since_epoch().count() == 0);
  start_timestamp_ = Clock::now();
}

void Timing::RecordFirstTokenTimestamp() {
  assert(first_token_timestamp_.time_since_epoch().count() == 0);
  first_token_timestamp_ = Clock::now();
}

void Timing::RecordEndTimestamp() {
  assert(end_timestamp_.time_since_epoch().count() == 0);
  end_timestamp_ = Clock::now();
}

void Timing::Log(const int prompt_tokens_length, const int new_tokens_length) {
  assert(start_timestamp_.time_since_epoch().count() != 0);
  assert(first_token_timestamp_.time_since_epoch().count() != 0);
  assert(end_timestamp_.time_since_epoch().count() != 0);

  Duration prompt_time = (first_token_timestamp_ - start_timestamp_);
  Duration run_time = (end_timestamp_ - first_token_timestamp_);

  const auto default_precision{std::cout.precision()};
  std::cout << std::endl;
  std::cout << "-------------" << std::endl;
  std::cout << std::fixed << std::showpoint << std::setprecision(2)
            << "Prompt length: " << prompt_tokens_length << ", New tokens: " << new_tokens_length
            << ", Time to first: " << prompt_time.count() << "s"
            << ", Prompt tokens per second: " << prompt_tokens_length / prompt_time.count() << " tps"
            << ", New tokens per second: " << new_tokens_length / run_time.count() << " tps"
            << std::setprecision(default_precision) << std::endl;
  std::cout << "-------------" << std::endl;
}

std::string Trim(const std::string& str) {
  const size_t first = str.find_first_not_of(' ');
  if (std::string::npos == first) {
    return str;
  }
  const size_t last = str.find_last_not_of(' ');
  return str.substr(first, (last - first + 1));
}

void to_json(nlohmann::ordered_json& j, const ToolSchema& tool) {
  j = nlohmann::ordered_json{ {"description", tool.description}, {"type", tool.type}, {"properties", tool.properties}, {"required", tool.required}, {"additionalProperties", tool.additionalProperties} };
}

void from_json(const nlohmann::ordered_json& j, ToolSchema& tool) {
  j.at("type").get_to(tool.type);

  if (j.contains("description")) {
    j.at("description").get_to(tool.description);
  }

  if (j.contains("properties")) {
    tool.properties = j.at("properties");
  }

  if (j.contains("required")) {
    j.at("required").get_to(tool.required);
  }

  if (j.contains("additionalProperties")) {
    j.at("additionalProperties").get_to(tool.additionalProperties);
  }
  else {
    tool.additionalProperties = false;
  }
}

void to_json(nlohmann::ordered_json& j, const JsonSchema& schema) {
  j = nlohmann::ordered_json{ {"x-guidance", schema.xGuidance}, {"type", schema.type}, {"items", schema.items}, {"minItems", schema.minItems} };
}

void from_json(const nlohmann::ordered_json& j, JsonSchema& schema) {
  j.at("x-guidance").get_to(schema.xGuidance);
  j.at("type").get_to(schema.type);
  j.at("items").get_to(schema.items);
  j.at("minItems").get_to(schema.minItems);
}

void to_json(nlohmann::ordered_json& j, const FunctionDefinition& func) {
  j = nlohmann::ordered_json{ {"name", func.name}, {"description", func.description}, {"parameters", func.parameters} };
}

void from_json(const nlohmann::ordered_json& j, FunctionDefinition& func) {
  j.at("name").get_to(func.name);

  if (j.contains("description")) {
    j.at("description").get_to(func.description);
  }

  if (j.contains("parameters")) {
    func.parameters = j.at("parameters");
  }
}

void to_json(nlohmann::ordered_json& j, const Tool& t) {
  j = nlohmann::ordered_json{ {"type", t.type}, {"function", t.function} };
}

void from_json(const nlohmann::ordered_json& j, Tool& t) {
  j.at("type").get_to(t.type);
  j.at("function").get_to(t.function);
}

void to_json(nlohmann::ordered_json& j, const GeneratorParamsArgs& a) {
  j = nlohmann::ordered_json{ {"batch_size", a.batch_size}, {"num_beams", a.num_beams}, {"num_return_sequences", a.num_return_sequences} };
  // Add optional generator params if provided
  if (a.chunk_size != 0) j["chunk_size"] = a.chunk_size;
  if (a.do_sample) j["do_sample"] = a.do_sample.value();
  if (a.min_length) j["min_length"] = a.min_length.value();
  if (a.max_length) j["max_length"] = a.max_length.value();
  if (a.repetition_penalty) j["repetition_penalty"] = a.repetition_penalty.value();
  if (a.temperature) j["temperature"] = a.temperature.value();
  if (a.top_k) j["top_k"] = a.top_k.value();
  if (a.top_p) j["top_p"] = a.top_p.value();
}

void from_json(const nlohmann::ordered_json& j, GeneratorParamsArgs& a) {
  if (j.contains("batch_size")) j.at("batch_size").get_to(a.batch_size);
  if (j.contains("chunk_size")) j.at("chunk_size").get_to(a.chunk_size);
  if (j.contains("do_sample")) j.at("do_sample").get_to(a.do_sample);
  if (j.contains("min_length")) j.at("min_length").get_to(a.min_length);
  if (j.contains("max_length")) j.at("max_length").get_to(a.max_length);
  if (j.contains("num_beams")) j.at("num_beams").get_to(a.num_beams);
  if (j.contains("num_return_sequences")) j.at("num_return_sequences").get_to(a.num_return_sequences);
  if (j.contains("repetition_penalty")) j.at("repetition_penalty").get_to(a.repetition_penalty);
  if (j.contains("temperature")) j.at("temperature").get_to(a.temperature);
  if (j.contains("top_k")) j.at("top_k").get_to(a.top_k);
  if (j.contains("top_p")) j.at("top_p").get_to(a.top_p);
}

static void PrintUsage(int /*argc*/, char** argv) {
  std::cerr << "usage: " << argv[0] << " <model_path> <execution_provider>" << std::endl;
  std::cerr << "  model_path: [required] Path to the folder containing onnx models, genai_config.json, etc." << std::endl;
  std::cerr << "  execution_provider: [optional] Force use of a particular execution provider (e.g. \"cpu\")" << std::endl;
  std::cerr << "                      If not specified, EP / provider options specified in genai_config.json will be used." << std::endl;
}

std::optional<int> ParseInt(const char* arg) {
  if (!arg) return std::nullopt;
  char* end = nullptr;
  long v = std::strtol(arg, &end, 10);
  if (end == arg || *end != '\0') {
    std::cout << "Invalid integer: \"" << arg << "\"" << std::endl;
    return std::nullopt;
  }
  return static_cast<int>(v);
}

std::optional<double> ParseDouble(const char* arg) {
  if (!arg) return std::nullopt;
  char* end = nullptr;
  double v = std::strtod(arg, &end);
  if (end == arg || *end != '\0') {
    std::cout << "Invalid double: \"" << arg << "\"" << std::endl;
    return std::nullopt;
  }
  return v;
}

std::optional<bool> ParseBool(const char* arg) {
  if (!arg) return std::nullopt;

  // Make lowercase string
  std::string s(arg);
  for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  
  if (s == "true" || s == "1")  return true;
  if (s == "false" || s == "0") return false;
  std::cout << "Invalid bool: \"" << arg << "\" (use true/false, 1/0)" << std::endl;
  return std::nullopt;
}

bool ParseArgs(
  int argc,
  char** argv,
  GeneratorParamsArgs& generator_params_args,
  GuidanceArgs& guidance_args,
  std::string& model_path,
  std::string& ep,
  std::string& system_prompt,
  bool& verbose,
  bool& interactive,
  bool& rewind
) {
  // Integer tokens for long-only options (avoid collision with ASCII)
  enum {
    RESPONSE_FORMAT = 1000,
    TOOLS_FILE,
    TEXT_OUTPUT,
    TOOL_OUTPUT,
    TOOL_CALL_START,
    TOOL_CALL_END,
    SYSTEM_PROMPT,
    REWIND,
    NON_INTERACTIVE
  };

  static struct option long_options[] = {
    // Generator params options
    {"batch_size",           optional_argument, nullptr, 'b'},
    {"chunk_size",           optional_argument, nullptr, 'c'},
    {"do_sample",            optional_argument, nullptr, 's'},
    {"min_length",           optional_argument, nullptr, 'i'},
    {"max_length",           optional_argument, nullptr, 'l'},
    {"num_beams",            optional_argument, nullptr, 'n'},
    {"num_return_sequences", optional_argument, nullptr, 'q'},
    {"repetition_penalty",   optional_argument, nullptr, 'r'},
    {"temperature",          optional_argument, nullptr, 't'},
    {"top_k",                optional_argument, nullptr, 'k'},
    {"top_p",                optional_argument, nullptr, 'p'},

    // Guidance options
    {"response_format",      optional_argument, nullptr, RESPONSE_FORMAT},
    {"tools_file",           optional_argument, nullptr, TOOLS_FILE},
    {"text_output",          no_argument,       nullptr, TEXT_OUTPUT},
    {"tool_output",          no_argument,       nullptr, TOOL_OUTPUT},
    {"tool_call_start",      optional_argument, nullptr, TOOL_CALL_START},
    {"tool_call_end",        optional_argument, nullptr, TOOL_CALL_END},

    // Main options
    {"model_path",           required_argument, nullptr, 'm'},
    {"execution_provider",   optional_argument, nullptr, 'e'},
    {"verbose",              no_argument,       nullptr, 'v'},
    {"system_prompt",        optional_argument, nullptr, SYSTEM_PROMPT},
    {"rewind",               no_argument,       nullptr, REWIND},
    {"non_interactive",      no_argument,       nullptr, NON_INTERACTIVE},
    {nullptr,                0,                 nullptr, 0}
  };
  const char* short_options = "b:c:s:i:l:n:q:r:t:k:p:m:e:v";

  int opt;
  int option_index = 0;
  while ((opt = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
    switch (opt) {
      // Generator params options
      case 'b': {
        auto v = ParseInt(optarg);
        if (v) generator_params_args.batch_size = *v;
        break;
      }
      case 'c': {
        auto v = ParseInt(optarg);
        if (v) generator_params_args.chunk_size = *v;
        break;
      }
      case 's': {
        auto v = ParseBool(optarg);
        if (v) generator_params_args.do_sample = *v;
        break;
      }
      case 'i': {
        auto v = ParseInt(optarg);
        if (v) generator_params_args.min_length = *v;
        break;
      }
      case 'l': {
        auto v = ParseInt(optarg);
        if (v) generator_params_args.max_length = *v;
        break;
      }
      case 'n': {
        auto v = ParseInt(optarg);
        if (v) generator_params_args.num_beams = *v;
        break;
      }
      case 'q': {
        auto v = ParseInt(optarg);
        if (v) generator_params_args.num_return_sequences = *v;
        break;
      }
      case 'r': {
        auto v = ParseDouble(optarg);
        if (v) generator_params_args.repetition_penalty = *v;
        break;
      }
      case 't': {
        auto v = ParseDouble(optarg);
        if (v) generator_params_args.temperature = *v;
        break;
      }
      case 'k': {
        auto v = ParseInt(optarg);
        if (v) generator_params_args.top_k = *v;
        break;
      }
      case 'p': {
        auto v = ParseInt(optarg);
        if (v) generator_params_args.top_p = *v;
        break;
      }

      // Guidance options
      case RESPONSE_FORMAT: {
        if (optarg) guidance_args.response_format = optarg;
        break;
      }
      case TOOLS_FILE: {
        if (optarg) guidance_args.tools_file = optarg;
        break;
      }
      case TEXT_OUTPUT: {
        guidance_args.text_output = true;
        break;
      }
      case TOOL_OUTPUT: {
        guidance_args.tool_output = true;
        break;
      }
      case TOOL_CALL_START: {
        if (optarg) guidance_args.tool_call_start = optarg;
        break;
      }
      case TOOL_CALL_END: {
        if (optarg) guidance_args.tool_call_end = optarg;
        break;
      }

      // Main options
      case 'm': {
        if (optarg) model_path = optarg;
        break;
      }
      case 'e': {
        if (optarg) ep = optarg;
        break;
      }
      case 'v': {
        verbose = true;
        break;
      }
      case NON_INTERACTIVE: {
        interactive = false; 
        break;
      }
      case SYSTEM_PROMPT: {
        if (optarg) system_prompt = optarg;
        break;
      }
      case REWIND: {
        rewind = true;
        break;
      }
      default: {
        std::cerr << "Error: Unknown option" << std::endl;
        return false;
      }
    }
  }

  if (model_path.empty()) {
    std::cerr << "Error: Model path was not provided" << std::endl;
    return false;
  }
  return true;
}

void SetLogger(bool inputs, bool outputs) {
  Oga::SetLogBool("enabled", true);
  Oga::SetLogBool("model_input_values", inputs);
  Oga::SetLogBool("model_output_values", outputs);
}

std::unique_ptr<OgaConfig> GetConfig(const std::string& path, const std::string& ep, const std::unordered_map<std::string, std::string>& ep_options, GeneratorParamsArgs& search_options) {
  auto config = OgaConfig::Create(path.c_str());
  if (ep.compare("follow_config") != 0) {
    config->ClearProviders();
    if (ep.compare("cpu") != 0) {
      std::cout << "Setting model to " << ep << std::endl;
      config->AppendProvider(ep.c_str());
    }

    // Set any EP-specific options
    for (const auto& [key, val] : ep_options) {
      if (key.compare("enable_cuda_graph") == 0 && (ep.compare("cuda") == 0 || ep.compare("NvTensorRtRtx") == 0) && search_options.num_beams > 1) {
        config->SetProviderOption(ep.c_str(), "enable_cuda_graph", "0");
      }
      else {
        config->SetProviderOption(ep.c_str(), key.c_str(), val.c_str());
      }
    }
  }

  // Set any search-specific options that need to be known before constructing a Model object
  // Otherwise they can be set with params.SetSearchOptions(search_options)
  nlohmann::ordered_json j = search_options;
  config.Overlay(j.dump());
  return config;
}

void SetSearchOptions(OgaGeneratorParams& generatorParams, GeneratorParamsArgs& args, bool verbose) {
  std::vector<std::string> opts;
  if (args.batch_size) {
    generatorParams.SetSearchOption("batch_size", args.batch_size);
    opts.push_back("batch_size: " + std::to_string(args.batch_size));
  }
  if (args.do_sample) {
    generatorParams.SetSearchOptionBool("do_sample", args.do_sample.value());
    opts.push_back("do_sample: " + std::to_string(args.do_sample.value()));
  }
  if (args.min_length) {
    generatorParams.SetSearchOption("min_length", args.min_length.value());
    opts.push_back("min_length: " + std::to_string(args.min_length.value()));
  }
  if (args.num_beams) {
    generatorParams.SetSearchOption("num_beams", args.num_beams);
    opts.push_back("num_beams: " + std::to_string(args.num_beams));
  }
  if (args.num_return_sequences) {
    generatorParams.SetSearchOption("num_return_sequences", args.num_return_sequences);
    opts.push_back("num_return_sequences: " + std::to_string(args.num_return_sequences));
  }
  if (args.repetition_penalty) {
    generatorParams.SetSearchOption("repetition_penalty", args.repetition_penalty.value());
    opts.push_back("repetition_penalty: " + std::to_string(args.repetition_penalty.value()));
  }
  if (args.temperature) {
    generatorParams.SetSearchOption("temperature", args.temperature.value());
    opts.push_back("temperature: " + std::to_string(args.temperature.value()));
  }
  if (args.top_k) {
    generatorParams.SetSearchOption("top_k", args.top_k.value());
    opts.push_back("top_k: " + std::to_string(args.top_k.value()));
  }
  if (args.top_p) {
    generatorParams.SetSearchOption("top_p", args.top_p.value());
    opts.push_back("top_p: " + std::to_string(args.top_p.value()));
  }
  if (verbose) {
    std::cout << "GeneratorParams created: {";
    for (int i = 0; i < opts.size(); i++) {
      std::cout << opts[i];
      if (i != opts.size()-1) std::cout << ", ";
    }
    std::cout << "}" << std::endl;
  }
}

std::string ApplyChatTemplate(const std::string& model_path, OgaTokenizer& tokenizer, const std::string& messages, bool add_generation_prompt, const std::string& tools) {
  std::string template_str = "";
  std::filesystem::path jinja_path = std::filesystem::path(model_path) / "chat_template.jinja";
  if (std::filesystem::exists(jinja_path)) {
    std::ifstream file(jinja_path, std::ios::binary);
    if (file) {
      std::ostringstream oss;
      oss << file.rdbuf();
      template_str = oss.str();
    } else {
      // If the file exists but can't be opened, fall back to empty template.
      template_str.clear();
    }
  }

  std::string prompt = tokenizer.ApplyChatTemplate(template_str.c_str(), messages.c_str(), tools.c_str(), add_generation_prompt);
  return prompt;
}

std::vector<ToolSchema> ToolsToSchemas(std::vector<Tool>& tools) {
  std::vector<ToolSchema> tool_schemas;
  for (Tool tool : tools) {
    std::unordered_map<std::string, std::string> name;
    name["const"] = tool.function.name;

    nlohmann::ordered_json properties = {};
    properties["name"] = name;

    bool tool_parameters_exist = tool.function.parameters.size() != 0;
    if (tool_parameters_exist) {
      nlohmann::ordered_json parameters = {};
      parameters["type"] = tool.function.parameters.contains("type") ? tool.function.parameters["type"] : "object";
      nlohmann::ordered_json empty_map = {};
      parameters["properties"] = tool.function.parameters.contains("properties") ? tool.function.parameters["properties"] : empty_map;
      std::vector<std::string> empty_list;
      parameters["required"] = tool.function.parameters.contains("required") ? tool.function.parameters["required"].get<std::vector<std::string>>() : empty_list;

      properties["parameters"] = parameters;
    }

    ToolSchema tool_schema;
    tool_schema.description = tool.function.description;
    tool_schema.type = "object";
    tool_schema.properties = properties;
    tool_schema.required = tool_parameters_exist ? std::vector<std::string>{"name", "parameters"} : std::vector<std::string>{"name"};
    tool_schema.additionalProperties = false;

    tool_schemas.push_back(tool_schema);
  }
  return tool_schemas;
}

std::string GetJsonSchema(std::vector<Tool>& tools, bool tool_output) {
  auto schemas = ToolsToSchemas(tools);

  nlohmann::ordered_json x_guidance = {};
  x_guidance["whitespace_flexible"] = false;
  x_guidance["key_separator"] = ": ";
  x_guidance["item_separator"] = ", ";

  std::unordered_map<std::string, std::vector<ToolSchema>> items;
  items["anyOf"] = schemas;

  JsonSchema json_schema;
  json_schema.xGuidance = x_guidance;
  json_schema.type = "array";
  json_schema.items = items;
  json_schema.minItems = tool_output ? 1 : 0;

  // Serialize JSON schema to string
  nlohmann::ordered_json j = json_schema;
  return j.dump();
}

std::string GetLarkGrammar(std::vector<Tool>& tools, bool text_output, bool tool_output, const std::string& tool_call_start, const std::string& tool_call_end) {
  bool known_tool_call_ids = tool_call_start != "" && tool_call_end != "";
  std::string call_type = known_tool_call_ids ? "toolcall" : "functioncall";

  std::vector<std::string> rows;
  std::string start_row;
  if (text_output && !tool_output) {
    start_row = "start: TEXT";
  }
  else if (!text_output && tool_output) {
    start_row = "start: " + call_type;
  }
  else if (text_output && tool_output) {
    start_row = "start: TEXT | " + call_type;
  }
  else {
    throw new std::runtime_error("At least one of 'text_output' and 'tool_output' must be true");
  }
  rows.push_back(start_row);

  if (text_output) {
    std::string text_row = "TEXT: /[^{<](.|\\n)*/";
    rows.push_back(text_row);
  }

  if (tool_output) {
    std::string schema = GetJsonSchema(tools, tool_output);
    if (known_tool_call_ids) {
      std::string tool_row = "toolcall: " + tool_call_start + " functioncall " + tool_call_end;
      rows.push_back(tool_row);
    }

    std::string func_row = "functioncall: %json " + schema;
    rows.push_back(func_row);
  }

  std::string grammar = "";
  for (int i = 0; i < rows.size(); i++) {
    grammar += rows[i];
    if (i != rows.size()-1) grammar += "\n";
  }
  return grammar;
}

std::vector<Tool> ToTool(std::vector<nlohmann::ordered_json>& tool_defs) {
  std::vector<Tool> tools;
  for (const auto& tool_def : tool_defs) {
    Tool tool = tool_def.get<Tool>();
    tools.push_back(tool);
  }
  return tools;
}

std::tuple<std::string, std::string, std::string> GetGuidance(
  const std::string& response_format,
  const std::string& filepath,
  const std::string& tools_str,
  std::vector<nlohmann::ordered_json>* tools,
  bool text_output,
  bool tool_output,
  const std::string& tool_call_start,
  const std::string& tool_call_end
) {
  std::string guidance_type = "";
  std::string guidance_data = "";
  std::vector<Tool> all_tools;

  // Get list of tools from a range of sources (filepath, JSON-serialized string, in-memory)
  if (tool_output) {
    if (std::filesystem::exists(filepath)) {
      std::string json_str;
      std::ifstream file(filepath, std::ios::binary);
      if (file) {
        std::ostringstream oss;
        oss << file.rdbuf();
        json_str = oss.str();
      }
      if (json_str.empty()) {
        throw new std::runtime_error("Error: JSON file is empty.");
      }

      nlohmann::ordered_json j = nlohmann::ordered_json::parse(json_str);
      if (j.empty()) {
        throw new std::runtime_error("Error: Tools did not de-serialize correctly");
      }

      std::vector<nlohmann::ordered_json> defs;
      defs.reserve(j.size());
      for (const auto& item : j) {
        defs.push_back(item);
      }
      all_tools = ToTool(defs);
    }
    else if (!tools_str.empty()) {
      nlohmann::ordered_json j = nlohmann::ordered_json::parse(tools_str);
      if (j.empty()) {
        throw new std::runtime_error("Error: Tools did not de-serialize correctly");
      }

      std::vector<nlohmann::ordered_json> defs;
      defs.reserve(j.size());
      for (const auto& item : j) {
        defs.push_back(item);
      }
      all_tools = ToTool(defs);
    }
    else if (tools && !tools->empty()) {
      try {
        all_tools = ToTool(*tools);
      }
      catch (...) {
        throw new std::runtime_error("Could not convert tools from vector<nlohmann::ordered_json> to vector<Tool>");
      }
    }
    else {
      throw new std::runtime_error("Error: Please provide the list of tools through a file, JSON-serialized string, or a list of tools");
    }

    if (all_tools.empty()) { 
      throw new std::runtime_error("Error: Could not obtain a list of tools in memory");
    }
  }

  if (response_format == "text" || response_format == "lark_grammar") {
    if (response_format == "text") {
      bool right_settings = text_output && !tool_output;
      if (!right_settings) {
        throw new std::runtime_error("Error: A response format of 'text' requires text_output = true and tool_output = false");
      }
    }

    guidance_type = "lark_grammar";
    guidance_data = GetLarkGrammar(all_tools, text_output, tool_output, tool_call_start, tool_call_end);
  }
  else if (response_format == "json_schema" || response_format == "json_object") {
    bool right_settings = tool_output && !text_output;
    if (!right_settings) {
      throw new std::runtime_error("Error: A response format of 'json_schema' or 'json_object' requires text_output = false and tool_output = true");
    }

    guidance_type = "json_schema";
    guidance_data = GetJsonSchema(all_tools, tool_output);
  }
  else {
    throw new std::runtime_error("Error: Invalid response format provided");
  }

  nlohmann::ordered_json j = all_tools;
  return std::make_tuple(guidance_type, guidance_data, j.dump());
}
