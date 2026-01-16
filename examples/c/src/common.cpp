// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>

#include "common.h"

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

// Define to_json and from_json for std::optional<T>
// Must be done within nlohmann::adl_serializer and not as standalone methods
namespace nlohmann {
template <class T>
struct adl_serializer<std::optional<T>> {
  static void to_json(nlohmann::ordered_json& j, const std::optional<T>& opt) {
    if (opt.has_value()) j = *opt;
    else j = nullptr;
  }
  static void from_json(const nlohmann::ordered_json& j, std::optional<T>& opt) {
    if (j.is_null()) { opt = std::nullopt; return; }
    opt = j.get<T>();
  }
};
}

void to_json(nlohmann::ordered_json& j, const ToolSchema& tool) {
  j = nlohmann::ordered_json{{"description", tool.description}, {"type", tool.type}, {"properties", tool.properties}, {"required", tool.required}, {"additionalProperties", tool.additionalProperties}};
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
  } else {
    tool.additionalProperties = false;
  }
}

void to_json(nlohmann::ordered_json& j, const JsonSchema& schema) {
  j = nlohmann::ordered_json{{"x-guidance", schema.xGuidance}, {"type", schema.type}, {"items", schema.items}, {"minItems", schema.minItems}};
}

void from_json(const nlohmann::ordered_json& j, JsonSchema& schema) {
  j.at("x-guidance").get_to(schema.xGuidance);
  j.at("type").get_to(schema.type);
  j.at("items").get_to(schema.items);
  j.at("minItems").get_to(schema.minItems);
}

void to_json(nlohmann::ordered_json& j, const FunctionDefinition& func) {
  j = nlohmann::ordered_json{{"name", func.name}, {"description", func.description}, {"parameters", func.parameters}};
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
  j = nlohmann::ordered_json{{"type", t.type}, {"function", t.function}};
}

void from_json(const nlohmann::ordered_json& j, Tool& t) {
  j.at("type").get_to(t.type);
  j.at("function").get_to(t.function);
}

void to_json(nlohmann::ordered_json& j, const GeneratorParamsArgs& a) {
  j = nlohmann::ordered_json{{"batch_size", a.batch_size}, {"num_beams", a.num_beams}, {"num_return_sequences", a.num_return_sequences}};
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
    bool& rewind) {
  
  CLI::App app{"Command-line arguments for ORT GenAI C/C++ examples"};
  argv = app.ensure_utf8(argv);

  std::string generator_params("Generator Params");
  std::string guidance("Guidance Arguments");

  app.add_option("-b,--batch_size", generator_params_args.batch_size, "Batch size used during inference.")->group(generator_params);
  app.add_option("-c,--chunk_size", generator_params_args.chunk_size, "Chunk size for prefill chunking during context processing (default: 0 = disabled, >0 = enabled)")->group(generator_params);
  app.add_option("-s,--do_sample", generator_params_args.do_sample, "Do random sampling. When false, greedy or beam search are used to generate the output. Defaults to false")->group(generator_params);
  app.add_option("-i,--min_length", generator_params_args.min_length, "Min number of tokens to generate including the prompt")->group(generator_params);
  app.add_option("-l,--max_length", generator_params_args.max_length, "Max number of tokens to generate including the prompt")->group(generator_params);
  app.add_option("-n,--num_beams", generator_params_args.num_beams, "Number of beams to create")->group(generator_params);
  app.add_option("-q,--num_return_sequences", generator_params_args.num_return_sequences, "Number of return sequences to produce")->group(generator_params);
  app.add_option("-r,--repetition_penalty", generator_params_args.repetition_penalty, "Repetition penalty to sample with")->group(generator_params);
  app.add_option("-t,--temperature", generator_params_args.temperature, "Temperature to sample with")->group(generator_params);
  app.add_option("-k,--top_k", generator_params_args.top_k, "Top k tokens to sample from")->group(generator_params);
  app.add_option("-p,--top_p", generator_params_args.top_p, "Top p probability to sample with")->group(generator_params);

  app.add_option("--response_format", guidance_args.response_format, "Provide response format for the model")->group(guidance);
  app.add_option("--tools_file", guidance_args.tools_file, "Path to file containing list of OpenAI-compatible tool definitions. Ex: test/test_models/tool-definitions/weather.json")->group(guidance);
  app.add_flag("--text_output", guidance_args.text_output, "Produce a text response in the output")->group(guidance);
  app.add_flag("--tool_output", guidance_args.tool_output, "Produce a tool call in the output")->group(guidance);
  app.add_option("--tool_call_start", guidance_args.tool_call_start, "String representation of tool call start (ex: <|tool_call|>). Needs to be marked as special in tokenizer.json for guidance to work.")->group(guidance);
  app.add_option("--tool_call_end", guidance_args.tool_call_end, "String representation of tool call end (ex: <|/tool_call|>). Needs to be marked as special in tokenizer.json for guidance to work.")->group(guidance);

  app.add_option("-m,--model_path", model_path, "ONNX model folder path (must contain genai_config.json and model.onnx)")->required();
  app.add_option("-e,--execution_provider", ep, "Execution provider to run the ONNX Runtime session with. Defaults to follow_config that uses the execution provider listed in the genai_config.json instead.");
  app.add_flag("-v,--verbose", verbose, "Print verbose output and timing information. Defaults to false");
  app.add_option("--system_prompt", system_prompt, "System prompt to use for the model.");
  app.add_flag("--rewind", rewind, "Rewind to the system prompt after each generation. Defaults to false");
  app.add_flag_callback("--non_interactive", [&]{ interactive = false; }, "Disable interactive mode");

  try {
    app.parse(argc, argv);
  } catch (...) {
    std::cout << app.help() << std::endl;
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
      } else {
        config->SetProviderOption(ep.c_str(), key.c_str(), val.c_str());
      }
    }
  }

  // Set any search-specific options that need to be known before constructing a Model object
  // Otherwise they can be set with params.SetSearchOptions(search_options)
  nlohmann::ordered_json j = search_options;
  std::string s = j.dump();
  config->Overlay(s.c_str());
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
      if (i != opts.size() - 1) std::cout << ", ";
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

  std::string prompt = std::string(tokenizer.ApplyChatTemplate(template_str.c_str(), messages.c_str(), tools.c_str(), add_generation_prompt));
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
  std::string s = j.dump();
  return s;
}

std::string GetLarkGrammar(std::vector<Tool>& tools, bool text_output, bool tool_output, const std::string& tool_call_start, const std::string& tool_call_end) {
  bool known_tool_call_ids = tool_call_start != "" && tool_call_end != "";
  std::string call_type = known_tool_call_ids ? "toolcall" : "functioncall";

  std::vector<std::string> rows;
  std::string start_row;
  if (text_output && !tool_output) {
    start_row = "start: TEXT";
  } else if (!text_output && tool_output) {
    start_row = "start: " + call_type;
  } else if (text_output && tool_output) {
    start_row = "start: TEXT | " + call_type;
  } else {
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
    if (i != rows.size() - 1) grammar += "\n";
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
    const std::string& tool_call_end) {
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
    } else if (!tools_str.empty()) {
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
    } else if (tools && !tools->empty()) {
      try {
        all_tools = ToTool(*tools);
      } catch (...) {
        throw new std::runtime_error("Could not convert tools from vector<nlohmann::ordered_json> to vector<Tool>");
      }
    } else {
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
  } else if (response_format == "json_schema" || response_format == "json_object") {
    bool right_settings = tool_output && !text_output;
    if (!right_settings) {
      throw new std::runtime_error("Error: A response format of 'json_schema' or 'json_object' requires text_output = false and tool_output = true");
    }

    guidance_type = "json_schema";
    guidance_data = GetJsonSchema(all_tools, tool_output);
  } else {
    throw new std::runtime_error("Error: Invalid response format provided");
  }

  nlohmann::ordered_json j = all_tools;
  std::string s = j.dump();
  return std::make_tuple(guidance_type, guidance_data, s);
}
