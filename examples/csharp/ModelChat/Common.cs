using Microsoft.ML.OnnxRuntimeGenAI;
using System.CommandLine;
using System.Reflection;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace ModelChat
{
    public static class Common
    {
        /// <summary>
        /// Set log options inside ORT GenAI
        /// </summary>
        /// <param name="inputs">Dump inputs to the model in the console</param>
        /// <param name="outputs">Dump outputs to the model in the console</param>
        /// <returns>
        /// None
        /// </returns>
        public static void SetLogger(bool inputs = true, bool outputs = true)
        {
            Utils.SetLogBool("enabled", true);
            Utils.SetLogBool("model_input_values", inputs);
            Utils.SetLogBool("model_output_values", outputs);
        }

        /// <summary>
        /// Get Config object and set EP-specific and search-specific options inside it
        /// </summary>
        /// <param name="path">Path to model folder containing GenAI config</param>
        /// <param name="ep">Name of execution provider to set</param>
        /// <param name="ep_options">Map of EP-specific option names and their values</param>
        /// <param name="search_options">Class of search-specific option names and their values</param>
        /// <returns>
        /// ORT GenAI config object with all options set
        /// </returns>
        public static Config GetConfig(string path, string ep, Dictionary<string, string>? ep_options, GeneratorParamsArgs search_options)
        {
            var config = new Config(path);
            if (ep != "follow_config")
            {
                config.ClearProviders();
                if (ep != "cpu")
                {
                    Console.WriteLine($"Setting model to {ep}");
                    config.AppendProvider(ep);
                }

                // Set any EP-specific options
                if (ep_options != null)
                {
                    foreach (var kvp in ep_options)
                    {
                        var k = kvp.Key;
                        var v = kvp.Value;
                        if (k == "enable_cuda_graph" && (ep == "cuda" || ep == "NvTensorRtRtx") && search_options.num_beams > 1)
                        {
                            // Disable CUDA graph if using beam search (num_beams > 1),
                            // num_beams > 1 requires past_present_share_buffer to be false so enable_cuda_graph must be false
                            config.SetProviderOption(ep, "enable_cuda_graph", "0");
                        }
                        else
                        {
                            config.SetProviderOption(ep, k, v);
                        }
                    }
                }
            }

            // Create serializer context to skip null attributes
            var options = new JsonSerializerOptions()
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            };
            var ctx = new ArgsSerializerContext(options);
            var json = JsonSerializer.Serialize(search_options, ctx.GeneratorParamsArgs);

            // Set any search-specific options that need to be known before constructing a Model object
            // Otherwise they can be set with params.SetSearchOptions(search_options)
            config.Overlay(json);
            return config;
        }

        /// <summary>
        /// Set search options for a generator's params during decoding
        /// </summary>
        /// <param name="generatorParams">Generator params object to set on</param>
        /// <param name="args">Arguments provided by user</param>
        /// <param name="verbose">Use verbose logging</param>
        /// <returns>
        /// None
        /// </returns>
        public static void SetSearchOptions(GeneratorParams generatorParams, GeneratorParamsArgs args, bool verbose)
        {
            var type = args.GetType();
            var options = new List<string>();
            foreach (var prop in type.GetProperties(BindingFlags.Instance | BindingFlags.Public))
            {
                var name = prop.Name;
                var value = prop.GetValue(args);
                if (value == null || name == "chunk_size") continue;

                if (name == "do_sample")
                {
                    var val = Convert.ToBoolean(value);
                    options.Add($"{name}: {val}");
                    generatorParams.SetSearchOption(name, val);
                }
                else
                {
                    var val = Convert.ToDouble(value);
                    options.Add($"{name}: {val}");
                    generatorParams.SetSearchOption(name, val);
                }
            }
            
            if (verbose) Console.WriteLine("GeneratorParams created: {" + string.Join(", ", options) + "}");
        }

        /// <summary>
        /// Apply the chat template with various fallback options
        /// </summary>
        /// <param name="model_path">Path to folder containing model</param>
        /// <param name="tokenizer">Tokenizer object to use</param>
        /// <param name="messages">String-encoded list of messages</param>
        /// <param name="add_generation_prompt">Add tokens to indicate the start of the AI's response</param>
        /// <param name="tools">String-encoded list of tools</param>
        /// <returns>
        /// Prompt to encode
        /// </returns>
        public static string ApplyChatTemplate(string model_path, Tokenizer tokenizer, string messages, bool add_generation_prompt, string tools = "")
        {
            var template_str = "";
            var jinja_path = Path.Combine(model_path, "chat_template.jinja");
            if (File.Exists(jinja_path))
            {
                template_str = File.ReadAllText(jinja_path, Encoding.UTF8);
            }

            var prompt = tokenizer.ApplyChatTemplate(
                messages: messages,
                tools: tools,
                add_generation_prompt: add_generation_prompt,
                template_str: template_str
            );
            return prompt;
        }

        /// <summary>
        /// Convert a list of tools to a list of tool schemas
        /// </summary>
        /// <param name="tools">List of OpenAI-compatible tools</param>
        /// <returns>
        /// List of JSON schema compatible tools
        /// </returns>
        public static IList<ToolSchema> ToolsToSchemas(IList<Tool> tools)
        {
            var tool_schemas = new List<ToolSchema> { };
            foreach (var tool in tools)
            {
                var name = new Dictionary<string, string>()
                {
                    { "const", tool.Function.Name }
                };
                var properties = new Dictionary<string, object>
                {
                    { "name", name }
                };

                var tool_parameters_exist = tool.Function.Parameters.Count != 0;
                if (tool_parameters_exist)
                {
                    var parameters = new Dictionary<string, object>
                    {
                        { "type", tool.Function.Parameters.GetValueOrDefault("type", "object") },
                        { "properties", tool.Function.Parameters.GetValueOrDefault("properties", new Dictionary<string, object>{}) },
                        { "required", tool.Function.Parameters.GetValueOrDefault("required", new List<string>{}) }
                    };
                    properties.Add("parameters", parameters);
                }

                var tool_schema = new ToolSchema()
                {
                    Description = tool.Function.Description,
                    Type = "object",
                    Properties = properties,
                    Required = tool_parameters_exist ? ["name", "parameters"] : ["name"],
                    AdditionalProperties = false
                };
                tool_schemas.Add(tool_schema);
            }
            return tool_schemas;
        }

        /// <summary>
        /// Create a JSON schema from a list of tools
        /// </summary>
        /// <param name="tools">List of OpenAI-compatible tools</param>
        /// <param name="tool_output">Output can have a tool call</param>
        /// <returns>
        /// JSON schema as a JSON-compatible string
        /// </returns>
        public static string GetJsonSchema(IList<Tool> tools, bool tool_output)
        {
            var schemas = ToolsToSchemas(tools);
            var x_guidance = new Dictionary<string, object>
            {
                { "whitespace_flexible", false },
                { "key_separator",  ": "},
                { "item_separator", ", " }
            };
            var json_schema = new JsonSchema
            {
                XGuidance = x_guidance,
                Type = "array",
                Items = new Dictionary<string, IList<ToolSchema>>{
                    { "anyOf", schemas }
                },
                MinItems = tool_output ? 1 : 0
            };

            // Create serializer context with encoder to not escape non-ASCII characters (e.g. don't convert '&' to \u0026)
            // and to skip null attributes
            var options = new JsonSerializerOptions()
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingDefault,
            };
            var ctx = new ToolSerializerContext(options);

            return JsonSerializer.Serialize(json_schema, ctx.JsonSchema);
        }

        /// <summary>
        /// Create a LARK grammar from a list of tools
        /// </summary>
        /// <param name="tools">List of OpenAI-compatible tools</param>
        /// <param name="text_output">Output can have text</param>
        /// <param name="tool_output">Output can have a tool call</param>
        /// <param name="tool_call_start">String representation of tool call starting token</param>
        /// <param name="tool_call_end">String representation of tool call ending token</param>
        /// <returns>
        /// LARK grammar as a string
        /// </returns>
        public static string GetLarkGrammar(IList<Tool> tools, bool text_output, bool tool_output, string tool_call_start, string tool_call_end)
        {
            var known_tool_call_ids = !string.IsNullOrEmpty(tool_call_start) && !string.IsNullOrEmpty(tool_call_end);
            var call_type = known_tool_call_ids ? "toolcall" : "functioncall";

            var rows = new List<string>();
            string? start_row;
            if (text_output && !tool_output)
            {
                start_row = "start: TEXT";
            } 
            else if (!text_output && tool_output)
            {
                start_row = $"start: {call_type}";
            }    
            else if (text_output && tool_output)
            {
                start_row = $"start: TEXT | {call_type}";
            }
            else
            {
                throw new Exception("At least one of 'text_output' and 'tool_output' must be true");
            }
            rows.Add(start_row);

            if (text_output)
            {
                var text_row = "TEXT: /[^{<](.|\\n)*/";
                rows.Add(text_row);
            }

            if (tool_output)
            {
                var schema = GetJsonSchema(tools: tools, tool_output: tool_output);
                if (known_tool_call_ids)
                {
                    var tool_row = $"toolcall: {tool_call_start} functioncall {tool_call_end}";
                    rows.Add(tool_row);
                }

                var func_row = $"functioncall: %json {schema}";
                rows.Add(func_row);
            }

            var grammar = string.Join("\n", rows);
            return grammar;
        }

        /// <summary>
        /// Convert a JSON-deserialized object of tools to a list of Tool objects
        /// </summary>
        /// <param name="tool_defs">JSON-deserialized object containing OpenAI-compatible tool definitions</param>
        /// <returns>
        /// List of Tool objects
        /// </returns>
        public static IList<Tool> ToTool(IList<Dictionary<string, object>> tool_defs)
        {
            var tools = new List<Tool> { };
            foreach (var tool_def in tool_defs)
            {
                if (tool_def.TryGetValue("function", out var functionObj))
                {
                    var functionStr = JsonSerializer.Serialize(functionObj);
                    var functionDict = JsonSerializer.Deserialize(functionStr, ToolSerializerContext.Default.DictionaryStringObject);
                    if (functionDict == null) continue;
                    
                    var name = functionDict.TryGetValue("name", out var nameObj) ? nameObj?.ToString() ?? string.Empty : string.Empty;
                    var description = functionDict.TryGetValue("description", out var descObj) ? descObj?.ToString() ?? string.Empty : string.Empty;

                    if (functionDict.TryGetValue("parameters", out var paramObj))
                    {
                        var paramStr = JsonSerializer.Serialize(paramObj);
                        var paramDict = JsonSerializer.Deserialize(paramStr, ToolSerializerContext.Default.DictionaryStringObject);
                        if (paramDict == null) continue;

                        var func = new FunctionDefinition
                        {
                            Name = name,
                            Description = description,
                            Parameters = paramDict
                        };
                        var tool = new Tool()
                        {
                            Type = "function",
                            Function = func
                        };
                        tools.Add(tool);
                    }
                }
            }
            return tools;
        }

        /// <summary>
        /// Create a grammar to use with LLGuidance
        /// </summary>
        /// <param name="response_format">Type of format requested</param>
        /// <param name="filepath">Path to file containing OpenAI-compatible tool definitions</param>
        /// <param name="tools_str">JSON-serialized string containing OpenAI-compatible tool definitions</param>
        /// <param name="tools">List of OpenAI-compatible tools defined in memory</param>
        /// <param name="text_output">Output can have text</param>
        /// <param name="tool_output">Output can have a tool call</param>
        /// <param name="tool_call_start">String representation of tool call starting token (e.g. <tool_call>)</param>
        /// <param name="tool_call_end">String representation of tool call ending token (e.g. </tool_call>)</param>
        /// <returns>
        /// (grammar type, grammar data, tools) as a tuple of strings
        /// </returns>
        public static (string, string, string) GetGuidance(
            string response_format = "",
            string filepath = "",
            string tools_str = "",
            List<object>? tools = null,
            bool text_output = true,
            bool tool_output = false,
            string tool_call_start = "",
            string tool_call_end = "")
        {
            var guidance_type = "";
            var guidance_data = "";
            IList<Tool> all_tools = [];

            // Get list of tools from a range of sources (filepath, JSON-serialized string, in-memory)
            if (tool_output)
            {
                if (File.Exists(filepath))
                {
                    var json_str = File.ReadAllText(filepath);
                    if (string.IsNullOrWhiteSpace(json_str))
                    {
                        throw new Exception("Error: JSON file is empty.");
                    }

                    var tool_defs = JsonSerializer.Deserialize(json_str, ToolSerializerContext.Default.IListDictionaryStringObject);
                    if (tool_defs == null)
                    {
                        throw new Exception("Error: Tools did not de-serialize correctly");
                    }
                    all_tools = ToTool(tool_defs);
                }
                else if (!string.IsNullOrEmpty(tools_str))
                {
                    var tool_defs = JsonSerializer.Deserialize(tools_str, ToolSerializerContext.Default.IListDictionaryStringObject);
                    if (tool_defs == null)
                    {
                        throw new Exception("Error: Tools did not de-serialize correctly");
                    }
                    all_tools = ToTool(tool_defs);
                }
                else if (tools != null && tools.Count > 0)
                {
                    try
                    {
                        all_tools = ToTool(tools.Cast<Dictionary<string, object>>().ToList());
                    }
                    catch
                    {
                        Console.WriteLine("Could not convert tools from List<object> to List<Dictionary<string, object>>");
                        try
                        {
                            all_tools = tools.Cast<Tool>().ToList();
                        }
                        catch
                        {
                            Console.WriteLine("Could not convert tools from List<object> to List<Tool>");
                        }
                    }
                }
                else
                {
                    throw new Exception("Error: Please provide the list of tools through a file, JSON-serialized string, or a list of tools");
                }

                if (all_tools.Count <= 0)
                { 
                    throw new Exception("Error: Could not obtain a list of tools in memory");
                }
            }

            // Create guidance based on user-provided response format
            if (response_format == "text" || response_format == "lark_grammar")
            {
                if (response_format == "text")
                {
                    var right_settings = text_output && !tool_output;
                    if (!right_settings)
                    {
                        throw new Exception("Error: A response format of 'text' requires text_output = true and tool_output = false");
                    }
                }

                guidance_type = "lark_grammar";
                guidance_data = GetLarkGrammar(
                    tools: all_tools,
                    text_output: text_output,
                    tool_output: tool_output,
                    tool_call_start: tool_call_start,
                    tool_call_end: tool_call_end);
            }
            else if (response_format == "json_schema" || response_format == "json_object")
            {
                var right_settings = tool_output && !text_output;
                if (!right_settings)
                {
                    throw new Exception("Error: A response format of 'json_schema' or 'json_object' requires text_output = false and tool_output = true");
                }

                guidance_type = "json_schema";
                guidance_data = GetJsonSchema(tools: all_tools, tool_output: tool_output);
            }
            else
            {
                throw new Exception("Error: Invalid response format provided");
            }

            return (guidance_type, guidance_data, JsonSerializer.Serialize(all_tools, ToolSerializerContext.Default.IListTool));
        }

        /// <summary>
        /// Add arguments for the generator params
        /// </summary>
        /// <param name="parser">Original parser object with existing arguments</param>
        /// <return>
        /// None
        /// </return>
        public static void GetGeneratorParamsArgs(RootCommand parser)
        {
            var batch_size = new Option<int>(
                name: "batch_size",
                aliases: ["-b", "--batch_size"]
            )
            {
                Arity = ArgumentArity.ExactlyOne,
                DefaultValueFactory = (_) => 1,
                Description = "Batch size for input payload"
            };

            var chunk_size = new Option<int>(
                name: "chunk_size",
                aliases: ["-c", "--chunk_size"]
            )
            {
                Arity = ArgumentArity.ExactlyOne,
                DefaultValueFactory = (_) => 0,
                Description = "Chunk size for prefill chunking during context processing (default: 0 = disabled, >0 = enabled)"
            };

            var do_sample = new Option<bool>(
                name: "do_sample",
                aliases: ["-s", "--do_sample"]
            )
            {
                Arity = ArgumentArity.Zero,
                DefaultValueFactory = (_) => false,
                Description = "Do random sampling. When false, greedy or beam search are used to generate the output. Defaults to false"
            };

            var min_length = new Option<int?>(
                name: "min_length",
                aliases: ["-i", "--min_length"]
            )
            {
                Arity = ArgumentArity.ExactlyOne,
                Description = "Min number of tokens to generate including the prompt"
            };

            var max_length = new Option<int?>(
                name: "max_length",
                aliases: ["-l", "--max_length"]
            )
            {
                Arity = ArgumentArity.ExactlyOne,
                Description = "Max number of tokens to generate including the prompt"
            };

            var num_beams = new Option<int>(
                name: "num_beams",
                aliases: ["-nb", "--num_beams"]
            )
            {
                Arity = ArgumentArity.ExactlyOne,
                DefaultValueFactory = (_) => 1,
                Description = "Number of beams to create"
            };

            var num_return_sequences = new Option<int>(
                name: "num_return_sequences",
                aliases: ["-rs", "--num_return_sequences"]
            )
            {
                Arity = ArgumentArity.ExactlyOne,
                DefaultValueFactory = (_) => 1,
                Description = "Number of return sequences to produce"
            };

            var repetition_penalty = new Option<double?>(
                name: "repetition_penalty",
                aliases: ["-r", "--repetition_penalty"]
            )
            {
                Arity = ArgumentArity.ExactlyOne,
                Description = "Repetition penalty to sample with"
            };

            var temperature = new Option<double?>(
                name: "temperature",
                aliases: ["-t", "--temperature"]
            )
            {
                Arity = ArgumentArity.ExactlyOne,
                Description = "Temperature to sample with"
            };

            var top_k = new Option<int?>(
                name: "top_k",
                aliases: ["-k", "--top_k"]
            )
            {
                Arity = ArgumentArity.ExactlyOne,
                Description = "Top k tokens to sample from"
            };

            var top_p = new Option<double?>(
                name: "top_p",
                aliases: ["-p", "--top_p"]
            )
            {
                Arity = ArgumentArity.ExactlyOne,
                Description = "Top p probability to sample with"
            };

            parser.Add(batch_size);
            parser.Add(chunk_size);
            parser.Add(do_sample);
            parser.Add(min_length);
            parser.Add(max_length);
            parser.Add(num_beams);
            parser.Add(num_return_sequences);
            parser.Add(repetition_penalty);
            parser.Add(temperature);
            parser.Add(top_k);
            parser.Add(top_p);
        }

        /// <summary>
        /// Add arguments for guidance options
        /// </summary>
        /// <param name="parser">Original parser object with existing arguments</param>
        /// <return>
        /// None
        /// </return>
        public static void GetGuidanceArgs(RootCommand parser)
        {
            var response_format = new Option<string>(
                name: "response_format",
                aliases: ["-rf", "--response_format"]
            )
            {
                Arity = ArgumentArity.ExactlyOne,
                DefaultValueFactory = (_) => "",
                Description = "Provide response format for the model",
            };
            response_format.Validators.Add(result => {
                var value = result.GetValue(response_format)!;
                if (string.IsNullOrEmpty(value)) return;

                var options = new List<string> { "text", "json_object", "json_schema", "lark_grammar" };
                if (!options.Contains(value))
                {
                    var options_str = string.Join(", ", options);
                    result.AddError($"Response format must be from one of the options: {options_str}");
                }
            });

            var tools_file = new Option<string>(
                name: "tools_file",
                aliases: ["-tf", "--tools_file"]
            )
            {
                Arity = ArgumentArity.ExactlyOne,
                DefaultValueFactory = (_) => "",
                Description = "Path to file containing list of OpenAI-compatible tool definitions. Ex: test/test_models/tool-definitions/weather.json"
            };
            tools_file.Validators.Add(result =>
            {
                var value = result.GetValue(tools_file)!;
                if (string.IsNullOrEmpty(value)) return;

                if (!value.EndsWith(".json"))
                {
                    result.AddError("Path must be to a .json file");
                }
                if (!File.Exists(value))
                {
                    result.AddError("JSON file does not exist");
                }
            });

            var text_output = new Option<bool>(
                name: "text_output",
                aliases: ["-text", "--text_output"]
            )
            {
                Arity = ArgumentArity.Zero,
                DefaultValueFactory = (_) => false,
                Description = "Produce a text response in the output"
            };

            var tool_output = new Option<bool>(
                name: "tool_output",
                aliases: ["-tool", "--tool_output"]
            )
            {
                Arity = ArgumentArity.Zero,
                DefaultValueFactory = (_) => false,
                Description = "Produce a tool call in the output"
            };

            var tool_call_start = new Option<string>(
                name: "tool_call_start",
                aliases: ["-tcs", "--tool_call_start"]
            )
            {
                Arity = ArgumentArity.ExactlyOne,
                DefaultValueFactory = (_) => "",
                Description = "String representation of tool call start (ex: <|tool_call|>). Needs to be marked as special in tokenizer.json for guidance to work."
            };

            var tool_call_end = new Option<string>(
                name: "tool_call_end",
                aliases: ["-tce", "--tool_call_end"]
            )
            {
                Arity = ArgumentArity.ExactlyOne,
                DefaultValueFactory = (_) => "",
                Description = "String representation of tool call end (ex: <|/tool_call|>). Needs to be marked as special in tokenizer.json for guidance to work."
            };

            parser.Add(response_format);
            parser.Add(tools_file);
            parser.Add(text_output);
            parser.Add(tool_output);
            parser.Add(tool_call_start);
            parser.Add(tool_call_end);
        }

        /// <summary>
        /// Set arguments for generator params and guidance
        /// </summary>
        /// <param name="parseResult">Parsed result with user-provided arguments</param>
        /// <return>
        /// (GeneratorParamsArgs, GuidanceArgs) as a tuple of user-provided arguments
        /// </return>
        public static (GeneratorParamsArgs, GuidanceArgs) SetGroupedArgs(ParseResult parseResult)
        {
            GeneratorParamsArgs generatorParamsArgs = new GeneratorParamsArgs
            {
                batch_size = parseResult.GetValue<int>("batch_size"),
                chunk_size = parseResult.GetValue<int>("chunk_size"),
                do_sample = parseResult.GetValue<bool>("do_sample"),
                min_length = parseResult.GetValue<int?>("min_length"),
                max_length = parseResult.GetValue<int?>("max_length"),
                num_beams = parseResult.GetValue<int>("num_beams"),
                num_return_sequences = parseResult.GetValue<int>("num_return_sequences"),
                repetition_penalty = parseResult.GetValue<double?>("repetition_penalty"),
                temperature = parseResult.GetValue<double?>("temperature"),
                top_k = parseResult.GetValue<int?>("top_k"),
                top_p = parseResult.GetValue<double?>("top_p")
            };

            GuidanceArgs guidanceArgs = new GuidanceArgs
            {
                response_format = parseResult.GetValue<string>("response_format") ?? "",
                tools_file = parseResult.GetValue<string>("tools_file") ?? "",
                text_output = parseResult.GetValue<bool>("text_output"),
                tool_output = parseResult.GetValue<bool>("tool_output"),
                tool_call_start = parseResult.GetValue<string>("tool_call_start") ?? "",
                tool_call_end = parseResult.GetValue<string>("tool_call_end") ?? ""
            };

            return (generatorParamsArgs, guidanceArgs);
        }
    }

    /// <summary>
    /// A class for defining a tool in a JSON schema compatible way
    /// </summary>
    public class ToolSchema
    {
        [JsonPropertyName("description")]
        public required string Description { get; set; }
        [JsonPropertyName("type")]
        public required string Type { get; set; }
        [JsonPropertyName("properties")]
        public required Dictionary<string, object> Properties { get; set; }
        [JsonPropertyName("required")]
        public required IList<string> Required { get; set; }
        [JsonPropertyName("additionalProperties")]
        public required bool AdditionalProperties { get; set; }
    }

    /// <summary>
    /// A class for defining a JSON schema for guidance
    /// </summary>
    public class JsonSchema
    {
        [JsonPropertyName("x-guidance")]
        public required Dictionary<string, object> XGuidance { get; set; }
        [JsonPropertyName("type")]
        public required string Type { get; set; }
        [JsonPropertyName("items")]
        public required Dictionary<string, IList<ToolSchema>> Items { get; set; }
        [JsonPropertyName("minItems")]
        public required int MinItems { get; set; }
    }

    /// <summary>
    /// A class for defining a function in an OpenAI-compatible way
    /// </summary>
    public class FunctionDefinition
    {
        [JsonPropertyName("name")]
        public required string Name { get; set; }
        [JsonPropertyName("description")]
        public required string Description { get; set; }
        [JsonPropertyName("parameters")]
        public required Dictionary<string, object> Parameters { get; set; }
    }

    /// <summary>
    /// A class for defining a tool in an OpenAI-compatible way
    /// </summary>
    public class Tool
    {
        [JsonPropertyName("type")]
        public required string Type { get; set; }
        [JsonPropertyName("function")]
        public required FunctionDefinition Function { get; set; }
    }

    [JsonSourceGenerationOptions(WriteIndented = true, PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
    [JsonSerializable(typeof(ToolSchema))]
    [JsonSerializable(typeof(JsonSchema))]
    [JsonSerializable(typeof(FunctionDefinition))]
    [JsonSerializable(typeof(Tool))]
    [JsonSerializable(typeof(JsonElement))]
    [JsonSerializable(typeof(Dictionary<string, string>))]
    [JsonSerializable(typeof(Dictionary<string, object>))]
    [JsonSerializable(typeof(IList<Dictionary<string, object>>))]
    [JsonSerializable(typeof(List<Dictionary<string, object>>))]
    [JsonSerializable(typeof(IList<Tool>))]
    [JsonSerializable(typeof(List<Tool>))]
    public sealed partial class ToolSerializerContext : JsonSerializerContext
    {
    }

    /// <summary>
    /// A class for holding parsed values for generator params
    /// </summary>
    public class GeneratorParamsArgs
    {
        // In case the user doesn't provide the batch size, set it to 1
        public int batch_size { get; set; } = 1;
        // In case the user doesn't provide the chunk size, set it to 0
        public int chunk_size { get; set; } = 0;
        public bool? do_sample { get; set; }
        public int? min_length { get; set; }
        public int? max_length { get; set; }
        // In case the user doesn't provide the number of beams, set it to 1
        public int num_beams { get; set; } = 1;
        // In case the user doesn't provide the number of return sequences, set it to 1
        public int num_return_sequences { get; set; } = 1;
        public double? repetition_penalty { get; set; }
        public double? temperature { get; set; }
        public int? top_k { get; set; }
        public double? top_p { get; set; }
    }

    /// <summary>
    /// A class for holding parsed values for guidance
    /// </summary>
    public class GuidanceArgs
    {
        public string response_format { get; set; } = "";
        public string tools_file { get; set; } = "";
        public bool text_output { get; set; } = false;
        public bool tool_output { get; set; } = false;
        public string tool_call_start { get; set; } = "";
        public string tool_call_end { get; set; } = "";
    }

    [JsonSourceGenerationOptions(WriteIndented = true, PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
    [JsonSerializable(typeof(GeneratorParamsArgs))]
    [JsonSerializable(typeof(GuidanceArgs))]
    public sealed partial class ArgsSerializerContext : JsonSerializerContext
    {
    }
}
