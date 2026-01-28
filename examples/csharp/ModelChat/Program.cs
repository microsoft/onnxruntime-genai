// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using ModelChat;
using Microsoft.ML.OnnxRuntimeGenAI;
using System.CommandLine;
using System.Text.Json;

/// <summary>
/// Get prompt from user
/// </summary>
/// <returns>
/// User prompt to use
/// </returns>
static string GetPrompt()
{
    string? prompt;
    do
    {
        Console.Write("Prompt (Use quit() to exit): ");
        prompt = Console.ReadLine();
    } while (string.IsNullOrEmpty(prompt));
    return prompt;
}

/// <summary>
/// Example of model-generate
/// </summary>
/// <param name="model">Model to use</param>
/// <param name="tokenizer">Tokenizer to use</param>
/// <param name="generatorParamsArgs">Generator params arguments to use</param>
/// <param name="modelPath">Path to folder containing model</param>
/// <param name="systemPrompt">System prompt to use with model</param>
/// <param name="userPrompt">User prompt to use with model</param>
/// <param name="interactive">Ask user or use pre-defined value</param>
/// <param name="verbose">Use verbose logging</param>
/// <returns>
/// None
/// </returns>
void ModelGenerate(
    Model model,
    Tokenizer tokenizer,
    GeneratorParamsArgs generatorParamsArgs,
    string modelPath,
    string systemPrompt,
    string userPrompt,
    bool interactive,
    bool verbose
)
{
    // Complete Q&A
    do
    {
        // Get prompt
        string user_prompt = interactive ? GetPrompt() : userPrompt;
        if (string.IsNullOrEmpty(user_prompt))
        {
            continue;
        }
        if (string.Compare(user_prompt, "quit()", StringComparison.OrdinalIgnoreCase) == 0)
        {
            break;
        }

        // Get input tokens
        string messages = $@"[{{""role"":""system"",""content"":""{systemPrompt}""}},{{""role"":""user"",""content"":""{user_prompt}""}}]";
        string prompt = Common.ApplyChatTemplate(modelPath, tokenizer, messages, add_generation_prompt: true);
        var sequences = tokenizer.Encode(prompt);
        if (verbose) Console.WriteLine($"Prompt encoded: {prompt}");

        // Set search options for generator params
        using GeneratorParams generatorParams = new GeneratorParams(model);
        Common.SetSearchOptions(generatorParams, generatorParamsArgs, verbose);

        // Create generator and append input tokens
        using Generator generator = new Generator(model, generatorParams);
        if (verbose) Console.WriteLine("Generator created");

        generator.AppendTokenSequences(sequences);
        if (verbose) Console.WriteLine("Input tokens added");

        // Run generation loop
        if (verbose) Console.WriteLine("Running generation loop...\n");
        var watch = System.Diagnostics.Stopwatch.StartNew();
        while (true)
        {
            generator.GenerateNextToken();
            if (generator.IsDone())
            {
                break;
            }
        }
        watch.Stop();
        var runTimeInSeconds = watch.Elapsed.TotalSeconds;

        // Get output tokens and decode to string
        var outputSequence = generator.GetSequence(0);
        var outputString = tokenizer.Decode(outputSequence);

        // Display output and timings
        Console.WriteLine("Output:");
        Console.WriteLine(outputString);
        var totalTokens = outputSequence.Length;
        Console.WriteLine($"Tokens: {totalTokens}, Time: {runTimeInSeconds:0.00}, Tokens per second: {totalTokens / runTimeInSeconds:0.00}");
        Console.WriteLine();

    } while (interactive);
}

/// <summary>
/// Example of model-qa
/// </summary>
/// <param name="model">Model to use</param>
/// <param name="tokenizer">Tokenizer to use</param>
/// <param name="tokenizerStream">Tokenizer stream to use</param>
/// <param name="generatorParamsArgs">Generator params arguments to use</param>
/// <param name="guidanceArgs">Guidance arguments to use</param>
/// <param name="modelPath">Path to folder containing model</param>
/// <param name="systemPrompt">System prompt to use with model</param>
/// <param name="userPrompt">User prompt to use with model</param>
/// <param name="interactive">Ask user or use pre-defined value</param>
/// <param name="verbose">Use verbose logging</param>
/// <returns>
/// None
/// </returns>
void ModelQA(
    Model model,
    Tokenizer tokenizer,
    TokenizerStream tokenizerStream,
    GeneratorParamsArgs generatorParamsArgs,
    GuidanceArgs guidanceArgs,
    string modelPath,
    string systemPrompt,
    string userPrompt,
    bool interactive,
    bool verbose
)
{
    // Creating running list of messages
    var system_message = new Dictionary<string, string>
    {
        { "role", "system" },
        { "content", systemPrompt }
    };
    var input_list = new List<Dictionary<string, string>>() { system_message };

    // Get and set guidance info if requested
    string guidance_type = "";
    string guidance_data = "";
    string tools = "";
    if (!string.IsNullOrEmpty(guidanceArgs.response_format))
    {
        Console.WriteLine("Make sure your tool call start id and tool call end id are marked as special in tokenizer.json");
        (guidance_type, guidance_data, tools) = Common.GetGuidance(
            response_format: guidanceArgs.response_format,
            filepath: guidanceArgs.tools_file,
            text_output: guidanceArgs.text_output,
            tool_output: guidanceArgs.tool_output,
            tool_call_start: guidanceArgs.tool_call_start,
            tool_call_end: guidanceArgs.tool_call_end
        );
        input_list[0]["tools"] = tools;
    }

    // Streaming Q&A
    do
    {
        // Get prompt
        string user_prompt = interactive ? GetPrompt() : userPrompt;
        if (string.IsNullOrEmpty(user_prompt))
        {
            continue;
        }
        if (string.Compare(user_prompt, "quit()", StringComparison.OrdinalIgnoreCase) == 0)
        {
            break;
        }

        // Add user message to list of messages
        var user_message = new Dictionary<string, string>
        {
            { "role", "user" },
            { "content", user_prompt }
        };
        input_list.Add(user_message);

        // Set search options for generator params
        using GeneratorParams generatorParams = new GeneratorParams(model);
        Common.SetSearchOptions(generatorParams, generatorParamsArgs, verbose);

        // Initialize guidance if requested
        if (!string.IsNullOrEmpty(guidance_type) && !string.IsNullOrEmpty(guidance_data))
        {
            generatorParams.SetGuidance(guidance_type, guidance_data);
            if (verbose)
            {
                Console.WriteLine();
                Console.WriteLine($"Guidance type is: {guidance_type}");
                Console.WriteLine($"Guidance data is: \n{guidance_data}");
                Console.WriteLine();
            }
        }

        // Create generator
        using Generator generator = new Generator(model, generatorParams);
        if (verbose) Console.WriteLine("Generator created");

        // Apply chat template
        string prompt = "";
        try
        {
            string messages = JsonSerializer.Serialize(input_list);
            prompt = Common.ApplyChatTemplate(modelPath, tokenizer, messages, add_generation_prompt: true, tools);
        }
        catch
        {
            prompt = user_prompt;
        }
        if (verbose) Console.WriteLine($"Prompt: {prompt}");

        // Encode combined system + user prompt and append tokens to model
        var sequences = tokenizer.Encode(prompt);
        generator.AppendTokenSequences(sequences);

        // Run generation loop
        if (verbose) Console.WriteLine("Running generation loop...\n");
        Console.Write("Output: ");
        var watch = System.Diagnostics.Stopwatch.StartNew();
        while (true)
        {
            generator.GenerateNextToken();
            if (generator.IsDone())
            {
                break;
            }
            // Decode and print the next token
            Console.Write(tokenizerStream.Decode(generator.GetNextTokens()[0]));
        }
        watch.Stop();
        var runTimeInSeconds = watch.Elapsed.TotalSeconds;

        // Remove user message from list of messages
        input_list.RemoveAt(input_list.Count - 1);

        // Display output and timings
        var outputSequence = generator.GetSequence(0);
        var totalTokens = outputSequence.Length;
        Console.WriteLine();
        Console.WriteLine($"Streaming Tokens: {totalTokens}, Time: {runTimeInSeconds:0.00}, Tokens per second: {totalTokens / runTimeInSeconds:0.00}");
        Console.WriteLine();

    } while (interactive);
}

/// <summary>
/// Example of model-chat
/// </summary>
/// <param name="model">Model to use</param>
/// <param name="tokenizer">Tokenizer to use</param>
/// <param name="tokenizerStream">Tokenizer stream to use</param>
/// <param name="generatorParamsArgs">Generator params arguments to use</param>
/// <param name="guidanceArgs">Guidance arguments to use</param>
/// <param name="modelPath">Path to folder containing model</param>
/// <param name="systemPrompt">System prompt to use with model</param>
/// <param name="userPrompt">User prompt to use with model</param>
/// <param name="interactive">Ask user or use pre-defined value</param>
/// <param name="rewind">Rewind to system prompt after each user prompt</param>
/// <param name="verbose">Use verbose logging</param>
/// <returns>
/// None
/// </returns>
void ModelChat(
    Model model,
    Tokenizer tokenizer,
    TokenizerStream tokenizerStream,
    GeneratorParamsArgs generatorParamsArgs,
    GuidanceArgs guidanceArgs,
    string modelPath,
    string systemPrompt,
    string userPrompt,
    bool interactive,
    bool rewind,
    bool verbose
)
{
    // Set search options for generator params
    using GeneratorParams generatorParams = new GeneratorParams(model);
    Common.SetSearchOptions(generatorParams, generatorParamsArgs, verbose);

    // Create system message
    var system_message = new Dictionary<string, string>
    {
        { "role", "system" },
        { "content", systemPrompt }
    };

    // Get and set guidance info if requested
    string tools = "";
    if (!string.IsNullOrEmpty(guidanceArgs.response_format))
    {
        Console.WriteLine("Make sure your tool call start id and tool call end id are marked as special in tokenizer.json");
        string guidance_type = "";
        string guidance_data = "";
        (guidance_type, guidance_data, tools) = Common.GetGuidance(
            response_format: guidanceArgs.response_format,
            filepath: guidanceArgs.tools_file,
            text_output: guidanceArgs.text_output,
            tool_output: guidanceArgs.tool_output,
            tool_call_start: guidanceArgs.tool_call_start,
            tool_call_end: guidanceArgs.tool_call_end
        );
        system_message["tools"] = tools;

        generatorParams.SetGuidance(guidance_type, guidance_data);
        if (verbose)
        {
            Console.WriteLine();
            Console.WriteLine($"Guidance type is: {guidance_type}");
            Console.WriteLine($"Guidance data is: \n{guidance_data}");
            Console.WriteLine();
        }
    }

    // Create generator
    using Generator generator = new Generator(model, generatorParams);
    if (verbose) Console.WriteLine("Generator created");

    // Apply chat template
    string prompt = "";
    try
    {
        string messages = JsonSerializer.Serialize(new List<Dictionary<string, string>> { system_message });
        prompt = Common.ApplyChatTemplate(modelPath, tokenizer, messages, add_generation_prompt: false, tools);
    }
    catch
    {
        prompt = systemPrompt;
    }
    if (verbose) Console.WriteLine($"System prompt: {prompt}\n");

    // Encode system prompt and append tokens to model
    var sequences = tokenizer.Encode(prompt);
    var system_prompt_length = sequences[0].Length;
    generator.AppendTokenSequences(sequences);

    // Streaming Chat
    var prevTotalTokens = 0;
    do
    {
        // Get prompt
        string user_prompt = interactive ? GetPrompt() : userPrompt;
        if (string.IsNullOrEmpty(user_prompt))
        {
            continue;
        }
        if (string.Compare(user_prompt, "quit()", StringComparison.OrdinalIgnoreCase) == 0)
        {
            break;
        }

        // Create user message
        var user_message = new Dictionary<string, string>
        {
            { "role", "user" },
            { "content", user_prompt }
        };

        // Apply chat template
        prompt = "";
        try
        {
            string messages = JsonSerializer.Serialize(new List<Dictionary<string, string>> { user_message });
            prompt = Common.ApplyChatTemplate(modelPath, tokenizer, messages, add_generation_prompt: true);
        }
        catch
        {
            prompt = systemPrompt;
        }
        if (verbose) Console.WriteLine($"User prompt: {prompt}");

        // Encode user prompt and append tokens to model
        sequences = tokenizer.Encode(prompt);
        generator.AppendTokenSequences(sequences);

        // Run generation loop
        if (verbose) Console.WriteLine("Running generation loop...\n");
        Console.Write("Output: ");
        var watch = System.Diagnostics.Stopwatch.StartNew();
        while (true)
        {
            generator.GenerateNextToken();
            if (generator.IsDone())
            {
                break;
            }
            Console.Write(tokenizerStream.Decode(generator.GetNextTokens()[0]));
        }
        watch.Stop();
        var runTimeInSeconds = watch.Elapsed.TotalSeconds;

        // Display output and timings
        var outputSequence = generator.GetSequence(0);
        var totalNewTokens = outputSequence.Length - prevTotalTokens;
        prevTotalTokens = totalNewTokens;
        Console.WriteLine();
        Console.WriteLine($"Streaming Tokens: {totalNewTokens}, Time: {runTimeInSeconds:0.00}, Tokens per second: {totalNewTokens / runTimeInSeconds:0.00}");
        Console.WriteLine();

        if (rewind)
        {
            generator.RewindTo((ulong)system_prompt_length);
        }

    } while (interactive);
}

/// <summary>
/// Get command-line arguments
/// </summary>
/// <returns>
/// RootCommand object with all possible command-line arguments
/// </returns>
RootCommand GetArgs()
{
    var parser = new RootCommand("ModelChat Arguments");

    var model_path = new Option<string>(
        name: "model_path",
        aliases: ["-m", "--model_path"]
    )
    {
        Arity = ArgumentArity.ExactlyOne,
        Description = "Path to the model",
        Required = true
    };
    model_path.Validators.Add(result =>
    {
        var value = result.GetValue(model_path);
        if (string.IsNullOrEmpty(value))
        {
            result.AddError("Model path must be specified");
        }
        else if (!Path.Exists(value))
        {
            result.AddError("Path must be to a model folder on disk");
        }
    });
    
    var execution_provider = new Option<string>(
        name: "execution_provider",
        aliases: ["-e", "--execution_provider"]
    )
    {
        Arity = ArgumentArity.ExactlyOne,
        DefaultValueFactory = (_) => "follow_config",
        Description = "Execution provider to run the model"
    };
    execution_provider.Validators.Add(result => {
        var value = result.GetValue(execution_provider);
        if (string.IsNullOrEmpty(value))
        {
            result.AddError("Execution provider must be specified. Use 'follow_config' to not specify one.");
        }
    });

    var verbose = new Option<bool>(
        name: "verbose",
        aliases: ["-v", "--verbose"]
    )
    {
        Arity = ArgumentArity.Zero,
        DefaultValueFactory = (_) => false,
        Description = "Print verbose output. Defaults to false"
    };

    var non_interactive = new Option<bool>(
        name: "non_interactive",
        aliases: ["--non_interactive"]
    )
    {
        Arity = ArgumentArity.Zero,
        DefaultValueFactory = (_) => false,
        Description = "Run in interactive mode"
    };
    
    var system_prompt = new Option<string>(
        name: "system_prompt",
        aliases: ["-sp", "--system_prompt"]
    )
    {
        Arity = ArgumentArity.ExactlyOne,
        DefaultValueFactory = (_) => "You are a helpful AI assistant.",
        Description = "System prompt to use for the model."
    };

    var user_prompt = new Option<string>(
        name: "user_prompt",
        aliases: ["-up", "--user_prompt"]
    )
    {
        Arity = ArgumentArity.ExactlyOne,
        DefaultValueFactory = (_) => "What color is the sky?",
        Description = "User prompt to use for the model."
    };

    var rewind = new Option<bool>(
        name: "rewind",
        aliases: ["-rw", "--rewind"]
    )
    {
        Arity = ArgumentArity.Zero,
        DefaultValueFactory = (_) => false,
        Description = "Rewind to the system prompt after each generation. Defaults to false"
    };

    parser.Add(model_path);
    parser.Add(execution_provider);
    parser.Add(system_prompt);
    parser.Add(user_prompt);
    parser.Add(verbose);
    parser.Add(non_interactive);
    parser.Add(rewind);

    Common.GetGeneratorParamsArgs(parser);
    Common.GetGuidanceArgs(parser);

    return parser;
}

/// <summary>
/// Main method for inference
/// </summary>
/// <param name="args">Command-line arguments</param>
/// <returns>
/// None
/// </returns>
void main(string[] args) {
    // Obtain and parse command-line arguments
    RootCommand parser = GetArgs();
    ParseResult parseResult = parser.Parse(args);
    parseResult.Invoke();

    // Validate command-line arguments
    if (args.Length < 1 || parseResult.Errors.Count > 0 || parseResult.Tokens.Any(t => t.Value is "-h" or "--help" or "-?"))
    {
        Console.WriteLine("Run this with -h/--help/-? to see which arguments you need to set.");
        foreach (var error in parseResult.Errors)
        {
            Console.WriteLine("Error: " + error.Message);
        }
        // Exit early
        return;
    }

    // Get main argument values
    string modelPath = parseResult.GetValue<string>("model_path")!;
    string executionProvider = parseResult.GetValue<string>("execution_provider")!;
    string systemPrompt = parseResult.GetValue<string>("system_prompt")!;
    string userPrompt = parseResult.GetValue<string>("user_prompt")!;
    bool verbose = parseResult.GetValue<bool>("verbose");
    bool interactive = !parseResult.GetValue<bool>("non_interactive");
    bool rewind = parseResult.GetValue<bool>("rewind");

    var (generatorParamsArgs, guidanceArgs) = Common.SetGroupedArgs(parseResult);

    // Print main argument values
    Console.WriteLine("-----------------");
    Console.WriteLine("Hello, ModelChat!");
    Console.WriteLine("-----------------");

    Console.WriteLine("Model path: " + modelPath);
    Console.WriteLine("Execution provider: " + executionProvider);
    Console.WriteLine("System prompt: " + systemPrompt);
    if (!interactive)
    {
        Console.WriteLine("User prompt: " + userPrompt);
    }
    Console.WriteLine("Verbose: " + verbose);
    Console.WriteLine("Interactive: " + interactive);
    Console.WriteLine("Rewind: " + rewind);
    Console.WriteLine("-----------------");
    Console.WriteLine();

    // Create model
    if (verbose) Console.WriteLine("Loading model...");
    using Config config = Common.GetConfig(path: modelPath, ep: executionProvider, null, generatorParamsArgs);
    using Model model = new Model(config);
    if (verbose) Console.WriteLine("Model loaded");

    // Create tokenizer
    using Tokenizer tokenizer = new Tokenizer(model);
    using TokenizerStream tokenizerStream = tokenizer.CreateStream();
    if (verbose) Console.WriteLine("Tokenizer created");

    // Get scenario to run from user
    var option = 2;
    if (interactive)
    {
        do
        {
            Console.WriteLine("Please enter option number:");
            Console.WriteLine("1. Complete Q&A");
            Console.WriteLine("2. Streaming Q&A");
            Console.WriteLine("3. Streaming Chat");
            Console.Write("> ");
            int.TryParse(Console.ReadLine(), out option);

            if (option < 1 || option > 3)
            {
                Console.WriteLine("Invalid option. Please try again.");
            }
        } while (option < 1 || option > 3);
    }

    // Get prompt and run chosen scenario
    if (option == 1)
    {
        if (verbose) Console.WriteLine("Entering option 1\n");
        ModelGenerate(model, tokenizer, generatorParamsArgs, modelPath, systemPrompt, userPrompt, interactive, verbose);
    }
    else if (option == 2)
    {
        if (verbose) Console.WriteLine("Entering option 2\n");
        ModelQA(model, tokenizer, tokenizerStream, generatorParamsArgs, guidanceArgs, modelPath, systemPrompt, userPrompt, interactive, verbose);
    }
    else
    {
        if (verbose) Console.WriteLine("Entering option 3\n");
        ModelChat(model, tokenizer, tokenizerStream, generatorParamsArgs, guidanceArgs, modelPath, systemPrompt, userPrompt, interactive, rewind, verbose);
    }
}

using OgaHandle ogaHandle = new OgaHandle();
main(args);
