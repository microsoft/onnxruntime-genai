// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntimeGenAI;
using System.Runtime.InteropServices;
using System.Text.Json;

static void PrintUsage()
{
    Console.WriteLine("Usage:");
    Console.WriteLine("  -m model_path");
    Console.WriteLine("\t\t\t\tPath to the model");
    Console.WriteLine("  -e execution_provider");
    Console.WriteLine("\t\t\t\tExecution provider to run the model");
    Console.WriteLine("  --verbose");
    Console.WriteLine("\t\t\t\tRun in verbose mode");
    Console.WriteLine("  --dump");
    Console.WriteLine("\t\t\t\tDump input and output data");
}

static void DebugMode()
{
    Utils.SetLogBool("enabled", true);
    Utils.SetLogBool("model_input_values", true);
    Utils.SetLogBool("model_output_values", true);
    Utils.SetLogBool("ansi_tags", RuntimeInformation.IsOSPlatform(OSPlatform.Windows));
}

static string GetInput(string description, string defaultInput)
{
    string userInput = "";
    Console.WriteLine(description);
    Console.Write("> ");
    userInput = Console.ReadLine();

    if (string.IsNullOrEmpty(userInput))
    {
        return defaultInput;
    }
    return userInput;
}

static void ValidateJson(string text, string description)
{
    try
    {
        var deserializedJson = JsonDocument.Parse(text); // requires RootElement so text must be {root_key: root_val} where root_val is anything
    }
    catch
    {
        throw new Exception($"Invalid JSON provided for {description}");
    }
}

static string GetJsonGrammar(string tools)
{
    string grammar = $@"{{ ""anyOf"": {tools} }}";  // Spacing here matters
    return grammar;
}

static string GetLarkGrammar(string tools, string toolCallToken)
{
    string startRow = "start: TEXT | fun_call";
    string textRow = "TEXT: /[^{](.|\\n)*/";
    string funcRow = "fun_call: " + toolCallToken + " %json ";
    string toolsJson = GetJsonGrammar(tools);
    string grammar = startRow + " \n" + textRow + " \n" + funcRow + toolsJson;
    return grammar;
}

static string GetMessages(string guidanceMode, string role, string content, string toolsList)
{
    string messages = "";
    if (string.Compare(guidanceMode, "json_schema", StringComparison.OrdinalIgnoreCase) == 0 ||
        string.Compare(guidanceMode, "lark_grammar", StringComparison.OrdinalIgnoreCase) == 0)
    {
        messages = $@"{{""role"": ""{role}"", ""content"": ""{content}"", ""tools"": ""{toolsList}""}}";
    }
    else
    {
        messages = $@"{{""role"": ""{role}"", ""content"": ""{content}""}}";
    }

    return "[" + messages + "]";
}

static void RunInference(Model model, Tokenizer tokenizer, int option, bool verbose)
{
    string systemPrompt = GetInput("System Prompt: (Press 'enter' to use the default)", "You are a helpful AI assistant.");
    string guidanceMode = GetInput("Guidance Mode: (Press 'enter' to use none)", "");
    string guidanceData = GetInput("Guidance Data: (Provide a list of tools in JSON format or press 'enter' to use none)", "");

    // Constants
    string toolCallToken = "<|tool_call|>";  // specific to Phi-4 mini
    int minLength = 0;
    int maxLength = 8192;

    // Create generator params and tokenizer stream
    using GeneratorParams generatorParams = new GeneratorParams(model);
    using var tokenizerStream = tokenizer.CreateStream();

    // Get and set guidance input
    if (!string.IsNullOrEmpty(guidanceMode))
    {
        if (string.IsNullOrEmpty(guidanceData))
        {
            throw new Exception("Guidance information is required if guidance type is provided.");
        }

        // Get guidance input
        string guidanceInput = "";
        if (string.Compare(guidanceMode, "json_schema", StringComparison.OrdinalIgnoreCase) == 0)
        {
            guidanceInput = GetJsonGrammar(guidanceData);
            ValidateJson(guidanceInput, "JSON schema");
        }
        else if (string.Compare(guidanceMode, "lark_grammar", StringComparison.OrdinalIgnoreCase) == 0)
        {
            ValidateJson(guidanceData, "LARK grammar");
            guidanceInput = GetLarkGrammar(guidanceData, toolCallToken);
        }
        else if (string.Compare(guidanceMode, "regex", StringComparison.OrdinalIgnoreCase) == 0)
        {
            guidanceInput = guidanceData;
        }
        else
        {
            throw new Exception("Guidance type must be one of the following: json_schema, lark_grammar, regex.");
        }

        // Set guidance input
        if (verbose)
        {
            Console.WriteLine($"Guidance mode is set to: \n{guidanceMode}");
            Console.WriteLine($"Guidance input is: \n{guidanceInput}");
        }
        generatorParams.SetGuidance(guidanceMode, guidanceInput);
    }
    else
    {
        guidanceMode = "";
    }

    // Get final system prompt in tokenized form
    string toolsListJson = guidanceData.Length == 0 ? "" : guidanceData[1..(guidanceData.Length - 1)].Replace("\"", "'");
    string systemMessages = GetMessages(guidanceMode: guidanceMode, role: "system", content: systemPrompt, toolsList: toolsListJson);
    if (verbose)
    {
        Console.WriteLine($"System messages are: {systemMessages}");
    }
    string finalSystemPrompt = tokenizer.ApplyChatTemplate(template_str: null, messages: systemMessages, add_generation_prompt: false);
    finalSystemPrompt = finalSystemPrompt.Replace("<|endoftext|>", "");
    if (verbose)
    {
        Console.WriteLine($"System prompt is: {finalSystemPrompt}");
    }
    var systemTokens = tokenizer.Encode(finalSystemPrompt);

    // Set generator params
    generatorParams.SetSearchOption("min_length", minLength);
    generatorParams.SetSearchOption("max_length", maxLength);

    // Create generator and append system prompt
    using var generator = new Generator(model, generatorParams);
    if (verbose)
    {
        Console.WriteLine("Generator created");
    }
    Console.WriteLine("Appending prompt...");
    generator.AppendTokenSequences(systemTokens);

    while (true)
    {
        // Get user prompt
        string userPrompt = GetInput("Prompt: (Use quit() to exit)", "");
        if (string.IsNullOrEmpty(userPrompt))
        {
            continue;
        }
        if (string.Compare(userPrompt, "quit()", StringComparison.OrdinalIgnoreCase) == 0)
        {
            break;
        }

        // Get final user prompt in tokenized form
        string userMessages = GetMessages(guidanceMode: "", role: "user", content: userPrompt, toolsList: "");
        if (verbose)
        {
            Console.WriteLine($"User messages are: {userMessages}");
        }
        string finalUserPrompt = tokenizer.ApplyChatTemplate(template_str: "", messages: userMessages, add_generation_prompt: true);
        if (verbose)
        {
            Console.WriteLine($"User prompt is: {finalUserPrompt}");
        }
        var userTokens = tokenizer.Encode(finalUserPrompt);

        // Append user tokens
        generator.AppendTokenSequences(userTokens);

        // Run generation loop
        var watch = System.Diagnostics.Stopwatch.StartNew();
        while (!generator.IsDone())
        {
            generator.GenerateNextToken();
            Console.Write(tokenizerStream.Decode(generator.GetSequence(0)[^1]));
        }
        Console.WriteLine();
        watch.Stop();
        var runTimeInSeconds = watch.Elapsed.TotalSeconds;
        var outputSequence = generator.GetSequence(0);
        var totalTokens = outputSequence.Length;
        Console.WriteLine($"Streaming Tokens: {totalTokens} Time: {runTimeInSeconds:0.00} Tokens per second: {totalTokens / runTimeInSeconds:0.00}");
    }
}

static void Main(string[] args)
{
    using OgaHandle ogaHandle = new OgaHandle();

    if (args.Length < 1)
    {
        PrintUsage();
        Environment.Exit(-1);
    }

    string modelPath = string.Empty;
    string executionProvider = string.Empty;
    bool verbose = false;

    uint i = 0;
    while (i < args.Length)
    {
        var arg = args[i];
        if (arg == "-m")
        {
            if (i + 1 < args.Length)
            {
                modelPath = Path.Combine(args[i + 1]);
            }
        }
        else if (arg == "-e")
        {
            if (i + 1 < args.Length)
            {
                executionProvider = Path.Combine(args[i + 1]);
            }
        }
        else if (arg == "--verbose")
        {
            verbose = true;
        }
        else if (arg == "--dump")
        {
            DebugMode();
        }
        i++;
    }

    if (string.IsNullOrEmpty(modelPath))
    {
        throw new Exception("Model path must be specified");
    }
    if (string.IsNullOrEmpty(executionProvider))
    {
        throw new Exception("Execution provider must be specified");
    }

    Console.WriteLine("-------------");
    Console.WriteLine("Hello, Phi!");
    Console.WriteLine("-------------");

    Console.WriteLine("Model path: " + modelPath);
    Console.WriteLine("Execution provider: " + executionProvider);

    using Config config = new Config(modelPath);
    config.ClearProviders();
    if (executionProvider != "cpu")
    {
        config.AppendProvider(executionProvider);
        if (executionProvider == "cuda")
        {
            config.SetProviderOption(executionProvider, "enable_cuda_graph", "0");
        }
    }

    Console.WriteLine("Loading model...");
    using Model model = new Model(config);
    Console.WriteLine("Model loaded");
    using Tokenizer tokenizer = new Tokenizer(model);

    var option = 2;
    //Console.WriteLine("Please enter option number:");
    //Console.WriteLine("1. Complete Q&A");
    //Console.WriteLine("2. Streaming Q&A");
    //Console.WriteLine("3. Streaming Chat (not supported for DirectML and QNN currently)");
    //Console.Write("> ");
    //int.TryParse(Console.ReadLine(), out option);

    RunInference(model, tokenizer, option, verbose);
}

Main(args);