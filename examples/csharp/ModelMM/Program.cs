// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using CommonUtils;
using Microsoft.ML.OnnxRuntimeGenAI;
using System.CommandLine;
using System.Text.Json;

/// <summary>
/// Example of model-mm
/// </summary>
/// <param name="model">Model to use</param>
/// <param name="tokenizer">Tokenizer to use</param>
/// <param name="tokenizerStream">Tokenizer stream to use</param>
/// <param name="processor">Processor to use</param>
/// <param name="generatorParamsArgs">Generator params arguments to use</param>
/// <param name="guidanceArgs">Guidance arguments to use</param>
/// <param name="imagePaths">File paths to images</param>
/// <param name="audioPaths">File paths to audios</param>
/// <param name="modelPath">Path to folder containing model</param>
/// <param name="systemPrompt">System prompt to use with model</param>
/// <param name="userPrompt">User prompt to use with model</param>
/// <param name="interactive">Ask user or use pre-defined value</param>
/// <param name="verbose">Use verbose logging</param>
/// <returns>
/// None
/// </returns>
void ModelMM(
    Model model,
    Tokenizer tokenizer,
    TokenizerStream tokenizerStream,
    MultiModalProcessor processor,
    GeneratorParamsArgs generatorParamsArgs,
    GuidanceArgs guidanceArgs,
    List<string> imagePaths,
    List<string> audioPaths,
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
        // Get images
        Images? images;
        int num_images;
        (images, num_images) = Common.GetUserImages(imagePaths, interactive);

        // Get audios
        Audios? audios;
        int num_audios;
        (audios, num_audios) = Common.GetUserAudios(audioPaths, interactive);

        // Get user prompt
        string text = Common.GetUserPrompt(userPrompt, interactive);
        if (string.Compare(text, "quit()", StringComparison.OrdinalIgnoreCase) == 0)
        {
            break;
        }

        // Construct user content based on inputs
        /**
         * TODO: Uncomment the below snippet to use model.GetType() once
         * the C# binding to Model.GetType() is in a stable package release.
         */
        //var user_content = Common.GetUserContent(model.GetType(), num_images, num_audios, text);
        var user_content = Common.GetUserContent("phi4mm", num_images, num_audios, text);

        // Add user message to list of messages
        var user_message = new Dictionary<string, string>
        {
            { "role", "user" },
            { "content", user_content }
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
            prompt = text;
        }
        if (verbose) Console.WriteLine($"Prompt: {prompt}");

        // Encode combined system + user prompt and append inputs to model
        using var inputTensors = processor.ProcessImagesAndAudios(prompt, images, audios);
        generator.SetInputs(inputTensors);

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

    var debug = new Option<bool>(
        name: "debug",
        aliases: ["-d", "--debug"]
    )
    {
        Arity = ArgumentArity.Zero,
        DefaultValueFactory = (_) => false,
        Description = "Dump input and output tensors with debug mode. Defaults to false"
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

    var ep_path = new Option<string>(
        name: "ep_path",
        aliases: ["--ep_path"]
    )
    {
        Arity = ArgumentArity.ExactlyOne,
        DefaultValueFactory = (_) => "",
        Description = "Path to execution provider DLL/SO for plug-in providers (ex: onnxruntime_providers_cuda.dll or onnxruntime_providers_tensorrt.dll)"
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

    var image_paths = new Option<List<string>>(
        name: "image_paths",
        aliases: ["--image_paths"]
    )
    {
        Arity = ArgumentArity.ZeroOrMore,
        AllowMultipleArgumentsPerToken = true,
        DefaultValueFactory = (_) => [],
        Description = "File paths to the images"
    };

    var audio_paths = new Option<List<string>>(
        name: "audio_paths",
        aliases: ["--audio_paths"]
    )
    {
        Arity = ArgumentArity.ZeroOrMore,
        AllowMultipleArgumentsPerToken = true,
        DefaultValueFactory = (_) => [],
        Description = "File paths to the audios"
    };

    parser.Add(model_path);
    parser.Add(execution_provider);
    parser.Add(ep_path);
    parser.Add(system_prompt);
    parser.Add(user_prompt);
    parser.Add(verbose);
    parser.Add(debug);
    parser.Add(non_interactive);
    parser.Add(rewind);
    parser.Add(image_paths);
    parser.Add(audio_paths);

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
    string epPath = parseResult.GetValue<string>("ep_path")!;
    string systemPrompt = parseResult.GetValue<string>("system_prompt")!;
    string userPrompt = parseResult.GetValue<string>("user_prompt")!;
    bool verbose = parseResult.GetValue<bool>("verbose");
    bool debug = parseResult.GetValue<bool>("debug");
    bool interactive = !parseResult.GetValue<bool>("non_interactive");
    bool rewind = parseResult.GetValue<bool>("rewind");
    List<string> imagePaths = parseResult.GetValue<List<string>>("image_paths") ?? [];
    List<string> audioPaths = parseResult.GetValue<List<string>>("audio_paths") ?? [];

    var (generatorParamsArgs, guidanceArgs) = Common.SetGroupedArgs(parseResult);

    // Print main argument values
    Console.WriteLine("-----------------");
    Console.WriteLine("Hello, ModelMM!");
    Console.WriteLine("-----------------");

    Console.WriteLine("Model path: " + modelPath);
    Console.WriteLine("Execution provider: " + executionProvider);
    if (!string.IsNullOrEmpty(epPath))
    {
        Console.WriteLine("Execution provider path: " + epPath);
    }
    Console.WriteLine("System prompt: " + systemPrompt);
    if (!interactive)
    {
        Console.WriteLine("User prompt: " + userPrompt);
    }
    Console.WriteLine("Verbose: " + verbose);
    Console.WriteLine("Debug: " + debug);
    Console.WriteLine("Interactive: " + interactive);
    Console.WriteLine("Rewind: " + rewind);
    Console.WriteLine("-----------------");
    Console.WriteLine();

    // Enable debugging if requested
    if (debug) Common.SetLogger();
    /**
     * TODO: Uncomment the below snippet to use Utils.RegisterEPLibrary once
     * the C# binding to Utils.RegisterEPLibrary is in a stable package release.
     */
    // RegisterEP(executionProvider, epPath);

    // Create model
    if (verbose) Console.WriteLine("Loading model...");
    using Config config = Common.GetConfig(path: modelPath, ep: executionProvider, null, generatorParamsArgs);
    using Model model = new Model(config);
    if (verbose) Console.WriteLine("Model loaded");

    // Create tokenizer
    using Tokenizer tokenizer = new Tokenizer(model);
    using TokenizerStream tokenizerStream = tokenizer.CreateStream();
    if (verbose) Console.WriteLine("Tokenizer created");

    // Create processor
    using MultiModalProcessor processor = new MultiModalProcessor(model);
    if (verbose) Console.WriteLine("Processor created");

    // Get prompt and run scenario
    if (verbose) Console.WriteLine("Entering model-mm\n");
    ModelMM(model, tokenizer, tokenizerStream, processor, generatorParamsArgs, guidanceArgs, imagePaths, audioPaths, modelPath, systemPrompt, userPrompt, interactive, verbose);
}

using OgaHandle ogaHandle = new OgaHandle();
main(args);
