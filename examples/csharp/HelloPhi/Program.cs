// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntimeGenAI;

void PrintUsage()
{
    Console.WriteLine("Usage:");
    Console.WriteLine("  -m model_path");
    Console.WriteLine("\t\t\t\tPath to the model");
    Console.WriteLine("  -e execution_provider");
    Console.WriteLine("\t\t\t\tExecution provider to run the model");
    Console.WriteLine("  --non-interactive (optional)");
    Console.WriteLine("\t\t\t\tInteractive mode");
}

using OgaHandle ogaHandle = new OgaHandle();

if (args.Length < 1)
{
    PrintUsage();
    Environment.Exit(-1);
}

bool interactive = true;
string modelPath = string.Empty;
string executionProvider = string.Empty;

uint i = 0;
while (i < args.Length)
{
    var arg = args[i];
    if (arg == "--non-interactive")
    {
        interactive = false;
    }
    else if (arg == "-m")
    {
        if (i + 1 < args.Length)
        {
            modelPath = Path.Combine(args[i+1]);
        }
    }
    else if (arg == "-e")
    {
        if (i + 1 < args.Length)
        {
            executionProvider = Path.Combine(args[i+1]);
        }
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
Console.WriteLine("Interactive: " + interactive);

using Config config = new Config(modelPath);
config.ClearProviders();
if (executionProvider != "cpu") {
    config.AppendProvider(executionProvider);
    if (executionProvider == "cuda") {
        config.SetProviderOption(executionProvider, "enable_cuda_graph", "0");
    }
}
using Model model = new Model(config);
using Tokenizer tokenizer = new Tokenizer(model);

var option = 2;
if (interactive)
{
    Console.WriteLine("Please enter option number:");
    Console.WriteLine("1. Complete Q&A");
    Console.WriteLine("2. Streaming Q&A");
    Console.WriteLine("3. Streaming Chat (not supported for DirectML and QNN currently)");
    int.TryParse(Console.ReadLine(), out option);
}

int minLength = 50;
int maxLength = 500;

static string GetPrompt(bool interactive)
{
    string prompt = "def is_prime(num):"; // Example prompt
    if (interactive)
    {
        Console.WriteLine("Prompt: (Use quit() to exit)");
        prompt = Console.ReadLine();
    }
    return prompt;
}

if (option == 1 || option == 2)
{
    do
    {
        string prompt = GetPrompt(interactive);
        if (string.IsNullOrEmpty(prompt))
        {
            continue;
        }
        if (string.Compare(prompt, "quit()", StringComparison.OrdinalIgnoreCase) == 0)
        {
            break;
        }
        var sequences = tokenizer.Encode(tokenizer.ApplyChatTemplate("", prompt, "", true));

        if (option == 1) // Complete Output
        {
            using GeneratorParams generatorParams = new GeneratorParams(model);
            generatorParams.SetSearchOption("min_length", minLength);
            generatorParams.SetSearchOption("max_length", maxLength);
            using var generator = new Generator(model, generatorParams);
            generator.AppendTokenSequences(sequences);
            var watch = System.Diagnostics.Stopwatch.StartNew();
            while (!generator.IsDone())
            {
                generator.GenerateNextToken();
            }

            var outputSequence = generator.GetSequence(0);
            var outputString = tokenizer.Decode(outputSequence);
            watch.Stop();
            var runTimeInSeconds = watch.Elapsed.TotalSeconds;
            Console.WriteLine("Output:");
            Console.WriteLine(outputString);
            var totalTokens = outputSequence.Length;
            Console.WriteLine($"Tokens: {totalTokens} Time: {runTimeInSeconds:0.00} Tokens per second: {totalTokens / runTimeInSeconds:0.00}");
        }

        else if (option == 2) //Streaming Output
        {
            using GeneratorParams generatorParams = new GeneratorParams(model);
            generatorParams.SetSearchOption("min_length", minLength);
            generatorParams.SetSearchOption("max_length", maxLength);
            using var tokenizerStream = tokenizer.CreateStream();
            using var generator = new Generator(model, generatorParams);
            generator.AppendTokenSequences(sequences);
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
    } while (interactive);
}

if (option == 3) // Streaming Chat
{
    using GeneratorParams generatorParams = new GeneratorParams(model);
    generatorParams.SetSearchOption("min_length", minLength);
    generatorParams.SetSearchOption("max_length", maxLength);
    using var tokenizerStream = tokenizer.CreateStream();
    using var generator = new Generator(model, generatorParams);
    var prevTotalTokens = 0;
    do{
        string prompt = GetPrompt(interactive);
        if (string.IsNullOrEmpty(prompt))
        {
            continue;
        }
        if (string.Compare(prompt, "quit()", StringComparison.OrdinalIgnoreCase) == 0)
        {
            break;
        }
        var sequences = tokenizer.Encode(tokenizer.ApplyChatTemplate("", prompt, "", true));
        var watch = System.Diagnostics.Stopwatch.StartNew();
        generator.AppendTokenSequences(sequences);
        while (!generator.IsDone())
        {
            generator.GenerateNextToken();
            Console.Write(tokenizerStream.Decode(generator.GetSequence(0)[^1]));
        }
        Console.WriteLine();
        watch.Stop();
        var runTimeInSeconds = watch.Elapsed.TotalSeconds;
        var outputSequence = generator.GetSequence(0);
        var totalNewTokens = outputSequence.Length - prevTotalTokens;
        prevTotalTokens = totalNewTokens;
        Console.WriteLine($"Streaming Tokens: {totalNewTokens} Time: {runTimeInSeconds:0.00} Tokens per second: {totalNewTokens / runTimeInSeconds:0.00}");
    } while (interactive);
}
