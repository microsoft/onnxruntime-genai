// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntimeGenAI;
using System.Linq;
using System.Runtime.CompilerServices;

static string GetDirectoryInTreeThatContains(string currentDirectory, string targetDirectoryName)
{
    bool found = false;
    foreach (string d in Directory.GetDirectories(currentDirectory, searchPattern: targetDirectoryName))
    {
        found = true;
        return Path.Combine(currentDirectory, targetDirectoryName);
    }
    if (!found)
    {
        DirectoryInfo dirInfo = new DirectoryInfo(currentDirectory);
        if (dirInfo.Parent != null)
        {
            return GetDirectoryInTreeThatContains(Path.GetFullPath(Path.Combine(currentDirectory, "..")), targetDirectoryName);
        }
        else
        {
            return null;
        }
    }
    return null;
}

void PrintUsage()
{
    Console.WriteLine("Usage:");
    Console.WriteLine("  -m model_path");
    Console.WriteLine("\t\t\t\tPath to the model");
    Console.WriteLine("  -e execution_provider");
    Console.WriteLine("\t\t\t\tExecution provider for the model");
    Console.WriteLine("  --image_paths");
    Console.WriteLine("\t\t\t\tPath to the images");
    Console.WriteLine("  --audio_paths");
    Console.WriteLine("\t\t\t\tPath to the audios");
    Console.WriteLine("  --non-interactive (optional), mainly for CI usage");
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
List<string> imagePaths = new List<string>();
List<string> audioPaths = new List<string>();

uint i_arg = 0;
while (i_arg < args.Length)
{
    var arg = args[i_arg];
    if (arg == "--non-interactive")
    {
        interactive = false;
    }
    else if (arg == "-m")
    {
        if (i_arg + 1 < args.Length)
        {
            modelPath = Path.Combine(args[i_arg+1]);
        }
    }
    else if (arg == "-e")
    {
        if (i_arg + 1 < args.Length)
        {
            executionProvider = Path.Combine(args[i_arg+1]);
        }
    }
    else if (arg == "--image_paths")
    {
        if (i_arg + 1 < args.Length)
        {
            imagePaths = args[i_arg + 1].Split(',').ToList<string>().Select(i => i.ToString().Trim()).ToList();
        }
    }
    else if (arg == "--audio_paths")
    {
        if (i_arg + 1 < args.Length)
        {
            audioPaths = args[i_arg + 1].Split(',').ToList<string>().Select(i => i.ToString().Trim()).ToList();
        }
    }
    i_arg++;
}

if (string.IsNullOrEmpty(modelPath))
{
    throw new Exception("Model path must be specified");
}
if (string.IsNullOrEmpty(executionProvider))
{
    throw new Exception("Execution provider must be specified");
}

Console.WriteLine("--------------------");
Console.WriteLine("Hello, Phi-4-Multimodal!");
Console.WriteLine("--------------------");

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
using MultiModalProcessor processor = new MultiModalProcessor(model);
using var tokenizerStream = processor.CreateStream();

do
{
    // Get images
    if (interactive)
    {
        Console.WriteLine("Image Path (comma separated; leave empty if no image):");
        imagePaths = Console.ReadLine().Split(',').ToList<string>().Select(i => i.ToString().Trim()).ToList();
    }
    if (imagePaths.Count == 0)
    {
        Console.WriteLine("No image provided. Using default image.");
        imagePaths.Add(Path.Combine(
            GetDirectoryInTreeThatContains(Directory.GetCurrentDirectory(), "test"), "test_models", "images", "australia.jpg"));
    }
    for (int i = 0; i < imagePaths.Count; i++)
    {
        string imagePath = Path.GetFullPath(imagePaths[i].Trim());
        if (!File.Exists(imagePath))
        {
            throw new Exception("Image file not found: " + imagePath);
        }
        Console.WriteLine("Using image: " + imagePath);
    }
    Images images = imagePaths.Count > 0 ? Images.Load(imagePaths.ToArray()) : null;

    // Get audios
    if (interactive)
    {
        Console.WriteLine("Audio Path (comma separated; leave empty if no audio):");
        audioPaths = Console.ReadLine().Split(',').ToList<string>().Select(i => i.ToString().Trim()).ToList();
    }
    if (audioPaths.Count == 0)
    {
        Console.WriteLine("No audio provided. Using default audio.");
        audioPaths.Add(Path.Combine(
            GetDirectoryInTreeThatContains(Directory.GetCurrentDirectory(), "test"), "test_models", "audios", "1272-141231-0002.mp3"));
    }
    for (int i = 0; i < audioPaths.Count; i++)
    {
        string audioPath = Path.GetFullPath(audioPaths[i].Trim());
        if (!File.Exists(audioPath))
        {
            throw new Exception("Audio file not found: " + audioPath);
        }
        Console.WriteLine("Using audio: " + audioPath);
    }
    Audios audios = audioPaths.Count > 0 ? Audios.Load(audioPaths.ToArray()) : null;

    // Get prompt
    string text = "Does the audio summarize what is in the picture? If not, what is different?";
    if (interactive) {
        Console.WriteLine("Prompt:");
        text = Console.ReadLine();
    }

    // Combine prompt, images, and audios
    string prompt = "<|user|>\n";
    if (images != null)
    {
        for (int i = 0; i < imagePaths.Count; i++)
        {
            prompt += "<|image_" + (i + 1) + "|>\n";
        }
    }
    if (audios != null)
    {
        for (int i = 0; i < audioPaths.Count; i++)
        {
            prompt += "<|audio_" + (i + 1) + "|>\n";
        }
    }
    prompt += text + "<|end|>\n<|assistant|>\n";

    Console.WriteLine("Processing inputs...");
    using var inputTensors = processor.ProcessImagesAndAudios(prompt, images, audios);

    Console.WriteLine("Generating response...");
    using GeneratorParams generatorParams = new GeneratorParams(model);
    generatorParams.SetSearchOption("max_length", 7680);
    generatorParams.SetInputs(inputTensors);

    using var generator = new Generator(model, generatorParams);
    var watch = System.Diagnostics.Stopwatch.StartNew();
    while (!generator.IsDone())
    {
        generator.GenerateNextToken();
        Console.Write(tokenizerStream.Decode(generator.GetSequence(0)[^1]));
    }
    watch.Stop();
    var runTimeInSeconds = watch.Elapsed.TotalSeconds;
    Console.WriteLine();
    Console.WriteLine($"Total Time: {runTimeInSeconds:0.00}");

    if (images != null)
    {
        images.Dispose();
    }
    if (audios != null)
    {
        audios.Dispose();
    }
} while (interactive);