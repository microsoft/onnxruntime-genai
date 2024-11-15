// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntimeGenAI;
using System.Linq;
using System.Runtime.CompilerServices;

void PrintUsage()
{
    Console.WriteLine("Usage:");
    Console.WriteLine("  -m model_path");
    Console.WriteLine("\t\t\t\tPath to the model");
    Console.WriteLine("  --image_paths");
    Console.WriteLine("\t\t\t\tPath to the images");
    Console.WriteLine("  --interactive (optional)");
    Console.WriteLine("\t\t\t\tInteractive mode");
}

using OgaHandle ogaHandle = new OgaHandle();

if (args.Length < 1)
{
    PrintUsage();
    Environment.Exit(-1);
}

bool interactive = false;
string modelPath = string.Empty;
string[] imagePaths = new string[0];

uint i_arg = 0;
while (i_arg < args.Length)
{
    var arg = args[i_arg];
    if (arg == "--interactive")
    {
        interactive = true;
    }
    else if (arg == "-m")
    {
        if (i_arg + 1 < args.Length)
        {
            modelPath = Path.Combine(args[i_arg+1]);
        }
    }
    else if (arg == "--image_paths")
    {
        if (i_arg + 1 < args.Length)
        {
            imagePaths = args[i_arg+1].Split(',').ToList<string>().Select(i => i.ToString().Trim()).ToArray();
        }
    }
    i_arg++;
}

// From https://stackoverflow.com/a/47841442
static string GetThisFilePath([CallerFilePath] string path = null)
{
    return path;
}

Console.WriteLine("--------------------");
Console.WriteLine("Hello, Phi-3-Vision!");
Console.WriteLine("--------------------");

Console.WriteLine("Model path: " + modelPath);
Console.WriteLine("Interactive: " + interactive);

using Model model = new Model(modelPath);
using MultiModalProcessor processor = new MultiModalProcessor(model);
using var tokenizerStream = processor.CreateStream();

do
{
    if (interactive)
    {
        Console.WriteLine("Image Path (comma separated; leave empty if no image):");
        imagePaths = Console.ReadLine().Split(',').ToList<string>().Select(i => i.ToString().Trim()).ToArray();
    }

    if (imagePaths.Length == 0)
    {
        Console.WriteLine("No image provided. Using default image.");
        imagePaths.Append(Path.GetFullPath(Path.Combine(
            GetThisFilePath(), "../../..", "test_models", "images", "australia.jpg")));
    }
    for (int i = 0; i < imagePaths.Length; i++)
    {
        string imagePath = Path.GetFullPath(imagePaths[i].Trim());
        if (!File.Exists(imagePath))
        {
            throw new Exception("Image file not found: " + imagePath);
        }
        Console.WriteLine("Using image: " + imagePath);
    }

    Images images = imagePaths.Length > 0 ? Images.Load(imagePaths) : null;

    string text = "What is shown in this image?";
    if (interactive) {
        Console.WriteLine("Prompt:");
        text = Console.ReadLine();
    }

    string prompt = "<|user|>\n";
    if (images != null)
    {
        for (int i = 0; i < imagePaths.Length; i++)
        {
            prompt += "<|image_" + (i + 1) + "|>\n";
        }
    }
    prompt += text + "<|end|>\n<|assistant|>\n";

    Console.WriteLine("Processing image and prompt...");
    using var inputTensors = processor.ProcessImages(prompt, images);

    Console.WriteLine("Generating response...");
    using GeneratorParams generatorParams = new GeneratorParams(model);
    generatorParams.SetSearchOption("max_length", 7680);
    generatorParams.SetInputs(inputTensors);

    using var generator = new Generator(model, generatorParams);
    while (!generator.IsDone())
    {
        generator.ComputeLogits();
        generator.GenerateNextToken();
        Console.Write(tokenizerStream.Decode(generator.GetSequence(0)[^1]));
    }

    if (images != null)
    {
        images.Dispose();
    }
} while (interactive);