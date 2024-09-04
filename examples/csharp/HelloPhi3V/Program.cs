// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntimeGenAI;
using System.Linq;

class Program
{
    static void Run(string modelPath)
    {
        using Model model = new Model(modelPath);
        using MultiModalProcessor processor = new MultiModalProcessor(model);
        using var tokenizerStream = processor.CreateStream();

        while (true)
        {
            Console.WriteLine("Image Path (comma separated; leave empty if no image):");
            string[] imagePaths = Console.ReadLine().Split(',').ToList<string>().Select(i => i.ToString().Trim()).ToArray();

            Images images = null;
            if (imagePaths.Length == 0)
            {
                Console.WriteLine("No image provided");
            }
            else
            {
                for (int i = 0; i < imagePaths.Length; i++)
                {
                    string imagePath = imagePaths[i].Trim();
                    if (!File.Exists(imagePath))
                    {
                        throw new Exception("Image file not found: " +  imagePath);
                    }
                }
                images = Images.Load(imagePaths);
            }

            Console.WriteLine("Prompt:");
            string text = Console.ReadLine();
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
            var inputTensors = processor.ProcessImages(prompt, images);

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
        }

    }

    static void Main(string[] args)
    {
        Console.WriteLine("--------------------");
        Console.WriteLine("Hello, Phi-3-Vision!");
        Console.WriteLine("--------------------");

        if (args.Length != 1)
        {
            throw new Exception("Usage: .\\HelloPhi3V <model_path>");
        }

        Run(args[0]);
    }
}