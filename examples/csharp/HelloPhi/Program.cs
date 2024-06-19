// See https://aka.ms/new-console-template for more information
using Microsoft.ML.OnnxRuntimeGenAI;

void PrintUsage()
{
    Console.WriteLine("Usage:");
    Console.WriteLine("  -m model_path");
    Console.WriteLine("  -i (optional): Intereactive mode");
}

GeneratorParams ConstructGeneratorParam(Model model, Tokenizer tokenizer, string prompt)
{
    var sequences = tokenizer.Encode($"<|user|>{prompt}<|end|><|assistant|>");
    var generatorParams = new GeneratorParams(model);
    generatorParams.SetSearchOption("max_length", 200);
    generatorParams.SetInputSequences(sequences);
    return generatorParams;
}

OgaHandle ogaHandle = new OgaHandle();

if (args.Length < 1)
{
    PrintUsage();
    Environment.Exit(-1);
}

bool intereactive = false;
string modelPath = string.Empty;

uint i = 0;
while (i < args.Length)
{
    var arg = args[i];
    if (arg == "-i")
    {
        intereactive = true;
    }
    else if (arg == "-m")
    {
        if (i + 1 < args.Length)
        {
            modelPath = Path.Combine(args[i+1]);
        }
    }
    i++;
}

if (string.IsNullOrEmpty(modelPath))
{
    throw new Exception("Model path must be specified");
}

Console.WriteLine("-------------");
Console.WriteLine("Hello, Phi!");
Console.WriteLine("-------------");

Console.WriteLine("Model path: " + modelPath);
Console.WriteLine("Intereactive: " + intereactive);

using Model model = new Model(modelPath);
using Tokenizer tokenizer = new Tokenizer(model);

var option = 2;
if (intereactive)
{
    Console.WriteLine("Please enter option number:");
    Console.WriteLine("1. Complete Output");
    Console.WriteLine("2. Streaming Output");
    int.TryParse(Console.ReadLine(), out option);
}

do
{
    string prompt = "def is_prime(num):"; // Example prompt
    if (intereactive)
    {
        Console.WriteLine("Prompt:");
        prompt = Console.ReadLine();
    }
    if (string.IsNullOrEmpty(prompt))
    {
        continue;
    }
    var generatorParams = ConstructGeneratorParam(model, tokenizer, prompt);
    if (option == 1) // Complete Output
    {
        var outputSequences = model.Generate(generatorParams);
        var outputString = tokenizer.Decode(outputSequences[0]);

        Console.WriteLine("Output:");
        Console.WriteLine(outputString);
    }

    else if (option == 2) //Streaming Output
    {
        using var tokenizerStream = tokenizer.CreateStream();
        using var generator = new Generator(model, generatorParams);
        while (!generator.IsDone())
        {
            generator.ComputeLogits();
            generator.GenerateNextToken();
            Console.Write(tokenizerStream.Decode(generator.GetSequence(0)[^1]));
        }
        Console.WriteLine();
    }
} while (intereactive);