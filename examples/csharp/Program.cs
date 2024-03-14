// See https://aka.ms/new-console-template for more information
using Microsoft.ML.OnnxRuntimeGenAI;

Console.WriteLine("-------------");
Console.WriteLine("Hello, Phi-2!");
Console.WriteLine("-------------");

string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "phi-2");
using Model model = new Model(modelPath);
using Tokenizer tokenizer = new Tokenizer(model);

while (true)
{
    Console.WriteLine("Prompt:");
    // Example prompt:
    // "def is_prime(num):"
    string prompt = Console.ReadLine();
    var sequences = tokenizer.Encode(prompt);

    using GeneratorParams generatorParams = new GeneratorParams(model);
    generatorParams.SetSearchOption("max_length", 200);
    generatorParams.SetInputSequences(sequences);

    var outputSequences = model.Generate(generatorParams);
    var outputString = tokenizer.Decode(outputSequences[0]);

    Console.WriteLine("Output:");
    Console.WriteLine(outputString);
}
