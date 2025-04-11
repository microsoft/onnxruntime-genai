## About

Run Llama, Phi (language and multi modal!), DeepSeek, Gemma, Mistral with ONNX Runtime.

This API gives you an easy, flexible and performant way of running LLMs on device using .NET/C#. 

It implements the generative AI loop for ONNX models, including pre and post processing, inference with ONNX Runtime, logits processing, search and sampling, and KV cache management.

## Key Features

* Language, vision, and audio pre and post processing
* Inference using ONNX Runtime
* Generation tuning with greedy, beam search and random sampling
* KV cache management to optimize performance
* Multi target execution (CPU, GPU, with NPU coming!)

## Sample

```csharp
// See https://aka.ms/new-console-template for more information
using Microsoft.ML.OnnxRuntimeGenAI;

using OgaHandle ogaHandle = new OgaHandle();

// Specify the location of your downloaded model.
// Many models are published on HuggingFace e.g. 
// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
string modelPath = "..."
Console.WriteLine("Model path: " + modelPath);

using Model model = new Model(modelPath);
using Tokenizer tokenizer = new Tokenizer(model);
using var tokenizerStream = tokenizer.CreateStream();

// Set your prompt here
string prompt = "public static bool IsPrime(int number)";
var sequences = tokenizer.Encode($"<|user|>{prompt}<|end|><|assistant|>");

using GeneratorParams generatorParams = new GeneratorParams(model);
generatorParams.SetSearchOption("max_length", 512);

using var generator = new Generator(model, generatorParams);
generator.AppendTokenSequences(sequences);
.
while (!generator.IsDone())
{
    generator.GenerateNextToken();
    Console.Write(tokenizerStream.Decode(generator.GetSequence(0)[^1]));
}
```

Generates the following output:


```
Here's a complete implementation of the `IsPrime` function in C# that checks if a given number is prime. The function includes basic input validation and comments for clarity.
```

```csharp
using System;

namespace PrimeChecker
{
    public class PrimeChecker
    {
        /// <summary>
        /// Checks if the given number is prime.
        /// </summary>
        /// <param name="number">The number to check.</param>
        /// <returns>true if the number is prime; otherwise, false.</returns>
        public static bool IsPrime(int number)
        {
            // Input validation
            if (number < 2)
            {
                return false;
            }

            // 2 is the only even prime number
            if (number == 2)
            {
                return true;
            }

            // Exclude even numbers greater than 2
            if (number % 2 == 0)
            {
                return false;
            }

            // Check for factors up to the square root of the number
            int limit = (int)Math.Floor(Math.Sqrt(number));
            for (int i = 3; i <= limit; i += 2)
            {
                if (number % i == 0)
                {
                    return false;
                }
            }

            return true;
        }

        static void Main(string[] args)
        {
            int number = 29;
            bool isPrime = PrimeChecker.IsPrime(number);

            Console.WriteLine($"Is {number} prime? {isPrime}");
        }
    }
}
```

```
This implementation checks if a number is prime by iterating only up to the square root of the number, which is an optimization over checking all numbers up to the number itself. It also excludes even numbers greater than 2, as they cannot be prime.
```

## Source code repository

ONNX Runtime is an open source project. See:
* [ONNX Runtime](https://github.com/microsoft/onnxruntime)
* [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai)

## Documentation

See [ONNX Runtime GenAI Documentation](https://onnxruntime.ai/docs/genai)


