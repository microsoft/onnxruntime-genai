// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;
using Microsoft.Extensions.AI;
using System.Reflection;
using Microsoft.ML.OnnxRuntime;

namespace Microsoft.ML.OnnxRuntimeGenAI.Tests
{
    public class OnnxRuntimeGenAITests
    {
        private readonly ITestOutputHelper output;

        private static string GetDirectoryInTreeThatContains(string currentDirectory, string targetDirectoryName)
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

        private static Lazy<string> _lazyPhi2Path = new Lazy<string>(() =>
        {
            string cpuModelPath = Path.Combine(GetDirectoryInTreeThatContains(Directory.GetCurrentDirectory(), "test"),
                                               "models", "phi-2", "int4", "cpu");
            string cudaModelPath = Path.Combine(GetDirectoryInTreeThatContains(Directory.GetCurrentDirectory(), "test"),
                                               "models", "phi-2", "int4", "cuda");
            // Prefer CUDA model if available.
            if (System.IO.Directory.Exists(cudaModelPath))
            {
                return cudaModelPath;
            }

            return cpuModelPath;
        });

        private static string _phi2Path => _lazyPhi2Path.Value;

        private static Lazy<string> _lazyTinyRandomGpt2ModelPath = new Lazy<string>(() =>
        {
            string modelPath = Path.Combine(GetDirectoryInTreeThatContains(Directory.GetCurrentDirectory(), "test"),
                                            "models", "hf-internal-testing", "tiny-random-gpt2-fp32");
            if (System.IO.Directory.Exists(modelPath))
            {
                return modelPath;
            }

            return null;
        });

        private static string _tinyRandomGpt2ModelPath => _lazyTinyRandomGpt2ModelPath.Value;

        private static Lazy<string> _lazyAdaptersPath = new Lazy<string>(() =>
        {
            string modelPath = Path.Combine(GetDirectoryInTreeThatContains(Directory.GetCurrentDirectory(), "test"),
                                            "models", "adapters");
            if (System.IO.Directory.Exists(modelPath))
            {
                return modelPath;
            }

            return null;
        });

        private static string _adaptersPath => _lazyAdaptersPath.Value;
        private static OgaHandle ogaHandle;

        private static int _epLibrariesRegistered = 0;

        // Execution providers that can be loaded as plugin libraries at test time, mapped to the
        // platform-independent stem of their library file. The full file name is built as
        // "<prefix><stem><suffix>", e.g. on Windows "webgpu" -> "onnxruntime_providers_webgpu.dll".
        private static readonly KeyValuePair<string, string>[] _knownEpLibraries = new[]
        {
            new KeyValuePair<string, string>("WebGpuExecutionProvider", "onnxruntime_providers_webgpu"),
        };

        // Resolves the directory containing the execution provider plugin libraries to register.
        // This is the EPDir MSBuild property (set via /p:EPDir), surfaced as assembly metadata. When
        // it is not set, no EP-plugins are registered.
        private static string GetEpDirectory()
        {
            return typeof(OnnxRuntimeGenAITests).Assembly
                .GetCustomAttributes<AssemblyMetadataAttribute>()
                .FirstOrDefault(a => a.Key == "EPDir")?.Value;
        }

        // ONNX Runtime's environment is a process-wide singleton, so registering an execution
        // provider plugin library with it, also makes the provider available to ONNX Runtime GenAI.
        private static void RegisterEpLibrariesFromDirectory()
        {
            if (System.Threading.Interlocked.Exchange(ref _epLibrariesRegistered, 1) != 0)
            {
                return;
            }

            string epDir = GetEpDirectory();
            if (string.IsNullOrEmpty(epDir) || !Directory.Exists(epDir))
            {
                return;
            }

            bool isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
            bool isMac = RuntimeInformation.IsOSPlatform(OSPlatform.OSX);
            string prefix = isWindows ? "" : "lib";
            string suffix = isWindows ? ".dll" : (isMac ? ".dylib" : ".so");

            foreach (KeyValuePair<string, string> ep in _knownEpLibraries)
            {
                string epName = ep.Key;
                string path = Path.Combine(epDir, prefix + ep.Value + suffix);
                if (!File.Exists(path))
                {
                    continue;
                }

                try
                {
                    OrtEnv.Instance().RegisterExecutionProviderLibrary(epName, path);
                    Console.WriteLine($"**** Registered execution provider library '{epName}' -> {path}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"**** Warning: failed to register execution provider library '{epName}': {ex.Message}");
                }
            }
        }

        public OnnxRuntimeGenAITests(ITestOutputHelper o)
        {
            Console.WriteLine("**** Running OnnxRuntimeGenAITests constructor");
            // Initialize GenAI and register a handler to dispose it on process exit
            ogaHandle = new OgaHandle();
            AppDomain.CurrentDomain.ProcessExit += (sender, e) => ogaHandle.Dispose();
            RegisterEpLibrariesFromDirectory();
            this.output = o;
            Console.WriteLine("**** OnnxRuntimeGenAI constructor completed");
        }

        private class IgnoreOnModelAbsenceFact : FactAttribute
        {
            public IgnoreOnModelAbsenceFact()
            {
                string modelPath = _phi2Path;
                bool exists = System.IO.Directory.Exists(modelPath);
                if (!System.IO.Directory.Exists(modelPath))
                {
                    // Skip this test on some machines since the model cannot be downloaded on those machines at runtime.
                    Skip = "Skipping this test since the model does not exist.";
                }
            }
        }

        [Fact(DisplayName = "TestConfig")]
        public void TestConfig()
        {
            string modelPath = _tinyRandomGpt2ModelPath;
            using (var config = new Config(modelPath))
            {
                config.ClearProviders();
                config.SetProviderOption("cuda", "device_id", "0");
                config.SetProviderOption("cuda", "catch_fire", "false");
                config.AppendProvider("pigeon");
                // At this point the providers is 'cuda' first and 'pigeon' as secondary.
                // Given some provider options are made up and there is no pigeon provider, the model won't load.
                // This tests the API
            }
        }

        [Fact(DisplayName = "TestGreedySearch")]
        public void TestGreedySearch()
        {
            ulong maxLength = 10;
            int[] inputIDs = new int[] { 0, 0, 0, 52, 0, 0, 195, 731 };
            var inputIDsShape = new ulong[] { 2, 4 };
            ulong batchSize = inputIDsShape[0];
            var expectedOutput = new int[] { 0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
                                             0, 0, 195, 731, 731, 114, 114, 114, 114, 114 };

            string modelPath = _tinyRandomGpt2ModelPath;
            using (var config = new Config(modelPath))
            {
                Assert.NotNull(config);
                using (var model = new Model(config))
                {
                    Assert.NotNull(model);
                    using (var generatorParams = new GeneratorParams(model))
                    {
                        Assert.NotNull(generatorParams);

                        generatorParams.SetSearchOption("max_length", maxLength);
                        generatorParams.SetSearchOption("batch_size", batchSize);
                        generatorParams.SetSpeculativeNumber("max_draft_tokens", 4);

                        using (var generator = new Generator(model, generatorParams))
                        {
                            Assert.NotNull(generator);
                            generator.AppendTokens(inputIDs);

                            Assert.False(generator.IsDone());
                            Assert.Equal(maxLength, generatorParams.GetSearchNumber("max_length"));
                            Assert.True(generatorParams.GetSearchBool("early_stopping"));
                            Assert.Equal(4.0, generatorParams.GetSpeculativeNumber("max_draft_tokens"));
                            using (SpeculativeStats stats = generator.GetSpeculativeStats())
                            {
                                Assert.Equal(0UL, stats.GetCount("rounds"));
                                Assert.False(stats.GetBool("formula_supported"));
                                Assert.Equal(0.0, stats.GetNumber("acceptance_rate"));
                            }
                            Assert.Equal(generator.GetSequence(0).Length, (int)generator.TokenCount());

                            while (!generator.IsDone())
                            {
                                generator.GenerateNextToken();
                            }

                            for (ulong i = 0; i < batchSize; i++)
                            {
                                var sequence = generator.GetSequence(i).ToArray();
                                var expectedSequence = expectedOutput.Skip((int)i * (int)maxLength).Take((int)maxLength);
                                Assert.Equal(expectedSequence, sequence);
                                Assert.Equal(generator.GetSequence(0).Length, (int)generator.TokenCount());
                            }
                        }
                    }
                }
            }
        }

        [Fact(DisplayName = "TestLoadModelFromMemory")]
        public void TestLoadModelFromMemory()
        {
            ulong maxLength = 10;
            int[] inputIDs = new int[] { 0, 0, 0, 52, 0, 0, 195, 731 };
            var inputIDsShape = new ulong[] { 2, 4 };
            ulong batchSize = inputIDsShape[0];
            var expectedOutput = new int[] { 0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
                                             0, 0, 195, 731, 731, 114, 114, 114, 114, 114 };

            string modelPath = _tinyRandomGpt2ModelPath;
            using (var config = new Config(modelPath))
            {
                Assert.NotNull(config);
                var modelData = File.ReadAllBytes(Path.Combine(modelPath, "past.onnx"));
                Assert.NotNull(modelData);
                config.AddModelData("past.onnx", modelData);
                using (var model = new Model(config))
                {
                    config.RemoveModelData("past.onnx");
                    Assert.NotNull(model);
                    using (var generatorParams = new GeneratorParams(model))
                    {
                        Assert.NotNull(generatorParams);

                        generatorParams.SetSearchOption("max_length", maxLength);
                        generatorParams.SetSearchOption("batch_size", batchSize);

                        using (var generator = new Generator(model, generatorParams))
                        {
                            Assert.NotNull(generator);

                            generator.AppendTokens(inputIDs);
                            Assert.False(generator.IsDone());
                            while (!generator.IsDone())
                            {
                                generator.GenerateNextToken();
                            }

                            for (ulong i = 0; i < batchSize; i++)
                            {
                                var sequence = generator.GetSequence(i).ToArray();
                                var expectedSequence = expectedOutput.Skip((int)i * (int)maxLength).Take((int)maxLength);
                                Assert.Equal(expectedSequence, sequence);
                            }
                        }
                    }
                }
            }
        }

        [Fact(DisplayName = "TestAudioOpenBytes")]
        public void TestAudioOpenBytes()
        {
            byte[] audioBytes = File.ReadAllBytes(Path.Combine(GetDirectoryInTreeThatContains(Directory.GetCurrentDirectory(), "test"),
                                                            "audios", "1272-141231-0002.mp3"));
            var audios = Audios.Load(audioBytes);
            Assert.NotNull(audios);
        }

        [Fact(DisplayName = "TestImageOpenBytes")]
        public void TestImageOpenBytes()
        {
            byte[] imageBytes = File.ReadAllBytes(Path.Combine(GetDirectoryInTreeThatContains(Directory.GetCurrentDirectory(), "test"),
                                                            "images", "10809054.jpg"));
            var images = Images.Load(imageBytes);
            Assert.NotNull(images);
        }

        [IgnoreOnModelAbsenceFact(DisplayName = "TestTopKSearch")]
        public void TestTopKSearch()
        {
            int topK = 100;
            float temp = 0.6f;
            ulong maxLength = 20;

            string modelPath = _phi2Path;
            using (var model = new Model(modelPath))
            {
                Assert.NotNull(model);
                using (var tokenizer = new Tokenizer(model))
                {
                    Assert.NotNull(tokenizer);

                    var strings = new string[] {
                        "This is a test.",
                        "Rats are awesome pets!",
                        "The quick brown fox jumps over the lazy dog."
                    };
                    ulong batchSize = (ulong)strings.Length;

                    var sequences = tokenizer.EncodeBatch(strings);
                    Assert.NotNull(sequences);
                    Assert.Equal((ulong)strings.Length, sequences.NumSequences);

                    using GeneratorParams generatorParams = new GeneratorParams(model);
                    Assert.NotNull(generatorParams);

                    generatorParams.SetSearchOption("max_length", maxLength);
                    generatorParams.SetSearchOption("batch_size", batchSize);
                    generatorParams.SetSearchOption("do_sample", true);
                    generatorParams.SetSearchOption("top_k", topK);
                    generatorParams.SetSearchOption("temperature", temp);

                    using (var generator = new Generator(model, generatorParams))
                    {
                        Assert.NotNull(generator);

                        generator.AppendTokenSequences(sequences);

                        while (!generator.IsDone())
                        {
                            generator.GenerateNextToken();
                        }

                        for (ulong i = 0; i < batchSize; i++)
                        {
                            var sequence = generator.GetSequence(i).ToArray();
                            Assert.NotNull(sequence);

                            var outputString = tokenizer.Decode(sequence);
                            Assert.NotNull(outputString);
                        }
                    }
                }
            }
        }

        [IgnoreOnModelAbsenceFact(DisplayName = "TestTopPSearch")]
        public void TestTopPSearch()
        {
            float topP = 0.6f;
            float temp = 0.6f;
            ulong maxLength = 20;

            string modelPath = _phi2Path;
            using (var model = new Model(modelPath))
            {
                Assert.NotNull(model);
                using (var tokenizer = new Tokenizer(model))
                {
                    Assert.NotNull(tokenizer);

                    var strings = new string[] {
                        "This is a test.",
                        "Rats are awesome pets!",
                        "The quick brown fox jumps over the lazy dog."
                    };
                    ulong batchSize = (ulong)strings.Length;

                    var sequences = tokenizer.EncodeBatch(strings);
                    Assert.NotNull(sequences);
                    Assert.Equal((ulong)strings.Length, sequences.NumSequences);

                    using GeneratorParams generatorParams = new GeneratorParams(model);
                    Assert.NotNull(generatorParams);

                    generatorParams.SetSearchOption("max_length", maxLength);
                    generatorParams.SetSearchOption("batch_size", batchSize);
                    generatorParams.SetSearchOption("do_sample", true);
                    generatorParams.SetSearchOption("top_p", topP);
                    generatorParams.SetSearchOption("temperature", temp);

                    using (var generator = new Generator(model, generatorParams))
                    {
                        Assert.NotNull(generator);

                        generator.AppendTokenSequences(sequences);

                        while (!generator.IsDone())
                        {
                            generator.GenerateNextToken();
                        }

                        for (ulong i = 0; i < batchSize; i++)
                        {
                            var sequence = generator.GetSequence(i).ToArray();
                            Assert.NotNull(sequence);

                            var outputString = tokenizer.Decode(sequence);
                            Assert.NotNull(outputString);
                        }
                    }
                }
            }
        }

        [IgnoreOnModelAbsenceFact(DisplayName = "TestTopKTopPSearch")]
        public void TestTopKTopPSearch()
        {
            int topK = 100;
            float topP = 0.6f;
            float temp = 0.6f;
            ulong maxLength = 20;

            string modelPath = _phi2Path;
            using (var model = new Model(modelPath))
            {
                Assert.NotNull(model);
                using (var tokenizer = new Tokenizer(model))
                {
                    Assert.NotNull(tokenizer);

                    var strings = new string[] {
                        "This is a test.",
                        "Rats are awesome pets!",
                        "The quick brown fox jumps over the lazy dog."
                    };
                    ulong batchSize = (ulong)strings.Length;

                    var sequences = tokenizer.EncodeBatch(strings);
                    Assert.NotNull(sequences);
                    Assert.Equal((ulong)strings.Length, sequences.NumSequences);

                    using GeneratorParams generatorParams = new GeneratorParams(model);
                    Assert.NotNull(generatorParams);

                    generatorParams.SetSearchOption("max_length", maxLength);
                    generatorParams.SetSearchOption("batch_size", batchSize);
                    generatorParams.SetSearchOption("do_sample", true);
                    generatorParams.SetSearchOption("top_k", topK);
                    generatorParams.SetSearchOption("top_p", topP);
                    generatorParams.SetSearchOption("temperature", temp);

                    using (var generator = new Generator(model, generatorParams))
                    {
                        Assert.NotNull(generator);

                        generator.AppendTokenSequences(sequences);

                        while (!generator.IsDone())
                        {
                            generator.GenerateNextToken();
                        }

                        for (ulong i = 0; i < batchSize; i++)
                        {
                            var sequence = generator.GetSequence(i).ToArray();
                            Assert.NotNull(sequence);

                            var outputString = tokenizer.Decode(sequence);
                            Assert.NotNull(outputString);
                        }
                    }
                }
            }
        }

        [IgnoreOnModelAbsenceFact(DisplayName = "TestChatClient")]
        public async Task TestChatClient()
        {
            OnnxRuntimeGenAIChatClientOptions options = new()
            {
                StopSequences = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"],
                PromptFormatter = static (messages, options) =>
                {
                    StringBuilder prompt = new();

                    foreach (var message in messages)
                        foreach (var content in message.Contents.OfType<TextContent>())
                            prompt.Append("<|").Append(message.Role.Value).Append("|>\n").Append(content.Text).Append("<|end|>\n");

                    return prompt.Append("<|assistant|>\n").ToString();
                },
            };

            using var client = new OnnxRuntimeGenAIChatClient(_phi2Path, options);

            var completion = await client.GetResponseAsync("The quick brown fox jumps over the lazy dog.", new()
            {
                MaxOutputTokens = 10,
                Temperature = 0f,
                StopSequences = ["."],
            });

            Assert.NotEmpty(completion.Text);
        }

        [IgnoreOnModelAbsenceFact(DisplayName = "TestTokenizerBatchEncodeDecode")]
        public void TestTokenizerBatchEncodeDecode()
        {
            string modelPath = _phi2Path;
            using (var model = new Model(modelPath))
            {
                Assert.NotNull(model);
                using (var tokenizer = new Tokenizer(model))
                {
                    Assert.NotNull(tokenizer);

                    var strings = new string[] {
                        "This is a test.",
                        "Rats are awesome pets!",
                        "The quick brown fox jumps over the lazy dog."
                    };

                    var sequences = tokenizer.EncodeBatch(strings);

                    Assert.NotNull(sequences);
                    Assert.Equal((ulong)strings.Length, sequences.NumSequences);

                    string[] decodedStrings = tokenizer.DecodeBatch(sequences);
                    Assert.NotNull(decodedStrings);
                    Assert.Equal(strings, decodedStrings);
                }
            }
        }

        [IgnoreOnModelAbsenceFact(DisplayName = "TestTokenizerBatchEncodeSingleDecode")]
        public void TestTokenizerBatchEncodeSingleDecode()
        {
            string modelPath = _phi2Path;
            using (var model = new Model(modelPath))
            {
                Assert.NotNull(model);
                using (var tokenizer = new Tokenizer(model))
                {
                    Assert.NotNull(tokenizer);

                    var strings = new string[] {
                        "This is a test.",
                        "Rats are awesome pets!",
                        "The quick brown fox jumps over the lazy dog."
                    };

                    var sequences = tokenizer.EncodeBatch(strings);

                    Assert.NotNull(sequences);
                    Assert.Equal((ulong)strings.Length, sequences.NumSequences);

                    for (ulong i = 0; i < sequences.NumSequences; i++)
                    {
                        var decodedString = tokenizer.Decode(sequences[i]);
                        Assert.Equal(strings[i], decodedString);
                    }
                }
            }
        }

        [IgnoreOnModelAbsenceFact(DisplayName = "TestTokenizerBatchEncodeStreamDecode")]
        public void TestTokenizerBatchEncodeStreamDecode()
        {
            string modelPath = _phi2Path;
            using (var model = new Model(modelPath))
            {
                Assert.NotNull(model);
                using (var tokenizer = new Tokenizer(model))
                {
                    Assert.NotNull(tokenizer);
                    var tokenizerStream = tokenizer.CreateStream();

                    var strings = new string[] {
                        "This is a test.",
                        "Rats are awesome pets!",
                        "The quick brown fox jumps over the lazy dog."
                    };

                    var sequences = tokenizer.EncodeBatch(strings);

                    Assert.NotNull(sequences);
                    Assert.Equal((ulong)strings.Length, sequences.NumSequences);

                    for (ulong i = 0; i < sequences.NumSequences; i++)
                    {
                        string decodedString = "";
                        for (int j = 0; j < sequences[i].Length; j++)
                        {
                            decodedString += tokenizerStream.Decode(sequences[i][j]);
                        }
                        Assert.Equal(strings[i], decodedString);
                    }
                }
            }
        }

        [IgnoreOnModelAbsenceFact(DisplayName = "TestTokenizerSingleEncodeDecode")]
        public void TestTokenizerSingleEncodeDecode()
        {
            string modelPath = _phi2Path;
            using (var model = new Model(modelPath))
            {
                Assert.NotNull(model);
                using (var tokenizer = new Tokenizer(model))
                {
                    Assert.NotNull(tokenizer);
                    var tokenizerStream = tokenizer.CreateStream();

                    var str = "She sells sea shells by the sea shore.";

                    var sequences = tokenizer.Encode(str);

                    Assert.NotNull(sequences);

                    string decodedString = tokenizer.Decode(sequences[0]);
                    Assert.Equal(str, decodedString);
                }
            }
        }

        [IgnoreOnModelAbsenceFact(DisplayName = "TestPhi2")]
        public void TestPhi2()
        {
            string modelPath = _phi2Path;
            using (var model = new Model(modelPath))
            {
                Assert.NotNull(model);
                using (var tokenizer = new Tokenizer(model))
                {
                    Assert.NotNull(tokenizer);

                    Assert.Equal(50256, tokenizer.GetBosTokenId());
                    Assert.Equal(50256, tokenizer.GetPadTokenId());
                    Assert.Equal(1, tokenizer.GetEosTokenIds().Length);
                    Assert.Equal(50256, tokenizer.GetEosTokenIds()[0]);

                    var strings = new string[] {
                        "This is a test.",
                        "Rats are awesome pets!",
                        "The quick brown fox jumps over the lazy dog."
                    };
                    var batchSize = (ulong)strings.Length;

                    var sequences = tokenizer.EncodeBatch(strings);
                    Assert.NotNull(sequences);
                    Assert.Equal((ulong)strings.Length, sequences.NumSequences);

                    using GeneratorParams generatorParams = new GeneratorParams(model);
                    Assert.NotNull(generatorParams);

                    generatorParams.SetSearchOption("max_length", 20);
                    generatorParams.SetSearchOption("batch_size", batchSize);

                    using (var generator = new Generator(model, generatorParams))
                    {
                        Assert.NotNull(generator);

                        generator.AppendTokenSequences(sequences);

                        while (!generator.IsDone())
                        {
                            generator.GenerateNextToken();
                        }

                        for (ulong i = 0; i < batchSize; i++)
                        {
                            var sequence = generator.GetSequence(i).ToArray();
                            Assert.NotNull(sequence);

                            var outputString = tokenizer.Decode(sequence);
                            Assert.NotNull(outputString);
                        }
                    }
                }
            }
        }

        [Fact(DisplayName = "TestTensorAndAddExtraInput")]
        public void TestTensorAndAddExtraInput()
        {
            string modelPath = _tinyRandomGpt2ModelPath;
            using var model = new Model(modelPath);
            Assert.NotNull(model);

            using var generatorParams = new GeneratorParams(model);
            Assert.NotNull(generatorParams);

            float[] data = { 0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24 };
            long[] shape = { 3, 5 };

            // Pin the array to get its pointer
            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                IntPtr data_pointer = handle.AddrOfPinnedObject();

                using var tensor = new Tensor(data_pointer, shape, ElementType.float32);
                Assert.NotNull(tensor);

                Assert.Equal(shape, tensor.Shape());
                Assert.Equal(ElementType.float32, tensor.Type());

                using var generator = new Generator(model, generatorParams);
                Assert.NotNull(generator);
                generator.SetModelInput("test_input", tensor);
            }
            finally
            {
                handle.Free();
            }
        }

        private class IgnoreOnAdaptersAbsentFact : FactAttribute
        {
            public IgnoreOnAdaptersAbsentFact()
            {
                string modelPath = _adaptersPath;
                bool exists = System.IO.Directory.Exists(modelPath);
                if (!System.IO.Directory.Exists(modelPath))
                {
                    // Skip this test on some machines since the model cannot be downloaded on those machines at runtime.
                    Skip = "Skipping this test since the model does not exist.";
                }
            }
        }

        // This model is dependent on the presense of Phi2 model
        // get this model generated and copied to the output
        // by running test_onnxruntime_genai.py
        [IgnoreOnAdaptersAbsentFact(DisplayName = "TestAdapters")]
        public void TestAdapters()
        {
            Console.WriteLine("**** Running TestAdapters");

            string modelPath = _adaptersPath;
            string adapterPath = Path.Combine(modelPath, "adapters.onnx_adapter");

            using var model = new Model(modelPath);
            Assert.NotNull(model);

            using var adapters = new Adapters(model);
            adapters.LoadAdapter(adapterPath, "adapters_a_and_b");

            var inputStrings = new string[]
            {
                "This is a test.",
                "Rats are awesome pets!",
                "The quick brown fox jumps over the lazy dog.",
            };

            using var tokenizer = new Tokenizer(model);
            using var sequences = tokenizer.EncodeBatch(inputStrings);

            Int64 outputSize = 0;
            Int64[] output_shape;
            float[] base_output = [];

            // Run base scenario
            {
                using var genParams = new GeneratorParams(model);
                genParams.SetSearchOption("max_length", 20);
                genParams.SetSearchOption("batch_size", 3);

                using var generator = new Generator(model, genParams);
                generator.AppendTokenSequences(sequences);
                while (!generator.IsDone())
                {
                    generator.GenerateNextToken();
                }

                using var logits = generator.GetOutput("logits");
                // The adapters model's precision depends on how it was generated (e.g. fp32 for CPU,
                // fp16 for CUDA/WebGPU), so branch on the actual logits type rather than assuming one.
                if (logits.Type() == ElementType.float32)
                {
                    base_output = logits.GetData<float>().ToArray();
                }
                // TODO: GetData with float16 to enable the base/adapter comparison for fp16 models.
                output_shape = logits.Shape();
                outputSize = logits.NumElements();
            }
            // Adapter scenario. The output must be affected
            {
                using var genParams = new GeneratorParams(model);
                genParams.SetSearchOption("max_length", 20);
                genParams.SetSearchOption("batch_size", 3);

                using var generator = new Generator(model, genParams);
                generator.SetActiveAdapter(adapters, "adapters_a_and_b");
                generator.AppendTokenSequences(sequences);
                while (!generator.IsDone())
                {
                    generator.GenerateNextToken();
                }
                using var logits = generator.GetOutput("logits");
                if (logits.Type() == ElementType.float32)
                {
                    var adapter_output = logits.GetData<float>().ToArray();
                    Assert.NotEqual(base_output, adapter_output);
                }
                // TODO: GetData with float16 to enable the base/adapter comparison for fp16 models.
                Assert.Equal(outputSize, logits.NumElements());
                Assert.Equal(output_shape, logits.Shape());
            }
        }
    }
}
