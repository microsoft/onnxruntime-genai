// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.IO;
using Xunit;
using Xunit.Abstractions;
using Microsoft.ML.OnnxRuntimeGenAI;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.OnnxRuntimeGenAI.Tests
{
    public partial class GeneratorTests
    {
        private readonly ITestOutputHelper output;

        public GeneratorTests(ITestOutputHelper o)
        {
            this.output = o;
        }

        [Fact(DisplayName = "TestGreedySearch")]
        public void TestGreedySearch()
        {
            ulong maxLength = 10;
            var inputIDs = new List<int> { 0, 0, 0, 52, 0, 0, 195, 731 };
            var inputIDsShape = new ulong[] { 2, 4 };
            ulong batchSize = inputIDsShape[0];
            ulong sequenceLength = inputIDsShape[1];
            var expectedOutput = new int[] { 0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
                                             0, 0, 195, 731, 731, 114, 114, 114, 114, 114 };

            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "testdata", "hf-internal-testing", "tiny-random-gpt2-fp32");
            using (var model = new Model(modelPath, DeviceType.CPU))
            {
                Assert.NotNull(model);
                using (var generatorParams = new GeneratorParams(model))
                {
                    Assert.NotNull(generatorParams);

                    generatorParams.SetMaxLength(maxLength);
                    generatorParams.SetInputIDs(inputIDs, sequenceLength, batchSize);

                    using (var generator = new Generator(model, generatorParams))
                    {
                        Assert.NotNull(generator);

                        while (!generator.IsDone())
                        {
                            generator.ComputeLogits();
                            generator.GenerateNextTokenTop();
                        }

                        for (ulong i = 0; i < batchSize; i++)
                        {
                            var sequence = generator.GetSequence(i).ToArray();
                            var expectedSequence = expectedOutput.Skip((int)i * (int)maxLength).Take((int)maxLength);
                            Assert.Equal(expectedSequence, sequence);
                        }
                    }

                    int[][] sequences = model.Generate(generatorParams).ToArray();
                    Assert.NotNull(sequences);

                    for (ulong i = 0; i < batchSize; i++)
                    {
                        var expectedSequence = expectedOutput.Skip((int)i * (int)maxLength).Take((int)maxLength);
                        Assert.Equal(expectedSequence, sequences[i]);
                    }
                }
            }
        }
    }
}
