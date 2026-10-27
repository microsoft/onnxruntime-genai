// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.CommandLine;
using System.IO;
using CommonUtils;
using Xunit;

namespace Microsoft.ML.OnnxRuntimeGenAI.Tests
{
    [Collection(nameof(OnnxRuntimeGenAICollection))]
    public class ExampleSearchOptionsTests
    {
        private static readonly string ModelPath = Path.Combine(
            AppContext.BaseDirectory, "models", "hf-internal-testing", "tiny-random-gpt2-fp32");

        [Theory]
        [InlineData(0)]
        [InlineData(-1)]
        [InlineData(int.MinValue)]
        public void RejectNonpositiveLengthBeforeConfig(int value)
        {
            var args = new GeneratorParamsArgs { max_length = value };
            var error = Assert.Throws<ArgumentException>(() => Common.GetConfig("missing-model", "cpu", null, args));
            Assert.Contains("max_length must be greater than 0", error.Message);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(-1)]
        [InlineData(33)]
        [InlineData(int.MaxValue)]
        public void RejectDimensionsBeforeConfig(int value)
        {
            var args = new GeneratorParamsArgs { batch_size = value, num_beams = 2 };
            var error = Assert.Throws<ArgumentException>(() => Common.GetConfig("missing-model", "cpu", null, args));
            Assert.Contains("batch_size", error.Message);
            Assert.Contains("must be in [1, 32]", error.Message);
            args.batch_size = 2;
            args.num_beams = value;
            error = Assert.Throws<ArgumentException>(() => Common.GetConfig("missing-model", "cpu", null, args));
            Assert.Contains("num_beams", error.Message);
            Assert.Contains("must be in [1, 32]", error.Message);
        }

        [Theory]
        [InlineData(1)]
        [InlineData(32)]
        public void AcceptDimensionBoundaries(int value)
        {
            using var config = Common.GetConfig(ModelPath, "cpu", null, new GeneratorParamsArgs
            {
                batch_size = value,
                num_beams = value
            });
        }

        [Theory]
        [InlineData(513)]
        [InlineData(int.MaxValue)]
        public void RejectLengthExceedingContext(int value)
        {
            var error = Assert.Throws<ArgumentException>(() => Common.GetConfig(
                ModelPath, "cpu", null, new GeneratorParamsArgs { max_length = value }));
            Assert.Contains("cannot be greater than model context_length", error.Message);
        }

        [Theory]
        [InlineData("-l", 16)]
        [InlineData("--max_length", 32)]
        [InlineData(null, 512)]
        public void LimitsReachModelAndGeneration(string flag, int limit)
        {
            var parser = new RootCommand();
            Common.GetGeneratorParamsArgs(parser);
            Common.GetGuidanceArgs(parser);
            var parsed = parser.Parse(flag == null ? Array.Empty<string>() : new[] { flag, limit.ToString() });
            Assert.Empty(parsed.Errors);
            var (args, _) = Common.SetGroupedArgs(parsed);
            using var config = Common.GetConfig(ModelPath, "cpu", null, args);
            using var model = new Model(config);
            using var parameters = new GeneratorParams(model);
            // Check the model's overlay before applying generator-time options.
            Assert.Equal((double)limit, parameters.GetSearchNumber("max_length"));
            Common.SetSearchOptions(parameters, args, false);
            parameters.SetSearchOption("min_length", limit);
            using var generator = new Generator(model, parameters);
            generator.AppendTokens(new int[] { 1, 2, 3, 4 });
            while (!generator.IsDone())
                generator.GenerateNextToken();
            Assert.Equal(limit, generator.GetSequence(0).Length);
        }
    }
}
