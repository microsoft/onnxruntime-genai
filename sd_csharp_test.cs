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
/*
namespace Microsoft.ML.OnnxRuntimeGenAI.Tests
{
    public class OnnxRuntimeGenAITests 
    {
        [Fact(DisplayName = "TestStableDiffusion")]
        public void TestStableDiffusion()
        {   
            string modelPath = "C:\\Users\\yangselena\\onnxruntime-genai\\onnxruntime-genai\\test\\test_models\\sd";
            using (var model = new Model(modelPath))
            {
              
                    using ImageGeneratorParams imageGeneratorParams = new ImageGeneratorParams(model);
                    Assert.NotNull(imageGeneratorParams);

                    imageGeneratorParams.SetPrompts("a photo of a cat");

                    var imageTensor = Generator.GenerateImage(model, imageGeneratorParams);

                    Assert.NotNull(imageTensor);
            }
        }
    }
}*/