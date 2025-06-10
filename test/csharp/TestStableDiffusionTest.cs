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

namespace Microsoft.ML.OnnxRuntimeGenAI.Tests
{
    public class ImageGeneratorParamsTests
    {
        private readonly ITestOutputHelper _output;
        private string _modelPath;

        public ImageGeneratorParamsTests(ITestOutputHelper output)
        {
            _output = output;
            // Set path to your test model directory - adjust as needed
            _modelPath = Path.Combine(
                GetDirectoryInTreeThatContains(Directory.GetCurrentDirectory(), "test"),
                "test", "test_models", "sd");
        }

        private static string GetDirectoryInTreeThatContains(string startDir, string targetDirName)
        {
            var currentDir = new DirectoryInfo(startDir);
            while (currentDir != null)
            {
                if (Directory.Exists(Path.Combine(currentDir.FullName, targetDirName)))
                {
                    return currentDir.FullName;
                }
                currentDir = currentDir.Parent;
            }
            return startDir; // Return original if not found
        }

        [Fact]
        public void TestCreateImageGeneratorParams()
        {
            // Skip the test if model doesn't exist
            /*
            if (!Directory.Exists(_modelPath))
            {
                _output.WriteLine($"Skipping test as model path does not exist: {_modelPath}");
                return;
            }*/

            _output.WriteLine($"Using model path: {_modelPath}");
            
            // Wrap in try-catch to report detailed errors
            try
            {
                using (var model = new Model(_modelPath))
                {
                    Assert.NotNull(model);
                   

                    using (var imageParams = new ImageGeneratorParams(model))
                    {
                        Assert.NotNull(imageParams);
                       
                    }
                }
            }
            catch (Exception ex)
            {
                _output.WriteLine($"Exception creating ImageGeneratorParams: {ex}");
                throw;
            }
        }

        [Fact(Skip = "Enable when model is not available")]
        public void TestSetPrompt()
        {
            _output.WriteLine($"Using model path: {_modelPath}");
            // Skip the test if model doesn't exist
            if (!Directory.Exists(_modelPath))
            {
                _output.WriteLine($"Skipping test as model path does not exist: {_modelPath}");
                return;
            }

            _output.WriteLine($"Using model path: {_modelPath}");
            
            try
            {
                using (var model = new Model(_modelPath))
                using (var imageParams = new ImageGeneratorParams(model))
                {
                    // Test with a simple prompt
                    string prompt = "a photo of a cat";
                    _output.WriteLine($"Setting prompt: '{prompt}'");
                    
                    imageParams.SetPrompts(prompt);
                    _output.WriteLine("SetPrompts completed successfully");
                }
            }
            catch (Exception ex)
            {
                _output.WriteLine($"Exception in SetPrompts: {ex}");
                if (ex is AccessViolationException)
                {
                    _output.WriteLine("ACCESS VIOLATION: This typically indicates a memory corruption issue in the native code");
                    _output.WriteLine("Check that the native library is properly loaded and compatible with this version");
                }
                throw;
            }
        }

        [Fact(Skip = "Enable when model is not available")]
        public void TestDispose()
        {
            if (!Directory.Exists(_modelPath))
            {
                _output.WriteLine($"Skipping test as model path does not exist: {_modelPath}");
                return;
            }

            Model model = null;
            ImageGeneratorParams imageParams = null;
            
            try
            {
                model = new Model(_modelPath);
                imageParams = new ImageGeneratorParams(model);
                
                // Test normal disposal
                imageParams.Dispose();
                
                // Verify handle is zeroed out (you may need to expose a way to check this)
                Assert.Throws<ObjectDisposedException>(() => imageParams.SetPrompts("test after dispose"));
            }
            finally
            {
                imageParams?.Dispose();
                model?.Dispose();
            }
        }
    }
}