using GennyMaui.Models;
using Microsoft.ML.OnnxRuntimeGenAI;


namespace GennyMaui.Services
{
    public interface IModelProvider
    {
        Model Model { get; }

        Tokenizer Tokenizer { get; }

        ConfigurationModel Configuration { get; }
    }
}
