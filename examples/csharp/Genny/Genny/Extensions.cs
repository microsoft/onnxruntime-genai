using Genny.ViewModel;
using Microsoft.ML.OnnxRuntimeGenAI;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;

namespace Genny
{
    internal static class Extensions
    {

        /// <summary>
        /// Applies the search options to the generator parameters.
        /// </summary>
        /// <param name="generatorParams">The generator parameters.</param>
        /// <param name="searchOptions">The search options.</param>
        internal static void ApplySearchOptions(this GeneratorParams generatorParams, SearchOptionsModel searchOptions)
        {
            generatorParams.SetSearchOption("top_p", searchOptions.TopP);
            generatorParams.SetSearchOption("top_k", searchOptions.TopK);
            generatorParams.SetSearchOption("temperature", searchOptions.Temperature);
            generatorParams.SetSearchOption("repetition_penalty", searchOptions.RepetitionPenalty);
            generatorParams.SetSearchOption("past_present_share_buffer", searchOptions.PastPresentShareBuffer);
            generatorParams.SetSearchOption("num_return_sequences", searchOptions.NumReturnSequences);
            generatorParams.SetSearchOption("no_repeat_ngram_size", searchOptions.NoRepeatNgramSize);
            generatorParams.SetSearchOption("min_length", searchOptions.MinLength);
            generatorParams.SetSearchOption("max_length", searchOptions.MaxLength);
            generatorParams.SetSearchOption("length_penalty", searchOptions.LengthPenalty);
            generatorParams.SetSearchOption("early_stopping", searchOptions.EarlyStopping);
            generatorParams.SetSearchOption("do_sample", searchOptions.DoSample);
            generatorParams.SetSearchOption("diversity_penalty", searchOptions.DiversityPenalty);
        }

        internal static Task<Sequences> EncodeAsync(this Tokenizer tokenizer, string input, CancellationToken cancellationToken = default)
        {
            return Application.Current.Dispatcher.Invoke(() =>
            {
                return Task.Run(() => tokenizer.Encode(input), cancellationToken);
            });
        }

        internal static Task<string> DecodeAsync(this Tokenizer tokenizer, int[] input, CancellationToken cancellationToken = default)
        {
            return Application.Current.Dispatcher.Invoke(() =>
            {
                return Task.Run(() => tokenizer.Decode(input), cancellationToken);
            });
        }
    }
}
