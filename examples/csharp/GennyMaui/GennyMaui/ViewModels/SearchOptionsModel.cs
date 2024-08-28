using CommunityToolkit.Mvvm.ComponentModel;
using System.Text.Json.Serialization;

namespace GennyMaui.ViewModels
{
    public partial class SearchOptionsModel : ObservableObject
    {
        [ObservableProperty]
        [JsonPropertyName("top_k")]
        private int _topK = 50;

        [ObservableProperty]
        [JsonPropertyName("top_p")]
        private float _topP = 0.9f;

        [ObservableProperty]
        [JsonPropertyName("temperature")]
        private float _temperature = 1;

        [ObservableProperty]
        [JsonPropertyName("repetition_penalty")]
        private float _repetitionPenalty = 1;

        [ObservableProperty]
        [JsonPropertyName("past_present_share_buffer")]
        private bool _pastPresentShareBuffer = false;

        [ObservableProperty]
        [JsonPropertyName("num_return_sequences")]
        private int _numReturnSequences = 1;

        [ObservableProperty]
        [JsonPropertyName("num_beams")]
        private int _numBeams = 1;

        [ObservableProperty]
        [JsonPropertyName("no_repeat_ngram_size")]
        private int _noRepeatNgramSize = 0;

        [ObservableProperty]
        [JsonPropertyName("min_length")]
        private int _minLength = 0;

        [ObservableProperty]
        [JsonPropertyName("max_length")]
        private int _maxLength = 200;

        [ObservableProperty]
        [JsonPropertyName("length_penalty")]
        private float _lengthPenalty = 1;

        [ObservableProperty]
        [JsonPropertyName("early_stopping")]
        private bool _earlyStopping = true;

        [ObservableProperty]
        [JsonPropertyName("do_sample")]
        private bool _doSample = false;

        [ObservableProperty]
        [JsonPropertyName("diversity_penalty")]
        private float _diversityPenalty = 0;
    }
}