using Microsoft.ML.OnnxRuntimeGenAI;
using CommunityToolkit.Mvvm.ComponentModel;
using GennyMaui.Models;
using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.Input;
using GennyMaui.Utils;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using CommunityToolkit.Mvvm.Messaging;
using CommunityToolkit.Mvvm.Messaging.Messages;
using GennyMaui.Services;

namespace GennyMaui.ViewModels
{
    public partial class ChatViewModel : ObservableRecipient
    {
        private readonly List<int> _pastTokens = [];

        private CancellationTokenSource _cancellationTokenSource;

        private Model _model;

        private Tokenizer _tokenizer;

        private SearchOptionsModel _searchOptions;

        private ModelOptionsModel _modelOptions;

        [ObservableProperty]
        private string _prompt;

        public ResultModel CurrentResult { get; set; }

        public ObservableCollection<ResultModel> ResultHistory { get; } = new ObservableCollection<ResultModel>();

        public ChatViewModel()
        {
            var modelProvider = MauiProgram.GetService<IModelProvider>();
            _model = modelProvider.Model;
            _tokenizer = modelProvider.Tokenizer;
            _searchOptions = modelProvider.Configuration?.SearchOptions;
            _modelOptions = modelProvider.Configuration?.ModelOptions;

            WeakReferenceMessenger.Default.Register<PropertyChangedMessage<Model>>(this, (r, m) =>
            {
                _model = m.NewValue;
                GenerateCommand.NotifyCanExecuteChanged();
            });

            WeakReferenceMessenger.Default.Register<PropertyChangedMessage<Tokenizer>>(this, (r, m) =>
            {
                _tokenizer = m.NewValue;
                GenerateCommand.NotifyCanExecuteChanged();
            });

            WeakReferenceMessenger.Default.Register<PropertyChangedMessage<SearchOptionsModel>>(this, (r, m) =>
            {
                _searchOptions = m.NewValue;
            });

            WeakReferenceMessenger.Default.Register<PropertyChangedMessage<ConfigurationModel>>(this, (r, m) =>
            {
                _modelOptions = m.NewValue.ModelOptions;
            });
        }

        [RelayCommand(CanExecute = "CanExecuteGenerate")]
        private async Task GenerateAsync()
        {
            try
            {
                var userInput = new ResultModel
                {
                    Content = Prompt,
                    IsUserInput = true
                };

                Prompt = null;
                CurrentResult = null;
                ResultHistory.Add(userInput);
                _cancellationTokenSource = new CancellationTokenSource();
                await foreach (var sentencePiece in RunInferenceAsync(userInput.Content, _cancellationTokenSource.Token))
                {
                    if (CurrentResult == null)
                    {
                        if (string.IsNullOrWhiteSpace(sentencePiece.Content)) // Ignore preceding '\n'
                            continue;

                        ResultHistory.Add(CurrentResult = new ResultModel());
                    }
                    CurrentResult.Content += sentencePiece.Content;
                }
            }
            catch (OperationCanceledException)
            {
                CurrentResult.Content += "\n\n[Operation Canceled]";
            }
            catch (Exception ex)
            {
                await Application.Current.MainPage.DisplayAlert("Inference Error", ex.Message, "OK");
            }
        }

        [RelayCommand]
        private Task ClearAsync()
        {
            _pastTokens.Clear();
            ResultHistory.Clear();
            return Task.CompletedTask;
        }

        private bool CanExecuteGenerate()
        {
            return _model != null && _tokenizer != null && !string.IsNullOrWhiteSpace(Prompt);
        }

        [RelayCommand]
        private Task CancelAsync()
        {
            _cancellationTokenSource?.Cancel();
            return Task.CompletedTask;
        }

        private async IAsyncEnumerable<TokenModel> RunInferenceAsync(string prompt, [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var sequences = await _tokenizer.EncodeAsync($"<|user|>{prompt}<|end|><|assistant|>", cancellationToken);

            // Add Tokens to history
            AddPastTokens(sequences);

            using var generatorParams = new GeneratorParams(_model);
            generatorParams.ApplySearchOptions(_searchOptions);

            // max_length is per message, so increment max_length for next call
            var newMaxLength = Math.Min(_pastTokens.Count + _searchOptions.MaxLength, _modelOptions.ContextLength);
            generatorParams.SetSearchOption("max_length", newMaxLength);

            generatorParams.SetInputIDs(CollectionsMarshal.AsSpan(_pastTokens), (ulong)_pastTokens.Count, 1);

            using var tokenizerStream = _tokenizer.CreateStream();
            using var generator = new Generator(_model, generatorParams);
            while (!generator.IsDone())
            {
                cancellationToken.ThrowIfCancellationRequested();

                yield return await Task.Run(() =>
                {
                    generator.ComputeLogits();
                    generator.GenerateNextToken();

                    var tokenId = generator.GetSequence(0)[^1];
                    return new TokenModel(tokenId, tokenizerStream.Decode(tokenId));
                }, cancellationToken);
            }
        }

        private void AddPastTokens(Sequences sequences)
        {
            _pastTokens.AddRange(sequences[0].ToArray());

            // Only keep (context_length - max_length) worth of history
            while (_pastTokens.Count > _modelOptions.ContextLength - _searchOptions.MaxLength)
            {
                _pastTokens.RemoveAt(0);
            }
        }
    }
}
