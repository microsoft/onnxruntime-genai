using Genny.Utils;
using Genny.ViewModel;
using Microsoft.ML.OnnxRuntimeGenAI;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace Genny.Views
{
    /// <summary>
    /// Interaction logic for StatelessView.xaml
    /// </summary>
    public partial class StatelessView : UserControl, INotifyPropertyChanged
    {
        private string _prompt;
        private CancellationTokenSource _cancellationTokenSource;

        public StatelessView()
        {
            ClearCommand = new RelayCommand(ClearAsync);
            CancelCommand = new RelayCommand(CancelAsync);
            GenerateCommand = new RelayCommand(GenerateAsync, CanExecuteGenerate);
            ResultHistory = new ObservableCollection<ResultModel>();
            InitializeComponent();
        }

        public static readonly DependencyProperty ModelProperty =
          DependencyProperty.Register(nameof(Model), typeof(Model), typeof(StatelessView));

        public static readonly DependencyProperty TokenizerProperty =
            DependencyProperty.Register(nameof(Tokenizer), typeof(Tokenizer), typeof(StatelessView));

        public static readonly DependencyProperty ModelOptionsProperty =
            DependencyProperty.Register(nameof(ModelOptions), typeof(ModelOptionsModel), typeof(StatelessView));

        public static readonly DependencyProperty SearchOptionsProperty =
            DependencyProperty.Register(nameof(SearchOptions), typeof(SearchOptionsModel), typeof(StatelessView));

        public RelayCommand ClearCommand { get; }
        public RelayCommand CancelCommand { get; }
        public RelayCommand GenerateCommand { get; }
        public ResultModel CurrentResult { get; set; }
        public ObservableCollection<ResultModel> ResultHistory { get; }

        public Model Model
        {
            get { return (Model)GetValue(ModelProperty); }
            set { SetValue(ModelProperty, value); }
        }

        public Tokenizer Tokenizer
        {
            get { return (Tokenizer)GetValue(TokenizerProperty); }
            set { SetValue(TokenizerProperty, value); }
        }

        public ModelOptionsModel ModelOptions
        {
            get { return (ModelOptionsModel)GetValue(ModelOptionsProperty); }
            set { SetValue(ModelOptionsProperty, value); }
        }

        public SearchOptionsModel SearchOptions
        {
            get { return (SearchOptionsModel)GetValue(SearchOptionsProperty); }
            set { SetValue(SearchOptionsProperty, value); }
        }

        public string Prompt
        {
            get { return _prompt; }
            set { _prompt = value; NotifyPropertyChanged(); }
        }


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
                        if (string.IsNullOrWhiteSpace(sentencePiece.Content)) // Ingore preceding '\n'
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
                MessageBox.Show(ex.Message, "Inference Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }


        private bool CanExecuteGenerate()
        {
            return !string.IsNullOrWhiteSpace(Prompt);
        }


        private Task CancelAsync()
        {
            _cancellationTokenSource?.Cancel();
            return Task.CompletedTask;
        }


        private Task ClearAsync()
        {
            ResultHistory.Clear();
            return Task.CompletedTask;
        }

        private async IAsyncEnumerable<TokenModel> RunInferenceAsync(string prompt, [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var sequences = await Tokenizer.EncodeAsync($"<|user|>{prompt}<|end|><|assistant|>", cancellationToken);

            using var generatorParams = new GeneratorParams(Model);
            generatorParams.ApplySearchOptions(SearchOptions);
            generatorParams.SetInputSequences(sequences);

            using var tokenizerStream = Tokenizer.CreateStream();
            using var generator = new Generator(Model, generatorParams);
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

        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;
        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }
        #endregion
    }
}
