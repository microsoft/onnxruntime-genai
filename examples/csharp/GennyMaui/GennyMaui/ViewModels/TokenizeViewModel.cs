using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Messaging.Messages;
using CommunityToolkit.Mvvm.Messaging;
using Microsoft.ML.OnnxRuntimeGenAI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.Input;
using GennyMaui.Utils;

namespace GennyMaui.ViewModels
{
    public partial class TokenizeViewModel : ObservableRecipient
    {
        private Tokenizer _tokenizer;

        [ObservableProperty]
        private string _encodeResult = string.Empty;

        [ObservableProperty]
        private string _decodeResult = string.Empty;

        public TokenizeViewModel()
        {
            WeakReferenceMessenger.Default.Register<PropertyChangedMessage<Tokenizer>>(this, (r, m) =>
            {
                _tokenizer = m.NewValue;
                EncodeCommand.NotifyCanExecuteChanged();
                DecodeCommand.NotifyCanExecuteChanged();
            });
        }

        [RelayCommand(CanExecute = "CanUseTokenizer")]
        private async Task EncodeAsync(string input)
        {
            if (string.IsNullOrWhiteSpace(input))
            {
                return;
            }

            EncodeResult = string.Empty;
            try
            {
                var sequences = await _tokenizer.EncodeAsync(input);
                EncodeResult = string.Join(", ", sequences[0].ToArray());
            }
            catch (Exception ex)
            {
                await Application.Current.MainPage.DisplayAlert("Tokenizer Encode Error", ex.Message, "OK");
            }
        }

        [RelayCommand(CanExecute = "CanUseTokenizer")]
        private async Task DecodeAsync(string input)
        {
            if (string.IsNullOrWhiteSpace(input))
            {
                return;
            }

            DecodeResult = string.Empty;
            try
            {
                var intArray = input
                     .Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
                     .Select(int.Parse)
                     .ToArray();
                DecodeResult = await _tokenizer.DecodeAsync(intArray);
            }
            catch (Exception ex)
            {
                await Application.Current.MainPage.DisplayAlert("Tokenizer Decode Error", ex.Message, "OK");
            }
        }

        private bool CanUseTokenizer()
        {
            return _tokenizer != null;
        }
    }
}
