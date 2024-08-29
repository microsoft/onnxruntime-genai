using GennyMaui.Models;
using Microsoft.ML.OnnxRuntimeGenAI;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Maui.Storage;
using System.Text.Json;
using CommunityToolkit.Mvvm.Messaging.Messages;
using CommunityToolkit.Mvvm.Messaging;
using Microsoft.Maui.Controls;

namespace GennyMaui.ViewModels
{
    public partial class LoadableModel: ObservableObject
    {
        [ObservableProperty]
        private Model? _model;

        [ObservableProperty]
        private Tokenizer? _tokenizer;

        [ObservableProperty]
        private ConfigurationModel? _configuration;

        [ObservableProperty]
        private string _modelPath = "D:\\Repositories\\phi2_onnx";

        [ObservableProperty]
        private bool _isModelLoaded;

        [ObservableProperty]
        private bool _isModelLoading;

        [RelayCommand]
        private async Task OpenModelAsync()
        {
#if ANDROID
#else
            var result = await FolderPicker.Default.PickAsync();

            if (result.IsSuccessful)
            {
                ModelPath = result.Folder.Path;
            }
            else
            {
                await Application.Current.MainPage.DisplayAlert("Folder Open Error", result.Exception.Message, "OK");
            }
#endif
        }

        [RelayCommand(CanExecute = "CanExecuteLoadModel")]
        private async Task LoadModelAsync()
        {
            await UnloadModelAsync();
            try
            {
                IsModelLoading = true;
                Configuration = await LoadConfigAsync(ModelPath);

                WeakReferenceMessenger.Default.Send(new PropertyChangedMessage<ConfigurationModel>(this, nameof(Configuration), null, Configuration));
                WeakReferenceMessenger.Default.Send(new PropertyChangedMessage<SearchOptionsModel>(this, nameof(SearchOptionsModel), null, Configuration.SearchOptions));

                await Task.Run(() =>
                {
                    Model = new Model(ModelPath);
                    Tokenizer = new Tokenizer(Model);
                });
                IsModelLoaded = true;

                WeakReferenceMessenger.Default.Send(new PropertyChangedMessage<Model>(this, nameof(Model), null, Model));
                WeakReferenceMessenger.Default.Send(new PropertyChangedMessage<Tokenizer>(this, nameof(Tokenizer), null, Tokenizer));
            }
            catch (Exception ex)
            {
                await Application.Current.MainPage.DisplayAlert("Model Load Error", ex.Message, "OK");
            }
            finally 
            {
                IsModelLoading = false;
            }
        }

        private bool CanExecuteLoadModel()
        {
            return !string.IsNullOrWhiteSpace(ModelPath);
        }

        private Task UnloadModelAsync()
        {
            Model?.Dispose();
            Tokenizer?.Dispose();
            IsModelLoaded = false;
            return Task.CompletedTask;
        }

        private static async Task<ConfigurationModel> LoadConfigAsync(string modelPath)
        {
            var configPath = Path.Combine(modelPath, "genai_config.json");
            var configJson = await File.ReadAllTextAsync(configPath);
            return JsonSerializer.Deserialize<ConfigurationModel>(configJson);
        }
    }
}
