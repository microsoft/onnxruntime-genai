using GennyMaui.Models;
using Microsoft.ML.OnnxRuntimeGenAI;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Maui.Storage;
using System.Text.Json;
using CommunityToolkit.Mvvm.Messaging.Messages;
using CommunityToolkit.Mvvm.Messaging;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using System.Collections.ObjectModel;
using System.Linq;


namespace GennyMaui.ViewModels
{
    internal enum ModelStatus {
        NotAvailble,

        ReadyToLoad,

        Loaded,

        Downloadable
    }

    public partial class LoadableModel : ObservableObject
    {
        [ObservableProperty]
        private Model? _model;

        [ObservableProperty]
        private Tokenizer? _tokenizer;

        [ObservableProperty]
        private ConfigurationModel? _configuration;

        [ObservableProperty]
        private string _modelPath = string.Empty;

        [ObservableProperty]
        private bool _isModelLoaded;

        [ObservableProperty]
        private bool _isModelLoading;

        [ObservableProperty]
        private bool _isLocalModelSelected;

        [ObservableProperty]
        private string _localModelStatusString = string.Empty;

        [ObservableProperty]
        private string _ortVersionString = string.Empty;

        public ObservableCollection<string> OrtEps { get; } = new ObservableCollection<string>();

        public LoadableModel()
        {
            var ortEnv = OrtEnv.Instance();
            _ortVersionString = ortEnv.GetVersionString();
            var providers = ortEnv.GetAvailableProviders();
            foreach (var provider in providers)
            {
                OrtEps.Add(provider);
            }

            ortEnv.Dispose();
        }

        public List<HuggingFaceModel> RemoteModels { get; } =
        [
            new()
            {
                RepoId = "microsoft/Phi-3-mini-4k-instruct-onnx",
                Subpath = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
            },
            new ()
            {
                RepoId = "microsoft/mistral-7b-instruct-v0.2-ONNX",
                Subpath = "onnx/cpu_and_mobile/mistral-7b-instruct-v0.2-cpu-int4-rtn-block-32-acc-level-4"
            },
            new ()
            {
                RepoId = "microsoft/Phi-3-small-8k-instruct-onnx-cuda",
                Subpath = "cuda-int4-rtn-block-32"
            }
        ];

        private async Task<bool> OpenModelAsync()
        {
#if ANDROID
            return false;
#else
            var result = await FolderPicker.Default.PickAsync();

            if (result.IsSuccessful)
            {
                ModelPath = result.Folder.Path;
                return true;
            }
            else
            {
                await Application.Current.MainPage.DisplayAlert("Folder Open Error", result.Exception.Message, "OK");
                return false;
            }
#endif
        }

        [RelayCommand(CanExecute = "CanExecuteLoadModel")]
        private async Task LoadModelAsync()
        {
            await UnloadModelAsync();
            try
            {
                var currentModelPath = CurrentSelectedModelPath();
                IsModelLoading = true;
                Configuration = await LoadConfigAsync(currentModelPath);

                WeakReferenceMessenger.Default.Send(new PropertyChangedMessage<ConfigurationModel>(this, nameof(Configuration), null, Configuration));
                WeakReferenceMessenger.Default.Send(new PropertyChangedMessage<SearchOptionsModel>(this, nameof(SearchOptionsModel), null, Configuration.SearchOptions));

                await Task.Run(() =>
                {
                    Model = new Model(currentModelPath);
                    Tokenizer = new Tokenizer(Model);
                });
                IsModelLoaded = true;

                RefreshLocalModelStatus();
                RefreshRemoteModelStatus();

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

        [RelayCommand]
        private async Task ShowOrtInfoAsync()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("Version: ");
            sb.Append(_ortVersionString);
            sb.Append('\n');

            sb.Append("Providers: ");
            sb.Append(string.Join(", ", OrtEps));
            sb.Append('\n');

            await Application.Current.MainPage.DisplayAlert("ORT Info", sb.ToString(), "OK");
        }

        private bool CanExecuteLoadModel()
        {
            var path = CurrentSelectedModelPath();
            return !string.IsNullOrWhiteSpace(path) && Path.Exists(path);
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

        private async Task<string> DownloadHuggingFaceModel(HuggingFaceModel hfModel)
        {
            try
            {
                hfModel.IsDownloading = true;
                return await HuggingfaceHub.HFDownloader.DownloadSnapshotAsync(
                    hfModel.RepoId,
                    allowPatterns: [hfModel.Include],
                    localDir: hfModel.DownloadPath
                    );
            }
            catch (Exception ex)
            {
                await Application.Current.MainPage.DisplayAlert("Model Download Error", ex.Message, "OK");
                return string.Empty;
            }
            finally
            {
                hfModel.IsDownloading = false;
            }
        }

        private string CurrentSelectedModelPath()
        {
            if (IsLocalModelSelected)
            {
                return ModelPath;
            }

            foreach (var item in RemoteModels)
            {
                if (item.IsChecked && item.Exists)
                {
                    return item.ModelPath;
                }
            }

            return string.Empty;
        }

        internal void ToggleLocalModel(bool ischecked)
        {
            if (!ischecked)
            {
                if (IsModelLoaded)
                {
                    UnloadModelAsync().ContinueWith(t =>
                    {
                        if (t.IsCompleted)
                        {
                            App.Current.Dispatcher.Dispatch(() =>
                            {
                                RefreshLocalModelStatus();
                            });
                        }
                    });
                }
                else
                {
                    RefreshLocalModelStatus();
                }
                return;
            }

            foreach (var item in RemoteModels)
            {
                item.IsChecked = false;
            }

            if (!string.IsNullOrWhiteSpace(ModelPath) || Path.Exists(ModelPath))
            {
                RefreshLocalModelStatus();
                return;
            }

            OpenModelAsync().ContinueWith(t =>
            {
                if (!t.Result)
                {
                    IsLocalModelSelected = false;
                }
                App.Current.Dispatcher.Dispatch(() =>
                {
                    RefreshLocalModelStatus();
                });
            });
        }

        internal void ToggleHuggingfaceModel(HuggingFaceModel hfModel, bool ischecked)
        {
            if (!ischecked)
            {
                if (IsModelLoaded)
                {
                    UnloadModelAsync().ContinueWith(t =>
                    {
                        if (t.IsCompleted)
                        {
                            App.Current.Dispatcher.Dispatch(() =>
                            {
                                RefreshRemoteModelStatus();
                            });
                        }
                    });
                }
                else
                {
                    RefreshRemoteModelStatus();
                }
                return;
            }

            IsLocalModelSelected = false;
            foreach (var item in RemoteModels)
            {
                if (item.RepoId != hfModel.RepoId)
                {
                    item.IsChecked = false;
                }
            }

            if (hfModel.Exists)
            {
                RefreshRemoteModelStatus();
                return;
            }
            else
            {
                DownloadHuggingFaceModel(hfModel).ContinueWith(t =>
                    {
                        if (string.IsNullOrEmpty(t.Result))
                        {
                            foreach (var item in RemoteModels)
                            {
                                if (item.RepoId == hfModel.RepoId)
                                {
                                    item.IsChecked = false;
                                }
                            }
                        }
                        App.Current.Dispatcher.Dispatch(() =>
                        {
                            RefreshRemoteModelStatus();
                        });
                    }); ;
            }
        }

        internal void RefreshLocalModelStatus()
        {
            if (!IsModelLoaded && !Path.Exists(ModelPath))
            {
                LocalModelStatusString = ModelStatusToString(ModelStatus.NotAvailble);
            }

            if (Path.Exists(ModelPath))
            {
                LocalModelStatusString = ModelStatusToString(ModelStatus.ReadyToLoad);
            }

            if (IsModelLoaded && IsLocalModelSelected)
            {
                LocalModelStatusString = ModelStatusToString(ModelStatus.Loaded);
            }

            LoadModelCommand.NotifyCanExecuteChanged();
        }

        internal void RefreshRemoteModelStatus()
        {
            foreach (var item in RemoteModels)
            {
                if (IsModelLoaded && CurrentSelectedModelPath() == item.ModelPath)
                {
                    item.StatusString = ModelStatusToString(ModelStatus.Loaded);
                    continue;
                }

                if (item.Exists)
                {
                    item.StatusString = ModelStatusToString(ModelStatus.ReadyToLoad);
                }
                else
                {
                    item.StatusString = ModelStatusToString(ModelStatus.Downloadable);
                }
            }

            LoadModelCommand.NotifyCanExecuteChanged();
        }

        internal string ModelStatusToString(ModelStatus status)
        {
            StringBuilder sb = new();

            switch (status)
            {
                case ModelStatus.NotAvailble:
                    sb.Append('❌');
                    break;
                case ModelStatus.Downloadable:
                    sb.Append("🔽");
                    break;
                case ModelStatus.ReadyToLoad:
                    sb.Append('⚡');
                    break;
                case ModelStatus.Loaded:
                    sb.Append('✅');
                    break;
            }

            return sb.ToString();

        }
    }
}
