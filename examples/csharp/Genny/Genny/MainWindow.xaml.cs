using Genny.Utils;
using Genny.ViewModel;
using Microsoft.ML.OnnxRuntimeGenAI;
using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;

namespace Genny
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        private Model _model;
        private Tokenizer _tokenizer;
        private ConfigurationModel _configuration;
        private string _modelPath = "D:\\Repositories\\phi2_onnx";
        private bool _isModelLoaded;

        public MainWindow()
        {
            OpenModelCommand = new RelayCommand(OpenModelAsync);
            LoadModelCommand = new RelayCommand(LoadModelAsync, CanExecuteLoadModel);
            InitializeComponent();
        }

        public RelayCommand OpenModelCommand { get; }
        public RelayCommand LoadModelCommand { get; }

        public Model Model
        {
            get { return _model; }
            set { _model = value; NotifyPropertyChanged(); }
        }

        public Tokenizer Tokenizer
        {
            get { return _tokenizer; }
            set { _tokenizer = value; NotifyPropertyChanged(); }
        }

        public ConfigurationModel Configuration
        {
            get { return _configuration; }
            set { _configuration = value; NotifyPropertyChanged(); }
        }


        public bool IsModelLoaded
        {
            get { return _isModelLoaded; }
            set { _isModelLoaded = value; NotifyPropertyChanged(); }
        }

        public string ModelPath
        {
            get { return _modelPath; }
            set { _modelPath = value; NotifyPropertyChanged(); }
        }


        private Task OpenModelAsync()
        {
            var folderBrowserDialog = new System.Windows.Forms.FolderBrowserDialog
            {
                Description = "Model Folder Path",
                UseDescriptionForTitle = true,
            };
            var dialogResult = folderBrowserDialog.ShowDialog();
            if (dialogResult == System.Windows.Forms.DialogResult.OK)
                ModelPath = folderBrowserDialog.SelectedPath;

            return Task.CompletedTask;
        }


        private async Task LoadModelAsync()
        {
            await UnloadModelAsync();
            try
            {
                Configuration = await LoadConfigAsync(ModelPath);
                await Task.Run(() =>
                {
                    Model = new Model(ModelPath);
                    Tokenizer = new Tokenizer(_model);
                });
                IsModelLoaded = true;
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Model Load Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }


        private bool CanExecuteLoadModel()
        {
            return !string.IsNullOrWhiteSpace(ModelPath);
        }


        private Task UnloadModelAsync()
        {
            _model?.Dispose();
            _tokenizer?.Dispose();
            IsModelLoaded = false;
            return Task.CompletedTask;
        }


        private static async Task<ConfigurationModel> LoadConfigAsync(string modelPath)
        {
            var configPath = Path.Combine(modelPath, "genai_config.json");
            var configJson = await File.ReadAllTextAsync(configPath);
            return JsonSerializer.Deserialize<ConfigurationModel>(configJson);
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