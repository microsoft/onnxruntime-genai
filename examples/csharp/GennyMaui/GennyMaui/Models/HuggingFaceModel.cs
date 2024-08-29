using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace GennyMaui.Models
{
    [ObservableObject]
    public partial class HuggingFaceModel
    {
        [ObservableProperty]
        private bool isChecked = false;

        [ObservableProperty]
        private string statusString = string.Empty;

        private bool _isDownloading = false;

        public bool IsDownloading
        {
            get
            {
                return _isDownloading;
            }
            set
            {
                SetProperty(ref _isDownloading, value);
                if (Exists)
                {
                    StatusString = "Ready";
                }
                else
                {
                    StatusString = "Downloadable";
                }
            }
        }

        public string RepoId { get; set; }

        public string Include { get; set; }

        public string Subpath { get; set; }

        public string ModelPath { 
            get
            {
                var mainDir = FileSystem.Current.AppDataDirectory;
                if (string.IsNullOrWhiteSpace(Subpath))
                {
                    return Path.Combine(mainDir, RepoId);
                }
                else
                {
                    return Path.Combine(mainDir, RepoId, Subpath);
                }
            }
        }

        public bool Exists => Path.Exists(ModelPath);
    }
}
