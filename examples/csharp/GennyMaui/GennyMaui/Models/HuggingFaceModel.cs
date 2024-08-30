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

        [ObservableProperty]
        private bool _isDownloading = false;

        public string RepoId { get; set; }

        public string Subpath { get; set; }

        public string Include => $"{Subpath}/*";

        public string DownloadPath
        {
            get {
                var mainDir = FileSystem.Current.AppDataDirectory;
                return Path.Combine(mainDir, RepoId);
            }
        }

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
