using CommunityToolkit.Mvvm.ComponentModel;

namespace GennyMaui.ViewModels
{
    public partial class ResultModel : ObservableObject
    {
        [ObservableProperty]
        private string? _content;

        [ObservableProperty]
        private bool _isUserInput;

        public DateTime Timestamp { get; } = DateTime.Now;
    }
}