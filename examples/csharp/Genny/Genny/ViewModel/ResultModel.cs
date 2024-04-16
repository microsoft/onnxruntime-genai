using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace Genny.ViewModel
{
    public class ResultModel : INotifyPropertyChanged
    {
        private string _content;
        private bool _isUserInput;

        public string Content
        {
            get { return _content; }
            set { _content = value; NotifyPropertyChanged(); }
        }

        public bool IsUserInput
        {
            get { return _isUserInput; }
            set { _isUserInput = value; NotifyPropertyChanged(); }
        }

        public DateTime Timestamp { get; } = DateTime.Now;

        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;
        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }
        #endregion
    }
}