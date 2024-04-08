using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace Genny.ViewModel
{
    public class SearchOptionsModel : INotifyPropertyChanged
    {
        private int _topK = 50;
        private float _topP = 0.9f;
        private float _temperature = 1;
        private float _repetitionPenalty = 1;
        private bool _pastPresentShareBuffer = false;
        private int _numReturnSequences = 1;
        private int _numBeams = 1;
        private int _noRepeatNgramSize = 0;
        private int _minLength = 0;
        private int _maxLength = 200;
        private float _lengthPenalty = 1;
        private bool _earlyStopping = true;
        private bool _doSample = false;
        private float _diversityPenalty = 0;

        public int TopK
        {
            get { return _topK; }
            set { _topK = value; NotifyPropertyChanged(); }
        }

        public float TopP
        {
            get { return _topP; }
            set { _topP = value; NotifyPropertyChanged(); }
        }

        public float Temperature
        {
            get { return _temperature; }
            set { _temperature = value; NotifyPropertyChanged(); }
        }

        public float RepetitionPenalty
        {
            get { return _repetitionPenalty; }
            set { _repetitionPenalty = value; NotifyPropertyChanged(); }
        }

        public bool PastPresentShareBuffer
        {
            get { return _pastPresentShareBuffer; }
            set { _pastPresentShareBuffer = value; NotifyPropertyChanged(); }
        }

        public int NumReturnSequences
        {
            get { return _numReturnSequences; }
            set { _numReturnSequences = value; NotifyPropertyChanged(); }
        }

        public int NumBeams
        {
            get { return _numBeams; }
            set { _numBeams = value; NotifyPropertyChanged(); }
        }

        public int NoRepeatNgramSize
        {
            get { return _noRepeatNgramSize; }
            set { _noRepeatNgramSize = value; NotifyPropertyChanged(); }
        }

        public int MinLength
        {
            get { return _minLength; }
            set { _minLength = value; NotifyPropertyChanged(); }
        }

        public int MaxLength
        {
            get { return _maxLength; }
            set { _maxLength = value; NotifyPropertyChanged(); }
        }

        public float LengthPenalty
        {
            get { return _lengthPenalty; }
            set { _lengthPenalty = value; NotifyPropertyChanged(); }
        }

        public float DiversityPenalty
        {
            get { return _diversityPenalty; }
            set { _diversityPenalty = value; NotifyPropertyChanged(); }
        }

        public bool EarlyStopping
        {
            get { return _earlyStopping; }
            set { _earlyStopping = value; NotifyPropertyChanged(); }
        }

        public bool DoSample
        {
            get { return _doSample; }
            set { _doSample = value; NotifyPropertyChanged(); }
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