using Genny.ViewModel;
using System.Windows;
using System.Windows.Controls;

namespace Genny.Controls
{
    /// <summary>
    /// Interaction logic for SearchOptionsControl.xaml
    /// </summary>
    public partial class SearchOptionsControl : UserControl
    {
        public SearchOptionsControl()
        {
            InitializeComponent();
        }

        public static readonly DependencyProperty SearchOptionsProperty =
           DependencyProperty.Register(nameof(SearchOptions), typeof(SearchOptionsModel), typeof(SearchOptionsControl), new PropertyMetadata(new SearchOptionsModel()));


        /// <summary>
        /// Gets or sets the search options.
        /// </summary>
        public SearchOptionsModel SearchOptions
        {
            get { return (SearchOptionsModel)GetValue(SearchOptionsProperty); }
            set { SetValue(SearchOptionsProperty, value); }
        }
    }
}
