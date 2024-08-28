using GennyMaui.ViewModels;

namespace GennyMaui.Pages.Views;

public partial class StatefulView : ContentView
{
    public StatefulView()
    {
        InitializeComponent();
    }

    public void PromptTextChanged(object sender, TextChangedEventArgs e)
    {
        ((StatefulChatViewModel)BindingContext).GenerateCommand.NotifyCanExecuteChanged();
    }
}