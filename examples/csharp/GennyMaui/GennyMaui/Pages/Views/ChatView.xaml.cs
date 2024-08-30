using GennyMaui.ViewModels;

namespace GennyMaui.Pages.Views;

public partial class ChatView : ContentView
{
    public ChatView()
    {
        InitializeComponent();
    }

    public void PromptTextChanged(object sender, TextChangedEventArgs e)
    {
        ((ChatViewModel)BindingContext).GenerateCommand.NotifyCanExecuteChanged();
    }
}