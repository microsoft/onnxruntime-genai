using GennyMaui.Models;
using GennyMaui.ViewModels;

namespace GennyMaui.Pages.Views;

public partial class ModelConfigView : ContentView
{
    public ModelConfigView()
    {
        InitializeComponent();

        var parentContext = (LoadableModel)this.BindingContext;
        foreach (var item in parentContext.RemoteModels)
        {
            item.RefreshStatus();
        }
    }

    private void LocalModelCheckBox_CheckedChanged(object sender, CheckedChangedEventArgs e)
    {
        var parentContext = (LoadableModel)this.BindingContext;
        parentContext.ToggleLocalModel(e.Value);
    }

    private void RemoteModelCheckBox_CheckedChanged(object sender, CheckedChangedEventArgs e)
    {
        var context = (HuggingFaceModel)((CheckBox)sender).BindingContext;
        var parentContext = (LoadableModel)this.BindingContext;
        parentContext.ToggleHuggingfaceModel(context, e.Value);
    }
}