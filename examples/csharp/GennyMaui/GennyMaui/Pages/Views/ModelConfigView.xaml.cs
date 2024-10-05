using GennyMaui.Models;
using GennyMaui.Services;
using GennyMaui.ViewModels;

namespace GennyMaui.Pages.Views;

public partial class ModelConfigView : ContentView
{
    public ModelConfigView()
    {
        InitializeComponent();
        this.BindingContext = MauiProgram.GetService<IModelProvider>();
        var parentContext = (LoadableModel)this.BindingContext;
        parentContext.RefreshLocalModelStatus();
        parentContext.RefreshRemoteModelStatus();
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

    private void LocalModelPathEntry_TextChanged(object sender, TextChangedEventArgs e)
    {
        var parentContext = (LoadableModel)this.BindingContext;
        parentContext.RefreshLocalModelStatus();
    }
}