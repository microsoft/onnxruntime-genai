using GennyMaui.Models;
using GennyMaui.ViewModels;

namespace GennyMaui.Pages.Views;

public partial class ModelConfigView : ContentView
{
	public ModelConfigView()
	{
		InitializeComponent();
	}

    private void CheckBox_CheckedChanged(object sender, CheckedChangedEventArgs e)
    {
		var context = (HuggingFaceModel)((CheckBox)sender).BindingContext;
		var parentContext = (LoadableModel)this.BindingContext;
        parentContext.ToggleHuggingfaceModel(context, e.Value);
    }
}