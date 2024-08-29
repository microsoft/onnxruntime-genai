namespace GennyMaui.Pages;

public partial class AppMainPageMobile : ContentPage
{
	public AppMainPageMobile()
	{
		InitializeComponent();
	}

    private void ChatButton_Clicked(object sender, EventArgs e)
    {
        Shell.Current.GoToAsync("//chat");
    }

    private void TokenizeButton_Clicked(object sender, EventArgs e)
    {

    }
}