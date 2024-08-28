namespace GennyMaui
{
    public partial class App : Application
    {
        public App()
        {
            InitializeComponent();

            if (DeviceInfo.Idiom == DeviceIdiom.Phone)
            {
                MainPage = new AppShellMobile();
            }
            else
            {
                MainPage = new AppShell();
            }
        }
    }
}
