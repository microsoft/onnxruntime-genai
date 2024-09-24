using Microsoft.Extensions.Logging;
using CommunityToolkit.Maui;
using GennyMaui.Services;
using GennyMaui.ViewModels;

namespace GennyMaui
{
    public static class MauiProgram
    {
        static IServiceProvider _serviceProvider;

        public static TService GetService<TService>()
            => _serviceProvider.GetService<TService>();

        public static MauiApp CreateMauiApp()
        {
            var builder = MauiApp.CreateBuilder();
            builder
                .UseMauiApp<App>()
                .UseMauiCommunityToolkit()
                .ConfigureFonts(fonts =>
                {
                    fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
                    fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
                });

#if DEBUG
    		builder.Logging.AddDebug();
#endif

            builder.Services.AddSingleton<IModelProvider>(new LoadableModel());

            var app = builder.Build();
            _serviceProvider = app.Services;

            return app;
        }
    }
}
