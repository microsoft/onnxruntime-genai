using System.Text.Json.Serialization;
using GennyMaui.ViewModels;

namespace GennyMaui.Models
{
    public class ConfigurationModel
    {
        [JsonPropertyName("model")]
        public ModelOptionsModel ModelOptions { get; set; }

        [JsonPropertyName("search")]
        public SearchOptionsModel SearchOptions { get; set; }
    }
}
