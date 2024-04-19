using System.Text.Json.Serialization;

namespace Genny.ViewModel
{
    public class ConfigurationModel
    {
        [JsonPropertyName("model")]
        public ModelOptionsModel ModelOptions { get; set; }

        [JsonPropertyName("search")]
        public SearchOptionsModel SearchOptions { get; set; }
    }
}
