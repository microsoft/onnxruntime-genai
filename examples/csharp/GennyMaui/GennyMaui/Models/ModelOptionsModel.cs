
using System.Text.Json.Serialization;

namespace GennyMaui.Models
{
    public class ModelOptionsModel
    {
        [JsonPropertyName("type")]
        public string Type { get; set; }

        [JsonPropertyName("context_length")]
        public int ContextLength { get; set; }
    }
}
