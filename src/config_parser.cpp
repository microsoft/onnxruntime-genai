#include "generators.h"
#include "json.h"
#include <fstream>

namespace Generators {

struct Root_Element : JSON::Element {
  Root_Element(Config& config) : config_{config} {}

  void OnString(std::string_view name, std::string_view value) override {
    if (name == "tokenizer_class")
      config_.tokenizer_class = value;
    else if (name == "prefix")
      config_.prefix = value;
    else if (name == "model_type")
      config_.model_type = value;
    else if (name == "ogai_model_decoder")
      config_.model_decoder = value;
  }

  void OnNumber(std::string_view name, double value) override {
    // Sequence Generation
    if (name == "max_length")
      config_.max_length = static_cast<int>(value);
    else if (name == "min_length")
      config_.min_length = static_cast<int>(value);
    else if (name == "num_beams")
      config_.num_beams = static_cast<int>(value);
    else if (name == "temperature")
      config_.temperature = static_cast<float>(value);
    else if (name == "top_k")
      config_.top_k = static_cast<int>(value);
    else if (name == "top_p")
      config_.top_p = static_cast<float>(value);
    else if (name == "repetition_penalty")
      config_.repetition_penalty = static_cast<float>(value);
    else if (name == "length_penalty")
      config_.length_penalty = static_cast<float>(value);

    // Tokenizer Parameters
    else if (name == "pad_token_id")
      config_.pad_token_id = static_cast<int>(value);
    else if (name == "eos_token_id")
      config_.eos_token_id = static_cast<int>(value);
    else if (name == "bos_token_id")
      config_.bos_token_id = static_cast<int>(value);
    else if (name == "decoder_start_token_id")
      config_.decoder_start_token_id = static_cast<int>(value);
    else if (name == "sep_token_id")
      config_.sep_token_id = static_cast<int>(value);

    // Model Class Attributes
    else if (name == "vocab_size")
      config_.vocab_size = static_cast<int>(value);
    else if (name == "hidden_size")
      config_.hidden_size = static_cast<int>(value);
    else if (name == "n_embed")
      config_.n_embed = static_cast<int>(value);
    else if (name == "num_attention_heads" || name == "num_heads")
      config_.num_attention_heads = static_cast<int>(value);
    else if (name == "n_head")
      config_.n_head = static_cast<int>(value);
    else if (name == "num_hidden_layers" || name == "num_layers")
      config_.num_hidden_layers = static_cast<int>(value);
    else if (name == "n_layer")
      config_.n_layer = static_cast<int>(value);
  }

  void OnBool(std::string_view name, bool value) override {
    if (name == "early_stopping")
      config_.early_stopping = value;
  }

  Config& config_;
};

struct RootObject_Element : JSON::Element {
  RootObject_Element(JSON::Element& t) : t_{t} {}

  Element& OnObject(std::string_view name) {
    return t_;
  }

  JSON::Element& t_;
};

void ParseConfig(const std::filesystem::path& filename, Config& config) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open())
    throw std::runtime_error("Error opening " + filename.string());
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size))
    throw std::runtime_error("Error reading " + filename.string());

  Root_Element root{config};
  RootObject_Element root_object{root};
  JSON::Parse(root_object, std::string_view(buffer.data(), buffer.size()));
}

}  // namespace Generators