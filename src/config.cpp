#include "generators.h"
#include "json.h"
#include <fstream>
#include <sstream>

namespace Generators {

ONNXTensorElementDataType TranslateTensorType(std::string_view value) {
  if (value == "float32") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  if (value == "float16") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  }

  throw std::runtime_error("Invalid tensor type: " + std::string(value));
}

struct Model_Element : JSON::Element {
  explicit Model_Element(Config::Model& model) : model_{model} {}

  void OnString(std::string_view name, std::string_view value) override {
    if (name == "type") {
      model_.type = value;
    } else if (name == "decoder") {
      model_.decoder = value;
    } else if (name == "encoder_decoder_init") {
      model_.encoder_decoder_init = value;
    } else if (name == "past_names_key") {
      model_.past_names_key = value;
    } else if (name == "past_names_value") {
      model_.past_names_value = value;
    } else if (name == "present_names_key") {
      model_.present_names_key = value;
    } else if (name == "present_names_value") {
      model_.present_names_value = value;
    } else if (name == "past_names") {
      model_.past_names = value;
    } else if (name == "present_names") {
      model_.present_names = value;
    } else if (name == "cross_past_names_key") {
      model_.cross_past_names_key = value;
    } else if (name == "cross_past_names_value") {
      model_.cross_past_names_value = value;
    } else if (name == "cross_present_names_key") {
      model_.cross_present_names_key = value;
    } else if (name == "cross_present_names_value") {
      model_.cross_present_names_value = value;
    } else if (name == "logits_type") {
      model_.logits_type = TranslateTensorType(value);
    } else if (name == "kv_type") {
      model_.kv_type = TranslateTensorType(value);
    } else {
      throw std::runtime_error("Unknown name: " + std::string(name));
    }
  }

  void OnNumber(std::string_view name, double value) override {
    if (name == "vocab_size") {
      model_.vocab_size = static_cast<int>(value);
    } else if (name == "hidden_size" || name == "n_embed") {
      model_.hidden_size = static_cast<int>(value);
    } else if (name == "num_attention_heads" || name == "num_heads" || name == "n_head") {
      model_.num_attention_heads = static_cast<int>(value);
    } else if (name == "num_hidden_layers" || name == "num_layers" || name == "n_layer") {
      model_.num_hidden_layers = static_cast<int>(value);
    }
  }

 private:
  Config::Model& model_;
};

struct Root_Element : JSON::Element {
  explicit Root_Element(Config& config) : config_{config} {}

  void OnString(std::string_view name, std::string_view value) override {
    if (name == "tokenizer_class") {
      config_.tokenizer_class = value;
    } else if (name == "prefix") {
      config_.prefix = value;
    }
  }

  void OnNumber(std::string_view name, double value) override {
    // Sequence Generation
    if (name == "max_length") {
      config_.max_length = static_cast<int>(value);
    } else if (name == "min_length") {
      config_.min_length = static_cast<int>(value);
    } else if (name == "num_beams") {
      config_.num_beams = static_cast<int>(value);
    } else if (name == "temperature") {
      config_.temperature = static_cast<float>(value);
    } else if (name == "top_k") {
      config_.top_k = static_cast<int>(value);
    } else if (name == "top_p") {
      config_.top_p = static_cast<float>(value);
    } else if (name == "repetition_penalty") {
      config_.repetition_penalty = static_cast<float>(value);
    } else if (name == "length_penalty") {
      config_.length_penalty = static_cast<float>(value);

      // Tokenizer Parameters
    } else if (name == "pad_token_id") {
      config_.pad_token_id = static_cast<int>(value);
    } else if (name == "eos_token_id") {
      config_.eos_token_id = static_cast<int>(value);
    } else if (name == "bos_token_id") {
      config_.bos_token_id = static_cast<int>(value);
    } else if (name == "decoder_start_token_id") {
      config_.decoder_start_token_id = static_cast<int>(value);
    } else if (name == "sep_token_id") {
      config_.sep_token_id = static_cast<int>(value);
    }
  }

  void OnBool(std::string_view name, bool value) override {
    if (name == "early_stopping") {
      config_.early_stopping = value;
    }
  }

  Element& OnObject(std::string_view name) override {
    if (name == "model") {
      return model_element_;
    }
    return Element::OnObject(name);
  }

  Config& config_;
  Model_Element model_element_{config_.model};
};

struct RootObject_Element : JSON::Element {
  explicit RootObject_Element(JSON::Element& t) : t_{t} {}

  Element& OnObject(std::string_view /*name*/) override {
    return t_;
  }

  JSON::Element& t_;
};

void ParseConfig(const std::filesystem::path& filename, Config& config) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Error opening " + filename.string());
  }
  std::streamsize const size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    throw std::runtime_error("Error reading " + filename.string());
  }

  Root_Element root{config};
  RootObject_Element root_object{root};
  try {
    JSON::Parse(root_object, std::string_view(buffer.data(), buffer.size()));
  } catch (const std::exception& message) {
    std::ostringstream oss;
    oss << "Error encountered while parsing '" << filename.string() << "' " << message.what();
    throw std::runtime_error(oss.str());
  }
}

Config::Config(const std::filesystem::path& path) : config_path{path} {
  ParseConfig(path / "config.json", *this);
}

}  // namespace Generators
