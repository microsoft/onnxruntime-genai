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

struct ProviderOptions_Element : JSON::Element {
  explicit ProviderOptions_Element(Config::ProviderOptions& v) : v_{v} {}

  void OnString(std::string_view name, std::string_view value) override {
    v_.options.emplace_back(name, value);
  }

 private:
  Config::ProviderOptions& v_;
};

struct ProviderOptionsObject_Element : JSON::Element {
  explicit ProviderOptionsObject_Element(std::vector<Config::ProviderOptions>& v) : v_{v} {}

  JSON::Element& OnObject(std::string_view name) override {
    if (options_element_)
      throw std::runtime_error("Each object in the provider_options array can only have one member (named value)");
    auto& options = v_.emplace_back();
    options.name = name;
    options_element_ = std::make_unique<ProviderOptions_Element>(options);
    return *options_element_;
  }

 private:
  std::vector<Config::ProviderOptions>& v_;
  std::unique_ptr<ProviderOptions_Element> options_element_;
};

struct ProviderOptionsArray_Element : JSON::Element {
  explicit ProviderOptionsArray_Element(std::vector<Config::ProviderOptions>& v) : v_{v} {}

  JSON::Element& OnObject(std::string_view name) override { return object_; }

 private:
  std::vector<Config::ProviderOptions>& v_;
  ProviderOptionsObject_Element object_{v_};
};

struct SessionOptions_Element : JSON::Element {
  explicit SessionOptions_Element(Config::SessionOptions& v) : v_{v} {}

  void OnString(std::string_view name, std::string_view value) override {
    if (name == "log_id")
      v_.log_id = value;
    else if (name == "enable_profiling")
      v_.enable_profiling = value;
    else
      throw JSON::unknown_value_error{};
  }

  void OnNumber(std::string_view name, double value) override {
    if (name == "intra_op_num_threads")
      v_.intra_op_num_threads = static_cast<int>(value);
    else if (name == "inter_op_num_threads")
      v_.inter_op_num_threads = static_cast<int>(value);
    else if (name == "log_severity_level")
      v_.log_severity_level = static_cast<int>(value);
    else
      throw JSON::unknown_value_error{};
  }

  void OnBool(std::string_view name, bool value) override {
    if (name == "enable_cpu_mem_arena")
      v_.enable_cpu_mem_arena = value;
    else if (name == "enable_mem_pattern")
      v_.enable_mem_pattern = value;
    else
      throw JSON::unknown_value_error{};
  }

  JSON::Element& OnArray(std::string_view name) override {
    if (name == "provider_options")
      return provider_options_;
    throw JSON::unknown_value_error{};
  }

 private:
  Config::SessionOptions& v_;
  ProviderOptionsArray_Element provider_options_{v_.provider_options};
};

struct EncoderDecoderInit_Element : JSON::Element {
  explicit EncoderDecoderInit_Element(Config::Model::EncoderDecoderInit& v) : v_{v} {}

  void OnString(std::string_view name, std::string_view value) override {
    if (name == "filename") {
      v_.filename = value;
    } else
      throw JSON::unknown_value_error{};
  }

 private:
  Config::Model::EncoderDecoderInit& v_;
};

struct Inputs_Element : JSON::Element {
  explicit Inputs_Element(Config::Model::Decoder::Inputs& v) : v_{v} {}

  void OnString(std::string_view name, std::string_view value) override {
    if (name == "input_ids") {
      v_.input_ids = value;
    } else if (name == "position_ids") {
      v_.position_ids = value;
    } else if (name == "attention_mask") {
      v_.attention_mask = value;
    } else if (name == "seqlens_k") {
      v_.seqlens_k = value;
    } else if (name == "total_seq_len") {
      v_.total_sequence_length = value;
    } else if (name == "past_key_names") {
      v_.past_key_names = value;
    } else if (name == "past_value_names") {
      v_.past_value_names = value;
    } else if (name == "past_names") {
      v_.past_names = value;
    } else if (name == "cross_past_key_names") {
      v_.cross_past_key_names = value;
    } else if (name == "cross_past_value_names") {
      v_.cross_past_value_names = value;
    } else
      throw JSON::unknown_value_error{};
  }

 private:
  Config::Model::Decoder::Inputs& v_;
};

struct Outputs_Element : JSON::Element {
  explicit Outputs_Element(Config::Model::Decoder::Outputs& v) : v_{v} {}

  void OnString(std::string_view name, std::string_view value) override {
    if (name == "logits") {
      v_.logits = value;
    } else if (name == "present_key_names") {
      v_.present_key_names = value;
    } else if (name == "present_value_names") {
      v_.present_value_names = value;
    } else if (name == "present_names") {
      v_.present_names = value;
    } else if (name == "cross_present_key_names") {
      v_.cross_present_key_names = value;
    } else if (name == "cross_present_value_names") {
      v_.cross_present_value_names = value;
    } else
      throw JSON::unknown_value_error{};
  }

 private:
  Config::Model::Decoder::Outputs& v_;
};

struct Decoder_Element : JSON::Element {
  explicit Decoder_Element(Config::Model::Decoder& v) : v_{v} {}

  void OnString(std::string_view name, std::string_view value) override {
    if (name == "filename") {
      v_.filename = value;
    } else
      throw JSON::unknown_value_error{};
  }

  void OnNumber(std::string_view name, double value) override {
    if (name == "hidden_size") {
      v_.hidden_size = static_cast<int>(value);
    } else if (name == "num_attention_heads") {
      v_.num_attention_heads = static_cast<int>(value);
    } else if (name == "num_key_value_heads") {
      v_.num_key_value_heads = static_cast<int>(value);
    } else if (name == "num_hidden_layers") {
      v_.num_hidden_layers = static_cast<int>(value);
    } else if (name == "head_size") {
      v_.head_size = static_cast<int>(value);
    } else
      throw JSON::unknown_value_error{};
  }

  Element& OnObject(std::string_view name) override {
    if (name == "session_options") {
      return session_options_;
    }
    if (name == "inputs") {
      return inputs_;
    }
    if (name == "outputs") {
      return outputs_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Model::Decoder& v_;
  SessionOptions_Element session_options_{v_.session_options};
  Inputs_Element inputs_{v_.inputs};
  Outputs_Element outputs_{v_.outputs};
};

struct Model_Element : JSON::Element {
  explicit Model_Element(Config::Model& v) : v_{v} {}

  void OnString(std::string_view name, std::string_view value) override {
    if (name == "type") {
      v_.type = value;
    } else
      throw JSON::unknown_value_error{};
  }

  void OnNumber(std::string_view name, double value) override {
    if (name == "vocab_size") {
      v_.vocab_size = static_cast<int>(value);
    } else if (name == "context_length") {
      v_.context_length = static_cast<int>(value);
    } else if (name == "pad_token_id") {
      v_.pad_token_id = static_cast<int>(value);
    } else if (name == "eos_token_id") {
      v_.eos_token_id = static_cast<int>(value);
    } else if (name == "bos_token_id") {
      v_.bos_token_id = static_cast<int>(value);
    } else if (name == "decoder_start_token_id") {
      v_.decoder_start_token_id = static_cast<int>(value);
    } else if (name == "sep_token_id") {
      v_.sep_token_id = static_cast<int>(value);
    } else
      throw JSON::unknown_value_error{};
  }

  Element& OnObject(std::string_view name) override {
    if (name == "encoder_decoder_init") {
      return encoder_decoder_init_;
    }
    if (name == "decoder") {
      return decoder_;
    }
    throw JSON::unknown_value_error{};
  }

 private:
  Config::Model& v_;
  EncoderDecoderInit_Element encoder_decoder_init_{v_.encoder_decoder_init};
  Decoder_Element decoder_{v_.decoder};
};

struct Search_Element : JSON::Element {
  explicit Search_Element(Config::Search& v) : v_{v} {}

  void OnString(std::string_view name, std::string_view value) override {
    throw JSON::unknown_value_error{};
  }

  void OnNumber(std::string_view name, double value) override {
    if (name == "min_length") {
      v_.min_length = static_cast<int>(value);
    } else if (name == "max_length") {
      v_.max_length = static_cast<int>(value);
    } else if (name == "num_beams") {
      v_.num_beams = static_cast<int>(value);
    } else if (name == "num_return_sequences") {
      v_.num_return_sequences = static_cast<int>(value);
    } else if (name == "top_k") {
      v_.top_k = static_cast<int>(value);
    } else if (name == "top_p") {
      v_.top_p = static_cast<float>(value);
    } else if (name == "temperature") {
      v_.temperature = static_cast<float>(value);
    } else if (name == "repetition_penalty") {
      v_.repetition_penalty = static_cast<float>(value);
    } else if (name == "length_penalty") {
      v_.length_penalty = static_cast<float>(value);
    } else if (name == "no_repeat_ngram_size") {
      v_.no_repeat_ngram_size = static_cast<int>(value);
    } else if (name == "diversity_penalty") {
      v_.diversity_penalty = static_cast<float>(value);
    } else if (name == "length_penalty") {
      v_.length_penalty = static_cast<float>(value);
    } else if (name == "random_seed") {
      v_.random_seed = static_cast<int>(value);
    } else
      throw JSON::unknown_value_error{};
  }

  void OnBool(std::string_view name, bool value) override {
    if (name == "do_sample") {
      v_.do_sample = value;
    } else if (name == "past_present_share_buffer") {
      v_.past_present_share_buffer = value;
    } else if (name == "early_stopping") {
      v_.early_stopping = value;
    } else
      throw JSON::unknown_value_error{};
  }

 private:
  Config::Search& v_;
};

void SetSearchNumber(Config::Search& search, std::string_view name, double value) {
  Search_Element(search).OnNumber(name, value);
}

void SetSearchBool(Config::Search& search, std::string_view name, bool value) {
  Search_Element(search).OnBool(name, value);
}

bool IsCudaGraphEnabled(Config::SessionOptions& session_options) {
  for (const auto& provider_options : session_options.provider_options) {
    if (provider_options.name == "cuda") {
      for (const auto& value : provider_options.options) {
        if (value.first == "enable_cuda_graph") {
          return value.second == "1";
        }
      }
    }
  }
  return false;
}

struct Root_Element : JSON::Element {
  explicit Root_Element(Config& config) : config_{config} {}

  void OnString(std::string_view name, std::string_view value) override {
  }

  void OnNumber(std::string_view name, double value) override {
  }

  Element& OnObject(std::string_view name) override {
    if (name == "model") {
      return model_element_;
    }
    if (name == "search") {
      return search_element_;
    }
    throw JSON::unknown_value_error{};
  }

  Config& config_;
  Model_Element model_element_{config_.model};
  Search_Element search_element_{config_.search};
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
  ParseConfig(path / "genai_config.json", *this);

  if (model.context_length == 0)
    throw std::runtime_error("model context_length is 0 or was not set. It must be greater than 0");

  if (search.max_length == 0)
    search.max_length = model.context_length;
}

}  // namespace Generators
