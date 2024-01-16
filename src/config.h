#pragma once
namespace Generators {

struct Config {
  Config() = default;
  Config(const std::filesystem::path& path);

  std::filesystem::path config_path;  // Path of the config directory

  // Sequence Generation
  int min_length{0};
  int max_length{20};
  bool early_stopping{false};  //  Whether to stop the beam search when at least num_beams sentences are finished per batch or not.
  int num_beams{1};            //  1 means no beam search.
  float temperature{1.0f};
  int top_k{50};                   // Number of highest probability vocabulary tokens to keep for top-k-filtering that will be used by default in the generate method of the model.
  float top_p{1.0f};               // If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
  float repetition_penalty{1.0f};  // 1.0 means no penalty.
  float length_penalty{1.0f};      // Exponential penalty to the length that is used with beam-based generation. length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.

  // Tokenizer Parameters
  std::string tokenizer_class;
  std::string prefix;
  int pad_token_id{};            // The id of the padding token.
  int eos_token_id{};            // The id of the end-of-stream token.
  int bos_token_id{};            // The id of the beginning-of-stream token.
  int decoder_start_token_id{};  // If an encoder-decoder model starts decoding with a different token than bos, the id of that token.
  int sep_token_id{};            // The id of the separation token.

  // Model Class Attributes
  std::string model_decoder;
  std::string model_encoder_decoder_init;
  std::string model_type;
  int vocab_size{};
  int hidden_size{};
  int num_attention_heads{};
  int num_hidden_layers{};
};

}  // namespace Generators