#pragma once
namespace Generators {

struct Config {
  Config() = default;
  Config(const std::filesystem::path& path);

  std::filesystem::path config_path;  // Path of the config directory

  struct Model {
    std::string type;

    int pad_token_id{};            // The id of the padding token.
    int eos_token_id{};            // The id of the end-of-stream token.
    int bos_token_id{};            // The id of the beginning-of-stream token.
    int sep_token_id{};            // The id of the separation token.
    int decoder_start_token_id{};  // If an encoder-decoder model starts decoding with a different token than bos, the id of that token.
    int vocab_size{};
    int context_length{};

    // For models like whisper
    struct EncoderDecoderInit {
      std::string filename;
    } encoder_decoder_init;

    struct Decoder {
      std::string filename;

      int hidden_size{};          // Not currently used, potentially useful for embeddings in the future
      int num_attention_heads{};  // Not currently used, potentially useful if num_key_value_heads isn't set
      int num_key_value_heads{};
      int num_hidden_layers{};
      int head_size{};

      struct Inputs {
        std::string input_ids{"input_ids"};
        std::string position_ids{"position_ids"};
        std::string attention_mask{"attention_mask"};
        std::string past_key_names{"past_key_values.%d.key"}, past_value_names{"past_key_values.%d.value"};
        std::string past_names;  // When key/value pairs are combined
        std::string cross_past_key_names, cross_past_value_names;
      } inputs;

      struct Outputs {
        std::string logits{"logits"};
        std::string present_key_names{"present.%d.key"}, present_value_names{"present.%d.value"};
        std::string present_names;  // When key/value pairs are combined
        std::string cross_present_key_names, cross_present_value_names;
      } outputs;

    } decoder;
  } model;

  struct Search {
    bool do_sample{};  // True to do randomized sampling through top_k and top_p, if false, the top logit score is chosen
    int min_length{};
    int max_length{};  // If omitted or 0 in json file, will be set to model.context_length on load
    int num_beams{1};  // 1 means no beam search.
    int num_return_sequences{1};
    float repetition_penalty{1.0f};  // 1.0 means no penalty.
    int top_k{};                     // Number of highest probability vocabulary tokens to keep for top-k-filtering that will be used by default in the generate method of the model.
    float top_p{1.0f};               // If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    float temperature{1.0f};
    bool early_stopping{true};  //  Whether to stop the beam search when at least num_beams sentences are finished per batch or not.
    int no_repeat_ngram_size{};
    float diversity_penalty{};
    float length_penalty{1.0f};        // Exponential penalty to the length that is used with beam-based generation. length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.
    bool past_present_share_buffer{};  // The past/present kv tensors are shared and allocated once to max_length (cuda only)
  } search;
};

void SetSearchNumber(Config::Search& search, std::string_view name, double value);
void SetSearchBool(Config::Search& search, std::string_view name, bool value);

}  // namespace Generators