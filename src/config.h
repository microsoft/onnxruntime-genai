// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace Generators {

struct Config {
  Config() = default;
  Config(const fs::path& path);

  struct Defaults {
    static constexpr std::string_view InputIdsName = "input_ids";
    static constexpr std::string_view PixelValuesName = "pixel_values";
    static constexpr std::string_view ImageSizesName = "image_sizes";
    static constexpr std::string_view InputFeaturesName = "encoder_input_ids";
    static constexpr std::string_view ImageFeaturesName = "image_features";
    static constexpr std::string_view CurrentSequenceLengthName = "current_sequence_length";
    static constexpr std::string_view PastSequenceLengthName = "past_sequence_length";
  };

  fs::path config_path;  // Path of the config directory

  using ProviderOption = std::pair<std::string, std::string>;
  struct ProviderOptions {
    std::string name;
    std::vector<ProviderOption> options;
  };

  struct SessionOptions {
    std::optional<int> intra_op_num_threads;
    std::optional<int> inter_op_num_threads;
    std::optional<bool> enable_cpu_mem_arena;
    std::optional<bool> enable_mem_pattern;
    std::optional<bool> disable_cpu_ep_fallback;
    std::optional<bool> disable_quant_qdq;
    std::optional<bool> enable_quant_qdq_cleanup;
    std::optional<bool> ep_context_enable;
    std::optional<std::string> ep_context_embed_mode;
    std::optional<std::string> ep_context_file_path;
    std::optional<std::string> log_id;
    std::optional<int> log_severity_level;
    std::optional<std::string> enable_profiling;
    bool use_env_allocators{true};

    std::vector<ProviderOptions> provider_options;
  };

  struct Model {
    std::string type;

    int pad_token_id{};              // The id of the padding token.
    int eos_token_id{};              // The id of the end-of-stream token.
    std::vector<int> eos_token_ids;  // If eos_token_id is passed as an array, this is where the values go (eos_token_id gets set to the first entry in the array)
    int bos_token_id{};              // The id of the beginning-of-stream token.
    int sep_token_id{};              // The id of the separation token.
    int decoder_start_token_id{};    // If an encoder-decoder model starts decoding with a different token than bos, the id of that token.
    int vocab_size{};
    int context_length{};

    // For models like whisper
    struct EncoderDecoderInit {
      std::string filename;

      struct Inputs {
        std::string input_features{Defaults::InputFeaturesName};
      } inputs;
    } encoder_decoder_init;

    struct Embedding {
      std::string filename;

      struct Inputs {
        std::string input_ids{Defaults::InputIdsName};
        std::string image_features{Defaults::ImageFeaturesName};
      } inputs;

      struct Outputs {
        std::string embeddings{"inputs_embeds"};
      } outputs;
    } embedding;

    struct Vision {
      std::string filename;

      struct Inputs {
        std::string pixel_values{Defaults::PixelValuesName};
        std::string image_sizes{Defaults::ImageSizesName};
      } inputs;

      struct Outputs {
        std::string image_features{Defaults::ImageFeaturesName};
      } outputs;
    } vision;

    struct Decoder {
      std::string filename;
      SessionOptions session_options;

      int hidden_size{};          // Not currently used, potentially useful for embeddings in the future
      int num_attention_heads{};  // Not currently used, potentially useful if num_key_value_heads isn't set
      int num_key_value_heads{};
      int num_hidden_layers{};
      int head_size{};

      struct Inputs {
        std::string input_ids{Defaults::InputIdsName};
        std::string embeddings{"inputs_embeds"};
        std::string position_ids{"position_ids"};
        std::string attention_mask{"attention_mask"};
        std::string past_key_names{"past_key_values.%d.key"}, past_value_names{"past_key_values.%d.value"};
        std::string past_names;  // When key/value pairs are combined
        std::string cross_past_key_names, cross_past_value_names;
        std::string current_sequence_length{Defaults::CurrentSequenceLengthName};
        std::string past_sequence_length{Defaults::PastSequenceLengthName};
      } inputs;

      struct Outputs {
        std::string logits{"logits"};
        std::string present_key_names{"present.%d.key"}, present_value_names{"present.%d.value"};
        std::string present_names;  // When key/value pairs are combined
        std::string cross_present_key_names, cross_present_value_names;
      } outputs;

      struct PipelineModel {
        std::string model_id;
        std::string filename;
        std::optional<SessionOptions> session_options;

        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        std::unordered_map<std::string, std::string> output_names_forwarder;
        bool run_on_prompt{true};
        bool run_on_token_gen{true};
      };

      std::vector<PipelineModel> pipeline;

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
    float top_p{};                   // If set to float >0 and <1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    float temperature{1.0f};
    bool early_stopping{true};  //  Whether to stop the beam search when at least num_beams sentences are finished per batch or not.
    int no_repeat_ngram_size{};
    float diversity_penalty{};
    float length_penalty{1.0f};        // Exponential penalty to the length that is used with beam-based generation. length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.
    bool past_present_share_buffer{};  // The past/present kv tensors are shared and allocated once to max_length (cuda only)
    int random_seed{-1};               // -1 = Seed with random device, otherwise use value to seed RNG
  } search;

  void AddMapping(const std::string& nominal_name, const std::string& graph_name);
  // Returns graph name and true if the nominal name is found in the mapping
  // otherwise returns the nominal name and false
  std::pair<std::string, bool> GetGraphName(const std::string& nominal_name) const;

  std::unordered_map<std::string, std::string> nominal_names_to_graph_names_;  // Mapping of nominal input/output names to graph input/output names
};

void SetSearchNumber(Config::Search& search, std::string_view name, double value);
void SetSearchBool(Config::Search& search, std::string_view name, bool value);
bool IsCudaGraphEnabled(Config::SessionOptions& session_options);

}  // namespace Generators