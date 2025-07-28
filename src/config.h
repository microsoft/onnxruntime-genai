// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

namespace Generators {

struct RuntimeSettings;

struct Config {
  Config() = default;
  Config(const fs::path& path, std::string_view json_overlay);

  struct Defaults {
    // Decoder names
    static constexpr std::string_view InputIdsName = "input_ids";
    static constexpr std::string_view AttentionMaskName = "attention_mask";
    static constexpr std::string_view PositionIdsName = "position_ids";
    static constexpr std::string_view PastKeyName = "past_key_values.%d.key";
    static constexpr std::string_view PastValueName = "past_key_values.%d.value";
    static constexpr std::string_view LogitsName = "logits";
    static constexpr std::string_view PresentKeyName = "present.%d.key";
    static constexpr std::string_view PresentValueName = "present.%d.value";
    static constexpr std::string_view RnnStatesName = "rnn_states";
    static constexpr std::string_view RnnStatesPrevName = "rnn_states_prev";

    // Speech encoder names
    static constexpr std::string_view AudioAttentionMaskName = "audio_attention_mask";
    static constexpr std::string_view AudioSizesName = "audio_sizes";
    static constexpr std::string_view AudioProjectionModeName = "audio_projection_mode";
    static constexpr std::string_view AudioFeaturesName = "audio_features";
    static constexpr std::string_view NumAudioTokens = "num_audio_tokens";

    // Vision encoder names
    static constexpr std::string_view PixelValuesName = "pixel_values";
    static constexpr std::string_view ImageSizesName = "image_sizes";
    static constexpr std::string_view ImageAttentionMaskName = "image_attention_mask";
    static constexpr std::string_view ImageFeaturesName = "image_features";
    static constexpr std::string_view NumImageTokens = "num_image_tokens";

    // Embedding names
    static constexpr std::string_view AudioEmbedsName = "audio_embeds";
    static constexpr std::string_view InputsEmbedsName = "inputs_embeds";

    // Generation names
    static constexpr std::string_view PastKeyValuesLengthName = "past_key_values_length";
    static constexpr std::string_view PastSequenceLengthName = "past_sequence_length";
    static constexpr std::string_view CurrentSequenceLengthName = "current_sequence_length";
    static constexpr std::string_view TotalSequenceLengthName = "total_sequence_length";
    static constexpr std::string_view CacheIndirectionName = "cache_indirection";
    static constexpr std::string_view AlignmentHeadsName = "alignment_heads";
    static constexpr std::string_view TokenTypeIdsName = "token_type_ids";

    // Encoder names
    static constexpr std::string_view EncoderHiddenStatesName = "encoder_hidden_states";
    static constexpr std::string_view EncoderOutputsName = "encoder_outputs";
    static constexpr std::string_view EncoderAttentionMaskName = "encoder_attention_mask";
  };

  fs::path config_path;  // Path of the config directory

  using NamedString = std::pair<std::string, std::string>;
  struct ProviderOptions {
    std::string name;
    std::vector<NamedString> options;
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
    std::optional<std::string> custom_ops_library;
    // TODO(baijumeswani): Sharing env allocators across sessions leads to crashes on windows and iOS.
    //                     Identify the reason for the crash to enable allocator sharing by default.
    bool use_env_allocators{};
    std::vector<NamedString> config_entries;  // Entries go into OrtSessionOptions::AddConfigEntry

    std::vector<ProviderOptions> provider_options;
    std::vector<std::string> providers;  // List of providers to use at runtime, not persisted in the json currently
    std::optional<GraphOptimizationLevel> graph_optimization_level;
  };

  struct Model {
    std::string type;

    int pad_token_id{};             // The id of the padding token.
    std::vector<int> eos_token_id;  // The end-of-stream tokens (when set as a single value it is converted to a vector with one value).
    int bos_token_id{};             // The id of the beginning-of-stream token.
    int sep_token_id{};             // The id of the separation token.
    int decoder_start_token_id{};   // If an encoder-decoder model starts decoding with a different token than bos, the id of that token.
    int vocab_size{};
    int context_length{};

    struct Encoder {
      std::string filename;
      SessionOptions session_options;

      int hidden_size{};
      int num_attention_heads{};
      int num_hidden_layers{};
      int num_key_value_heads{};
      int head_size{};

      struct Inputs {
        std::string input_ids{Defaults::InputIdsName};
        std::string embeddings{Defaults::InputsEmbedsName};
        std::string attention_mask{Defaults::AttentionMaskName};
        std::string position_ids{Defaults::PositionIdsName};
        std::string audio_features{Defaults::AudioFeaturesName};
      } inputs;

      struct Outputs {
        std::string encoder_outputs{Defaults::EncoderOutputsName};
        std::string hidden_states{Defaults::EncoderHiddenStatesName};
        std::string cross_present_key_names{"present_key_cross_%d"}, cross_present_value_names{"present_value_cross_%d"};
      } outputs;
    } encoder;

    struct Embedding {
      std::string filename;

      struct Inputs {
        std::string input_ids{Defaults::InputIdsName};
        std::string image_features{Defaults::ImageFeaturesName};
        std::string audio_features{Defaults::AudioFeaturesName};
      } inputs;

      struct Outputs {
        std::string embeddings{Defaults::InputsEmbedsName};
      } outputs;
    } embedding;

    struct Vision {
      std::string filename;
      std::string config_filename{"processor_config.json"};
      std::optional<std::string> adapter_filename{};

      struct Inputs {
        std::string pixel_values{Defaults::PixelValuesName};
        std::string image_sizes{Defaults::ImageSizesName};
        std::string attention_mask{Defaults::ImageAttentionMaskName};  // image attention mask
      } inputs;

      struct Outputs {
        std::string image_features{Defaults::ImageFeaturesName};
      } outputs;
    } vision;

    struct Speech {
      std::string filename;
      std::string config_filename{"audio_processor_config.json"};
      std::optional<std::string> adapter_filename{};

      struct Inputs {
        std::string audio_embeds{Defaults::AudioEmbedsName};
        std::string attention_mask{Defaults::AudioAttentionMaskName};
        std::string audio_sizes{Defaults::AudioSizesName};
        std::string audio_projection_mode{Defaults::AudioProjectionModeName};
      } inputs;

      struct Outputs {
        std::string audio_features{Defaults::AudioFeaturesName};
      } outputs;
    } speech;

    struct Decoder {
      std::string filename;
      SessionOptions session_options;

      int hidden_size{};          // Not currently used, potentially useful for embeddings in the future
      int num_attention_heads{};  // Not currently used, potentially useful if num_key_value_heads isn't set
      int num_key_value_heads{};
      int num_hidden_layers{};
      int head_size{};

      struct SlidingWindow {               // Sliding window parameters for models that process input prompt in chunks
        int window_size{};                 // The size of the window to slide over the input prompt
        int pad_value{};                   // The key-value cache padding value to use for the sliding window for inactive tokens
        std::string alignment{"right"};    // The alignment of the window, either "left" or "right"
        bool slide_key_value_cache{true};  // Whether to slide the key-value cache along with the input prompt
        bool slide_inputs{true};           // Whether to slide the input prompt along with the key-value cache
      };
      std::optional<SlidingWindow> sliding_window;

      struct Inputs {
        std::string input_ids{Defaults::InputIdsName};
        std::string embeddings{Defaults::InputsEmbedsName};
        std::string attention_mask{Defaults::AttentionMaskName};
        std::string position_ids{Defaults::PositionIdsName};
        std::string past_key_names{Defaults::PastKeyName};
        std::string past_value_names{Defaults::PastValueName};
        std::string past_names;  // When key/value pairs are combined
        std::string cross_past_key_names, cross_past_value_names;

        std::string past_key_values_length{Defaults::PastKeyValuesLengthName};
        std::string past_sequence_length{Defaults::PastSequenceLengthName};
        std::string current_sequence_length{Defaults::CurrentSequenceLengthName};
        std::string total_sequence_length{Defaults::TotalSequenceLengthName};
        std::string cache_indirection{Defaults::CacheIndirectionName};
        std::string encoder_hidden_states{Defaults::EncoderHiddenStatesName};
        std::string rnn_prev_states{Defaults::RnnStatesPrevName};
        std::string encoder_attention_mask{Defaults::EncoderAttentionMaskName};
      } inputs;

      struct Outputs {
        std::string logits{Defaults::LogitsName};
        std::string present_key_names{Defaults::PresentKeyName};
        std::string present_value_names{Defaults::PresentValueName};
        std::string present_names;  // When key/value pairs are combined
        std::string output_cross_qk_names{"output_cross_qk_%d"};
        std::string rnn_states{Defaults::RnnStatesName};
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
        int reset_session_idx{-1};  // Some models cannot keep all the ort sessions in memory at once due to memory constraints.
                                    // This is the index of the session that needs to be reset during the execution of the current session.
                                    // This is a temporary solution until the QNN driver updates are available.
                                    // Once the driver updates are available, this option will be deprecated.
      };

      std::vector<PipelineModel> pipeline;

    } decoder;

  } model;

  struct Search {
    bool do_sample{};                  // True to do randomized sampling through top_k and top_p, if false, the top logit score is chosen
    int min_length{};                  // Minimum length for final sequence length
    int max_length{};                  // If omitted or 0 in json file, will be set to model.context_length on load
    int batch_size{1};                 // Batch size of inputs. Default is 1.
    int num_beams{1};                  // 1 means no beam search.
    int num_return_sequences{1};       // Number of sequences to return after search. Default is 1.
    float repetition_penalty{1.0f};    // 1.0 means no penalty.
    int top_k{50};                     // Number of highest probability vocabulary tokens to keep for top-k-filtering that will be used by default in the generate method of the model.
    float top_p{};                     // If set to float >0 and <1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    float temperature{1.0f};           // Temperature to control during generation. Default is 1.0.
    bool early_stopping{true};         //  Whether to stop the beam search when at least num_beams sentences are finished per batch or not.
    int no_repeat_ngram_size{};        // Unused param
    float diversity_penalty{};         // Unused param
    float length_penalty{1.0f};        // Exponential penalty to the length that is used with beam-based generation. length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.
    bool past_present_share_buffer{};  // The past/present kv tensors are shared and allocated once to max_length (cuda only)
    int random_seed{-1};               // -1 = Seed with random device, otherwise use value to seed RNG
  } search;

  void AddMapping(const std::string& nominal_name, const std::string& graph_name);
  // Returns graph name and true if the nominal name is found in the mapping
  // otherwise returns the nominal name and false
  std::pair<std::string, bool> GetGraphName(const std::string& nominal_name) const;

  std::unordered_map<std::string, std::string> nominal_names_to_graph_names_;     // Mapping of nominal input/output names to graph input/output names
  std::unordered_map<std::string, std::span<const std::byte>> model_data_spans_;  // Model bytes to support loading a model from memory
};

void SetSearchNumber(Config::Search& search, std::string_view name, double value);
void SetSearchBool(Config::Search& search, std::string_view name, bool value);
void ClearProviders(Config& config);
void SetProviderOption(Config& config, std::string_view provider_name, std::string_view option_name, std::string_view option_value);
void OverlayConfig(Config& config, std::string_view json);
bool IsGraphCaptureEnabled(const Config::SessionOptions& session_options);
bool IsMultiProfileEnabled(const Config::SessionOptions& session_options);

}  // namespace Generators
