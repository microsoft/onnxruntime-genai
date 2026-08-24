// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Portions of this file consist of AI generated content.
#pragma once
#include "filesystem.h"
#include "provider_options.h"

#include <functional>
#include <memory>

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
    static constexpr std::string_view PastConvName = "past_key_values.%d.conv_state";
    static constexpr std::string_view PastRecurrentName = "past_key_values.%d.recurrent_state";
    static constexpr std::string_view LogitsName = "logits";
    static constexpr std::string_view PresentKeyName = "present.%d.key";
    static constexpr std::string_view PresentValueName = "present.%d.value";
    static constexpr std::string_view PresentConvName = "present.%d.conv_state";
    static constexpr std::string_view PresentRecurrentName = "present.%d.recurrent_state";
    static constexpr std::string_view StateUpdateCaptureCountName = "state_update_capture_count";
    static constexpr std::string_view StateUpdateActiveName = "state_update_active";
    static constexpr std::string_view StateUpdateConvValueName = "state_update.%d.conv_value";
    static constexpr std::string_view StateUpdateRecurrentCapsuleName = "state_update.%d.recurrent_capsule";
    static constexpr std::string_view HiddenStatesName = "hidden_states";
    static constexpr std::string_view RnnStatesName = "rnn_states";
    static constexpr std::string_view RnnStatesPrevName = "rnn_states_prev";
    static constexpr std::string_view CumulativeSequenceLengthsName = "cumulative_sequence_lengths";
    static constexpr std::string_view SequenceLengthsName = "sequence_lengths";
    static constexpr std::string_view PastSequenceLengthsName = "past_sequence_lengths";
    static constexpr std::string_view BlockTableName = "block_table";
    static constexpr std::string_view BlockTableWindowedName = "block_table_windowed";
    static constexpr std::string_view AttentionMetadataName = "attention_metadata";

    // Speech encoder names
    static constexpr std::string_view AudioAttentionMaskName = "audio_attention_mask";
    static constexpr std::string_view AudioSizesName = "audio_sizes";
    static constexpr std::string_view AudioProjectionModeName = "audio_projection_mode";
    static constexpr std::string_view AudioFeaturesName = "audio_features";
    static constexpr std::string_view NumAudioTokens = "num_audio_tokens";
    static constexpr std::string_view OutputCrossQKName = "output_cross_qk_%d";

    // Vision encoder names
    static constexpr std::string_view PixelValuesName = "pixel_values";
    static constexpr std::string_view ImageSizesName = "image_sizes";
    static constexpr std::string_view ImageGridThwName = "image_grid_thw";
    static constexpr std::string_view ImageAttentionMaskName = "image_attention_mask";
    static constexpr std::string_view PixelPositionIdsName = "pixel_position_ids";
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

    // Cache-aware streaming encoder names
    static constexpr std::string_view EncoderInputLengthsName = "length";
    static constexpr std::string_view CacheLastChannelName = "cache_last_channel";
    static constexpr std::string_view CacheLastTimeName = "cache_last_time";
    static constexpr std::string_view CacheLastChannelLenName = "cache_last_channel_len";
    static constexpr std::string_view LangIdName = "lang_id";
    static constexpr std::string_view EncoderOutputLengthsName = "encoded_lengths";
    static constexpr std::string_view CacheLastChannelNextName = "cache_last_channel_next";
    static constexpr std::string_view CacheLastTimeNextName = "cache_last_time_next";
    static constexpr std::string_view CacheLastChannelLenNextName = "cache_last_channel_len_next";

    // Cross present key/value names
    static constexpr std::string_view CrossPresentKeyName = "present_key_cross_%d";
    static constexpr std::string_view CrossPresentValueName = "present_value_cross_%d";

    // Joiner names
    static constexpr std::string_view JoinerEncoderOutputsName = "encoder_outputs";
    static constexpr std::string_view JoinerDecoderOutputsName = "decoder_outputs";
    static constexpr std::string_view JoinerLogitsName = "outputs";

    // Tool-calling and reasoning token ID config field names.
    //   bot = beginning of tool (call), eot = end of tool (call)
    //   bor = beginning of reasoning,   eor = end of reasoning
    static constexpr std::string_view BotTokenIdName = "bot_token_id";
    static constexpr std::string_view EotTokenIdName = "eot_token_id";
    static constexpr std::string_view BorTokenIdName = "bor_token_id";
    static constexpr std::string_view EorTokenIdName = "eor_token_id";
  };

  fs::path config_path;   // Path of the config directory
  fs::path package_root;  // Package root if loaded from a model package, otherwise empty.

  // When loaded from a model package, resolves path-shaped genai_config.json values through
  // ORT's package resolver: a "sha256:<hex>[/tail]" content-addressed shared-asset reference
  // (honoring manifest overrides) or a plain relative path against base_dir. Empty for flat
  // model directories. Captures the OrtModelPackageContext to keep it alive for resolution.
  std::function<fs::path(const fs::path& base_dir, std::string_view value)> package_resolver;

  // Resolves a path-like string from genai_config.json. Empty -> config_path. When loaded
  // from a package, delegates to package_resolver (sha256: shared assets, relative paths);
  // otherwise the value is joined with config_path.
  fs::path ResolvePath(std::string_view value) const;

  using NamedString = Generators::NamedString;
  using DeviceFilteringOptions = Generators::DeviceFilteringOptions;
  using ProviderOptions = Generators::ProviderOptions;

  struct SessionOptions {
    std::optional<int> intra_op_num_threads;
    std::optional<int> inter_op_num_threads;
    std::optional<bool> enable_cpu_mem_arena;
    std::optional<bool> enable_mem_pattern;
    std::optional<std::string> log_id;
    std::optional<int> log_severity_level;
    std::optional<int> log_verbosity_level;
    std::optional<std::string> enable_profiling;
    std::optional<std::string> custom_ops_library;
    std::optional<GraphOptimizationLevel> graph_optimization_level;

    // TODO(baijumeswani): Sharing env allocators across sessions leads to crashes on windows and iOS.
    //                     Identify the reason for the crash to enable allocator sharing by default.

    std::vector<NamedString> config_entries;  // Entries go into OrtSessionOptions::AddConfigEntry
    std::vector<ProviderOptions> provider_options;
    std::vector<std::string> providers;  // List of providers to use at runtime, not persisted in the json currently
  };

  using RunOptions = std::vector<NamedString>;  // Entries go into OrtRunOptions::AddConfigEntry

  struct Model {
    std::string type;

    std::string tokenizer_dir;  // Directory containing tokenizer files. Empty means alongside genai_config.json. Resolved via Config::ResolvePath.

    int pad_token_id{};             // The id of the padding token.
    std::vector<int> eos_token_id;  // The end-of-stream tokens (when set as a single value it is converted to a vector with one value).
    int bos_token_id{};             // The id of the beginning-of-stream token.
    int sep_token_id{};             // The id of the separation token.
    int decoder_start_token_id{};   // If an encoder-decoder model starts decoding with a different token than bos, the id of that token.

    // Multimodal token IDs (used by Qwen-VL, Gemma4, and other VLM/MMM models)
    int image_token_id{};
    int audio_token_id{};
    int boa_token_id{};  // Beginning-of-audio token ID
    int video_token_id{};
    int vision_start_token_id{};

    // Tool-calling and reasoning token IDs.
    // Follows the bos/eos/pad naming convention:
    //   bot = beginning of tool (call), eot = end of tool (call)
    //   bor = beginning of reasoning,   eor = end of reasoning
    std::optional<int> bot_token_id;
    std::optional<int> eot_token_id;
    std::optional<int> bor_token_id;
    std::optional<int> eor_token_id;

    int vocab_size{};
    int context_length{};

    // Streaming ASR / RNNT model parameters
    int num_mels{};
    int fft_size{};
    int hop_length{};
    int win_length{};
    float preemph{};
    float log_eps{};
    float norm_eps{};
    int subsampling_factor{};
    int left_context{};
    int conv_context{};
    int pre_encode_cache_size{};
    int sample_rate{};
    int chunk_samples{};
    int blank_id{};
    int max_symbols_per_step{};

    // Parakeet TDT (Token-and-Duration Transducer) parameters
    int left_context_samples{};
    int right_context_samples{};
    std::vector<int> tdt_durations;  // e.g., {0, 1, 2, 3, 4}

    struct Encoder {
      std::string filename;
      std::optional<SessionOptions> session_options;
      std::optional<RunOptions> run_options;

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
        // Cache-aware streaming encoder I/O names
        std::string input_lengths{Defaults::EncoderInputLengthsName};
        std::string cache_last_channel{Defaults::CacheLastChannelName};
        std::string cache_last_time{Defaults::CacheLastTimeName};
        std::string cache_last_channel_len{Defaults::CacheLastChannelLenName};
        std::string lang_id{Defaults::LangIdName};
      } inputs;

      struct Outputs {
        std::string encoder_outputs{Defaults::EncoderOutputsName};
        std::string hidden_states{Defaults::EncoderHiddenStatesName};
        std::string cross_present_key_names{Defaults::CrossPresentKeyName}, cross_present_value_names{Defaults::CrossPresentValueName};
        // Cache-aware streaming encoder output names
        std::string output_lengths{Defaults::EncoderOutputLengthsName};
        std::string cache_last_channel_next{Defaults::CacheLastChannelNextName};
        std::string cache_last_time_next{Defaults::CacheLastTimeNextName};
        std::string cache_last_channel_len_next{Defaults::CacheLastChannelLenNextName};
      } outputs;
    } encoder;

    struct Embedding {
      std::string filename;
      std::optional<SessionOptions> session_options;
      std::optional<RunOptions> run_options;

      struct Inputs {
        std::string input_ids{Defaults::InputIdsName};
        std::string image_features{Defaults::ImageFeaturesName};
        std::string audio_features{Defaults::AudioFeaturesName};
      } inputs;

      struct Outputs {
        std::string embeddings{Defaults::InputsEmbedsName};
        std::string per_layer_inputs;  // Gemma4: per-layer conditioning from embedding to decoder
      } outputs;
    } embedding;

    struct Vision {
      std::string filename;
      std::optional<SessionOptions> session_options;
      std::optional<RunOptions> run_options;

      // Qwen VL specific vision config values.
      // These are only needed for the QNN 3-stage pipeline (patch_embed → vision_attn → patch_merger),
      // where the C++ runtime computes window attention indices between stages.
      // For standard single-ONNX CUDA/CPU models, windowing is baked into the ONNX graph
      // and these values are unused.
      int spatial_merge_size{2};
      float tokens_per_second{2.0f};
      int num_visual_tokens{0};  // Fixed visual tokens per image; must be > 0 for videochat_flash_qwen
      int patch_size{14};        // Qwen2.5-VL uses 14, Qwen3-VL uses 16
      int window_size{0};        // Used by CalculateWindowIndex() in QNN pipeline only.
                                 // 0 = auto-compute as patch_size * spatial_merge_size * 2
                                 // Qwen2.5-VL default: 56 (14*4), Qwen3-VL default: 64 (16*4)

      std::string config_filename{"processor_config.json"};
      std::optional<std::string> adapter_filename{};

      // Vision pipeline support (patch embed -> vision attn -> patch merger)
      struct PipelineModel {
        std::string filename;
        std::optional<SessionOptions> session_options;
        std::optional<RunOptions> run_options;
        std::string model_id;              // Identifier used to link outputs to subsequent stages
        std::vector<std::string> inputs;   // Graph input names
        std::vector<std::string> outputs;  // Graph output names
        bool run_on_cpu{false};            // If true force CPU EP when multiple EPs are configured
      };
      std::vector<PipelineModel> pipeline;  // Ordered pipeline models

      struct Inputs {
        std::string pixel_values{Defaults::PixelValuesName};
        std::string pixel_position_ids{Defaults::PixelPositionIdsName};
        std::string image_sizes{Defaults::ImageSizesName};
        std::string image_grid_thw{Defaults::ImageSizesName};          // Qwen2.5-VL uses image_grid_thw, defaults to image_sizes
        std::string attention_mask{Defaults::ImageAttentionMaskName};  // image attention mask
      } inputs;

      struct Outputs {
        std::string image_features{Defaults::ImageFeaturesName};
      } outputs;
    } vision;

    struct Speech {
      std::string filename;
      std::optional<SessionOptions> session_options;
      std::optional<RunOptions> run_options;

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

    struct Joiner {
      std::string filename;
      std::optional<SessionOptions> session_options;
      std::optional<RunOptions> run_options;

      struct Inputs {
        std::string encoder_outputs{Defaults::JoinerEncoderOutputsName};
        std::string decoder_outputs{Defaults::JoinerDecoderOutputsName};
      } inputs;

      struct Outputs {
        std::string logits{Defaults::JoinerLogitsName};
      } outputs;
    } joiner;

    struct VAD {
      std::string filename;
      float threshold{0.5f};
      int silence_duration_ms{500};
      int prefix_padding_ms{300};
      std::optional<SessionOptions> session_options;
      std::optional<RunOptions> run_options;
    } vad;

    struct SharedInitializer {
      std::string name;
      std::string data_file;
      std::string offset;
      std::string length;
      int data_type{};
      std::vector<int64_t> shape;
    };

    struct Decoder {
      std::string filename;
      SessionOptions session_options;
      std::optional<RunOptions> run_options;
      std::vector<SharedInitializer> shared_initializers;

      int hidden_size{};          // Not currently used, potentially useful for embeddings in the future
      int num_attention_heads{};  // Not currently used, potentially useful if num_key_value_heads isn't set
      int num_key_value_heads{};
      int num_hidden_layers{};
      int head_size{};
      // Compact per-token state transitions a forward captures so a partial accept can be replayed
      // without rerunning the model. 0 means the model does not export the state-update bindings.
      int state_update_capacity{};

      // Hybrid SSM+Attention (LFM2) parameters
      std::vector<std::string> layer_types;  // Per-layer type: "conv" or "full_attention"
      int conv_cache_size{};                 // Convolution cache width (conv_L_cache from HF config)

      struct SlidingWindow {               // Sliding window parameters for models that process input prompt in chunks
        int window_size{};                 // The size of the window to slide over the input prompt
        int pad_value{};                   // The key-value cache padding value to use for the sliding window for inactive tokens
        std::string alignment{"right"};    // The alignment of the window, either "left" or "right"
        bool slide_key_value_cache{true};  // Whether to slide the key-value cache along with the input prompt
        bool slide_inputs{true};           // Whether to slide the input prompt along with the key-value cache
        std::vector<int> layers;           // Layer indices that use sliding window attention (for models with alternating patterns)
        // Extra key-value cache positions allocated beyond window_size on execution providers that
        // own eviction themselves (CUDA and CPU GroupQueryAttention with sliding_window_cache=1).
        // 0 means "use the EP default": 0 for CUDA (optimal — launch overhead dominates, attention
        // is O(W) regardless of C), 16 for CPU (optimal — amortises O(C) shift traffic at W+16).
        // Set explicitly to cover a whole prefill chunk or to tune the amortisation tradeoff.
        int cache_slack{0};
      };
      std::optional<SlidingWindow> sliding_window;

      enum class StateGroupKind {
        Invalid,
        PagedKeyValue,
        FixedConv,
        FixedRecurrent,
      };

      enum class StateUpdateKind {
        Invalid,
        CausalConv,
        GatedDeltaNet,
      };

      static constexpr int MaxStateUpdateCapacity = 8;

      struct StateUpdate {
        int capacity{};
        bool enabled{true};
        int key_head_count{};
      };

      struct StateGroup {
        StateGroupKind kind{StateGroupKind::Invalid};
        std::vector<int> layer_ids;
        std::optional<StateUpdate> state_update;
      };

      // Absence preserves the legacy dense, sequential paged-KV contract.
      std::optional<std::vector<StateGroup>> state_groups;

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
        std::string cumulative_sequence_lengths{Defaults::CumulativeSequenceLengthsName};
        std::string past_sequence_lengths{Defaults::PastSequenceLengthsName};
        std::string block_table{Defaults::BlockTableName};
        // Second block table read by the sliding-window layers. Their cache is a ring of blocks, so
        // this table repeats a request's few blocks across the columns instead of listing distinct
        // ones. Empty when the model has no windowed paged layers.
        std::string block_table_windowed{Defaults::BlockTableWindowedName};
        std::string attention_metadata{Defaults::AttentionMetadataName};
        std::string past_conv_names{Defaults::PastConvName};  // Conv cache input name template (LFM2)
        std::string past_recurrent_names{Defaults::PastRecurrentName};
        std::string state_update_capture_count{Defaults::StateUpdateCaptureCountName};  // Per-sequence capture count
        std::string state_update_active{Defaults::StateUpdateActiveName};               // Capture enable flag

        // Last hidden-state input (e.g. the MTP head consumes the main model's hidden state).
        // Empty unless the model graph takes a hidden_states input.
        std::string hidden_states;

        // RNNT decoder inputs
        std::string targets;
        std::string lstm_hidden_state;
        std::string lstm_cell_state;

        // Gemma4 per-layer inputs (e.g. per-layer embeddings from embedding model)
        std::string per_layer_inputs;

        // Parakeet TDT decoder (prediction network) extra inputs
        std::string targets_length;
      } inputs;

      struct Outputs {
        std::string logits{Defaults::LogitsName};
        std::string present_key_names{Defaults::PresentKeyName};
        std::string present_value_names{Defaults::PresentValueName};
        std::string present_names;  // When key/value pairs are combined
        std::string output_cross_qk_names{Defaults::OutputCrossQKName};
        std::string rnn_states{Defaults::RnnStatesName};
        std::string present_conv_names{Defaults::PresentConvName};  // Conv cache output name template (LFM2)
        std::string present_recurrent_names{Defaults::PresentRecurrentName};
        std::string state_update_conv_value_names{Defaults::StateUpdateConvValueName};
        std::string state_update_recurrent_capsule_names{Defaults::StateUpdateRecurrentCapsuleName};
        std::string hidden_states;  // Last hidden state output (when exported with include_hidden_states; e.g. fed to the MTP head)
        // Residual streams tapped at model.dflash2.aux_hidden_state_layers, concatenated on the
        // last axis. Empty unless the model was exported with aux_hidden_state_layers.
        std::string aux_hidden_states;

        // RNNT decoder outputs
        std::string outputs;
        std::string lstm_hidden_state;
        std::string lstm_cell_state;

        // Parakeet TDT decoder (prediction network) extra outputs
        std::string outputs_length;
      } outputs;

      struct PipelineModel {
        std::string filename;
        std::optional<SessionOptions> session_options;
        std::optional<RunOptions> run_options;

        std::string model_id;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        std::unordered_map<std::string, std::string> output_names_forwarder;
        bool run_on_prompt{true};
        bool run_on_token_gen{true};
        bool is_lm_head{false};
        bool inherit_session_options{false};  // If true, the top level (decoder) session options are used as the
                                              // base for this component's session options, which are then overlaid
                                              // on top of them.
        int reset_session_idx{-1};            // Some models cannot keep all the ort sessions in memory at once due to memory constraints.
                                              // This is the index of the session that needs to be reset during the execution of the current session.
                                              // This is a temporary solution until the QNN driver updates are available.
                                              // Once the driver updates are available, this option will be deprecated.
      };

      std::vector<PipelineModel> pipeline;

    } decoder;

    // Multi-token-prediction (MTP) self-speculative head metadata (e.g. Qwen3.6). The caller
    // loads the head as a separate Model; MtpGenerator uses this block to map the main model's
    // hidden-state output and the head's feedback output.
    struct Mtp {
      std::string filename;  // e.g. "mtp.onnx"; used by model packaging/building tools
      std::optional<SessionOptions> session_options;
      std::optional<RunOptions> run_options;
      // Empty intentionally means the head does not share the main decoder's initializers.
      std::vector<SharedInitializer> shared_initializers;

      int num_hidden_layers{1};  // The MTP head has a single decoder layer.
      int num_key_value_heads{};
      int head_size{};

      // Name of the main decoder's hidden-states output that feeds the MTP head.
      // The main model must be exported with this output exposed (include_hidden_states).
      std::string main_hidden_states{Defaults::HiddenStatesName};

      struct Inputs {
        std::string input_ids{Defaults::InputIdsName};
        std::string hidden_states{Defaults::HiddenStatesName};
        std::string attention_mask{Defaults::AttentionMaskName};
        std::string position_ids{Defaults::PositionIdsName};
        std::string past_key_names{Defaults::PastKeyName};
        std::string past_value_names{Defaults::PastValueName};
      } inputs;

      struct Outputs {
        std::string logits{Defaults::LogitsName};
        std::string hidden_states{"hidden_states_out"};
        std::string present_key_names{Defaults::PresentKeyName};
        std::string present_value_names{Defaults::PresentValueName};
      } outputs;
    } mtp;

    // DFlash 2 block-drafter metadata. Unlike MTP the drafter is not decoder-shaped: it reads the
    // main model's auxiliary hidden states, predicts a whole block of tokens at once, and returns
    // a candidate lattice (top-k ids per slot plus the pairwise edge scores) that the engine walks
    // greedily. The Engine drives its session directly rather than through a Model.
    struct Dflash2 {
      std::string filename;  // e.g. "dflash2.onnx"
      std::optional<SessionOptions> session_options;
      std::optional<RunOptions> run_options;
      std::vector<SharedInitializer> shared_initializers;

      int num_hidden_layers{};
      int num_key_value_heads{};
      int head_size{};
      int block_size{};        // Query rows per request: the anchor token plus one mask per draft.
      int num_draft_tokens{};  // block_size - 1
      int selector_top_k{};
      int mask_token_id{};
      int sliding_window{-1};
      std::vector<int> aux_hidden_state_layers;

      // Name of the main decoder's auxiliary hidden-states output that feeds the drafter.
      std::string main_aux_hidden_states{"aux_hidden_states"};

      struct Inputs {
        std::string aux_hidden_states{"aux_hidden_states"};
        std::string input_ids{Defaults::InputIdsName};
        std::string q_row_map{"q_row_map"};
        std::string qkv_row_map{"qkv_row_map"};
        std::string block_row_index{"block_row_index"};
        std::string cumulative_sequence_lengths{Defaults::CumulativeSequenceLengthsName};
        std::string past_sequence_lengths{Defaults::PastSequenceLengthsName};
        std::string block_table{Defaults::BlockTableName};
        std::string attention_metadata{Defaults::AttentionMetadataName};
        std::string past_key_names{Defaults::PastKeyName};
        std::string past_value_names{Defaults::PastValueName};
      } inputs;

      struct Outputs {
        std::string candidate_ids{"draft_candidate_ids"};
        std::string scores{"draft_scores"};
        std::string present_key_names{Defaults::PresentKeyName};
        std::string present_value_names{Defaults::PresentValueName};
      } outputs;
    } dflash2;

    std::optional<Decoder> draft;

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
    bool early_stopping{true};         // Whether to stop the beam search when at least num_beams sentences are finished per batch or not.
    int no_repeat_ngram_size{};        // If > 0, no n-gram of this size may repeat in the generated sequence. 0 disables.
    float diversity_penalty{};         // Unused param
    float length_penalty{1.0f};        // Exponential penalty to the length that is used with beam-based generation. length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.
    bool past_present_share_buffer{};  // The past/present kv tensors are shared and allocated once to max_length (cuda only)
    int random_seed{-1};               // -1 = Seed with random device, otherwise use value to seed RNG
    std::optional<size_t> chunk_size;  // Chunk size for prefill chunking during context processing. If present, chunking is enabled with the chunk size > 0.
    float blank_penalty{};             // Penalty applied to blank token logits in CTC/RNNT decoding. Default 0 means no penalty.
  } search;

  struct Speculative {
    // Fixed proposal width when min_adaptive_k is 0. Four conservatively amortizes target
    // verification without excessive draft work; the best value depends on acceptance and EP cost.
    int max_draft_tokens{4};
    int ngram_size{};             // 0 disables n-gram decoding; 2-16 matches the last N-1 tokens.
    bool ngram_chained_lookup{};  // Refill the proposal by repeatedly looking up synthetic context.
    // 0 disables adaptation. Values 1-16 enable it and set the starting width and floor;
    // adjacent-width probes may grow the effective width up to the hard limit of 16.
    int min_adaptive_k{};
    bool cooldown{};  // Skip one speculative attempt after three zero-accept rounds.
  } speculative;

  struct Engine {
    struct DynamicBatching {
      size_t block_size{256};                       // Total number of slots per block.
      std::optional<size_t> num_blocks;             // Total number of blocks per layer.
      std::optional<float> gpu_utilization_factor;  // Fraction of free GPU memory to use for key-value cache.
      size_t max_batch_size{16};                    // Maximum batch size for dynamically batching requests.
      size_t max_scheduled_tokens{2048};            // Maximum tokens in one dynamically batched model run.
    };
    std::optional<DynamicBatching> dynamic_batching;  // Dynamic batching settings

    struct StaticBatching {
      size_t max_batch_size{4};  // Maximum batch size for static batching
    };
    std::optional<StaticBatching> static_batching;  // Static batching settings

    // Runtime-only capability flag, never parsed from genai_config.json. The Engine sets it on
    // every decoder whose packed hidden states it consumes: the target decoder feeds the MTP head,
    // and the head feeds the next link of a chained draft. Models that merely export hidden states
    // leave it false so an ordinary step does not pay for the extra output.
    bool hidden_states_output_required{};
  } engine;  // Engine settings

  void AddMapping(const std::string& nominal_name, const std::string& graph_name);
  // Returns graph name and true if the nominal name is found in the mapping
  // otherwise returns the nominal name and false
  std::pair<std::string, bool> GetGraphName(const std::string& nominal_name) const;

  std::unordered_map<std::string, std::string> nominal_names_to_graph_names_;     // Mapping of nominal input/output names to graph input/output names
  std::unordered_map<std::string, std::span<const std::byte>> model_data_spans_;  // Model bytes to support loading a model from memory
};

void SetSearchNumber(Config::Search& search, std::string_view name, double value);
void SetSearchBool(Config::Search& search, std::string_view name, bool value);
void SetSpeculativeNumber(Config::Speculative& speculative, std::string_view name, double value);
void SetSpeculativeBool(Config::Speculative& speculative, std::string_view name, bool value);
// Build the decoder-model view used to run model.mtp as an internal session. The projection keeps
// the main model's device, batching and paged-attention contract, but replaces model-specific
// decoder state with the single MTP attention layer.
std::unique_ptr<Config> CreateMtpDecoderConfig(const Config& config);
void ClearProviders(Config& config);
void SetProviderOption(Config& config, std::string_view provider_name, std::string_view option_name, std::string_view option_value);
void OverlayConfig(Config& config, std::string_view json);
int SafeDoubleToInt(double x, std::string_view name);

// Normalizes historical casings, short aliases, and full ORT names (e.g.
// "CUDAExecutionProvider") to the canonical dispatch-table name; unknown names pass through.
std::string_view NormalizeProviderName(std::string_view name);
bool IsGraphCaptureEnabled(const Config::SessionOptions& session_options);
bool IsMultiProfileEnabled(const Config::SessionOptions& session_options);

void SetDecoderProviderOptionsHardwareDeviceType(Config& config, std::string_view provider_name, std::string_view hardware_device_type);
void SetDecoderProviderOptionsHardwareDeviceId(Config& config, std::string_view provider_name, uint32_t hardware_device_id);
void SetDecoderProviderOptionsHardwareVendorId(Config& config, std::string_view provider_name, uint32_t hardware_vendor_id);
void ClearDecoderProviderOptionsHardwareDeviceType(Config& config, std::string_view provider_name);
void ClearDecoderProviderOptionsHardwareDeviceId(Config& config, std::string_view provider_name);
void ClearDecoderProviderOptionsHardwareVendorId(Config& config, std::string_view provider_name);

}  // namespace Generators
