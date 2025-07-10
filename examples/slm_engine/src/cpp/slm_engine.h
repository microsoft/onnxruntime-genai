#pragma once

#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <functional>

#if defined(_WIN32) || defined(_WIN64)
#include <string.h>
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#else
#include <strings.h>
#endif

#include "input_decoder.h"
#include "ort_genai.h"

#ifdef _WIN32
#ifdef BUILDING_SLM_ENGINE
#define SLM_ENGINE_EXPORT __declspec(dllexport)
#else
#define SLM_ENGINE_EXPORT __declspec(dllimport)
#endif
#else
// To make symbols visible on macOS/iOS
#ifdef __APPLE__
#define SLM_ENGINE_EXPORT __attribute__((visibility("default")))
#else
#define SLM_ENGINE_EXPORT
#endif
#endif

namespace microsoft {
namespace slm_engine {

/// @brief SLM Engine class to interact with the GenAI Model
///
/// The SLM Engine class is responsible for loading the GenAI Model and
/// interacting with it to generate responses to user prompts. The class
/// provides a complete() function that takes a user prompt and returns the
/// generated response.
///
/// The class provides a Create() function to create a new instance
/// of the SLM Engine and initialize it.
///
/// The class also provides a struct to hold the runtime performance metrics
/// of the SLM Engine.
///
/// Example Usage:
/// @code
/// // Create a new instance of the SLM Engine
/// auto slm_engine = SLMEngine::Create("path/to/model", true);
/// if (!slm_engine) {
///     std::cout << "Error creating the SLM Engine" << std::endl;
///     return -1;
/// }
///
/// // Generate a response to a user prompt
/// std::string prompt =
///     "{\"role\": \"user\", \"content\": \"Hello, how are you?\"}";
/// std::string response = slm_engine->complete(prompt.c_str());
/// std::cout << "Response: " << response["response"]["answer"] << std::endl;
/// @endcode
///

class SLM_ENGINE_EXPORT SLMEngine {
 public:
  /// @brief Status of the operation
  /// @param code True if the operation was successful, false otherwise
  /// @param message Message providing additional information about the status
  /// @note The message is empty if the operation was successful
  struct SLM_ENGINE_EXPORT Status {
    bool code;
    std::string message;
  };

  /// @brief Struct to represent a function tool parameter
  struct SLM_ENGINE_EXPORT FunctionParameter {
    std::string description;
    std::string type;
    std::string default_value;

    FunctionParameter() : type("string") {}
    FunctionParameter(const std::string& desc, const std::string& param_type = "string",
                      const std::string& default_val = "")
        : description(desc), type(param_type), default_value(default_val) {}
  };

  /// @brief Struct to represent a function tool
  struct SLM_ENGINE_EXPORT FunctionTool {
    std::string name;
    std::string description;
    std::map<std::string, FunctionParameter> parameters;

    FunctionTool() = default;
    FunctionTool(const std::string& tool_name, const std::string& tool_desc)
        : name(tool_name), description(tool_desc) {}
  };

  /// @brief Function calling generation options
  struct SLM_ENGINE_EXPORT FunctionCallOptions {
    std::vector<FunctionTool> tools;
    bool force_function_call;

    FunctionCallOptions() : force_function_call(false) {}
  };

  /// @brief Single function call information
  struct SLM_ENGINE_EXPORT FunctionCall {
    std::string function_name;
    std::string parameters_json;

    FunctionCall() = default;
    FunctionCall(const std::string& name, const std::string& params)
        : function_name(name), parameters_json(params) {}
  };

  /// @brief Function call result structure supporting multiple calls
  struct SLM_ENGINE_EXPORT FunctionCallResult {
    std::vector<FunctionCall> function_calls;
    bool is_function_call;
    std::string text_response;

    FunctionCallResult() : is_function_call(false) {}

    // Legacy compatibility methods
    std::string function_name() const {
      return function_calls.empty() ? "" : function_calls[0].function_name;
    }

    std::string parameters_json() const {
      return function_calls.empty() ? "" : function_calls[0].parameters_json;
    }
  };

  /// @brief Get the version of the SLM Engine
  /// @param slm_version SLM Engine version
  /// @param ortga_version ORT GenAI version
  /// @param ort_version ORT version
  static void GetVersion(
      std::string& slm_version,
      std::string& ortga_version,
      std::string& ort_version);

  /// @brief Creates a new instance of the SLM Engine and initializes it
  /// @param model_path Path to ONNX GenAI Model Directory
  /// @param verbose When set, the LLM Generated output is displayed on stdout
  /// @return New object or null if unsuccessful
  static std::unique_ptr<SLMEngine> Create(
      const char* model_path, bool verbose);

  struct SLM_ENGINE_EXPORT LoRAAdapter {
    std::string name;
    std::string adapter_path;
    explicit LoRAAdapter(const std::string& name,
                         const std::string& adapter_path)
        : name(name), adapter_path(adapter_path) {}
    // Copy constructor
    LoRAAdapter(const LoRAAdapter& other)
        : name(other.name), adapter_path(other.adapter_path) {}
  };

  /// @brief Create SLMEngine, loads the model and adapters
  /// @param model_path Path to ONNX GenAI Model Directory
  /// @param adapters List of LoRA adapters in NN format
  /// @param verbose When set to true, the LLM Generated output is displayed on stdout
  /// @param status_msg Provides information about cause of failure to load model or
  ///         adapters when applicable.
  /// @return A new object or nullptr if unsuccessful. When unsuccessful, status_msg
  ///         will contain information about the cause of failure.
  static std::unique_ptr<SLMEngine> Create(
      const char* model_path,
      const std::vector<LoRAAdapter> adapters,
      bool verbose,
      Status& status_msg);

  /// @brief  Get the current memory usage of the SLM Engine
  /// @return Current memory usage in MB
  static uint32_t GetMemoryUsage();

  /// @brief Get the model family from the model path
  /// @param model_path Path to the model file
  /// @return Model family as a string
  static std::string GetModelFamily(const std::string& model_path);

  std::string get_model_path() const { return m_model_path; }
  std::vector<LoRAAdapter> get_adapter_list();

  /// @brief Generates a response to the user prompt using the GenAI Model
  /// @param prompt User prompt to generate response for. The format for this
  /// string is exactly the same as the OpenAI Text Completion API
  /// @return Generated response JSON object as a string.
  ///
  /// The complete() function takes a user prompt and generates a response
  /// using the GenAI Model. The function returns the generated response as a
  /// string.
  ///
  ///  In SLM engine - the model is loaded at the create time. So we are re-purposing
  ///  the OpenAI API "model" parameter to indicate name as the LoRA adapter if
  ///  the adapter was loaded. If this parameter is not provided, the default
  ///  then the base model without the apdapter is used.
  ///
  /// The user prompt should be in the following format:
  /// {
  ///     "model": "LoRA adapter name",
  ///     "messages": [
  ///         {
  ///             "role": "system",
  ///             "content": "System message"
  ///         },
  ///         {
  ///             "role": "user",
  ///             "content": "User message"
  ///         }
  ///     ],
  ///     "temperature": 0.0,
  ///     "stop": ["Stop token 1", "Stop token 2"],
  ///     "max_tokens": 250
  /// }
  ///
  /// Format of the response string when the call succeeds
  /// {
  ///     "status": "success",
  ///     "response": {
  ///         "answer": "Generated response",
  ///         "kpi": {
  ///             "prompt_toks": 10,
  ///             "response_toks": 20,
  ///             "ttft": 1000,
  ///             "tok_rate": 10,
  ///             "total_time": 10000,
  ///             "memory_usage": 100
  ///         }
  ///     }
  /// }
  ///
  /// Format of the response string when the call fails
  /// {
  ///     "status": "error",
  ///     "message": "Error message"
  /// }
  ///
  /// @note To support multi-turn conversations, the history should be
  /// maintained by the caller and submitted just like how the OpenAI API
  /// works
  std::string complete(const char* prompt);

  /// @brief Struct to hold the runtime performance metrics of the SLM Engine
  /// @param PromptTokenCount Number of tokens in the prompt
  /// @param TimeToFirstToken Time taken to generate the first token (milliseconds)
  /// @param GeneratedTokenCount Number of tokens generated
  /// @param TokenRate Number of tokens generated per second
  /// @param TotalTime Total time taken to generate the response (milliseconds)
  /// @param LoRAAdapterSwitchTime Time taken to "SetActiveAdapter" (milliseconds)
  /// @param CurrentMemoryUsed Current memory used by the SLM Engine
  struct RuntimePerf {
    uint32_t PromptTokenCount;
    uint32_t TimeToFirstToken;
    uint32_t GeneratedTokenCount;
    uint32_t TokenRate;
    uint32_t TotalTime;
    uint32_t GenerationTimePerToken;
    uint32_t CurrentMemoryUsed;
    RuntimePerf()
        : PromptTokenCount(0),
          TimeToFirstToken(0),
          GeneratedTokenCount(0),
          TokenRate(0),
          TotalTime(0),
          GenerationTimePerToken(0),
          CurrentMemoryUsed(0) {}
    RuntimePerf(const RuntimePerf& other)
        : PromptTokenCount(other.PromptTokenCount),
          TimeToFirstToken(other.TimeToFirstToken),
          GeneratedTokenCount(other.GeneratedTokenCount),
          TokenRate(other.TokenRate),
          TotalTime(other.TotalTime),
          GenerationTimePerToken(other.GenerationTimePerToken),
          CurrentMemoryUsed(other.CurrentMemoryUsed) {}
    RuntimePerf& operator=(const RuntimePerf& other) = delete;
    RuntimePerf(RuntimePerf&& other) = delete;
    RuntimePerf& operator=(RuntimePerf&& other) = delete;
  };

  /// @brief Struct to hold the generation options for the GenAI Model
  /// @param MaxGeneratedTokens Maximum number of tokens to generate
  /// @param TopK Top K sampling
  /// @param TopP Top P sampling
  /// @param Temperature Temperature for sampling
  struct GenerationOptions {
    uint32_t MaxGeneratedTokens;
    uint32_t TopK;
    float TopP;
    float Temperature;
    explicit GenerationOptions() {
      MaxGeneratedTokens = 2048;
      Temperature = 0.00000000000001f;
      TopK = 50;
      TopP = 0.1f;
    }
  };

  /// @brief Asks the GenAI Model for a response
  /// @param formatted_prompt Formatted prompt to generate response for
  /// @param generation_options Generation options for the GenAI Model
  /// @param response_str Generated response
  /// @param kpi Runtime performance metrics of the SLM Engine
  SLMEngine::Status generate(
      const std::string& formatted_prompt,
      const GenerationOptions& generation_options,
      std::string& response_str,
      RuntimePerf& kpi);

  /// @brief Asks the GenAI Model for a response using the given LoRA adapter
  /// @param adapter_name Name of the LoRA adapter to use
  /// @param formatted_prompt Formatted prompt to generate response for
  /// @param generation_options Generation options for the GenAI Model
  /// @param response_str Generated response
  /// @param kpi Runtime performance metrics of the SLM Engine
  Status generate(
      const std::string& adapter_name,
      const std::string& formatted_prompt,
      const GenerationOptions& generation_options,
      std::string& response_str,
      RuntimePerf& kpi);

  /// @brief Generate response with function calling support
  /// @param formatted_prompt Formatted prompt
  /// @param generation_options Generation options
  /// @param function_options Function calling options
  /// @param response_str Generated response
  /// @param function_result Function call result if any
  /// @param kpi Runtime performance metrics
  /// @return Status of the operation
  Status generate_with_functions(
      const std::string& formatted_prompt,
      const GenerationOptions& generation_options,
      const FunctionCallOptions& function_options,
      std::string& response_str,
      FunctionCallResult& function_result,
      RuntimePerf& kpi);

  /// @brief Generate response with function calling using adapter
  /// @param adapter_name Name of the LoRA adapter
  /// @param formatted_prompt Formatted prompt
  /// @param generation_options Generation options
  /// @param function_options Function calling options
  /// @param response_str Generated response
  /// @param function_result Function call result if any
  /// @param kpi Runtime performance metrics
  /// @return Status of the operation
  Status generate_with_functions(
      const std::string& adapter_name,
      const std::string& formatted_prompt,
      const GenerationOptions& generation_options,
      const FunctionCallOptions& function_options,
      std::string& response_str,
      FunctionCallResult& function_result,
      RuntimePerf& kpi);

  /// @brief Given a system and an user prompt, formats the prompt by adding the
  /// necessary control strings for the current LLM Model
  /// @param system_prompt
  /// @param user_prompt
  /// @return
  std::string format_prompt(
      const std::string& system_prompt,
      const std::string& user_prompt);

  /// @brief Parse tools from JSON string
  /// @param tools_json JSON string containing tools definition
  /// @return Vector of FunctionTool objects
  std::vector<FunctionTool> parse_tools_from_json(const std::string& tools_json);

  SLMEngine(const SLMEngine&) = delete;
  SLMEngine& operator=(const SLMEngine&) = delete;

  /// @brief Destructor for the SLM Engine
  ~SLMEngine();

 private:
  SLMEngine(bool verbose) : m_verbose(verbose) {}

  /// @brief
  /// @param model_path
  /// @return
  bool load_model(const char* model_path);

  /// @brief Given the user input parameters formats by adding the necessary
  /// control strings for the current LLM Model (Phi3)
  /// @param input_params Input parameters to use
  /// @return Complete prompt to be fed to the LLM
  std::string format_input(const InputDecoder::InputParams& input_params);

  /// @brief Format input with tools for function calling
  /// @param input_params Input parameters with tools
  /// @return Complete prompt with tools information
  std::string format_input_with_tools(const InputDecoder::InputParams& input_params);

  // Define the Model related prompts
  struct PromptFormat {
    std::string prefix;
    std::string suffix;
  };

  struct PromptFormatDictionary {
    std::string llm_type;
    std::map<InputDecoder::InputParams::Role, PromptFormat> prompt_format;
  };

  /// @brief Enum to define the supported model types
  enum class SupportedModelType { PHI,
                                  Llama,
                                  Qwen,
                                  CUSTOM,
                                  UNKNOWN };

  /// @param model_type String representation of the model type
  /// @return SupportedModelType enum value
  /// @note The string comparison is case-insensitive
  static SupportedModelType StringToModelType(const std::string& model_type);

  /// @brief  Converts SupportedModelType enum to string
  /// @param model_type SupportedModelType enum value
  /// @note The string representation is in lowercase
  static std::string ModelTypeToString(SupportedModelType model_type);

  bool parse_prompt_format_dict(SupportedModelType model_type,
                                const std::string& json_dict,
                                PromptFormatDictionary& prompt_format_dict);

  std::unique_ptr<OgaGenerator> create_generator(
      const std::string& formatted_prompt,
      const GenerationOptions& generation_options,
      uint32_t& time_to_prefill);

  /// @brief Generate the response using the GenAI Model
  /// @param formatted_prompt Formatted prompt to generate response for
  /// @param generator OgaGenerator object to use for generation
  /// @param generation_callback Callback function to use for generation
  /// @param response_str Generated response
  /// @param kpi Runtime performance metrics of the SLM Engine
  /// @return Status of the operation
  /// @note The generation_callback function (if provided) is called for each token generated
  Status generate(
      OgaGenerator* generator,
      std::function<bool(const std::string&, OgaTensor* logits)> generation_callback,
      std::string& response_str,
      RuntimePerf& kpi);

  /// @brief Create Lark grammar for function calling
  /// @param tools List of function tools
  /// @param prompt_tool_input Output parameter for tool input prompt
  /// @param grammar_input Output parameter for grammar input
  /// @return True if successful
  bool create_lark_grammar(const std::vector<FunctionTool>& tools,
                           std::string& prompt_tool_input,
                           std::string& grammar_input);

  /// @brief Convert tool to grammar input format
  /// @param tool Function tool to convert
  /// @return JSON schema for the tool
  std::string convert_tool_to_grammar_input(const FunctionTool& tool);

  /// @brief Create tool input prompt
  /// @param tools List of function tools
  /// @return Tool input prompt string
  std::string create_prompt_tool_input(const std::vector<FunctionTool>& tools);

  /// @brief Parse function call from generated text
  /// @param generated_text Generated text to parse
  /// @param function_result Output function call result
  /// @return True if function call was found and parsed
  bool parse_function_call(const std::string& generated_text,
                           FunctionCallResult& function_result);

  /// @brief Create generator with function calling support
  /// @param formatted_prompt Formatted prompt
  /// @param generation_options Generation options
  /// @param function_options Function calling options
  /// @param time_to_prefill Output time to prefill
  /// @return Generator object
  std::unique_ptr<OgaGenerator> create_function_generator(
      const std::string& formatted_prompt,
      const GenerationOptions& generation_options,
      const FunctionCallOptions& function_options,
      uint32_t& time_to_prefill);

  std::unique_ptr<OgaModel> m_onnx_model;
  std::unique_ptr<OgaAdapters> m_adapters;
  std::unique_ptr<OgaTokenizer> m_tokenizer;
  std::unique_ptr<OgaTokenizerStream> m_tokenizer_stream;
  std::unique_ptr<InputDecoder> m_input_decoder;
  PromptFormatDictionary m_prompt_format;

  std::vector<LoRAAdapter> m_adapters_list;
  std::string m_model_path;
  std::string m_model_type;

  bool m_verbose;
  std::ofstream m_llm_input_dbg_stream;
  std::ofstream m_llm_output_dbg_stream;

  // Need a scoped mutex to ensure only one complete() call at a time
  std::mutex m_mutex;
};
}  // namespace slm_engine
}  // namespace microsoft
