#pragma once

#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "input_decoder.h"
#include "ort_genai.h"

namespace microsoft {
namespace aias {

/// @brief SLM Engine class to interact with the GenAI Model
///
/// The SLM Engine class is responsible for loading the GenAI Model and
/// interacting with it to generate responses to user prompts. The class
/// provides a complete() function that takes a user prompt and returns the
/// generated response.
///
/// The class also provides a CreateEngine() function to create a new instance
/// of the SLM Engine and initialize it.
///
/// The class also provides utility functions to convert between the model type
/// enum and string representations.
///
/// The class also provides a struct to hold the runtime performance metrics
/// of the SLM Engine.
///
/// Example Usage:
/// @code
/// // Create a new instance of the SLM Engine
/// auto slm_engine = SLMEngine::CreateEngine("path/to/model", "phi3", true);
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

class SLMEngine {
 public:
  /// @brief Enum to define the supported model types
  enum class SupportedModelType { PHI3,
                                  Llama3_2,
                                  CUSTOM,
                                  UNKNOWN };

  // Utility to convert string to SupportedModelType
  static SupportedModelType StringToModelType(const std::string& model_type) {
    if (model_type == "phi3") {
      return SupportedModelType::PHI3;
    } else if (model_type == "llama3.2") {
      return SupportedModelType::Llama3_2;
    } else if (model_type == "custom") {
      return SupportedModelType::CUSTOM;
    }
    return SupportedModelType::UNKNOWN;
  }
  // Utility to convert SupportedModelType to string
  static std::string ModelTypeToString(SupportedModelType model_type) {
    switch (model_type) {
      case SupportedModelType::PHI3:
        return "phi3";
      case SupportedModelType::Llama3_2:
        return "llama3.2";
      case SupportedModelType::CUSTOM:
        return "custom";
      case SupportedModelType::UNKNOWN:
      default:
        return "unknown";
    }
  }

  /// @brief Creates a new instance of the SLM Engine and initializes it
  /// @param model_path Path to ONNX GenAI Model Directory
  /// @param model_family_name Model family name (phi3 or llama3.2)
  /// @param verbose When set, the LLM Generated output is displayed on stdout
  /// @return New object or null if unsuccessful

  static std::unique_ptr<SLMEngine> CreateEngine(
      const char* model_path, const std::string& model_family_name, bool verbose);

  /// @brief Generates a response to the user prompt using the GenAI Model
  /// @param prompt User prompt to generate response for. The format for this
  /// string is exactly the same as the OpenAI Text Completion API
  /// @return Generated response JSON object as a string.
  ///
  /// The complete() function takes a user prompt and generates a response
  /// using the GenAI Model. The function returns the generated response as a
  /// string.
  ///
  /// The user prompt should be in the following format:
  /// {
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

  SLMEngine(const SLMEngine&) = delete;
  SLMEngine& operator=(const SLMEngine&) = delete;
  static std::string GetVersion() { return std::string(SW_VERSION_NUMBER); }

 private:
  SLMEngine(bool verbose) : m_verbose(verbose) {}

  static uint32_t GetMemoryUsage();

  /// @brief
  /// @param model_path
  /// @return
  bool load_model(const char* model_path, SupportedModelType model_type);

  // Define the Model related prompts
  struct PromptFormat {
    std::string prefix;
    std::string suffix;
  };

  struct PromptFormatDictionary {
    std::string llm_type;
    std::map<InputDecoder::InputParams::Role, PromptFormat> prompt_format;
  };

  bool parse_prompt_format_dict(SupportedModelType model_type,
                                const std::string& json_dict,
                                PromptFormatDictionary& prompt_format_dict);

  /// @brief Given the user input parameters formats by adding the necessary
  /// control strings for the current LLM Model (Phi3)
  /// @param input_params Input parameters to use
  /// @return Complete prompt to be fed to the LLM
  std::string format_input(const InputDecoder::InputParams& input_params);

  std::unique_ptr<OgaModel> m_onnx_model;
  std::unique_ptr<OgaTokenizer> m_tokenizer;
  std::unique_ptr<OgaTokenizerStream> m_tokenizer_stream;
  std::unique_ptr<InputDecoder> m_input_decoder;
  PromptFormatDictionary m_prompt_format;

  bool m_verbose;
  std::ofstream m_llm_input_dbg_stream;
  std::ofstream m_llm_output_dbg_stream;

  // Need a scoped mutex to ensure only one complete() call at a time
  std::mutex m_mutex;
};
}  // namespace aias
}  // namespace microsoft
