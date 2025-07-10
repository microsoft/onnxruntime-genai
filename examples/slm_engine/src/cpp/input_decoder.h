#pragma once

#include <stdint.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace microsoft {
namespace slm_engine {
/// @brief An abstract class defining the interface to various types of
///        input decoder such as OpenAI and so on.
class InputDecoder {
 public:
  /// @brief Creates an instance of a specific decoder that conforms to a
  /// specific API (such as OpenAI)
  /// @param name Name of the API provider
  /// @return An instance of the decoder. If the given decoder is not
  /// supported, a nullptr is returned
  static std::unique_ptr<InputDecoder> CreateDecoder(const std::string& name);

  /// @brief Data structure representing input parameters
  struct InputParams {
    enum class Role { SYSTEM,
                      USER,
                      ASSISTANT,
                      TOOL };

    // Utility function to convert string to Role
    static Role ToRole(const std::string& role) {
      if (role == "system") {
        return Role::SYSTEM;
      } else if (role == "user") {
        return Role::USER;
      } else if (role == "tool") {
        return Role::TOOL;
      } else {
        return Role::ASSISTANT;
      }
    }

    // The first message is the system prompt and subsequent messages are
    // sets of user followed by assistant messages
    std::vector<std::pair<Role, std::string>> Messages;
    // The user prompt is the last message in the sequence
    std::string UserPrompt;
    // The LoRAAdapterName is sent by the client as "model" in the
    // OpenAI API. In our implementation, this is the name of the adapter that will be used
    std::string LoRAAdapterName;
    uint32_t MaxGeneratedTokens;
    std::vector<std::string> StopTokens;
    float Temperature;
    float TopP;
    uint32_t TopK;

    // Function calling support
    std::string ToolsJson;  // Raw tools JSON string from input
    bool HasTools;

    explicit InputParams() {
      MaxGeneratedTokens = 512;
      Temperature = 0.00000000000001f;
      TopK = 50;
      TopP = 1.0f;
      HasTools = false;
    }

    std::string get_messages() {
      std::ostringstream output;
      for (const auto& msg : Messages) {
        switch (msg.first) {
          case Role::SYSTEM:
            output << "{\"role\": \"system\", ";
            break;
          case Role::USER:
            output << "{\"role\": \"user\", ";
            break;
          case Role::TOOL:
            output << "{\"role\": \"tool\", ";
            break;
          case Role::ASSISTANT:
            output << "{\"role\": \"assistant\", ";
            break;
        }
        output << "\"" << msg.second << "\"}\n";
      }
      return output.str();
    }

    std::string to_string() {
      // std::string operator<<(const InputParams& that) {
      std::ostringstream output;
      for (const auto& msg : Messages) {
        output << "Role: ";
        switch (msg.first) {
          case Role::SYSTEM:
            output << "SYSTEM";
            break;
          case Role::USER:
            output << "USER";
            break;
          case Role::TOOL:
            output << "TOOL";
            break;
          case Role::ASSISTANT:
            output << "ASSISTANT";
            break;
        }
        output << " Content: " << msg.second << std::endl;
      }
      return output.str();
    }
  };

  /// @brief Default destructor for needed to clean up derived classes
  virtual ~InputDecoder() = default;

  /// @brief Given the message, extracts various fields from the message
  /// @param message A message encoded in JSON that represents a specific API
  /// that this decoder understands and can decode
  /// @param decoded_params The decoded parameters from the message above
  /// @return True when all the mandatory parameters are specified, False
  /// otherwise
  virtual bool decode(const std::string& message,
                      InputParams& decoded_params) = 0;
};
}  // namespace slm_engine
}  // namespace microsoft