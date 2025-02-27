#include "input_decoder.h"

#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#define RED "\033[31;1m"
#define GREEN "\033[32m"
#define CLEAR "\033[0m"

using namespace std;

namespace microsoft {
namespace slm_engine {

// clang-format off
// OpenAI API example
// {
//     "messages": [
// 		{
// 			"role": "system",
// 			"content": "You are an in car virtual assistant that maps user's inputs to the corresponding function call in the vehicle. You must respond with only a JSON object matching the following schema: {\"function_name\": <name of the function>, \"arguments\": <arguments of the function>}"
// 		},
// 		{
// 			"role": "user",
// 			"content": "Would you mind changing the radio station to BBC Radio 1, please?"
// 		}
// 	],
// 	"temperature": 0,
// 	"stop": [
// 		"\n\n",
// 		"\n\n\n"
// 	],
// 	"max_tokens": 250
// }

// clang-format on

class OpenAIInputDecoder : public InputDecoder {
 public:
  bool decode(const string& message, InputParams& decoded_params) override {
    try {
      auto json_msg = json::parse(message);
      // Look for "messages"
      if (!json_msg.contains("messages")) {
        cout << RED << "Required node 'messages' not found!" << CLEAR
             << endl;
        return false;
      } else {
        auto messages = json_msg.at("messages");
        if (messages.size() == 0) {
          cout << RED << "Empty \"messages\" node\n"
               << CLEAR;
          return false;
        }
        if (!extract_messages(messages, decoded_params)) {
          cout << RED << "Error extracting messages\n"
               << CLEAR;
          return false;
        }
      }
      if (json_msg.contains("temperature")) {
        decoded_params.Temperature =
            json_msg["temperature"].get<float_t>();
      }
      if (json_msg.contains("max_tokens")) {
        decoded_params.MaxGeneratedTokens =
            json_msg["max_tokens"].get<uint32_t>();
      }
      if (json_msg.contains("top_k")) {
        decoded_params.TopK = json_msg["top_k"].get<uint32_t>();
      }
      if (json_msg.contains("top_p")) {
        decoded_params.TopP = json_msg["top_p"].get<float_t>();
      }

      if (json_msg.contains("stop")) {
        auto stop_tokens = json_msg.at("stop");
        if (stop_tokens.size() > 0 && stop_tokens.size() < 5) {
          for (auto& next_token : stop_tokens) {
            decoded_params.StopTokens.push_back(next_token);
          }
        } else {
          cout << RED
               << "Wrong size of stop tokens: " << stop_tokens.size()
               << CLEAR << endl;
        }
      }
    } catch (json::parse_error& err) {
      cout << RED << "Error in JSON At: " << err.what() << CLEAR << endl;
      return false;
    }
    return true;
  }

 private:
  bool extract_messages(const nlohmann::json& messages,
                        InputParams& decoded_params) {
    bool user_msg_found = false;
    bool system_msg_found = false;

    for (auto& next_msg : messages) {
      if (next_msg.contains("role")) {
        auto role = next_msg.at("role");
        if (!next_msg.contains("content")) {
          cout << RED << "Error: No content for role: " << role
               << CLEAR << endl;
          return false;
        }
        if (role == "system") {  // system message
          if (system_msg_found) {
            cout << RED << "Error: System message already exists"
                 << CLEAR << endl;
            return false;
          }
          decoded_params.Messages.push_back(
              {InputParams::Role::SYSTEM, next_msg["content"]});
          system_msg_found = true;
        } else if (role == "user" || role == "assistant") {
          // Check to see if the next messages what we expect
          // Meaning - the sequence is:
          // system, user, assistant, user, assistant, ...
          // The validity of the system message is checked above. So
          // we just need to check the user and assistant messages
          if (role == "user") {
            if (user_msg_found) {
              cout << RED << "Error: User message already exists"
                   << CLEAR << endl;
              return false;
            } else {
              decoded_params.Messages.push_back(
                  {InputParams::Role::USER, next_msg["content"]});
              user_msg_found = true;
              decoded_params.UserPrompt = next_msg["content"];
            }
          } else {
            // assistant message found
            decoded_params.Messages.push_back(
                {InputParams::Role::ASSISTANT,
                 next_msg["content"]});
            user_msg_found = false;  // reset the flag
          }
        } else {  // unknown role
          cout << RED << "Unknown role: " << role << CLEAR << endl;
          return false;
        }
      } else {  // role not found
        cout << RED << "Role not found in message" << CLEAR << endl;
        return false;
      }
    }
    return true;
  }
};

unique_ptr<InputDecoder> InputDecoder::CreateDecoder(const string& name) {
  // check to see if this decoder exists
  if (name == "openai") {
    return make_unique<OpenAIInputDecoder>();
  }

  // Instantiate and return
  cout << RED << "Decoder not available: " << name << CLEAR << endl;
  return nullptr;
}

}  // namespace slm_engine
}  // namespace microsoft
