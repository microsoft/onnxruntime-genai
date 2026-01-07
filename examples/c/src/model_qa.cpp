// -----------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// C++ API Example for Model Question-Answering
// This example demonstrates how to use the C++ API of the ONNX Runtime GenAI library
// to perform model question-answering tasks. It includes functionalities to create a model,
// tokenizer, and generator, and to handle user input for generating responses based on prompts.
// -----------------------------------------------------------------------------------------------

#include <csignal>
#include <iomanip>
#include <string>

#include "common.h"

OgaGenerator* g_generator = nullptr;

void TerminateGeneration(int signum) {
  if (g_generator == nullptr) {
    return;
  }
  g_generator->SetRuntimeOption("terminate_session", "1");
}

void CXX_API(
  GeneratorParamsArgs& generator_params_args,
  GuidanceArgs& guidance_args,
  const std::string& model_path,
  const std::string& ep,
  const std::string& system_prompt,
  bool verbose,
  bool interactive
) {
  if (verbose) std::cout << "Creating config..." << std::endl;
  std::unordered_map<std::string, std::string> ep_options;
  auto config = GetConfig(model_path, ep, ep_options, generator_params_args);

  if (verbose) std::cout << "Creating model..." << std::endl;
  auto model = OgaModel::Create(*config);

  if (verbose) std::cout << "Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);
  auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);

  // Create running list of messages
  std::vector<nlohmann::ordered_json> input_list;
  nlohmann::ordered_json system_message = nlohmann::ordered_json{ {"role", "system"}, {"content", system_prompt} };
  input_list.push_back(system_message);

  // Get and set guidance info if requested
  std::string guidance_type, guidance_data, tools;
  if (!guidance_args.response_format.empty()) {
    std::cout << "Make sure your tool call start id and tool call end id are marked as special in tokenizer.json" << std::endl;
    std::tie(guidance_type, guidance_data, tools) = GetGuidance(
      guidance_args.response_format,
      guidance_args.tools_file,
      "",  // tools_str
      nullptr,  // tools
      guidance_args.text_output, 
      guidance_args.tool_output,
      guidance_args.tool_call_start,
      guidance_args.tool_call_end
    );
    
    input_list[0]["tools"] = tools;
  }

  // Keep asking for input prompts in a loop
  while (true) {
    // Get user prompt
    std::string text;
    std::cout << "Prompt (Use quit() to exit):" << std::endl;
    // Clear Any cin error flags because of SIGINT
    std::cin.clear();
    std::getline(std::cin, text);

    if (text.empty()) {
      std::cout << "Empty input. Please enter a valid prompt." << std::endl;
      continue;  // Skip to the next iteration if input is empty
    } else if (text == "quit()") {
      break;  // Exit the loop
    }

    signal(SIGINT, TerminateGeneration);

    // Add user message to list of messages
    nlohmann::ordered_json user_message = nlohmann::ordered_json{ {"role", "user"}, {"content", text} };
    input_list.push_back(user_message);
    nlohmann::ordered_json j = input_list;
    std::string messages = j.dump();

    // Start timings
    bool is_first_token = true;
    Timing timing;
    timing.RecordStartTimestamp();

    // Initialize generator params
    auto params = OgaGeneratorParams::Create(*model);
    SetSearchOptions(*params, generator_params_args, verbose);

    // Initialize guidance info
    if (!guidance_args.response_format.empty()) {
      params->SetGuidance(guidance_type.c_str(), guidance_data.c_str());
      if (verbose) {
        std::cout << std::endl;
        std::cout << "Guidance type is: " << guidance_type << std::endl;
        std::cout << "Guidance data is: \n" << guidance_data << std::endl;
        std::cout << std::endl;
      }
    }

    // Create generator
    auto generator = OgaGenerator::Create(*model, *params);
    g_generator = generator.get();  // Store the current generator for termination
    if (verbose) std::cout << "Generator created" << std::endl;

    // Apply chat template
    std::string prompt;
    try {
      bool add_generation_prompt = true;
      prompt = ApplyChatTemplate(model_path, *tokenizer, messages, add_generation_prompt, tools);
    }
    catch (...) {
      prompt = text;
    }
    if (verbose) std::cout << "Prompt: " << prompt << "\n" << std::endl;

    // Encode combined system + user prompt and append tokens to model
    auto sequences = OgaSequences::Create();
    tokenizer->Encode(prompt.c_str(), *sequences);
    generator->AppendTokenSequences(*sequences);

    // Run generation loop
    if (verbose) std::cout << "Running generation loop..." << std::endl;
    std::cout << std::endl;
    std::cout << "Output: ";
    try {
      while (true) {
        generator->GenerateNextToken();

        if (is_first_token) {
          timing.RecordFirstTokenTimestamp();
          is_first_token = false;
        }

        if (generator->IsDone()) {
          break;
        }

        const auto num_tokens = generator->GetSequenceCount(0);
        const auto new_token = generator->GetSequenceData(0)[num_tokens - 1];
        std::cout << tokenizer_stream->Decode(new_token) << std::flush;
      }
    } catch (const std::exception& e) {
      std::cout << "\n" << "Terminating generation: " << e.what() << std::endl;
    }
    timing.RecordEndTimestamp();

    // Clear the generator after use
    g_generator = nullptr;

    // Remove user message from list of messages
    input_list.pop_back();

    const int prompt_tokens_length = sequences->SequenceCount(0);
    const int new_tokens_length = generator->GetSequenceCount(0) - prompt_tokens_length;
    timing.Log(prompt_tokens_length, new_tokens_length);

    std::cout << "\n\n" << std::endl;
  }
}

int main(int argc, char** argv) {
  // Get command-line args
  GeneratorParamsArgs generator_params_args;
  GuidanceArgs guidance_args;
  std::string model_path, ep = "follow_config", system_prompt = "You are a helpful AI assistant.";
  bool verbose = false, interactive = false, rewind = true;
  if (!ParseArgs(argc, argv, generator_params_args, guidance_args, model_path, ep, system_prompt, verbose, interactive, rewind)) {
    return -1;
  }

  // Responsible for cleaning up the library during shutdown
  OgaHandle handle;

  std::cout << "--------------------------" << std::endl;
  std::cout << "Hello, ORT GenAI Model-QA!" << std::endl;
  std::cout << "--------------------------" << std::endl;

  std::cout << "Model path: " << model_path << std::endl;
  std::cout << "Execution provider: " << ep << std::endl;
  std::cout << "System prompt: " << system_prompt << std::endl;
  std::cout << "Verbose: " << verbose << std::endl;
  std::cout << "Interactive: " << interactive << std::endl;
  std::cout << "--------------------------" << std::endl;
  std::cout << std::endl;

  try {
    CXX_API(generator_params_args, guidance_args, model_path, ep, system_prompt, verbose, interactive);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}