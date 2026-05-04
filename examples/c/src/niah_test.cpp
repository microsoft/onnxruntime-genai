// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Needle-in-a-Haystack (NIAH) test for attention quality validation.
// Embeds a secret fact in filler text at various depths, then asks the model
// to retrieve it. Compares baseline vs TurboQuant (or any config change).
//
// Usage:
//   niah_test -m <model_path> [-e <ep>] [--max_tokens 256] [--depths 5]

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ort_genai.h"

struct NiahConfig {
  std::string model_path;
  std::string ep = "webgpu";
  int max_new_tokens = 128;
  int num_depth_levels = 5;  // How many depth positions to test the needle
  int haystack_sentences = 40;  // Number of filler sentences
};

// Generate a deterministic filler sentence by index
static std::string FillerSentence(int idx) {
  // Diverse filler topics to simulate realistic context
  static const char* topics[] = {
      "The global economy experienced significant shifts during the early 21st century.",
      "Advances in renewable energy have accelerated the transition away from fossil fuels.",
      "Machine learning algorithms continue to improve in both accuracy and efficiency.",
      "Urban planning in modern cities increasingly prioritizes pedestrian accessibility.",
      "The history of mathematics spans thousands of years across many cultures.",
      "Ocean currents play a critical role in regulating the Earth's climate patterns.",
      "Biodiversity loss remains one of the most pressing environmental challenges today.",
      "The development of antibiotics revolutionized modern medicine in the 20th century.",
      "Space exploration has yielded numerous technological spin-offs for everyday life.",
      "Agricultural practices have evolved dramatically since the industrial revolution.",
      "The printing press fundamentally changed the dissemination of knowledge in Europe.",
      "Quantum computing promises to solve problems intractable for classical computers.",
      "The water cycle is essential for sustaining life on Earth.",
      "International trade agreements shape economic relationships between nations.",
      "Volcanic activity has influenced climate and human migration throughout history.",
      "The architecture of ancient Rome has inspired builders for over two millennia.",
      "Photosynthesis converts sunlight into chemical energy stored in glucose molecules.",
      "Artificial intelligence raises important ethical questions about bias and fairness.",
      "The human genome project mapped all genes in human DNA by the year 2003.",
      "Transportation infrastructure is a key driver of economic growth and development.",
  };
  return topics[idx % 20];
}

// The needle: a specific, verifiable fact
static const char* NEEDLE_FACT = "The secret code for Project Aurora is 7492-BLUE-DELTA.";
static const char* NEEDLE_QUERY = "What is the secret code for Project Aurora?";
static const char* NEEDLE_ANSWER = "7492-BLUE-DELTA";

// Build prompt with needle at a given depth (0.0 = beginning, 1.0 = end)
static std::string BuildPrompt(int haystack_sentences, float depth) {
  int needle_pos = static_cast<int>(depth * (haystack_sentences - 1));
  needle_pos = std::max(0, std::min(needle_pos, haystack_sentences - 1));

  std::ostringstream oss;
  for (int i = 0; i < haystack_sentences; ++i) {
    if (i == needle_pos) {
      oss << NEEDLE_FACT << " ";
    }
    oss << FillerSentence(i) << " ";
  }
  oss << "\n\nBased on the information above, answer the following question in one short sentence.\n"
      << "Question: " << NEEDLE_QUERY << "\nAnswer:";
  return oss.str();
}

// Generate response from model
static std::string Generate(OgaModel& model, OgaTokenizer& tokenizer,
                            const std::string& prompt, int max_new_tokens) {
  auto params = OgaGeneratorParams::Create(model);
  params->SetSearchOption("max_length", static_cast<double>(max_new_tokens + 4096));  // prompt + gen
  params->SetSearchOption("min_length", 1.0);

  auto generator = OgaGenerator::Create(model, *params);
  auto stream = OgaTokenizerStream::Create(tokenizer);

  auto sequences = OgaSequences::Create();
  tokenizer.Encode(prompt.c_str(), *sequences);
  generator->AppendTokenSequences(*sequences);

  int prompt_len = generator->TokenCount();
  std::string response;
  int generated = 0;

  while (!generator->IsDone() && generated < max_new_tokens) {
    generator->GenerateNextToken();
    auto new_token = generator->GetNextTokens()[0];
    const char* piece = stream->Decode(new_token);
    response += piece;
    generated++;
  }

  return response;
}

// Check if response contains the needle answer
static bool ContainsAnswer(const std::string& response) {
  // Case-insensitive search for the key parts
  std::string lower_response = response;
  std::transform(lower_response.begin(), lower_response.end(),
                 lower_response.begin(), ::tolower);

  // Check for the full code or significant parts of it
  return lower_response.find("7492") != std::string::npos &&
         lower_response.find("blue") != std::string::npos &&
         lower_response.find("delta") != std::string::npos;
}

static void PrintUsage(const char* prog) {
  std::cout << "Usage: " << prog << " -m <model_path> [options]\n"
            << "  -m, --model_path    Path to model directory (required)\n"
            << "  -e, --ep            Execution provider (default: webgpu)\n"
            << "  --max_tokens        Max new tokens to generate (default: 128)\n"
            << "  --depths            Number of depth levels to test (default: 5)\n"
            << "  --sentences         Number of filler sentences (default: 40)\n";
}

int main(int argc, char** argv) {
  NiahConfig config;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << arg << "\n";
        std::exit(1);
      }
      return argv[++i];
    };

    if (arg == "-m" || arg == "--model_path") config.model_path = next();
    else if (arg == "-e" || arg == "--ep") config.ep = next();
    else if (arg == "--max_tokens") config.max_new_tokens = std::stoi(next());
    else if (arg == "--depths") config.num_depth_levels = std::stoi(next());
    else if (arg == "--sentences") config.haystack_sentences = std::stoi(next());
    else if (arg == "-h" || arg == "--help") { PrintUsage(argv[0]); return 0; }
    else { std::cerr << "Unknown argument: " << arg << "\n"; PrintUsage(argv[0]); return 1; }
  }

  if (config.model_path.empty()) {
    std::cerr << "Error: --model_path is required\n";
    PrintUsage(argv[0]);
    return 1;
  }

  OgaHandle handle;

  std::cout << "=== Needle-in-a-Haystack Attention Quality Test ===\n"
            << "Model: " << config.model_path << "\n"
            << "EP: " << config.ep << "\n"
            << "Haystack: " << config.haystack_sentences << " sentences\n"
            << "Depths: " << config.num_depth_levels << "\n"
            << "Needle: \"" << NEEDLE_FACT << "\"\n"
            << "Query: \"" << NEEDLE_QUERY << "\"\n"
            << "Expected: \"" << NEEDLE_ANSWER << "\"\n"
            << "================================================\n\n";

  try {
    // Create model with EP
    auto oga_config = OgaConfig::Create(config.model_path.c_str());
    if (config.ep != "follow_config") {
      oga_config->ClearProviders();
      if (config.ep != "cpu") {
        oga_config->AppendProvider(config.ep.c_str());
      }
    }
    auto model = OgaModel::Create(*oga_config);
    auto tokenizer = OgaTokenizer::Create(*model);

    int passed = 0;
    int total = config.num_depth_levels;

    for (int d = 0; d < config.num_depth_levels; ++d) {
      float depth = (config.num_depth_levels == 1) ? 0.5f
                    : static_cast<float>(d) / (config.num_depth_levels - 1);

      std::string prompt = BuildPrompt(config.haystack_sentences, depth);

      // Tokenize to show prompt length
      auto seq = OgaSequences::Create();
      tokenizer->Encode(prompt.c_str(), *seq);
      int prompt_tokens = static_cast<int>(seq->SequenceCount(0));

      std::cout << "[Depth " << static_cast<int>(depth * 100) << "%] "
                << prompt_tokens << " tokens | ";
      std::cout.flush();

      auto t0 = std::chrono::steady_clock::now();
      std::string response = Generate(*model, *tokenizer, prompt, config.max_new_tokens);
      auto t1 = std::chrono::steady_clock::now();
      float elapsed = std::chrono::duration<float>(t1 - t0).count();

      bool found = ContainsAnswer(response);
      if (found) passed++;

      // Trim response for display
      std::string display = response.substr(0, 200);
      // Replace newlines for compact display
      for (auto& c : display) if (c == '\n') c = ' ';

      std::cout << (found ? "PASS" : "FAIL")
                << " (" << elapsed << "s) | "
                << display << "\n";
    }

    std::cout << "\n=== Results: " << passed << "/" << total << " passed ===\n";

    if (passed == total) {
      std::cout << "All needle retrievals successful. Attention quality OK.\n";
    } else {
      std::cout << "WARNING: " << (total - passed) << " retrievals failed. "
                << "Attention quality may be degraded.\n";
    }

    return (passed == total) ? 0 : 1;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
