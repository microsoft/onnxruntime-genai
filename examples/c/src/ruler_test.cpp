// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// RULER-inspired attention quality benchmark for comparing baseline vs TurboQuant.
// Implements 4 task categories from NVIDIA RULER (arXiv:2404.06654):
//   1. Multi-Key NIAH - retrieve specific values from multiple needles
//   2. Multi-Value NIAH - retrieve multiple values for the same key
//   3. Variable Tracking - trace variable assignment chains
//   4. Common Word Extraction - identify frequently inserted words
//
// Usage:
//   ruler_test -m <model_path> [-e <ep>] [--context_tokens 512,1024,2048]
//
// Run twice (baseline config vs TQ config) and compare scores.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "ort_genai.h"

// ============================================================================
// Utilities
// ============================================================================

static std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  return s;
}

// Deterministic filler sentences
static const char* kFillerSentences[] = {
    "The global economy experienced significant shifts during the early 21st century as trade patterns evolved.",
    "Advances in renewable energy have accelerated the transition away from fossil fuels across many nations.",
    "Machine learning algorithms continue to improve in both accuracy and efficiency for complex tasks.",
    "Urban planning in modern cities increasingly prioritizes pedestrian accessibility and green spaces.",
    "The history of mathematics spans thousands of years across many cultures and civilizations.",
    "Ocean currents play a critical role in regulating the Earth's climate patterns and weather systems.",
    "Biodiversity loss remains one of the most pressing environmental challenges today worldwide.",
    "The development of antibiotics revolutionized modern medicine in the 20th century significantly.",
    "Space exploration has yielded numerous technological spin-offs for everyday life and industry.",
    "Agricultural practices have evolved dramatically since the industrial revolution began in Europe.",
    "The printing press fundamentally changed the dissemination of knowledge throughout Europe.",
    "Quantum computing promises to solve problems that are intractable for classical computers today.",
    "The water cycle is essential for sustaining all forms of life on the planet Earth.",
    "International trade agreements shape economic relationships between nations across the globe.",
    "Volcanic activity has influenced climate patterns and human migration throughout recorded history.",
    "The architecture of ancient Rome has inspired builders and designers for over two millennia.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose molecules in plants.",
    "Digital communication networks have transformed how people interact and share information globally.",
    "The human genome project successfully mapped all genes in human DNA by the year 2003.",
    "Transportation infrastructure is widely recognized as a key driver of economic growth.",
    "Coral reefs support approximately one quarter of all known marine species in the ocean.",
    "The invention of the transistor in 1947 laid the foundation for all modern electronics.",
    "Climate models predict significant changes in precipitation patterns over the coming decades.",
    "Medieval European universities established many academic traditions that persist to this day.",
    "Tectonic plate movements continuously reshape the surface of the Earth over geological timescales.",
    "The discovery of penicillin by Alexander Fleming marked a turning point in medical treatments.",
    "Artificial satellites provide essential services including GPS navigation and weather forecasting.",
    "The Amazon rainforest produces approximately twenty percent of the world's oxygen supply.",
    "Ancient Egyptian civilization flourished along the Nile River for over three thousand years.",
    "Modern cryptography relies on mathematical problems that are computationally difficult to solve.",
};
static constexpr int kNumFillers = 30;

static std::string Filler(int idx) {
  return kFillerSentences[idx % kNumFillers];
}

// Build a haystack of N filler sentences
static std::string BuildHaystack(int n, std::mt19937& rng) {
  std::ostringstream oss;
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), rng);
  for (int i = 0; i < n; ++i) {
    oss << Filler(order[i]) << " ";
  }
  return oss.str();
}

// Insert text at position in haystack (by sentence count)
static std::string InsertAtDepth(const std::string& haystack_base, const std::string& needle, float depth) {
  // Split into sentences (rough: by ". ")
  std::vector<std::string> sentences;
  std::istringstream iss(haystack_base);
  std::string buf;
  while (std::getline(iss, buf, '.')) {
    if (!buf.empty()) {
      sentences.push_back(buf + ".");
    }
  }
  int pos = static_cast<int>(depth * sentences.size());
  pos = std::max(0, std::min(pos, static_cast<int>(sentences.size())));
  std::ostringstream oss;
  for (int i = 0; i < static_cast<int>(sentences.size()); ++i) {
    if (i == pos) oss << needle << " ";
    oss << sentences[i] << " ";
  }
  return oss.str();
}

// Generate response
static std::string Generate(OgaModel& model, OgaTokenizer& tokenizer,
                            const std::string& prompt, int max_new_tokens) {
  auto params = OgaGeneratorParams::Create(model);
  params->SetSearchOption("max_length", static_cast<double>(max_new_tokens + 8192));
  params->SetSearchOption("min_length", 1.0);

  auto generator = OgaGenerator::Create(model, *params);
  auto stream = OgaTokenizerStream::Create(tokenizer);

  auto sequences = OgaSequences::Create();
  tokenizer.Encode(prompt.c_str(), *sequences);
  generator->AppendTokenSequences(*sequences);

  std::string response;
  int generated = 0;
  while (!generator->IsDone() && generated < max_new_tokens) {
    generator->GenerateNextToken();
    auto new_token = generator->GetNextTokens()[0];
    response += stream->Decode(new_token);
    generated++;
  }
  return response;
}

static int CountTokens(OgaTokenizer& tokenizer, const std::string& text) {
  auto seq = OgaSequences::Create();
  tokenizer.Encode(text.c_str(), *seq);
  return static_cast<int>(seq->SequenceCount(0));
}

// ============================================================================
// Task 1: Multi-Key NIAH
// Insert N key-value pairs in the haystack, ask for a specific one.
// ============================================================================
struct MultiKeyResult {
  bool passed;
  std::string response;
  std::string expected;
};

static MultiKeyResult RunMultiKeyNIAH(OgaModel& model, OgaTokenizer& tokenizer,
                                       int num_sentences, int num_keys, int query_key_idx,
                                       std::mt19937& rng) {
  // Generate key-value pairs
  static const char* colors[] = {"red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"};
  static const char* objects[] = {"apple", "bicycle", "candle", "diamond", "elephant", "falcon", "guitar", "hammer"};
  int codes[] = {3847, 5291, 7163, 4028, 6835, 9412, 1574, 2906};

  std::string haystack = BuildHaystack(num_sentences, rng);

  // Insert needles at varying depths
  for (int k = 0; k < num_keys; ++k) {
    float depth = static_cast<float>(k + 1) / (num_keys + 1);
    std::ostringstream needle;
    needle << "The special code for the " << colors[k] << " " << objects[k]
           << " is " << codes[k] << ".";
    haystack = InsertAtDepth(haystack, needle.str(), depth);
  }

  // Query for one specific key
  std::ostringstream prompt;
  prompt << haystack
         << "\n\nBased only on the information above, what is the special code for the "
         << colors[query_key_idx] << " " << objects[query_key_idx]
         << "? Answer with just the number.\nAnswer:";

  std::string response = Generate(model, tokenizer, prompt.str(), 32);
  std::string expected = std::to_string(codes[query_key_idx]);
  bool passed = response.find(expected) != std::string::npos;

  return {passed, response, expected};
}

// ============================================================================
// Task 2: Multi-Value NIAH
// Insert one key with multiple values, ask for all of them.
// ============================================================================
struct MultiValueResult {
  int found;
  int total;
  std::string response;
  std::vector<std::string> expected;
};

static MultiValueResult RunMultiValueNIAH(OgaModel& model, OgaTokenizer& tokenizer,
                                           int num_sentences, int num_values,
                                           std::mt19937& rng) {
  static const char* cities[] = {"Paris", "Tokyo", "Sydney", "Cairo", "Lima", "Oslo", "Seoul", "Dublin"};

  std::string haystack = BuildHaystack(num_sentences, rng);

  // Insert the same key with different values at different depths
  std::vector<std::string> expected_cities;
  for (int v = 0; v < num_values; ++v) {
    float depth = static_cast<float>(v + 1) / (num_values + 1);
    std::ostringstream needle;
    needle << "Agent Falcon visited " << cities[v] << " on a classified mission.";
    haystack = InsertAtDepth(haystack, needle.str(), depth);
    expected_cities.push_back(cities[v]);
  }

  std::ostringstream prompt;
  prompt << haystack
         << "\n\nBased only on the information above, list ALL cities that Agent Falcon visited. "
         << "Answer with just the city names separated by commas.\nAnswer:";

  std::string response = Generate(model, tokenizer, prompt.str(), 64);
  std::string lower_resp = ToLower(response);

  int found = 0;
  for (auto& city : expected_cities) {
    if (lower_resp.find(ToLower(city)) != std::string::npos) found++;
  }

  return {found, num_values, response, expected_cities};
}

// ============================================================================
// Task 3: Variable Tracking
// Create chains of variable assignments: X1=Y, Y=Z, Z=W. Ask what X1 equals.
// ============================================================================
struct VarTrackResult {
  bool passed;
  std::string response;
  std::string expected;
};

static VarTrackResult RunVariableTracking(OgaModel& model, OgaTokenizer& tokenizer,
                                           int num_sentences, int num_hops,
                                           std::mt19937& rng) {
  // Variable names
  static const char* vars[] = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"};
  // Final value
  std::string final_value = "42";

  std::string haystack = BuildHaystack(num_sentences, rng);

  // Build chain: vars[0] = vars[1], vars[1] = vars[2], ..., vars[n-1] = 42
  // Insert in REVERSE order so they appear in reading order
  for (int h = num_hops - 1; h >= 0; --h) {
    float depth = static_cast<float>(h + 1) / (num_hops + 1);
    std::ostringstream needle;
    if (h == num_hops - 1) {
      needle << "The variable " << vars[h] << " is set to " << final_value << ".";
    } else {
      needle << "The variable " << vars[h] << " is set to the value of " << vars[h + 1] << ".";
    }
    haystack = InsertAtDepth(haystack, needle.str(), depth);
  }

  std::ostringstream prompt;
  prompt << haystack
         << "\n\nBased only on the variable assignments above, what is the final numeric value of "
         << vars[0] << "? Trace through all assignments and give just the number.\nAnswer:";

  std::string response = Generate(model, tokenizer, prompt.str(), 64);
  bool passed = response.find(final_value) != std::string::npos;

  return {passed, response, final_value};
}

// ============================================================================
// Task 4: Common Word Extraction
// Insert a specific word N times throughout the text. Ask which word was
// repeated. Also insert distractors fewer times.
// ============================================================================
struct CommonWordResult {
  bool passed;
  std::string response;
  std::string expected;
};

static CommonWordResult RunCommonWordExtraction(OgaModel& model, OgaTokenizer& tokenizer,
                                                 int num_sentences, int target_freq,
                                                 int distractor_freq,
                                                 std::mt19937& rng) {
  std::string target_word = "ZENITHRON";
  static const char* distractor_words[] = {"QUASOREX", "VELMITHRA", "PAXIFLUX"};

  std::string haystack = BuildHaystack(num_sentences, rng);

  // Insert target word target_freq times
  for (int i = 0; i < target_freq; ++i) {
    float depth = static_cast<float>(i) / target_freq;
    std::string needle = "Remember the word: " + target_word + ".";
    haystack = InsertAtDepth(haystack, needle, depth);
  }

  // Insert distractors fewer times
  for (int d = 0; d < 3; ++d) {
    for (int i = 0; i < distractor_freq; ++i) {
      float depth = static_cast<float>(i + d * 3) / (distractor_freq * 3 + 3);
      std::string needle = std::string("Remember the word: ") + distractor_words[d] + ".";
      haystack = InsertAtDepth(haystack, needle, depth);
    }
  }

  std::ostringstream prompt;
  prompt << haystack
         << "\n\nIn the text above, several special words were mentioned with 'Remember the word:'. "
         << "Which word appeared the MOST times? Answer with just that one word.\nAnswer:";

  std::string response = Generate(model, tokenizer, prompt.str(), 32);
  std::string lower_resp = ToLower(response);
  bool passed = lower_resp.find(ToLower(target_word)) != std::string::npos;

  return {passed, response, target_word};
}

// ============================================================================
// Main
// ============================================================================

struct Config {
  std::string model_path;
  std::string ep = "webgpu";
  int max_new_tokens = 64;
  std::vector<int> haystack_sizes = {20, 40, 80};  // number of filler sentences
  int seed = 42;
};

static void PrintUsage(const char* prog) {
  std::cout << "RULER-inspired attention quality test\n"
            << "Usage: " << prog << " -m <model_path> [options]\n"
            << "  -m, --model_path     Path to model directory (required)\n"
            << "  -e, --ep             Execution provider (default: webgpu)\n"
            << "  --sizes              Comma-separated haystack sizes in sentences (default: 20,40,80)\n"
            << "  --seed               Random seed (default: 42)\n";
}

static std::vector<int> ParseIntList(const std::string& s) {
  std::vector<int> result;
  std::istringstream iss(s);
  std::string token;
  while (std::getline(iss, token, ',')) {
    result.push_back(std::stoi(token));
  }
  return result;
}

int main(int argc, char** argv) {
  Config config;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { std::cerr << "Missing value for " << arg << "\n"; std::exit(1); }
      return argv[++i];
    };
    if (arg == "-m" || arg == "--model_path") config.model_path = next();
    else if (arg == "-e" || arg == "--ep") config.ep = next();
    else if (arg == "--sizes") config.haystack_sizes = ParseIntList(next());
    else if (arg == "--seed") config.seed = std::stoi(next());
    else if (arg == "-h" || arg == "--help") { PrintUsage(argv[0]); return 0; }
    else { std::cerr << "Unknown: " << arg << "\n"; PrintUsage(argv[0]); return 1; }
  }

  if (config.model_path.empty()) {
    std::cerr << "Error: --model_path required\n";
    PrintUsage(argv[0]);
    return 1;
  }

  OgaHandle handle;

  std::cout << "========================================================\n"
            << "  RULER-Inspired Attention Quality Test\n"
            << "========================================================\n"
            << "Model: " << config.model_path << "\n"
            << "EP: " << config.ep << "\n"
            << "Haystack sizes: ";
  for (auto s : config.haystack_sizes) std::cout << s << " ";
  std::cout << "sentences\n"
            << "========================================================\n\n";

  try {
    auto oga_config = OgaConfig::Create(config.model_path.c_str());
    if (config.ep != "follow_config") {
      oga_config->ClearProviders();
      if (config.ep != "cpu") {
        oga_config->AppendProvider(config.ep.c_str());
      }
    }
    auto model = OgaModel::Create(*oga_config);
    auto tokenizer = OgaTokenizer::Create(*model);

    int total_pass = 0, total_tests = 0;

    // Results table
    struct Row {
      std::string task;
      int sentences;
      int tokens;
      bool passed;
      std::string detail;
      float time_s;
    };
    std::vector<Row> results;

    for (int ns : config.haystack_sizes) {
      std::mt19937 rng(config.seed);

      std::cout << "--- Haystack: " << ns << " sentences ---\n";

      // Estimate tokens
      std::string sample_haystack = BuildHaystack(ns, rng);
      int approx_tokens = CountTokens(*tokenizer, sample_haystack);
      rng.seed(config.seed);  // reset for determinism

      // Task 1: Multi-Key NIAH (4 keys, query key at depth 25%)
      {
        auto t0 = std::chrono::steady_clock::now();
        auto r = RunMultiKeyNIAH(*model, *tokenizer, ns, 4, 1, rng);
        float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();

        std::string detail = "expected=" + r.expected + " got=" + r.response.substr(0, 40);
        results.push_back({"MultiKey-NIAH", ns, approx_tokens, r.passed, detail, elapsed});
        total_tests++;
        if (r.passed) total_pass++;
        std::cout << "  [" << (r.passed ? "PASS" : "FAIL") << "] MultiKey-NIAH   ("
                  << elapsed << "s) expected=" << r.expected << "\n";
      }

      // Task 2: Multi-Value NIAH (3 values to recall)
      {
        auto t0 = std::chrono::steady_clock::now();
        auto r = RunMultiValueNIAH(*model, *tokenizer, ns, 3, rng);
        float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();

        std::ostringstream detail;
        detail << r.found << "/" << r.total << " cities found";
        bool passed = (r.found == r.total);
        results.push_back({"MultiValue-NIAH", ns, approx_tokens, passed, detail.str(), elapsed});
        total_tests++;
        if (passed) total_pass++;
        std::cout << "  [" << (passed ? "PASS" : "FAIL") << "] MultiValue-NIAH ("
                  << elapsed << "s) " << detail.str() << "\n";
      }

      // Task 3: Variable Tracking (3 hops)
      {
        auto t0 = std::chrono::steady_clock::now();
        auto r = RunVariableTracking(*model, *tokenizer, ns, 3, rng);
        float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();

        std::string detail = "expected=" + r.expected + " got=" + r.response.substr(0, 40);
        results.push_back({"VarTracking-3hop", ns, approx_tokens, r.passed, detail, elapsed});
        total_tests++;
        if (r.passed) total_pass++;
        std::cout << "  [" << (r.passed ? "PASS" : "FAIL") << "] VarTracking     ("
                  << elapsed << "s) expected=" << r.expected << "\n";
      }

      // Task 4: Common Word Extraction (target 5x, distractors 2x each)
      {
        auto t0 = std::chrono::steady_clock::now();
        auto r = RunCommonWordExtraction(*model, *tokenizer, ns, 5, 2, rng);
        float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();

        std::string detail = "expected=" + r.expected + " got=" + r.response.substr(0, 40);
        results.push_back({"CommonWord", ns, approx_tokens, r.passed, detail, elapsed});
        total_tests++;
        if (r.passed) total_pass++;
        std::cout << "  [" << (r.passed ? "PASS" : "FAIL") << "] CommonWord      ("
                  << elapsed << "s) expected=" << r.expected << "\n";
      }

      std::cout << "\n";
    }

    // Summary table
    std::cout << "========================================================\n"
              << "  SUMMARY\n"
              << "========================================================\n"
              << std::left
              << std::setw(20) << "Task"
              << std::setw(10) << "Sents"
              << std::setw(10) << "~Tokens"
              << std::setw(8) << "Result"
              << std::setw(8) << "Time"
              << "\n"
              << std::string(56, '-') << "\n";

    for (auto& r : results) {
      std::cout << std::setw(20) << r.task
                << std::setw(10) << r.sentences
                << std::setw(10) << r.tokens
                << std::setw(8) << (r.passed ? "PASS" : "FAIL")
                << std::fixed << std::setprecision(1) << std::setw(8) << r.time_s
                << "\n";
    }

    std::cout << std::string(56, '-') << "\n"
              << "Score: " << total_pass << "/" << total_tests
              << " (" << std::fixed << std::setprecision(1)
              << (100.0f * total_pass / total_tests) << "%)\n"
              << "========================================================\n";

    return (total_pass == total_tests) ? 0 : 1;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
