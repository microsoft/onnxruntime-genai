// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <functional>
#include <algorithm>
#include <cmath>
#include <memory>
#include <iomanip>
#include <sstream>
#include "common.h"
#include "ort_genai.h"

// C++ API Example

void PrintTimestampTokens(const int32_t* tokens, size_t count) {
  constexpr int32_t timestamp_begin = 50364;
  bool printed = false;
  for (size_t index = 0; index < count; ++index) {
    if (tokens[index] >= timestamp_begin) {
      if (!printed) {
        std::cout << "    timestamps:";
        printed = true;
      }
      std::cout << " " << std::fixed << std::setprecision(2)
                << (tokens[index] - timestamp_begin) * 0.02;
    }
  }
  if (printed) {
    std::cout << " seconds" << std::endl;
  }
}

constexpr int32_t kTimestampBegin = 50364;
constexpr double kTimestampPrecision = 0.02;

struct WhisperSegment {
  double start{};
  double end{};
  std::string text;
};

struct WhisperOptions {
  std::string language{"en"};
  std::string task{"transcribe"};
  std::string initial_prompt;
  std::string output_dir;
  std::vector<std::string> output_formats{"txt"};
  int max_length{448};
  int top_k{1};
  double temperature{};
  double top_p{1.0};
  double repetition_penalty{1.0};
  bool timestamps{};
};

std::string FormatTimestamp(double seconds, char separator) {
  auto milliseconds = static_cast<int64_t>(std::round(seconds * 1000));
  auto hours = milliseconds / 3600000;
  milliseconds %= 3600000;
  auto minutes = milliseconds / 60000;
  milliseconds %= 60000;
  auto secs = milliseconds / 1000;
  milliseconds %= 1000;
  std::ostringstream result;
  result << std::setfill('0') << std::setw(2) << hours << ":" << std::setw(2) << minutes << ":"
         << std::setw(2) << secs << separator << std::setw(3) << milliseconds;
  return result.str();
}

std::vector<WhisperSegment> GetSegments(const int32_t* tokens, size_t count,
                                        const std::function<std::string(const int32_t*, size_t)>& decode) {
  std::vector<WhisperSegment> segments;
  const auto first_timestamp = std::find_if(tokens, tokens + count, [](int32_t token) { return token >= kTimestampBegin; });
  if (first_timestamp == tokens + count) return {{0, 0, decode(tokens, count)}};
  size_t start = static_cast<size_t>(first_timestamp - tokens);
  for (size_t index = 1; index < count; ++index) {
    if (tokens[index - 1] >= kTimestampBegin && tokens[index] >= kTimestampBegin) {
      std::vector<int32_t> text_tokens;
      for (size_t token = start; token < index; ++token) {
        if (tokens[token] < kTimestampBegin) text_tokens.push_back(tokens[token]);
      }
      const auto text = decode(text_tokens.data(), text_tokens.size());
      if (!text.empty()) {
        segments.push_back({(tokens[start] - kTimestampBegin) * kTimestampPrecision,
                            (tokens[index - 1] - kTimestampBegin) * kTimestampPrecision, text});
      }
      start = index;
    }
  }
  if (segments.empty()) segments.push_back({0, 0, decode(tokens, count)});
  return segments;
}

void WriteResults(const std::vector<WhisperSegment>& segments, const std::string& output_dir,
                  const std::string& stem, const std::vector<std::string>& formats) {
  if (output_dir.empty()) return;
  std::filesystem::create_directories(output_dir);
  for (const auto& format : formats) {
    std::ofstream output(std::filesystem::path(output_dir) / (stem + "." + format));
    if (!output) throw std::runtime_error("Unable to create output file.");
    if (format == "txt") {
      for (const auto& segment : segments) output << segment.text << "\n";
    } else if (format == "json" || format == "jsonl") {
      nlohmann::ordered_json records = nlohmann::ordered_json::array();
      for (size_t i = 0; i < segments.size(); ++i) {
        records.push_back({{"id", i}, {"start", segments[i].start}, {"end", segments[i].end}, {"text", segments[i].text}});
      }
      if (format == "json") output << nlohmann::ordered_json{{"segments", records}}.dump(2) << "\n";
      else for (const auto& record : records) output << record.dump() << "\n";
    } else if (format == "tsv") {
      output << "start\tend\ttext\n";
      for (const auto& segment : segments) output << std::round(segment.start * 1000) << "\t"
                                                  << std::round(segment.end * 1000) << "\t" << segment.text << "\n";
    } else if (format == "srt" || format == "vtt") {
      if (format == "vtt") output << "WEBVTT\n\n";
      for (size_t i = 0; i < segments.size(); ++i) {
        if (format == "srt") output << i + 1 << "\n";
        const char separator = format == "srt" ? ',' : '.';
        output << FormatTimestamp(segments[i].start, separator) << " --> "
               << FormatTimestamp(segments[i].end, separator) << "\n" << segments[i].text << "\n\n";
      }
    } else {
      throw std::runtime_error("Unsupported output format: " + format);
    }
  }
}

void CXX_API(const char* model_path, int32_t num_beams, const WhisperOptions& options) {
  std::cout << "Creating model..." << std::endl;
  auto model = OgaModel::Create(model_path);
  std::cout << "Creating multimodal processor..." << std::endl;
  auto processor = OgaMultiModalProcessor::Create(*model);

  while (true) {
    std::string audio_paths_str;
    std::cout << "Audio Paths (comma separated):" << std::endl;
    std::getline(std::cin, audio_paths_str);
    std::unique_ptr<OgaAudios> audios;
    std::vector<std::string> audio_paths;
    for (size_t start = 0, end = 0; end < audio_paths_str.size(); start = end + 1) {
      end = audio_paths_str.find(',', start);
      audio_paths.push_back(Trim(audio_paths_str.substr(start, end - start)));
    }
    if (audio_paths.empty()) {
      throw std::runtime_error("No audio file provided.");
    } else {
      std::cout << "Loading audios..." << std::endl;
      for (const auto& audio_path : audio_paths) {
        std::filesystem::path p(audio_path);
        if (!std::filesystem::exists(p)) {
          throw std::runtime_error(std::string("Audio file not found: ") + audio_path);
        }
      }
      std::vector<const char*> audio_paths_c;
      for (const auto& audio_path : audio_paths) audio_paths_c.push_back(audio_path.c_str());
      audios = OgaAudios::Load(audio_paths_c);
    }

    std::cout << "Processing inputs..." << std::endl;
    const size_t batch_size = audio_paths.size();
    std::string prompt_tokens = "<|startoftranscript|><|" + options.language + "|><|" + options.task + "|>";
    if (!options.initial_prompt.empty()) prompt_tokens += " " + options.initial_prompt;
    if (!options.timestamps) {
      prompt_tokens += "<|notimestamps|>";
    }
    const std::vector<const char*> prompts(batch_size, prompt_tokens.c_str());
    auto inputs = processor->ProcessAudios(prompts, audios.get());

    std::cout << "Generating response..." << std::endl;
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("batch_size", static_cast<double>(batch_size));
    params->SetSearchOption("max_length", options.max_length);
    params->SetSearchOptionBool("do_sample", options.temperature > 0);
    params->SetSearchOption("num_beams", num_beams);
    params->SetSearchOption("num_return_sequences", num_beams);
    params->SetSearchOption("temperature", options.temperature);
    params->SetSearchOption("top_k", options.top_k);
    params->SetSearchOption("top_p", options.top_p);
    params->SetSearchOption("repetition_penalty", options.repetition_penalty);

    auto generator = OgaGenerator::Create(*model, *params);
    generator->SetInputs(*inputs);

    while (!generator->IsDone()) {
      generator->GenerateNextToken();
    }

    for (size_t i = 0; i < static_cast<size_t>(num_beams * batch_size); ++i) {
      std::cout << "Transcription:" << std::endl;
      std::cout << "    batch " << i / num_beams << ", beam " << i % num_beams << ":";
      const auto num_tokens = generator->GetSequenceCount(i);
      const auto tokens = generator->GetSequenceData(i);
      std::cout << processor->Decode(tokens, num_tokens) << std::endl;
      if (options.timestamps) {
        PrintTimestampTokens(tokens, num_tokens);
      }
      auto segments = GetSegments(tokens, num_tokens, [&processor](const int32_t* data, size_t size) {
        return processor->Decode(data, size).p_;
      });
      auto stem = std::filesystem::path(audio_paths[i / num_beams]).stem().string();
      WriteResults(segments, options.output_dir, stem + ".beam" + std::to_string(i % num_beams), options.output_formats);
    }

    std::cout << "\n\n\n";
  }
}

// C API Example

void CheckResult(OgaResult* result) {
  if (result) {
    std::string string = OgaResultGetError(result);
    OgaDestroyResult(result);
    throw std::runtime_error(string);
  }
}

void C_API(const char* model_path, int32_t num_beams, const WhisperOptions& options) {
  OgaModel* model;
  std::cout << "Creating model..." << std::endl;
  CheckResult(OgaCreateModel(model_path, &model));

  OgaMultiModalProcessor* processor;
  std::cout << "Creating multimodal processor..." << std::endl;
  CheckResult(OgaCreateMultiModalProcessor(model, &processor));
  OgaTokenizer* tokenizer;
  std::cout << "Creating tokenizer..." << std::endl;
  CheckResult(OgaCreateTokenizer(model, &tokenizer));

  while (true) {
    std::string audio_paths_str;
    std::cout << "Audio Paths (comma separated):" << std::endl;
    std::getline(std::cin, audio_paths_str);
    OgaAudios* audios = nullptr;
    std::vector<std::string> audio_paths;
    for (size_t start = 0, end = 0; end < audio_paths_str.size(); start = end + 1) {
      end = audio_paths_str.find(',', start);
      audio_paths.push_back(Trim(audio_paths_str.substr(start, end - start)));
    }
    if (audio_paths.empty()) {
      throw std::runtime_error("No audio file provided.");
    } else {
      std::cout << "Loading audios..." << std::endl;
      for (const auto& audio_path : audio_paths) {
        std::filesystem::path p(audio_path);
        if (!std::filesystem::exists(p)) {
          throw std::runtime_error(std::string("Audio file not found: ") + audio_path);
        }
        std::vector<const char*> audio_paths_c;
        for (const auto& audio_path : audio_paths) audio_paths_c.push_back(audio_path.c_str());
        OgaStringArray* audio_paths_string_array;
        CheckResult(OgaCreateStringArrayFromStrings(audio_paths_c.data(), audio_paths_c.size(), &audio_paths_string_array));
        CheckResult(OgaLoadAudios(audio_paths_string_array, &audios));
        OgaDestroyStringArray(audio_paths_string_array);
      }
    }

    std::cout << "Processing audio..." << std::endl;
    OgaNamedTensors* inputs;
    const size_t batch_size = audio_paths.size();
    std::string prompt_tokens = "<|startoftranscript|><|" + options.language + "|><|" + options.task + "|>";
    if (!options.initial_prompt.empty()) prompt_tokens += " " + options.initial_prompt;
    if (!options.timestamps) {
      prompt_tokens += "<|notimestamps|>";
    }
    std::vector<const char*> prompts(batch_size, prompt_tokens.c_str());
    OgaStringArray* prompts_string_array;
    CheckResult(OgaCreateStringArrayFromStrings(prompts.data(), prompts.size(), &prompts_string_array));
    CheckResult(OgaProcessorProcessAudiosAndPrompts(processor, prompts_string_array, audios, &inputs));
    OgaDestroyStringArray(prompts_string_array);

    std::cout << "Generating response..." << std::endl;
    OgaGeneratorParams* params;
    CheckResult(OgaCreateGeneratorParams(model, &params));
    CheckResult(OgaGeneratorParamsSetSearchNumber(params, "batch_size", static_cast<double>(batch_size)));
    CheckResult(OgaGeneratorParamsSetSearchNumber(params, "max_length", options.max_length));
    CheckResult(OgaGeneratorParamsSetSearchBool(params, "do_sample", options.temperature > 0));
    CheckResult(OgaGeneratorParamsSetSearchNumber(params, "num_beams", num_beams));
    CheckResult(OgaGeneratorParamsSetSearchNumber(params, "num_return_sequences", num_beams));
    CheckResult(OgaGeneratorParamsSetSearchNumber(params, "temperature", options.temperature));
    CheckResult(OgaGeneratorParamsSetSearchNumber(params, "top_k", options.top_k));
    CheckResult(OgaGeneratorParamsSetSearchNumber(params, "top_p", options.top_p));
    CheckResult(OgaGeneratorParamsSetSearchNumber(params, "repetition_penalty", options.repetition_penalty));

    OgaGenerator* generator;
    CheckResult(OgaCreateGenerator(model, params, &generator));
    CheckResult(OgaGenerator_SetInputs(generator, inputs));

    while (!OgaGenerator_IsDone(generator)) {
      CheckResult(OgaGenerator_GenerateNextToken(generator));
    }

    for (size_t i = 0; i < static_cast<size_t>(num_beams * batch_size); ++i) {
      std::cout << "Transcription:" << std::endl;
      std::cout << "    batch " << i / num_beams << ", beam " << i % num_beams << ":";
      const int32_t num_tokens = OgaGenerator_GetSequenceCount(generator, i);
      const int32_t* tokens = OgaGenerator_GetSequenceData(generator, i);

      const char* str;
      CheckResult(OgaProcessorDecode(processor, tokens, num_tokens, &str));
      std::cout << str << std::endl;
      if (options.timestamps) {
        PrintTimestampTokens(tokens, num_tokens);
      }
      auto segments = GetSegments(tokens, num_tokens, [processor](const int32_t* data, size_t size) {
        const char* decoded;
        CheckResult(OgaProcessorDecode(processor, data, size, &decoded));
        return std::string(decoded);
      });
      auto stem = std::filesystem::path(audio_paths[i / num_beams]).stem().string();
      WriteResults(segments, options.output_dir, stem + ".beam" + std::to_string(i % num_beams), options.output_formats);
    }

    std::cout << "\n\n"
              << std::endl;

    OgaDestroyGenerator(generator);
    OgaDestroyGeneratorParams(params);
    OgaDestroyNamedTensors(inputs);
    OgaDestroyAudios(audios);
  }

  OgaDestroyTokenizer(tokenizer);
  OgaDestroyMultiModalProcessor(processor);
  OgaDestroyModel(model);
}

static void print_usage_whisper(int /*argc*/, char** argv) {
  std::cerr << "usage: " << argv[0] << " <model_path> <num_beams> [--language <language>] "
            << "[--task <transcribe|translate>] [--timestamps] [--output-dir <dir>] "
            << "[--output-format <txt|json|jsonl|srt|tsv|vtt,...>] [--initial-prompt <text>] "
            << "[--temperature <float>] [--top-k <int>] [--top-p <float>] "
            << "[--repetition-penalty <float>] [--max-length <int>]" << std::endl;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    print_usage_whisper(argc, argv);
    return -1;
  }

  WhisperOptions options;
  for (int i = 3; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--timestamps") {
      options.timestamps = true;
    } else if ((option == "--language" || option == "--task" || option == "--output-dir" || option == "--output-format" ||
                option == "--initial-prompt" || option == "--temperature" || option == "--top-k" || option == "--top-p" ||
                option == "--repetition-penalty" || option == "--max-length") && ++i < argc) {
      if (option == "--language") options.language = argv[i];
      else if (option == "--task") options.task = argv[i];
      else if (option == "--output-dir") options.output_dir = argv[i];
      else if (option == "--initial-prompt") options.initial_prompt = argv[i];
      else if (option == "--temperature") options.temperature = std::stod(argv[i]);
      else if (option == "--top-k") options.top_k = std::stoi(argv[i]);
      else if (option == "--top-p") options.top_p = std::stod(argv[i]);
      else if (option == "--repetition-penalty") options.repetition_penalty = std::stod(argv[i]);
      else if (option == "--max-length") options.max_length = std::stoi(argv[i]);
      else {
        options.output_formats.clear();
        std::istringstream formats(argv[i]);
        for (std::string format; std::getline(formats, format, ',');) options.output_formats.push_back(format);
      }
    } else {
      print_usage_whisper(argc, argv);
      return -1;
    }
  }
  if (options.task != "transcribe" && options.task != "translate") {
    std::cerr << "--task must be transcribe or translate." << std::endl;
    return -1;
  }
  if (options.temperature > 0 && std::stoi(argv[2]) != 1) {
    std::cerr << "Sampling requires num_beams to be 1." << std::endl;
    return -1;
  }
  if (options.top_p <= 0 || options.top_p > 1 || options.max_length <= 0) {
    std::cerr << "--top-p must be in (0, 1] and --max-length must be positive." << std::endl;
    return -1;
  }

  // Uncomment for debugging purposes
  // Oga::SetLogBool("enabled", true);
  // Oga::SetLogBool("model_input_values", true);
  // Oga::SetLogBool("model_output_values", true);

  std::cout << "---------------" << std::endl;
  std::cout << "Hello, Whisper!" << std::endl;
  std::cout << "---------------" << std::endl;

#ifdef USE_CXX
  std::cout << "C++ API" << std::endl;
  CXX_API(argv[1], std::stoi(argv[2]), options);
#else
  std::cout << "C API" << std::endl;
  C_API(argv[1], std::stoi(argv[2]), options);
#endif

  return 0;
}