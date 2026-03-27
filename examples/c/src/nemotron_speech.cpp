// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// nemotron_speech.cpp — Streaming ASR example using StreamingProcessor + Generator API.
//
// Usage:
//   ./nemotron_speech --model_path /path/to/nemotron-model --audio_file /path/to/audio.wav

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "ort_genai.h"

struct AudioConfig {
  int sample_rate;
  int chunk_samples;
};

AudioConfig LoadConfig(const std::string& model_path) {
  std::string config_path = model_path + "/genai_config.json";
  std::ifstream f(config_path);
  if (!f.is_open()) {
    throw std::runtime_error("Cannot open " + config_path);
  }
  auto config = nlohmann::json::parse(f);
  return {
      config["model"]["sample_rate"].get<int>(),
      config["model"]["chunk_samples"].get<int>(),
  };
}

// Simple WAV loader — expects 16-bit PCM, mono or stereo.
// Returns float32 samples normalized to [-1, 1].
std::vector<float> LoadWav(const std::string& path, int target_sample_rate) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open audio file: " + path);
  }

  // Read WAV header
  char riff[4];
  file.read(riff, 4);
  if (std::memcmp(riff, "RIFF", 4) != 0) {
    throw std::runtime_error("Not a valid WAV file (missing RIFF header)");
  }

  file.seekg(4, std::ios::cur);  // Skip file size

  char wave[4];
  file.read(wave, 4);
  if (std::memcmp(wave, "WAVE", 4) != 0) {
    throw std::runtime_error("Not a valid WAV file (missing WAVE marker)");
  }

  // Find fmt chunk
  int16_t num_channels = 0;
  int32_t sample_rate = 0;
  int16_t bits_per_sample = 0;

  while (file.good()) {
    char chunk_id[4];
    int32_t chunk_size;
    file.read(chunk_id, 4);
    file.read(reinterpret_cast<char*>(&chunk_size), 4);

    if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
      int16_t audio_format;
      file.read(reinterpret_cast<char*>(&audio_format), 2);
      file.read(reinterpret_cast<char*>(&num_channels), 2);
      file.read(reinterpret_cast<char*>(&sample_rate), 4);
      file.seekg(6, std::ios::cur);  // Skip byte rate + block align
      file.read(reinterpret_cast<char*>(&bits_per_sample), 2);
      if (chunk_size > 16) {
        file.seekg(chunk_size - 16, std::ios::cur);
      }
    } else if (std::memcmp(chunk_id, "data", 4) == 0) {
      int num_samples = chunk_size / (bits_per_sample / 8) / num_channels;
      std::vector<float> audio(num_samples);

      if (bits_per_sample == 16) {
        std::vector<int16_t> raw(num_samples * num_channels);
        file.read(reinterpret_cast<char*>(raw.data()), chunk_size);
        for (int i = 0; i < num_samples; i++) {
          if (num_channels == 1) {
            audio[i] = raw[i] / 32768.0f;
          } else {
            // Average channels
            float sum = 0.0f;
            for (int c = 0; c < num_channels; c++) {
              sum += raw[i * num_channels + c];
            }
            audio[i] = (sum / num_channels) / 32768.0f;
          }
        }
      } else if (bits_per_sample == 32) {
        // Assume float32
        std::vector<float> raw(num_samples * num_channels);
        file.read(reinterpret_cast<char*>(raw.data()), chunk_size);
        for (int i = 0; i < num_samples; i++) {
          if (num_channels == 1) {
            audio[i] = raw[i];
          } else {
            float sum = 0.0f;
            for (int c = 0; c < num_channels; c++) {
              sum += raw[i * num_channels + c];
            }
            audio[i] = sum / num_channels;
          }
        }
      } else {
        throw std::runtime_error("Unsupported bits per sample: " + std::to_string(bits_per_sample));
      }

      // Basic resampling if needed (linear interpolation)
      if (sample_rate != target_sample_rate) {
        int new_len = static_cast<int>(audio.size() * static_cast<double>(target_sample_rate) / sample_rate);
        std::vector<float> resampled(new_len);
        for (int i = 0; i < new_len; i++) {
          double src_idx = i * static_cast<double>(audio.size() - 1) / (new_len - 1);
          int idx0 = static_cast<int>(src_idx);
          int idx1 = std::min(idx0 + 1, static_cast<int>(audio.size()) - 1);
          double frac = src_idx - idx0;
          resampled[i] = static_cast<float>(audio[idx0] * (1.0 - frac) + audio[idx1] * frac);
        }
        return resampled;
      }

      return audio;
    } else {
      file.seekg(chunk_size, std::ios::cur);
    }
  }

  throw std::runtime_error("No data chunk found in WAV file");
}

std::string DecodeTokens(OgaGenerator& generator, OgaTokenizerStream& tokenizer_stream) {
  std::string text;
  while (!generator.IsDone()) {
    generator.GenerateNextToken();
    auto next_tokens = generator.GetNextTokens();
    if (!next_tokens.empty()) {
      const char* token_text = tokenizer_stream.Decode(next_tokens[0]);
      if (token_text && token_text[0] != '\0') {
        std::cout << token_text << std::flush;
        text += token_text;
      }
    }
  }
  return text;
}

void StreamingTranscribe(const std::string& model_path, const std::string& audio_path, bool enable_vad = false) {
  auto [sample_rate, chunk_samples] = LoadConfig(model_path);

  std::cout << "Loading audio: " << audio_path << std::endl;
  auto audio = LoadWav(audio_path, sample_rate);
  double duration = static_cast<double>(audio.size()) / sample_rate;

  std::cout << "Loading model: " << model_path << std::endl;
  auto config = OgaConfig::Create(model_path.c_str());
  auto model = OgaModel::Create(*config);
  auto processor = OgaStreamingProcessor::Create(*model);

  // VAD is disabled by default. Enable via genai_config.json or --enable_vad.
  if (enable_vad && std::string(processor->GetOption("vad_enabled")) != "true") {
    processor->SetOption("vad_enabled", "true");
  }
  if (std::string(processor->GetOption("vad_enabled")) == "true") {
    std::cout << "  VAD: enabled" << std::endl;
  } else {
    std::cout << "  VAD: disabled" << std::endl;
  }

  auto tokenizer = OgaTokenizer::Create(*model);
  auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);
  auto params = OgaGeneratorParams::Create(*model);
  auto generator = OgaGenerator::Create(*model, *params);

  std::cout << "  Sample rate: " << sample_rate << ", Chunk: " << chunk_samples << " samples" << std::endl;
  std::cout << "  Audio duration: " << duration << "s" << std::endl;
  std::cout << std::string(60, '-') << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  std::string full_transcript;

  // Stream audio in chunks
  for (size_t i = 0; i < audio.size(); i += chunk_samples) {
    size_t remaining = std::min(static_cast<size_t>(chunk_samples), audio.size() - i);
    auto inputs = processor->Process(audio.data() + i, remaining);
    if (inputs) {
      generator->SetInputs(*inputs);
      full_transcript += DecodeTokens(*generator, *tokenizer_stream);
    }
  }

  // Flush remaining audio
  {
    auto inputs = processor->Flush();
    if (inputs && inputs.get()) {
      generator->SetInputs(*inputs);
      full_transcript += DecodeTokens(*generator, *tokenizer_stream);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  double wall_time = std::chrono::duration<double>(end - start).count();

  std::cout << "\n"
            << std::string(60, '=') << std::endl;
  std::cout << "  " << full_transcript << std::endl;
  std::cout << std::string(60, '=') << std::endl;
  std::cout << "  Audio: " << duration << "s | Wall: " << wall_time << "s | RTF: " << (duration / wall_time) << "x" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " --model_path <path> --audio_file <path>" << std::endl;
    return 1;
  }

  std::string model_path;
  std::string audio_file;
  bool enable_vad = false;

  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--model_path" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (std::string(argv[i]) == "--audio_file" && i + 1 < argc) {
      audio_file = argv[++i];
    } else if (std::string(argv[i]) == "--enable_vad") {
      enable_vad = true;
    }
  }

  if (model_path.empty() || audio_file.empty()) {
    std::cerr << "Both --model_path and --audio_file are required." << std::endl;
    return 1;
  }

  try {
    StreamingTranscribe(model_path, audio_file, enable_vad);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
