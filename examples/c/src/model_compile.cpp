// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Model Compile example: runs the same model under different EP and compile configurations
// (CPU, CPU+overlay, NvTensorRtRtx no-compile / 4 options / all options). Use -v for verbose,
// -d for ORT verbose logging (ORTGENAI_ORT_VERBOSE_LOGGING=1).

#include <chrono>
#include <cstdlib>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

#include "common.h"

namespace fs = std::filesystem;

// Enable ONNX Runtime verbose logging. Must be set before any Oga/ORT API use.
// Alternatively set env ORTGENAI_ORT_VERBOSE_LOGGING=1 before launching.
static void SetOrtVerboseLogging() {
#ifdef _WIN32
  _putenv("ORTGENAI_ORT_VERBOSE_LOGGING=1");
#else
  setenv("ORTGENAI_ORT_VERBOSE_LOGGING", "1", 1);
#endif
}

static const char* kCpuEp = "cpu";
static const char* kNvTensorRtRtxEp = "NvTensorRtRtx";

static const char* kDefaultPrompt = "Tell me about AI and ML";

static double RunOneGeneration(OgaModel& model, OgaTokenizer& tokenizer, bool verbose) {
  auto stream = OgaTokenizerStream::Create(tokenizer);
  auto sequences = OgaSequences::Create();
  tokenizer.Encode(kDefaultPrompt, *sequences);

  auto params = OgaGeneratorParams::Create(model);
  params->SetSearchOption("max_length", 128);
  params->SetSearchOption("batch_size", 1);

  auto generator = OgaGenerator::Create(model, *params);
  generator->AppendTokenSequences(*sequences);

  if (verbose) std::cout << "Prompt: " << kDefaultPrompt << std::endl;
  std::cout << "Output: " << std::flush;
  auto t0 = Clock::now();
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
    std::cout << stream->Decode(generator->GetNextTokens()[0]) << std::flush;
  }
  std::cout << std::endl;
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

static void PrintTimings(const char* label, double load_time_sec, double inference_time_sec) {
  const auto default_precision = std::cout.precision();
  std::cout << "  " << label << ": "
            << std::fixed << std::setprecision(3)
            << "model load " << load_time_sec << "s, "
            << "inference " << inference_time_sec << "s"
            << std::setprecision(default_precision) << std::endl;
}

// 1) Run model with CPU execution provider only (no compile overlay).
void RunWithCpu(const std::string& model_path, const std::string& ep_path, bool verbose) {
  (void)ep_path;
  if (verbose) std::cout << "[RunWithCpu] Creating config (CPU, no compile overlay)..." << std::endl;
  std::unordered_map<std::string, std::string> ep_options;
  GeneratorParamsArgs search_options;
  auto config = GetConfig(model_path, kCpuEp, ep_options, search_options);
  if (verbose) std::cout << "[RunWithCpu] Creating model..." << std::endl;
  auto load_t0 = Clock::now();
  auto model = OgaModel::Create(*config);
  double load_time = std::chrono::duration<double>(Clock::now() - load_t0).count();
  if (verbose) std::cout << "[RunWithCpu] Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);
  double inference_time = RunOneGeneration(*model, *tokenizer, verbose);
  PrintTimings("RunWithCpu (CPU, no overlay)", load_time, inference_time);
}

// 2) Run model with CPU execution provider and compile config passed via config_overlay.
void RunWithCpuAndCompileOverlay(const std::string& model_path, const std::string& ep_path, bool verbose) {
  (void)ep_path;
  if (verbose) std::cout << "[RunWithCpuAndCompileOverlay] Creating config (CPU + compile overlay)..." << std::endl;
  std::unordered_map<std::string, std::string> ep_options;
  GeneratorParamsArgs search_options;
  auto config = GetConfig(model_path, kCpuEp, ep_options, search_options);
  config->Overlay(R"({
    "model": {
      "decoder": {
        "compile_options": {
          "enable_ep_context": true,
          "ep_context_embed_mode": false,
          "force_compile_if_needed": true,
          "graph_optimization_level": 99
        }
      }
    }
  })");
  if (verbose) std::cout << "[RunWithCpuAndCompileOverlay] Creating model..." << std::endl;
  auto load_t0 = Clock::now();
  auto model = OgaModel::Create(*config);
  double load_time = std::chrono::duration<double>(Clock::now() - load_t0).count();
  if (verbose) std::cout << "[RunWithCpuAndCompileOverlay] Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);
  double inference_time = RunOneGeneration(*model, *tokenizer, verbose);
  PrintTimings("RunWithCpuAndCompileOverlay (CPU + overlay)", load_time, inference_time);
}

// 3) Run model with NvTensorRtRtx EP without compile options.
void RunWithNvTensorRtRtxNoCompile(const std::string& model_path, const std::string& ep_path, bool verbose) {
  if (ep_path.empty() && verbose) {
    std::cout << "Warning: --ep_path not set; NvTensorRTRTX may not be available (only CPU)." << std::endl;
  }
  if (verbose) std::cout << "[RunWithNvTensorRtRtxNoCompile] Creating config (NvTensorRtRtx, no compile)..." << std::endl;
  std::unordered_map<std::string, std::string> ep_options;
  GeneratorParamsArgs search_options;
  auto config = GetConfig(model_path, kNvTensorRtRtxEp, ep_options, search_options);
  if (verbose) std::cout << "[RunWithNvTensorRtRtxNoCompile] Creating model..." << std::endl;
  auto load_t0 = Clock::now();
  auto model = OgaModel::Create(*config);
  double load_time = std::chrono::duration<double>(Clock::now() - load_t0).count();
  if (verbose) std::cout << "[RunWithNvTensorRtRtxNoCompile] Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);
  double inference_time = RunOneGeneration(*model, *tokenizer, verbose);
  PrintTimings("RunWithNvTensorRtRtxNoCompile (NvTensorRtRtx, no compile)", load_time, inference_time);
}

// 4) Run model with NvTensorRtRtx EP and minimum compile options.
void RunWithNvTensorRtRtxMinimumCompileOptions(const std::string& model_path, const std::string& ep_path, bool verbose) {
  if (ep_path.empty() && verbose) {
    std::cout << "Warning: --ep_path not set; NvTensorRTRTX may not be available (only CPU)." << std::endl;
  }
  if (verbose) std::cout << "[RunWithNvTensorRtRtxMinimumCompileOptions] Creating config (NvTensorRtRtx + minimum compile options)..." << std::endl;
  std::unordered_map<std::string, std::string> ep_options;
  GeneratorParamsArgs search_options;
  auto config = GetConfig(model_path, kNvTensorRtRtxEp, ep_options, search_options);
  // ep_context_embed_mode must be false for larger models(>2GB) or compilation will error
  config->Overlay(R"({
    "model": {
      "decoder": {
        "compile_options": {
          "enable_ep_context": true,
          "ep_context_embed_mode": false
        }
      }
    }
  })");
  if (verbose) std::cout << "[RunWithNvTensorRtRtxMinimumCompileOptions] Creating model..." << std::endl;
  auto load_t0 = Clock::now();
  auto model = OgaModel::Create(*config);
  double load_time = std::chrono::duration<double>(Clock::now() - load_t0).count();
  if (verbose) std::cout << "[RunWithNvTensorRtRtxMinimumCompileOptions] Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);
  double inference_time = RunOneGeneration(*model, *tokenizer, verbose);
  PrintTimings("RunWithNvTensorRtRtxMinimumCompileOptions (minimum options)", load_time, inference_time);
}

// 5) Run model with NvTensorRtRtx EP and all compile options.
void RunWithNvTensorRtRtxCompileAllOptions(const std::string& model_path, const std::string& ep_path, bool verbose) {
  if (ep_path.empty() && verbose) {
    std::cout << "Warning: --ep_path not set; NvTensorRTRTX may not be available (only CPU)." << std::endl;
  }
  if (verbose) std::cout << "[RunWithNvTensorRtRtxCompileAllOptions] Creating config (NvTensorRtRtx + all compile options)..." << std::endl;
  std::unordered_map<std::string, std::string> ep_options;
  GeneratorParamsArgs search_options;
  auto config = GetConfig(model_path, kNvTensorRtRtxEp, ep_options, search_options);
  // Single config: ep_context_file_path is full path (relative to model dir) including filename, e.g. "contexts/model_ctx.onnx"
  config->Overlay(R"({
    "model": {
      "decoder": {
        "compile_options": {
          "enable_ep_context": true,
          "graph_optimization_level": 99,
          "ep_context_file_path": "contexts/ep_context_output/model_ctx.onnx",
          "ep_context_embed_mode": false,
          "force_compile_if_needed": true
        }
      }
    }
  })");
  if (verbose) std::cout << "[RunWithNvTensorRtRtxCompileAllOptions] Creating model..." << std::endl;
  auto load_t0 = Clock::now();
  auto model = OgaModel::Create(*config);
  double load_time = std::chrono::duration<double>(Clock::now() - load_t0).count();
  if (verbose) std::cout << "[RunWithNvTensorRtRtxCompileAllOptions] Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);
  double inference_time = RunOneGeneration(*model, *tokenizer, verbose);
  PrintTimings("RunWithNvTensorRtRtxCompileAllOptions (all options)", load_time, inference_time);
}

int main(int argc, char** argv) {
  GeneratorParamsArgs generator_params_args;
  GuidanceArgs guidance_args;
  std::string model_path, ep = "follow_config", ep_path, system_prompt, user_prompt;
  bool verbose = false, debug = false, interactive = false, rewind = true;
  std::vector<std::string> image_paths, audio_paths;

  if (!ParseArgs(argc, argv, generator_params_args, guidance_args, model_path, ep, ep_path, system_prompt, user_prompt, verbose, debug, interactive, rewind, image_paths, audio_paths)) {
    return -1;
  }

  if (ep.compare(kNvTensorRtRtxEp) == 0 && ep_path.empty()) {
#if defined(_WIN32)
    ep_path = (fs::current_path() / "onnxruntime_providers_nv_tensorrt_rtx.dll").string();
#else
    ep_path = (fs::current_path() / "libonnxruntime_providers_nv_tensorrt_rtx.so").string();
#endif
  }

  if (debug) {
    SetOrtVerboseLogging();
    SetLogger();
  }

  if (!ep_path.empty()) {
    RegisterEP(kNvTensorRtRtxEp, ep_path);
  }

  OgaHandle handle;

  if (verbose) {
    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "EP path: " << (ep_path.empty() ? "(none)" : ep_path) << std::endl;
  }
  std::cout << "Timings (model load, inference):" << std::endl;

  try {
    // RunWithCpu(model_path, ep_path, verbose);
    // RunWithCpuAndCompileOverlay(model_path, ep_path, verbose);
    //First run the no-compile case
    RunWithNvTensorRtRtxNoCompile(model_path, ep_path, verbose);
    //Then run for first time compile case, Model load time will be load time at no compile + compile time
    RunWithNvTensorRtRtxMinimumCompileOptions(model_path, ep_path, verbose);
    //Then run for second time compile case, Model load time must be very less as it is already compiled
    RunWithNvTensorRtRtxMinimumCompileOptions(model_path, ep_path, verbose);
    //Then run for all compile options,With different ep_context_file_path, ep_context_embed_mode, force_compile_if_needed, graph_optimization_level
    RunWithNvTensorRtRtxCompileAllOptions(model_path, ep_path, verbose);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
