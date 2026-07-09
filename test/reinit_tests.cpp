// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Dedicated test binary for GenAI shutdown / re-initialization.
//
// These tests call OgaShutdown() to prove GenAI can be torn down and re-initialized in-process
// (the Foundry Local `Manager` recreate scenario — see docs/PluginEPSharedAllocator.md §2/§3).
// They live in their OWN executable rather than in unit_tests because OgaShutdown() resets
// process-global GenAI state (the ORT environment, the device interfaces, the genai add-on
// libraries, and any registered plugin EP libraries). Running them inside the shared suite would
// couple every other test to their teardown and depend on test ordering; isolating them removes
// that fragility. This binary deliberately does not register any plugin EPs — re-init is exercised
// on the always-present CPU path.

#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "ort_genai.h"
#include "generators.h"  // Generators::GetDeviceInterface / DeviceType (from the genai object library)

#include <gtest/gtest.h>

#include "ep_registration.h"
#include "test_utils.h"

// Plugin EPs discovered at startup (WinML packages by default, plus any --ep_dir). Registered on
// genai's env at the start of each re-init cycle by the test below, so their device interfaces are
// created and then torn down by OgaShutdown() each cycle. (generators.h defines a global `fs` type,
// so this file qualifies std::filesystem explicitly rather than adding its own alias.)
test_ep::EpRegistrar g_ep_registrar;

namespace {

// Runs a minimal CPU generate workload and validates the output. Every GenAI object is destroyed
// before returning, so nothing holding device memory outlives a subsequent OgaShutdown() (§5
// lifetime contract).
void RunCpuWorkload() {
  std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};
  std::vector<int32_t> expected_output{
      0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
      0, 0, 195, 731, 731, 114, 114, 114, 114, 114};
  const int batch_size = 2;
  const int max_length = 10;

  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", batch_size);
  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids.data(), input_ids.size());
  while (!generator->IsDone())
    generator->GenerateNextToken();

  for (int i = 0; i < batch_size; ++i) {
    const auto sequence_length = generator->GetSequenceCount(i);
    const auto* sequence_data = generator->GetSequenceData(i);
    ASSERT_LE(sequence_length, static_cast<size_t>(max_length));
    EXPECT_TRUE(0 == std::memcmp(&expected_output[i * max_length], sequence_data,
                                 sequence_length * sizeof(int32_t)));
  }
}

// Forces creation of the DeviceInterface for `provider` directly via genai's internal
// GetDeviceInterface (no model load), so the create -> teardown -> recreate path is exercised across
// re-init cycles (for the CUDA add-on this also loads and later unloads the add-on library).
// The created interface is owned by OrtGlobals, so the subsequent OgaShutdown() tears it down.
void ExerciseDeviceInterface(const std::string& provider) {
  Generators::DeviceType device_type;
  if (provider == "WebGPU") {
    device_type = Generators::DeviceType::WEBGPU;
  } else if (provider == "cuda") {
    device_type = Generators::DeviceType::CUDA;
  } else if (provider == "OpenVINO") {
    device_type = Generators::DeviceType::OpenVINO;
  } else if (provider == "NvTensorRtRtx") {
    device_type = Generators::DeviceType::NvTensorRtRtx;
  } else {
    return;  // no genai DeviceInterface mapping for this provider
  }

  // Creating the interface registers it with OrtGlobals (so OgaShutdown tears it down / unloads the
  // CUDA add-on). It may throw if the EP/add-on is not usable on this machine (e.g. the CUDA add-on
  // library is absent), which the caller treats as skip.
  Generators::GetDeviceInterface(device_type);
}

}  // namespace

// Primary driver (§2/§3): after OgaShutdown() GenAI returns to a just-loaded state and the next
// GenAI call re-initializes cleanly with a fresh OrtEnv. Each cycle registers the discovered plugin
// EPs (on the fresh env), runs a CPU workload plus a workload per available provider (to create and
// then tear down each EP's device interface), and calls OgaShutdown(). Repeating this proves that
// repeated register -> use -> shutdown -> re-init is stable, including reloading the CUDA add-on.
TEST(ReInitTests, ShutdownReInitCycle) {
  const int kCycles = 3;
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    SCOPED_TRACE("re-init cycle " + std::to_string(cycle));

    // Register discovered plugin EPs on the current (fresh, after the previous OgaShutdown) env.
    const std::vector<std::string>& providers = g_ep_registrar.RegisterAll();

    RunCpuWorkload();  // always-present CPU path (full model/session/generator teardown)

    // Force creation + teardown of each available EP's device interface. Best-effort: an EP's genai
    // add-on may be absent (e.g. CUDA without onnxruntime-genai-cuda) or its allocator unavailable,
    // so a failure just skips that provider this cycle.
    for (const std::string& provider : providers) {
      try {
        ExerciseDeviceInterface(provider);
        std::cout << "cycle " << cycle << ": exercised device interface for '" << provider << "'"
                  << std::endl;
      } catch (const std::exception& e) {
        std::cout << "cycle " << cycle << ": skipped provider '" << provider << "': " << e.what()
                  << std::endl;
      }
    }

    OgaShutdown();  // full teardown: env + all created device interfaces + add-on libraries
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // --ep_dir <dir>: also register plugin EP libraries found under <dir> (recursive).
  std::filesystem::path ep_dir;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--ep_dir" && i + 1 < argc)
      ep_dir = argv[++i];
  }

  // Register any WinML-installed EP packages by default (Windows) so re-init is exercised with real
  // device interfaces where available, plus any --ep_dir. Discovery is filesystem/PowerShell only
  // (no genai env); the test registers these on genai's env at the start of each cycle.
#if defined(_WIN32)
  g_ep_registrar.DiscoverWinML();
#endif
  g_ep_registrar.DiscoverFromDirectory(ep_dir);

  const int result = RUN_ALL_TESTS();
  OgaShutdown();
  return result;
}
