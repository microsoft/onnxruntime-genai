// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Portions of this file consist of AI generated content.

// Unit tests for the pipeline plugin loader escape hatch (issue #2114, PR4). The default build ships
// with USE_GENAI_PLUGINS=OFF, so the dlopen body is compiled out and any plugin config must fail
// loudly with a clear "plugin support is not enabled" error rather than being silently misrouted to a
// built-in model. These tests construct a Config with `pipeline.plugin` set and assert the loader and
// the structural CreatePipeline() dispatch both throw. When built with USE_GENAI_PLUGINS=ON the same
// configs still throw (the bogus library path cannot be loaded), so the tests are valid in both modes.

#include <string>
#include <gtest/gtest.h>

#include "generators.h"
#include "models/plugin_loader.h"

namespace {

std::unique_ptr<Generators::Config> MakePluginConfig() {
  auto config = std::make_unique<Generators::Config>();
  config->pipeline.present = true;
  Generators::Config::Pipeline::Plugin plugin;
  plugin.library = "libgenai_nonexistent_plugin.so";
  plugin.entry_point = "OgaCreatePipeline";
  config->pipeline.plugin = plugin;
  return config;
}

}  // namespace

// LoadPluginPipeline always throws in the default (USE_GENAI_PLUGINS=OFF) build, and in the ON build
// it throws because the library cannot be loaded. Either way a plugin config never silently succeeds.
TEST(PluginLoaderTests, LoadPluginPipelineThrows) {
  EXPECT_THROW(
      { Generators::LoadPluginPipeline(*MakePluginConfig()->pipeline.plugin, MakePluginConfig(), Generators::GetOrtEnv()); },
      std::runtime_error);
}

// The structural CreatePipeline() dispatch recognizes pipeline.plugin and routes it to the loader
// (rather than to any built-in model), so it surfaces the same throw.
TEST(PluginLoaderTests, CreatePipelineRoutesPluginToLoader) {
  EXPECT_THROW(
      { Generators::CreatePipeline(Generators::GetOrtEnv(), MakePluginConfig()); },
      std::runtime_error);
}

#if !USE_GENAI_PLUGINS
// In the default build the error message must clearly state that plugin support is disabled and how
// to enable it, so misconfigurations are actionable.
TEST(PluginLoaderTests, DisabledBuildReportsClearMessage) {
  try {
    Generators::CreatePipeline(Generators::GetOrtEnv(), MakePluginConfig());
    FAIL() << "Expected CreatePipeline to throw for a plugin config in a USE_GENAI_PLUGINS=OFF build.";
  } catch (const std::runtime_error& e) {
    const std::string message = e.what();
    EXPECT_NE(message.find("not enabled"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("USE_GENAI_PLUGINS=ON"), std::string::npos) << "message: " << message;
  }
}
#endif
