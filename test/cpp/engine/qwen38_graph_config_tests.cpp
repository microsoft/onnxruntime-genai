// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "config.h"
#include "models/model_type.h"

namespace Generators::test {
namespace {

struct GraphConfigDirectory {
  GraphConfigDirectory() {
    path = std::filesystem::temp_directory_path() /
           ("ortgenai_qwen38_graph_config_test_" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(path);
  }
  ~GraphConfigDirectory() { std::filesystem::remove_all(path); }

  void Write(const std::string& graph) const {
    std::ofstream out(path / "genai_config.json");
    out << R"({"model":{"type":"qwen38_flash_graph","vocab_size":16,"context_length":32,)"
        << R"("decoder":{"filename":"model.onnx"},"qwen38_graph":)" << graph
        << R"(},"search":{}})";
  }

  std::filesystem::path path;
};

TEST(Qwen38GraphConfig, ParsesGraphSessionsAndProviderOptions) {
  GraphConfigDirectory directory;
  directory.Write(R"({
    "plane_library":"libs/libqwen38_plane.so","ple_table":"ple_table.txt","kv_capacity":128,
    "target_s1_graph_id":11,"target_s2_graph_id":21,"head_s1_graph_id":31,"head_s2_graph_id":41,
    "warmup":false,
    "head":{"filename_s1":"head1.onnx","filename_s2":"head2.onnx",
            "session_options":{"provider_options":[{"CUDA":{"device_id":"0"}}]}},
    "mtp":{"enabled":true,"prefill_head":false,"skip_refeed":true},
    "fold":{"enabled":true,"target_graph_id":51,"force_reject":true,
            "head_r2":{"filename":"head_r2.onnx","graph_id":61,
                       "session_options":{"provider_options":[{"CUDA":{"device_id":"0"}}]}}},
    "prefill":{"enabled":true,"chunk":64,"weights_pool":"weights.txt","constants":"constants.txt",
               "warmup_lengths":[64,16],
               "dense":{"filename":"dense.onnx",
                        "session_options":{"provider_options":[{"CUDA":{"device_id":"0"}}]}},
               "head_kv":{"filename":"head_kv.onnx",
                          "session_options":{"provider_options":[{"CUDA":{"device_id":"0"}}]}}}
  })");
  Config config(directory.path.string(), {});
  const auto& graph = config.model.qwen38_graph;
  EXPECT_TRUE(ModelType::IsLLM(config.model.type));
  EXPECT_FALSE(ModelType::IsVLM(config.model.type));
  EXPECT_TRUE(graph.IsConfigured());
  EXPECT_EQ(graph.kv_capacity, 128);
  EXPECT_FALSE(graph.warmup);
  EXPECT_FALSE(graph.mtp.prefill_head);
  EXPECT_TRUE(graph.mtp.skip_refeed);
  EXPECT_TRUE(graph.fold.force_reject);
  EXPECT_EQ(graph.fold.head_r2.graph_id, 61);
  EXPECT_EQ(graph.prefill.chunk, 64);
  EXPECT_EQ(graph.prefill.warmup_lengths, (std::vector<int>{64, 16}));
  for (const auto* options : {&graph.head.session_options, &graph.fold.head_r2.session_options,
                              &graph.prefill.dense.session_options, &graph.prefill.head_kv.session_options}) {
    ASSERT_TRUE(options->has_value());
    EXPECT_EQ((*options)->providers, (std::vector<std::string>{"cuda"}));
  }
}

TEST(Qwen38GraphConfig, RejectsUnknownSettingsAndEscapingPaths) {
  GraphConfigDirectory directory;
  for (const auto* graph : {R"({"unknown":true})",
                            R"({"plane_library":"../plane.so"})",
                            R"({"head":{"filename_s1":"../head.onnx"}})",
                            R"({"prefill":{"constants":"../constants.txt"}})"}) {
    directory.Write(graph);
    EXPECT_THROW((Config{directory.path.string(), {}}), std::runtime_error);
  }
}

TEST(Qwen38GraphConfig, FoldedConfigurationDoesNotRequireUnusedMtpHeadFilename) {
  GraphConfigDirectory directory;
  directory.Write(R"({
    "plane_library":"libs/libdflash_plane_v2.so","ple_table":"ple_table.txt",
    "fold":{"enabled":true},"prefill":{"enabled":true,"unified":true}
  })");
  Config config(directory.path.string(), {});
  EXPECT_TRUE(config.model.qwen38_graph.IsConfigured());
  EXPECT_TRUE(config.model.qwen38_graph.fold.head_r2.filename.empty());
  EXPECT_TRUE(config.model.qwen38_graph.head.filename_s1.empty());
}

}  // namespace
}  // namespace Generators::test
