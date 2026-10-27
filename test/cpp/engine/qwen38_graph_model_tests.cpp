// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "config.h"
#include "models/model_type.h"
#include "models/qwen38_decode_device.h"
#include "models/qwen38_graph_plane.h"

namespace Generators::test {

namespace {

std::filesystem::path MakeScratchDir(const std::string& suffix) {
  const auto dir = std::filesystem::temp_directory_path() / ("genai_qwen38_graph_" + suffix);
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

std::filesystem::path WriteFile(const std::filesystem::path& path, const std::string& text) {
  std::ofstream out{path};
  out << text;
  return path;
}

constexpr const char* kValidPleTable =
    "qwen38_ple_table 1\n"
    "n_heads 2\n"
    "head_dim 4\n"
    "heads_per_ngram 1\n"
    "eos 7\n"
    "scale 0.5\n"
    "mult 1 2 3 0\n"
    "vocab 11 13\n"
    "offset 0 11\n"
    "shard ple/a.safetensors 128 11\n"
    "shard ple/b.safetensors 0 13\n";

}  // namespace

// ---- config ----------------------------------------------------------------------------------

TEST(Qwen38GraphConfigTest, DefaultsAreUnconfigured) {
  Config config;
  EXPECT_FALSE(config.model.qwen38_graph.IsConfigured());
  EXPECT_EQ(config.model.qwen38_graph.kv_capacity, 33028);
  EXPECT_EQ(config.model.qwen38_graph.target_s1_graph_id, 41);
  EXPECT_EQ(config.model.qwen38_graph.target_s2_graph_id, 51);
  EXPECT_EQ(config.model.qwen38_graph.head_s1_graph_id, 61);
  EXPECT_EQ(config.model.qwen38_graph.head_s2_graph_id, 63);
  EXPECT_EQ(config.model.qwen38_graph.prefill_chunk_tokens, 2048);
  EXPECT_TRUE(config.model.qwen38_graph.warmup);
  EXPECT_TRUE(config.model.qwen38_graph.mtp.enabled);
  EXPECT_TRUE(config.model.qwen38_graph.mtp.prefill_head);
  EXPECT_FALSE(config.model.qwen38_graph.mtp.skip_refeed);
}

TEST(Qwen38GraphConfigTest, ParsesEverySetting) {
  Config config;
  OverlayConfig(config, R"({"model":{"type":"qwen38_flash_graph","qwen38_graph":{
    "plane_library":"libs/libqwen38_plane.so",
    "ple_table":"ple_table.txt",
    "kv_capacity":1024,
    "target_s1_graph_id":10,
    "target_s2_graph_id":20,
    "head_s1_graph_id":30,
    "head_s2_graph_id":40,
    "prefill_chunk_tokens":512,
    "warmup":false,
    "mtp":{"enabled":false,"prefill_head":false,"skip_refeed":true},
    "head":{"filename_s1":"head/s1.onnx","filename_s2":"head/s2.onnx",
            "session_options":{"custom_ops_libraries":["libs/a.so","libs/b.so"],
                               "provider_options":[{"cuda":{"enable_cuda_graph":"1"}}]}}
  }}})");

  const auto& q = config.model.qwen38_graph;
  EXPECT_TRUE(q.IsConfigured());
  EXPECT_EQ(q.plane_library, "libs/libqwen38_plane.so");
  EXPECT_EQ(q.ple_table, "ple_table.txt");
  EXPECT_EQ(q.kv_capacity, 1024);
  EXPECT_EQ(q.target_s1_graph_id, 10);
  EXPECT_EQ(q.target_s2_graph_id, 20);
  EXPECT_EQ(q.head_s1_graph_id, 30);
  EXPECT_EQ(q.head_s2_graph_id, 40);
  EXPECT_EQ(q.prefill_chunk_tokens, 512);
  EXPECT_FALSE(q.warmup);
  EXPECT_FALSE(q.mtp.enabled);
  EXPECT_FALSE(q.mtp.prefill_head);
  EXPECT_TRUE(q.mtp.skip_refeed);
  EXPECT_EQ(q.head.filename_s1, "head/s1.onnx");
  EXPECT_EQ(q.head.filename_s2, "head/s2.onnx");
  ASSERT_TRUE(q.head.session_options.has_value());
  EXPECT_EQ(q.head.session_options->custom_ops_libraries, (std::vector<std::string>{"libs/a.so", "libs/b.so"}));
}

TEST(Qwen38GraphConfigTest, FoldAndPrefillDefaultToOff) {
  Config config;
  const auto& q = config.model.qwen38_graph;
  EXPECT_FALSE(q.fold.enabled);
  EXPECT_FALSE(q.fold.force_reject);
  EXPECT_EQ(q.fold.target_graph_id, 71);
  EXPECT_EQ(q.fold.head_r2.graph_id, 81);
  EXPECT_FALSE(q.prefill.enabled);
  EXPECT_EQ(q.prefill.chunk, 8192);
  EXPECT_EQ(q.prefill.warmup_lengths, (std::vector<int>{8192, 2048, 512, 128, 37}));
}

TEST(Qwen38GraphConfigTest, ParsesTheFoldAndPrefillSections) {
  Config config;
  OverlayConfig(config, R"({"model":{"type":"qwen38_flash_graph","qwen38_graph":{
    "plane_library":"libs/libqwen38_plane.so",
    "ple_table":"ple_table.txt",
    "fold":{"enabled":true,"target_graph_id":90,"force_reject":true,
            "head_r2":{"filename":"head_r2/model.onnx","graph_id":95,
                       "session_options":{"custom_ops_libraries":["libs/a.so"]}}},
    "prefill":{"enabled":true,"chunk":4096,"weights_pool":"pool.txt","constants":"qsa.txt",
               "dense":{"filename":"dense/model.onnx","session_options":{"provider_options":[{"cuda":{}}]}},
               "head_kv":{"filename":"head_kv/model.onnx"},
               "warmup_lengths":[64,16]}
  }}})");

  const auto& q = config.model.qwen38_graph;
  EXPECT_TRUE(q.IsConfigured());  // no head.filename_s1: the fold head_r2 session is enough
  EXPECT_TRUE(q.fold.enabled);
  EXPECT_TRUE(q.fold.force_reject);
  EXPECT_EQ(q.fold.target_graph_id, 90);
  EXPECT_EQ(q.fold.head_r2.filename, "head_r2/model.onnx");
  EXPECT_EQ(q.fold.head_r2.graph_id, 95);
  ASSERT_TRUE(q.fold.head_r2.session_options.has_value());
  EXPECT_EQ(q.fold.head_r2.session_options->custom_ops_libraries, (std::vector<std::string>{"libs/a.so"}));
  EXPECT_TRUE(q.prefill.enabled);
  EXPECT_EQ(q.prefill.chunk, 4096);
  EXPECT_EQ(q.prefill.weights_pool, "pool.txt");
  EXPECT_EQ(q.prefill.constants, "qsa.txt");
  EXPECT_EQ(q.prefill.dense.filename, "dense/model.onnx");
  ASSERT_TRUE(q.prefill.dense.session_options.has_value());
  EXPECT_EQ(q.prefill.head_kv.filename, "head_kv/model.onnx");
  EXPECT_FALSE(q.prefill.head_kv.session_options.has_value());
  EXPECT_EQ(q.prefill.warmup_lengths, (std::vector<int>{64, 16}));

  // Also verify "mtp" and "drafter" aliases under fold
  Config config_mtp;
  OverlayConfig(config_mtp, R"({"model":{"type":"qwen38_flash_graph","qwen38_graph":{
    "fold":{"enabled":true,"mtp":{"filename":"mtp/model.onnx","graph_id":81}}
  }}})");
  EXPECT_EQ(config_mtp.model.qwen38_graph.fold.head_r2.filename, "mtp/model.onnx");

  Config config_drafter;
  OverlayConfig(config_drafter, R"({"model":{"type":"qwen38_flash_graph","qwen38_graph":{
    "fold":{"enabled":true,"drafter":{"filename":"drafter/model.onnx","graph_id":82}}
  }}})");
  EXPECT_EQ(config_drafter.model.qwen38_graph.fold.head_r2.filename, "drafter/model.onnx");
  EXPECT_EQ(config_drafter.model.qwen38_graph.fold.head_r2.graph_id, 82);
}

TEST(Qwen38GraphConfigTest, RejectsUnknownFoldAndPrefillKeys) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({"model":{"qwen38_graph":{"fold":{"unknown":true}}}})"), std::exception);
  EXPECT_THROW(OverlayConfig(config, R"({"model":{"qwen38_graph":{"fold":{"head_r2":{"unknown":1}}}}})"), std::exception);
  EXPECT_THROW(OverlayConfig(config, R"({"model":{"qwen38_graph":{"prefill":{"unknown":1}}}})"), std::exception);
  EXPECT_THROW(OverlayConfig(config, R"({"model":{"qwen38_graph":{"prefill":{"dense":{"unknown":1}}}}})"), std::exception);
}

TEST(Qwen38GraphConfigTest, FoldAndPrefillPathsMustStayInsideTheModelFolder) {
  const auto dir = MakeScratchDir("fold_paths");
  const auto write_config = [&](const std::string& qwen_section) {
    WriteFile(dir / "genai_config.json",
              R"({"model":{"type":"qwen38_flash_graph","vocab_size":8,"context_length":16,
                  "decoder":{"filename":"model.onnx","hidden_size":4,"num_attention_heads":1,"num_hidden_layers":1,
                             "num_key_value_heads":1,"head_size":4},
                  "qwen38_graph":)" +
                  qwen_section + R"(},
                  "search":{"max_length":16}})");
  };
  write_config(R"({"plane_library":"p.so","ple_table":"ple.txt","fold":{"head_r2":{"filename":"r2.onnx"}},
                   "prefill":{"weights_pool":"pool.txt","constants":"c.txt","dense":{"filename":"d.onnx"},
                              "head_kv":{"filename":"k.onnx"}}})");
  EXPECT_NO_THROW(Config(fs::path(dir.string()), {}));
  for (const char* section : {R"("fold":{"head_r2":{"filename":"../r2.onnx"}})", R"("prefill":{"weights_pool":"/abs/pool.txt"})",
                              R"("prefill":{"constants":"../c.txt"})", R"("prefill":{"dense":{"filename":"/abs/d.onnx"}})",
                              R"("prefill":{"head_kv":{"filename":"../k.onnx"}})"}) {
    write_config(std::string(R"({"plane_library":"p.so","ple_table":"ple.txt",)") + section + "}");
    EXPECT_THROW(Config(fs::path(dir.string()), {}), std::runtime_error) << section;
  }
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Qwen38GraphConfigTest, RejectsUnknownKeys) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({"model":{"qwen38_graph":{"unknown":1}}})"), std::exception);
  EXPECT_THROW(OverlayConfig(config, R"({"model":{"qwen38_graph":{"mtp":{"unknown":true}}}})"), std::exception);
  EXPECT_THROW(OverlayConfig(config, R"({"model":{"qwen38_graph":{"head":{"unknown":"x"}}}})"), std::exception);
}

TEST(Qwen38GraphConfigTest, HeadProviderOptionsBecomeProviders) {
  Config config;
  OverlayConfig(config, R"({"model":{"qwen38_graph":{"head":{"filename_s1":"h.onnx","session_options":{
    "provider_options":[{"cuda":{"enable_cuda_graph":"1"}}]}}}}})");
  // The Config constructor normalises provider_options into providers; an overlay alone must at least keep the options.
  ASSERT_TRUE(config.model.qwen38_graph.head.session_options.has_value());
  ASSERT_EQ(config.model.qwen38_graph.head.session_options->provider_options.size(), 1u);
  EXPECT_EQ(config.model.qwen38_graph.head.session_options->provider_options[0].name, "cuda");
}

TEST(Qwen38GraphConfigTest, TypeIsRegisteredAsALanguageModel) {
  EXPECT_TRUE(ModelType::IsLLM("qwen38_flash_graph"));
}

TEST(Qwen38GraphConfigTest, ModelPathsMustStayInsideTheModelFolder) {
  const auto dir = MakeScratchDir("paths");
  const auto write_config = [&](const std::string& qwen_section) {
    WriteFile(dir / "genai_config.json",
              R"({"model":{"type":"qwen38_flash_graph","vocab_size":8,"context_length":16,
                  "decoder":{"filename":"model.onnx","hidden_size":4,"num_attention_heads":1,"num_hidden_layers":1,
                             "num_key_value_heads":1,"head_size":4},
                  "qwen38_graph":)" +
                  qwen_section + R"(},
                  "search":{"max_length":16}})");
  };
  std::error_code ec;
  write_config(R"({"plane_library":"libs/plane.so","ple_table":"ple.txt","head":{"filename_s1":"h1.onnx"}})");
  EXPECT_NO_THROW(Config(fs::path(dir.string()), {}));
  for (const char* key : {"plane_library", "ple_table"}) {
    write_config(std::string(R"({")") + key + R"(":"../escape","head":{"filename_s1":"h1.onnx"}})");
    EXPECT_THROW(Config(fs::path(dir.string()), {}), std::runtime_error) << key;
    write_config(std::string(R"({")") + key + R"(":"/abs/path","head":{"filename_s1":"h1.onnx"}})");
    EXPECT_THROW(Config(fs::path(dir.string()), {}), std::runtime_error) << key;
  }
  write_config(R"({"head":{"filename_s1":"../h1.onnx"}})");
  EXPECT_THROW(Config(fs::path(dir.string()), {}), std::runtime_error);
  write_config(R"({"head":{"filename_s1":"h1.onnx","filename_s2":"/abs/h2.onnx"}})");
  EXPECT_THROW(Config(fs::path(dir.string()), {}), std::runtime_error);
  std::filesystem::remove_all(dir, ec);
}

// ---- weight pool manifest and QSA constants index --------------------------------------------

TEST(Qwen38WeightPoolManifestTest, ParsesEntriesAndSessions) {
  const auto dir = MakeScratchDir("pool");
  const auto path = WriteFile(dir / "pool.txt",
                              "qwen38_weights_pool 1\n"
                              "pool 0 16 1000 64 2 4 8 target/w0.safetensors\n"
                              "pool 1 2 0 128 3 2 4 8 target/experts.data\n"
                              "session target model.embed.weight 0\n"
                              "session target experts.w 1\n"
                              "session dense model.embed.weight 0\n");
  const auto manifest = ParseQwen38WeightPoolManifest(path);
  ASSERT_EQ(manifest.entries.size(), 2u);
  EXPECT_EQ(manifest.entries[0].dtype, 16);
  EXPECT_EQ(manifest.entries[0].offset, 1000u);
  EXPECT_EQ(manifest.entries[0].length, 64u);
  EXPECT_EQ(manifest.entries[0].shape, (std::vector<int64_t>{4, 8}));
  EXPECT_EQ(manifest.entries[0].file, "target/w0.safetensors");
  EXPECT_EQ(manifest.entries[1].shape, (std::vector<int64_t>{2, 4, 8}));
  ASSERT_EQ(manifest.sessions.at("target").size(), 2u);
  EXPECT_EQ(manifest.sessions.at("target")[1], (std::pair<std::string, int>{"experts.w", 1}));
  EXPECT_EQ(manifest.sessions.at("dense").size(), 1u);
  EXPECT_EQ(manifest.sessions.count("head_kv"), 0u);
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Qwen38WeightPoolManifestTest, RejectsMalformedManifests) {
  const auto dir = MakeScratchDir("pool_bad");
  const auto parse = [&](const std::string& text) { return ParseQwen38WeightPoolManifest(WriteFile(dir / "p.txt", text)); };
  EXPECT_THROW(ParseQwen38WeightPoolManifest(dir / "missing.txt"), std::runtime_error);
  EXPECT_THROW(parse("qwen38_weights_pool 2\n"), std::runtime_error);
  EXPECT_THROW(parse("qwen38_weights_pool 1\npool 1 16 0 8 1 4 a.bin\n"), std::runtime_error);     // ids must start at 0
  EXPECT_THROW(parse("qwen38_weights_pool 1\npool 0 16 0 8 1 4 ../a.bin\n"), std::runtime_error);  // escapes the folder
  EXPECT_THROW(parse("qwen38_weights_pool 1\npool 0 16 0 8 1 4 /abs/a.bin\n"), std::runtime_error);
  EXPECT_THROW(parse("qwen38_weights_pool 1\npool 0 16 0 8 1 4\n"), std::runtime_error);                       // no file
  EXPECT_THROW(parse("qwen38_weights_pool 1\npool 0 16 0 8 1 4 a.bin\nsession t n 1\n"), std::runtime_error);  // unknown id
  EXPECT_NO_THROW(parse("qwen38_weights_pool 1\n"));
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Qwen38QsaConstantsIndexTest, ParsesTheIndex) {
  const auto dir = MakeScratchDir("qsa");
  const auto path = WriteFile(dir / "qsa.txt",
                              "qwen38_qsa_consts 1\n"
                              "file qsa_consts.bin\n"
                              "epsilon 1e-06\n"
                              "width 64\n"
                              "tensor target cos 0 1024\n"
                              "tensor head knorm.0 1024 256\n");
  const auto description = ParseQwen38QsaConstantsIndex(path);
  EXPECT_EQ(description.file, "qsa_consts.bin");
  EXPECT_FLOAT_EQ(description.epsilon, 1e-6f);
  EXPECT_EQ(description.width, 64);
  ASSERT_EQ(description.items.size(), 2u);
  EXPECT_EQ(description.items[1].set, "head");
  EXPECT_EQ(description.items[1].name, "knorm.0");
  EXPECT_EQ(description.items[1].offset, 1024u);
  EXPECT_EQ(description.items[1].bytes, 256u);
  EXPECT_THROW(ParseQwen38QsaConstantsIndex(WriteFile(dir / "bad.txt", "qwen38_qsa_consts 2\n")), std::runtime_error);
  EXPECT_THROW(ParseQwen38QsaConstantsIndex(WriteFile(dir / "bad2.txt", "qwen38_qsa_consts 1\nfile ../x.bin\n")),
               std::runtime_error);
  EXPECT_THROW(ParseQwen38QsaConstantsIndex(WriteFile(dir / "bad3.txt", "qwen38_qsa_consts 1\nfile x.bin\ntensor target cos 0\n")),
               std::runtime_error);
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

// ---- PLE table description -------------------------------------------------------------------

TEST(Qwen38PleDescriptionTest, ParsesAValidTable) {
  const auto dir = MakeScratchDir("ple_valid");
  const auto description = ParseQwen38PleDescription(WriteFile(dir / "ple_table.txt", kValidPleTable));

  EXPECT_EQ(description.params.n_heads, 2);
  EXPECT_EQ(description.params.head_dim, 4);
  EXPECT_EQ(description.params.heads_per_ngram, 1);
  EXPECT_EQ(description.params.eos, 7);
  EXPECT_FLOAT_EQ(description.params.scale, 0.5f);
  EXPECT_EQ(description.params.mult[2], 3);
  EXPECT_EQ(description.params.vocab[1], 13);
  EXPECT_EQ(description.params.offset[1], 11);
  EXPECT_EQ(description.params.n_shards, 2);
  ASSERT_EQ(description.shards.size(), 2u);
  EXPECT_EQ(description.shards[0].path, "ple/a.safetensors");
  EXPECT_EQ(description.shards[0].offset, 128u);
  EXPECT_EQ(description.shards[1].rows, 13);
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Qwen38PleDescriptionTest, RejectsMalformedTables) {
  const auto dir = MakeScratchDir("ple_invalid");
  const auto table = dir / "ple_table.txt";
  const auto parse = [&](const std::string& text) { return ParseQwen38PleDescription(WriteFile(table, text)); };

  EXPECT_THROW(ParseQwen38PleDescription(dir / "missing.txt"), std::runtime_error);
  EXPECT_THROW(parse("qwen38_ple_table 2\n"), std::runtime_error);                         // header
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\n"), std::runtime_error);  // no shards
  EXPECT_THROW(parse("qwen38_ple_table 1\nvocab 1 2\nn_heads 2\n"), std::runtime_error);   // vocab before n_heads
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 33\n"), std::runtime_error);
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nshard ple/a.bin 0\n"), std::runtime_error);  // rows
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nshard ple/a.bin 0 0\n"), std::runtime_error);

  // Shards must stay inside the model folder.
  for (const char* path : {"../a.bin", "ple/../../a.bin", "/etc/a.bin"}) {
    EXPECT_THROW(parse(std::string("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nshard ") + path + " 0 4\n"),
                 std::runtime_error)
        << path;
  }

  std::string too_many = "qwen38_ple_table 1\nn_heads 2\nhead_dim 4\n";
  for (int i = 0; i < 257; ++i) too_many += "shard ple/a.bin 0 1\n";
  EXPECT_THROW(parse(too_many), std::runtime_error);
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

// ---- plane library loader --------------------------------------------------------------------

TEST(Qwen38PlaneTest, LoadingAMissingLibraryFailsWithItsPath) {
  const auto missing = std::filesystem::path{"definitely"} / "missing" / "libqwen38_plane.so";
  try {
    Qwen38Plane plane{missing};
    FAIL() << "loading a missing library must throw";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("qwen38"), std::string::npos) << message;
  }
}

TEST(Qwen38PlaneTest, LoadsBuiltInGenaiCudaLibraryWhenPathIsEmpty) {
  Qwen38Plane plane{};
  EXPECT_EQ(plane.abi_version(), QWEN38_PLANE_ABI_VERSION);
  EXPECT_EQ(plane.sizeof_ple_params(), static_cast<int>(sizeof(Qwen38PleParams)));
  EXPECT_NE(plane.quantize_kv, nullptr);
  EXPECT_NE(plane.embed_rows, nullptr);
  EXPECT_NE(plane.host_get_device_pointer, nullptr);
  EXPECT_NE(plane.dflash_select, nullptr);
  EXPECT_NE(plane.replay_state_updates, nullptr);
}

TEST(Qwen38DecodeLedgerTest, CompactPingPongAlternatesBanksAcrossOneToEightRowSteps) {
  Qwen38DecodeLedger ledger{64};
  ledger.SetCompactPingPong(true, 8);
  EXPECT_EQ(ledger.NumBanks(), 2);
  EXPECT_EQ(ledger.Bank(), 0);
  const size_t steps[] = {8, 1, 5, 7, 3, 8, 2};
  size_t length = 0;
  for (size_t i = 0; i < std::size(steps); ++i) {
    ledger.Advance(steps[i]);
    length += steps[i];
    EXPECT_EQ(ledger.Bank(), static_cast<int>((i + 1) % 2));
    EXPECT_EQ(ledger.Length(), length);
  }
  EXPECT_THROW(ledger.Advance(9), std::logic_error);
}

// ---- host ledger -----------------------------------------------------------------------------

TEST(Qwen38DecodeLedgerTest, RingBankFollowsTheCommittedRows) {
  Qwen38DecodeLedger ledger{64};
  EXPECT_EQ(ledger.Bank(), 0);
  EXPECT_EQ(ledger.Length(), 0u);

  // Steps of 2, 1, 2, 2, 1, 1, 1: bank = (bank + step) % 3, length = sum.
  const size_t steps[] = {2, 1, 2, 2, 1, 1, 1};
  const int expected_bank[] = {2, 0, 2, 1, 2, 0, 1};
  size_t length = 0;
  for (size_t i = 0; i < std::size(steps); ++i) {
    ledger.Advance(steps[i]);
    length += steps[i];
    EXPECT_EQ(ledger.Bank(), expected_bank[i]) << "step " << i;
    EXPECT_EQ(ledger.Length(), length) << "step " << i;
  }
  ledger.Reset();
  EXPECT_EQ(ledger.Bank(), 0);
  EXPECT_EQ(ledger.Length(), 0u);
}

TEST(Qwen38DecodeLedgerTest, DensePromptRowsAdvanceTheLengthAtBankZero) {
  Qwen38DecodeLedger ledger{64};
  ledger.AdvancePrompt(8);
  ledger.AdvancePrompt(5);
  EXPECT_EQ(ledger.Length(), 13u);
  EXPECT_EQ(ledger.Bank(), 0);
  ledger.Advance(2);  // the decode loop continues from the hand-off bank 0
  EXPECT_EQ(ledger.Length(), 15u);
  EXPECT_EQ(ledger.Bank(), 2);
  EXPECT_THROW(ledger.AdvancePrompt(1), std::logic_error);  // a dense chunk after decoding / S=2 steps is a bug
  EXPECT_EQ(ledger.Length(), 15u);
  EXPECT_NO_THROW(Qwen38DecodeLedger{8}.RequireRoom(8, "x"));
}

TEST(Qwen38DecodeLedgerTest, AStepCommitsOneOrTwoRows) {
  Qwen38DecodeLedger ledger{64};
  EXPECT_THROW(ledger.Advance(0), std::logic_error);
  EXPECT_THROW(ledger.Advance(3), std::logic_error);
  EXPECT_EQ(ledger.Length(), 0u);
}

TEST(Qwen38DecodeLedgerTest, CapacityIncludesTheReservedTail) {
  Qwen38DecodeLedger ledger{10};
  EXPECT_NO_THROW(ledger.RequireRoom(10, "x"));
  EXPECT_THROW(ledger.RequireRoom(11, "x"), std::runtime_error);
  EXPECT_NO_THROW(ledger.RequireRoom(7, "x", 3));
  EXPECT_THROW(ledger.RequireRoom(8, "x", 3), std::runtime_error);
  for (int i = 0; i < 4; ++i) ledger.Advance(2);
  EXPECT_NO_THROW(ledger.RequireRoom(2, "x"));
  EXPECT_THROW(ledger.RequireRoom(3, "x"), std::runtime_error);
}

TEST(Qwen38DecodeLedgerTest, DeviceLengthMustMatchTheHost) {
  Qwen38DecodeLedger ledger{16};
  ledger.Advance(2);
  EXPECT_NO_THROW(ledger.CheckDeviceLength(2));
  EXPECT_THROW(ledger.CheckDeviceLength(3), std::runtime_error);
}

TEST(Qwen38FoldIoTest, StateOutputsBindOverTheMatchingInputs) {
  EXPECT_STREQ(Qwen38FoldStateInput("fold.ctl_out"), "fold.ctl");
  EXPECT_STREQ(Qwen38FoldStateInput("fold.ctx_commit_out"), "fold.ctx_commit");
  EXPECT_EQ(Qwen38FoldStateInput("fold.tail_done"), nullptr);
  EXPECT_EQ(Qwen38FoldStateInput("fold.ctl"), nullptr);
  EXPECT_EQ(Qwen38FoldStateInput("fold.ctx_b"), nullptr);
  EXPECT_EQ(Qwen38FoldStateInput("next_tokens"), nullptr);
}

TEST(Qwen38CycleRecordTest, AcceptedCyclesCommitTwoTokens) {
  Qwen38CycleRecord record;
  EXPECT_EQ(record.TokenCount(), 1u);
  record.accepted = true;
  EXPECT_EQ(record.TokenCount(), 2u);
}

}  // namespace Generators::test
