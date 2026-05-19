// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include "../src/generators.h"
#include "../src/config.h"
#include "../src/models/model_package.h"

// --- JSON Merge Patch (RFC 7386) tests ---

TEST(JsonMergePatch, SimpleOverwrite) {
  std::string base = R"({"a":1,"b":"hello"})";
  std::string patch = R"({"b":"world"})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("\"a\":1"), std::string::npos);
  EXPECT_NE(result.find("\"b\":\"world\""), std::string::npos);
  EXPECT_EQ(result.find("\"hello\""), std::string::npos);
}

TEST(JsonMergePatch, AddNewKey) {
  std::string base = R"({"a":1})";
  std::string patch = R"({"b":2})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("\"a\":1"), std::string::npos);
  EXPECT_NE(result.find("\"b\":2"), std::string::npos);
}

TEST(JsonMergePatch, DeleteKey) {
  std::string base = R"({"a":1,"b":2})";
  std::string patch = R"({"b":null})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("\"a\":1"), std::string::npos);
  EXPECT_EQ(result.find("\"b\""), std::string::npos);
}

TEST(JsonMergePatch, NestedMerge) {
  std::string base = R"({"model":{"decoder":{"hidden_size":768},"type":"decoder_only"}})";
  std::string patch = R"({"model":{"decoder":{"hidden_size":1024,"num_heads":16}}})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("1024"), std::string::npos);
  EXPECT_NE(result.find("16"), std::string::npos);
  EXPECT_NE(result.find("decoder_only"), std::string::npos);
  EXPECT_EQ(result.find("768"), std::string::npos);
}

TEST(JsonMergePatch, ArrayReplacement) {
  std::string base = R"({"arr":[1,2,3]})";
  std::string patch = R"({"arr":[4,5]})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("[4,5]"), std::string::npos);
  EXPECT_EQ(result.find("[1,2,3]"), std::string::npos);
}

TEST(JsonMergePatch, EmptyPatch) {
  std::string base = R"({"a":1})";
  std::string result = Generators::JsonMergePatch(base, "");
  EXPECT_EQ(result, base);
}

TEST(JsonMergePatch, EmptyBase) {
  std::string patch = R"({"a":1})";
  std::string result = Generators::JsonMergePatch("", patch);
  EXPECT_EQ(result, patch);
}

TEST(JsonMergePatch, PatchReplacesNonObject) {
  std::string base = R"("hello")";
  std::string patch = R"({"a":1})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("\"a\":1"), std::string::npos);
}

TEST(JsonMergePatch, BoolAndNullValues) {
  std::string base = R"({"a":true,"b":false,"c":null})";
  std::string patch = R"({"a":false,"c":"restored"})";
  std::string result = Generators::JsonMergePatch(base, patch);
  EXPECT_NE(result.find("\"a\":false"), std::string::npos);
  EXPECT_NE(result.find("\"c\":\"restored\""), std::string::npos);
}

// --- Package detection tests ---

TEST(IsModelPackage, NonExistentPath) {
  EXPECT_FALSE(Generators::IsModelPackage(fs::path("/tmp/nonexistent_model_package_test_path_12345")));
}

TEST(IsModelPackage, FlatDirectory) {
  // A temp dir without manifest.json should not be detected as a package
  std::string tmp_dir = "/tmp/genai_test_flat_dir";
  std::system(("mkdir -p " + tmp_dir).c_str());
  EXPECT_FALSE(Generators::IsModelPackage(fs::path(tmp_dir)));
  std::system(("rm -rf " + tmp_dir).c_str());
}

TEST(IsModelPackage, PackageDirectory) {
  // A temp dir with manifest.json should be detected as a package
  std::string tmp_dir = "/tmp/genai_test_package_dir";
  std::system(("mkdir -p " + tmp_dir).c_str());
  std::system(("touch " + tmp_dir + "/manifest.json").c_str());
  EXPECT_TRUE(Generators::IsModelPackage(fs::path(tmp_dir)));
  std::system(("rm -rf " + tmp_dir).c_str());
}

// --- Config::FromPackage tests ---

TEST(ConfigFromPackage, ParsesMergedJsonAndSetsConfigPath) {
  // Test that Config::FromPackage correctly parses a merged JSON string
  // and sets the config_path to the provided path
  std::string merged_json = R"({
    "model": {
      "type": "phi3",
      "context_length": 4096,
      "bos_token_id": 1,
      "eos_token_id": [2],
      "pad_token_id": 0,
      "vocab_size": 32000,
      "decoder": {
        "component": "decoder",
        "head_size": 128,
        "hidden_size": 3072,
        "num_attention_heads": 24,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8
      }
    },
    "search": {
      "max_length": 4096,
      "past_present_share_buffer": true
    }
  })";

  auto config = Generators::Config::FromPackage(
      fs::path("/tmp/fake_package/configs"), merged_json, nullptr);

  EXPECT_EQ(config->model.type, "phi3");
  EXPECT_EQ(config->model.context_length, 4096);
  EXPECT_EQ(config->model.vocab_size, 32000);
  EXPECT_EQ(config->model.decoder.component, "decoder");
  EXPECT_EQ(config->model.decoder.hidden_size, 3072);
  EXPECT_EQ(config->model.decoder.num_attention_heads, 24);
  EXPECT_EQ(config->model.decoder.num_hidden_layers, 32);
  EXPECT_EQ(config->search.max_length, 4096);
  EXPECT_EQ(config->config_path.string(), "/tmp/fake_package/configs");
  EXPECT_FALSE(config->IsPackage());  // no package_state passed
}

TEST(ConfigFromPackage, OverlayAppliedBeforeParsing) {
  // Test that JSON merge patch correctly modifies the config
  std::string base_json = R"({
    "model": {
      "type": "phi3",
      "context_length": 131072,
      "bos_token_id": 1,
      "eos_token_id": [2],
      "pad_token_id": 0,
      "vocab_size": 32000,
      "decoder": {
        "component": "decoder",
        "head_size": 128,
        "hidden_size": 3072,
        "num_attention_heads": 24,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8
      }
    },
    "search": { "max_length": 131072 }
  })";

  // Overlay changes context_length and adds an input name
  std::string overlay = R"({
    "model": {
      "context_length": 4096,
      "decoder": {
        "inputs": { "position_ids": "position_ids" }
      }
    },
    "search": { "max_length": 4096 }
  })";

  std::string merged = Generators::JsonMergePatch(base_json, overlay);
  auto config = Generators::Config::FromPackage(
      fs::path("/tmp/fake_package/configs"), merged, nullptr);

  EXPECT_EQ(config->model.context_length, 4096);
  EXPECT_EQ(config->search.max_length, 4096);
  // Original values should be preserved
  EXPECT_EQ(config->model.decoder.hidden_size, 3072);
  EXPECT_EQ(config->model.decoder.num_hidden_layers, 32);
  // Overlay should add position_ids input
  EXPECT_EQ(config->model.decoder.inputs.position_ids, "position_ids");
}
