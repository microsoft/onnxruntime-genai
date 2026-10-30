// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telemetry/generation_telemetry.h"
#include "telemetry/telemetry.h"

#include <chrono>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>

namespace {

bool sampled = true;
int baseline_reads = 0;
int resource_reads = 0;
int64_t cpu_time_ms = 0;
std::vector<Generators::GenerateEndInfo> events;
std::vector<std::string> modalities;

class GenerationResourceTests : public ::testing::Test {
 protected:
  void SetUp() override {
    sampled = true;
    baseline_reads = 0;
    resource_reads = 0;
    cpu_time_ms = 0;
    events.clear();
    modalities.clear();
  }
};

}  // namespace

// Link only the generation tracker, replacing telemetry emission and OS probes so tests
// observe collection work without an uploader, driver, model, or process-wide singleton state.
namespace Generators {

struct GenAiTelemetry::Impl {};
GenAiTelemetry::GenAiTelemetry() = default;
GenAiTelemetry::~GenAiTelemetry() = default;

GenAiTelemetry& GenAiTelemetry::Instance() {
  static GenAiTelemetry instance;
  return instance;
}

bool GenAiTelemetry::IsDestroyed() { return false; }
bool GenAiTelemetry::IsEnabled() const { return true; }
bool GenAiTelemetry::IsSessionSampled(uint32_t) { return sampled; }
void GenAiTelemetry::LogGeneratorCreate(uint32_t, uint32_t, int, int, int, int,
                                        float, float, bool, bool, bool) {}
void GenAiTelemetry::LogAdapterActivated(uint32_t, uint32_t) {}

void GenAiTelemetry::LogGeneration(uint32_t, uint32_t, int64_t, std::string_view modality,
                                   const GenerateEndInfo& info, int64_t, int64_t) {
  events.push_back(info);
  modalities.emplace_back(modality);
}

int64_t ReadProcessCpuTimeMs() noexcept {
  ++baseline_reads;
  return cpu_time_ms;
}

ProcessResourceUsage CollectProcessResourceUsage() noexcept {
  ++resource_reads;
  ProcessResourceUsage usage;
  usage.cpu_time_ms = cpu_time_ms;
  usage.resident_memory_mb = 100;
  return usage;
}

}  // namespace Generators

namespace {

using Generators::GenerationTelemetry;

TEST_F(GenerationResourceTests, UnsampledSessionsPerformNoResourceQueries) {
  sampled = false;
  {
    GenerationTelemetry generation{1, false};
    ASSERT_TRUE(generation.BeginAppend());
    generation.CompleteAppend(4, 1, "text");
    generation.OnTokenGenerated(1);
    generation.OnGenerationCompleted();
  }
  EXPECT_EQ(baseline_reads, 0);
  EXPECT_EQ(resource_reads, 0);
}

TEST_F(GenerationResourceTests, StartReadsOnlyCpuAndCompletionIsCached) {
  {
    GenerationTelemetry generation{2, false};
    generation.BeginAppend();
    generation.CompleteAppend(4, 1, "text");
    EXPECT_EQ(baseline_reads, 1);
    EXPECT_EQ(resource_reads, 0);
    cpu_time_ms = 10;
    generation.OnTokenGenerated(1);
    generation.OnGenerationCompleted();
    cpu_time_ms = 999;
    generation.OnGenerationCompleted();
  }
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events.front().process_cpu_time_ms, 10);
  EXPECT_EQ(resource_reads, 1);
}

TEST_F(GenerationResourceTests, TokenFreeAudioInputsDelimitWindowsAndRetainEncoderTime) {
  {
    GenerationTelemetry generation{3, true};
    generation.BeginAppend();
    generation.CompleteAppend(0, 1, "audio");
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    cpu_time_ms = 10;
    generation.OnTokenGenerated(1);
    generation.OnGenerationCompleted();
    cpu_time_ms = 100;
    generation.BeginAppend();
    generation.CompleteAppend(0, 1, "audio");
    cpu_time_ms = 110;
    generation.OnTokenGenerated(1);
    generation.OnGenerationCompleted();
  }
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(baseline_reads, 2);
  EXPECT_EQ(resource_reads, 2);
  EXPECT_EQ(events[0].process_cpu_time_ms, 10);
  EXPECT_EQ(events[1].process_cpu_time_ms, 10);
  EXPECT_GE(events[0].time_to_first_token_ms, 10);
  EXPECT_EQ(modalities, (std::vector<std::string>{"audio", "audio"}));
}

TEST_F(GenerationResourceTests, SilentAudioCompletionDiscardsBaselineBeforeNextChunk) {
  {
    GenerationTelemetry generation{4, true};
    generation.BeginAppend();
    generation.CompleteAppend(0, 1, "audio");
    cpu_time_ms = 10;
    generation.OnTokenGenerated(0);
    generation.OnGenerationCompleted();
    EXPECT_EQ(resource_reads, 0);
    EXPECT_TRUE(events.empty());
    cpu_time_ms = 100;
    generation.BeginAppend();
    generation.CompleteAppend(0, 1, "audio");
    cpu_time_ms = 110;
    generation.OnTokenGenerated(1);
    generation.OnGenerationCompleted();
  }
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(baseline_reads, 2);
  EXPECT_EQ(resource_reads, 1);
  EXPECT_EQ(events.front().process_cpu_time_ms, 10);
}

}  // namespace
