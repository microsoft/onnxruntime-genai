// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "generators.h"
#include "models/multi_gpu/tensor_parallel.h"

#if defined(__linux__)

#include <cstdio>
#include <fstream>
#include <sstream>
#include <unistd.h>

namespace {

std::string StubWorkerPath() {
  char buffer[4096];
  const ssize_t length = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
  EXPECT_GT(length, 0);
  buffer[length > 0 ? length : 0] = '\0';
  std::string path{buffer};
  return path.substr(0, path.rfind('/') + 1) + "tp_stub_worker";
}

Generators::Config MakeConfig(int world_size, const std::string& log_path, const char* fail_on = nullptr) {
  Generators::Config config;
  config.model.decoder.filename = "model.onnx";
  config.model.multi_gpu.world_size = world_size;
  config.model.multi_gpu.worker_executable = StubWorkerPath();
  ::setenv("TP_STUB_LOG", log_path.c_str(), 1);
  ::unsetenv("TP_STUB_FAIL_READY");
  ::unsetenv("TP_STUB_DIE_BEFORE_READY");
  if (fail_on)
    ::setenv("TP_STUB_FAIL_ON", fail_on, 1);
  else
    ::unsetenv("TP_STUB_FAIL_ON");
  return config;
}

std::vector<std::string> ReadLines(const std::string& path) {
  std::vector<std::string> lines;
  std::ifstream file{path};
  std::string line;
  while (std::getline(file, line))
    lines.push_back(line);
  return lines;
}

struct TempFile {
  std::string path{"/tmp/ortgenai_tp_test_" + std::to_string(::getpid()) + ".log"};
  ~TempFile() { std::remove(path.c_str()); }
};

}  // namespace

TEST(TensorParallel, ConfigIsUntouchedWithoutMultiGpu) {
  Generators::Config config;
  config.model.decoder.filename = "model.onnx";
  Generators::PrepareTensorParallelConfig(config);
  EXPECT_EQ(config.model.decoder.filename, "model.onnx");
  EXPECT_EQ(Generators::TensorParallelGroup::Launch(config), nullptr);
}

TEST(TensorParallel, DecoderPointsAtThisRanksGraph) {
  Generators::Config config;
  config.model.decoder.filename = "model.onnx";
  config.model.multi_gpu.world_size = 4;
  Generators::PrepareTensorParallelConfig(config);
  EXPECT_EQ(config.model.decoder.filename, "rank_0/model.onnx");

  // PrepareTensorParallelConfig publishes the NCCL bootstrap variables into this process; leaving
  // them set would make any later session in this test binary try to join a group of four.
  for (const char* name : {"LOCAL_RANK", "LOCAL_WORLD_SIZE", "RANK0_IP", "RANK0_PORT"})
    ::unsetenv(name);
}

TEST(TensorParallel, RankOutsideWorldIsRejected) {
  ::setenv("ORTGENAI_TP_RANK", "9", 1);
  Generators::Config config;
  config.model.multi_gpu.world_size = 4;
  EXPECT_THROW(Generators::PrepareTensorParallelConfig(config), std::runtime_error);
  ::unsetenv("ORTGENAI_TP_RANK");
}

TEST(TensorParallel, WorkersMirrorEveryOperation) {
  TempFile log;
  auto config = MakeConfig(3, log.path);

  {
    auto group = Generators::TensorParallelGroup::Launch(config);
    ASSERT_NE(group, nullptr);
    EXPECT_EQ(group->WorldSize(), 3);

    group->BeginGenerator(128, 1);

    const std::vector<int32_t> prompt{11, 22, 33};
    group->SendForward(prompt);
    group->Wait();

    const std::vector<int32_t> next{44};
    group->SendForward(next);
    group->Wait();

    group->SendRewind(2);
    group->Wait();

    group->EndGenerator();
  }

  const auto lines = ReadLines(log.path);
  // Two workers x (BeginGenerator, 2 Forwards, Rewind, EndGenerator, Shutdown).
  ASSERT_EQ(lines.size(), 12u);
  for (const char* rank : {"1", "2"}) {
    std::vector<std::string> mine;
    for (const auto& line : lines)
      if (line.compare(0, 2, std::string{rank} + " ") == 0)
        mine.push_back(line.substr(2));
    ASSERT_EQ(mine.size(), 6u);
    EXPECT_EQ(mine[0], "BeginGenerator 128 1");
    EXPECT_EQ(mine[1], "Forward 0 11 22 33");
    EXPECT_EQ(mine[2], "Forward 0 44");
    EXPECT_EQ(mine[3], "Rewind 2");
    EXPECT_EQ(mine[4], "EndGenerator 0");
    EXPECT_EQ(mine[5], "Shutdown 0");
  }
}

TEST(TensorParallel, RankFailureSurfacesToTheCaller) {
  TempFile log;
  // 3 is Op::Forward.
  auto config = MakeConfig(2, log.path, "3");

  auto group = Generators::TensorParallelGroup::Launch(config);
  ASSERT_NE(group, nullptr);
  group->BeginGenerator(128, 1);

  const std::vector<int32_t> prompt{7};
  group->SendForward(prompt);
  EXPECT_THROW(group->Wait(), std::runtime_error);

  // The group latches the failure instead of deadlocking on the next collective.
  EXPECT_THROW(group->SendForward(prompt), std::runtime_error);
  ::unsetenv("TP_STUB_FAIL_ON");
}

// The cases below are why Launch waits for a greeting: without it these failures would only show
// up as rank 0 blocking forever inside session creation.

TEST(TensorParallel, StartupFailureIsReportedByRank) {
  TempFile log;
  auto config = MakeConfig(2, log.path);
  ::setenv("TP_STUB_FAIL_READY", "cannot read the model", 1);

  try {
    Generators::TensorParallelGroup::Launch(config);
    FAIL() << "Launch should have thrown";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string{e.what()}.find("rank 1: cannot read the model"), std::string::npos) << e.what();
  }
  ::unsetenv("TP_STUB_FAIL_READY");
}

TEST(TensorParallel, WorkerThatDiesDuringStartupIsReported) {
  TempFile log;
  auto config = MakeConfig(2, log.path);
  ::setenv("TP_STUB_DIE_BEFORE_READY", "1", 1);

  try {
    Generators::TensorParallelGroup::Launch(config);
    FAIL() << "Launch should have thrown";
  } catch (const std::runtime_error& e) {
    const std::string message{e.what()};
    EXPECT_NE(message.find("rank 1: died during startup"), std::string::npos) << message;
    EXPECT_NE(message.find("exit code 7"), std::string::npos) << message;
  }
  ::unsetenv("TP_STUB_DIE_BEFORE_READY");
}

TEST(TensorParallel, MissingWorkerExecutableIsReported) {
  Generators::Config config;
  config.model.multi_gpu.world_size = 2;
  config.model.multi_gpu.worker_executable = "/nonexistent/tp-worker";
  EXPECT_THROW(Generators::TensorParallelGroup::Launch(config), std::runtime_error);
}

#endif  // __linux__
