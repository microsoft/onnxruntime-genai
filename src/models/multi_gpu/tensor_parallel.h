// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <atomic>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace Generators {

struct Config;

// Multi-GPU support for tensor/expert parallel exports: one ONNX graph per rank, all ranks
// meeting in the collective ops of every layer. ORT's NCCL context is a process-wide singleton
// keyed off the LOCAL_RANK environment variable, so a rank cannot be a thread - each one needs
// its own process. Rank 0 is the process the application runs in and owns the tokenizer, the
// search and the sampling; ranks 1..world_size-1 are worker processes that mirror rank 0's
// forward passes so the collectives have a partner.

// This process's rank, from ORTGENAI_TP_RANK. 0 in an ordinary (single-GPU) process.
int TensorParallelRank();

// Points config.model.decoder.filename at this rank's graph and publishes LOCAL_RANK,
// LOCAL_WORLD_SIZE, RANK0_IP and RANK0_PORT for ORT's NCCL bootstrap. Must run before any
// session is created. No-op when world_size <= 1.
void PrepareTensorParallelConfig(Config& config);

struct TensorParallelGroup {
  // Launches ranks 1..world_size-1, or returns nullptr if this process is not the rank 0 of a
  // multi-GPU group. Must be called *before* rank 0's session is created: rank 0 hands out the
  // NCCL unique id from inside session creation and blocks there until every worker connects.
  static std::unique_ptr<TensorParallelGroup> Launch(const Config& config);

  ~TensorParallelGroup();

  int WorldSize() const { return world_size_; }

  // Call once rank 0's session exists. Until then a watchdog reports workers that die early,
  // because rank 0 has no way to notice: it is blocked inside session creation waiting for a
  // connection that a dead worker will never make.
  void SessionCreated();

  void BeginGenerator(int max_length, int batch_size);
  void EndGenerator();

  // Split on purpose: all ranks must be inside the same session run at the same time, so the
  // caller sends, runs rank 0 locally, and only then collects the acks.
  void SendForward(std::span<const int32_t> tokens);
  void SendRewind(size_t new_length);
  void Wait();

 private:
  struct Worker;

  TensorParallelGroup(int world_size, std::string log_dir, std::vector<std::unique_ptr<Worker>> workers);
  void Broadcast(uint32_t op, int64_t arg, std::span<const int32_t> payload);
  // Collects the greeting every worker sends before it goes looking for rank 0, so that a bad
  // executable or an unreadable model turns into an exception instead of a deadlock.
  void WaitForReady(int timeout_s);
  void StartWatchdog();
  void StopWatchdog();

  int world_size_{};
  std::string log_dir_;
  std::vector<std::unique_ptr<Worker>> workers_;
  bool pending_{};  // a Broadcast is in flight and its acks have not been collected
  std::atomic<bool> failed_{};
  std::thread watchdog_;
  std::atomic<bool> watchdog_stop_{};
};

}  // namespace Generators
