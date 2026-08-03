// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tensor_parallel.h"

#include "../../generators.h"
#include "../../config.h"
#include "../../logging.h"
#include "tp_protocol.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

#if defined(__linux__)
#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace Generators {

namespace {

constexpr const char* kRankEnv = "ORTGENAI_TP_RANK";
constexpr const char* kFdEnv = "ORTGENAI_TP_FD";
constexpr const char* kModelEnv = "ORTGENAI_TP_MODEL";
constexpr const char* kWorkerEnv = "ORTGENAI_TP_WORKER";
constexpr int kWorkerFd = 3;        // where the control socket lands in the worker
constexpr int kReadyTimeoutS = 60;  // a worker greets rank 0 before it loads anything, so this is generous
constexpr int kWatchdogPollMs = 200;

const char* EnvValue(const char* name) {
  const char* v = std::getenv(name);
  return (v && *v) ? v : nullptr;
}

std::string RankDirectory(const Config::Model::MultiGpu& mg, int rank) {
  const auto pos = mg.rank_dir.find("%d");
  if (pos == std::string::npos)
    return mg.rank_dir;
  return mg.rank_dir.substr(0, pos) + std::to_string(rank) + mg.rank_dir.substr(pos + 2);
}

#if defined(__linux__)

// Where the worker executable lives. Config wins, then the environment, then the directory
// holding the GenAI shared library (that is where the build and the wheel both put it).
std::string ResolveWorkerExecutable(const Config& config) {
  if (!config.model.multi_gpu.worker_executable.empty())
    return config.model.multi_gpu.worker_executable;
  if (const char* env = EnvValue(kWorkerEnv))
    return env;

  Dl_info info{};
  if (dladdr(reinterpret_cast<const void*>(&ResolveWorkerExecutable), &info) && info.dli_fname) {
    fs::path candidate = fs::path{info.dli_fname}.parent_path() / "onnxruntime-genai-tp-worker";
    if (fs::exists(candidate))
      return candidate.string();
  }
  throw std::runtime_error(
      "Multi-GPU is enabled but the rank worker executable was not found next to the "
      "onnxruntime-genai shared library. Set model.multi_gpu.worker_executable in "
      "genai_config.json or the ORTGENAI_TP_WORKER environment variable.");
}

// A socketpair end that is guaranteed not to collide with the fd number the child expects.
int MoveAboveWorkerFd(int fd) {
  while (fd <= kWorkerFd) {
    int higher = ::fcntl(fd, F_DUPFD_CLOEXEC, kWorkerFd + 1);
    if (higher < 0)
      return fd;
    ::close(fd);
    fd = higher;
  }
  return fd;
}

std::vector<std::string> BuildWorkerEnvironment(const Config& config, int rank) {
  const auto& mg = config.model.multi_gpu;
  std::vector<std::string> overrides{
      std::string{"CUDA_VISIBLE_DEVICES="} + std::to_string(rank),
      std::string{"LOCAL_RANK="} + std::to_string(rank),
      std::string{"LOCAL_WORLD_SIZE="} + std::to_string(mg.world_size),
      std::string{"RANK0_IP="} + mg.master_ip,
      std::string{"RANK0_PORT="} + std::to_string(mg.master_port),
      std::string{kRankEnv} + "=" + std::to_string(rank),
      std::string{kFdEnv} + "=" + std::to_string(kWorkerFd),
      std::string{kModelEnv} + "=" + config.config_path.string(),
      std::string{"ORTGENAI_TP_STARTUP_TIMEOUT="} + std::to_string(mg.startup_timeout_s),
  };

  std::vector<std::string> result;
  for (char** e = environ; e && *e; ++e) {
    std::string entry{*e};
    const auto eq = entry.find('=');
    const std::string key = entry.substr(0, eq == std::string::npos ? entry.size() : eq);
    const bool overridden = std::any_of(overrides.begin(), overrides.end(), [&](const std::string& o) {
      return o.compare(0, key.size() + 1, key + "=") == 0;
    });
    if (!overridden)
      result.push_back(std::move(entry));
  }
  result.insert(result.end(), overrides.begin(), overrides.end());
  return result;
}

#endif  // __linux__

}  // namespace

int TensorParallelRank() {
  if (const char* v = EnvValue(kRankEnv))
    return std::atoi(v);
  return 0;
}

void PrepareTensorParallelConfig(Config& config) {
  auto& mg = config.model.multi_gpu;
  if (mg.world_size <= 1)
    return;

  const int rank = TensorParallelRank();
  if (rank < 0 || rank >= mg.world_size)
    throw std::runtime_error("ORTGENAI_TP_RANK (" + std::to_string(rank) + ") is outside the world size (" +
                             std::to_string(mg.world_size) + ")");

  const std::string rank_dir = RankDirectory(mg, rank);
  if (!rank_dir.empty() && !config.model.decoder.filename.empty())
    config.model.decoder.filename = rank_dir + "/" + config.model.decoder.filename;

#if defined(__linux__)
  // ORT reads these from the environment when it bootstraps the NCCL communicator. The worker
  // processes get them from their spawn environment; rank 0 has to set its own.
  if (rank == 0) {
    ::setenv("LOCAL_RANK", "0", 1);
    ::setenv("LOCAL_WORLD_SIZE", std::to_string(mg.world_size).c_str(), 1);
    ::setenv("RANK0_IP", mg.master_ip.c_str(), 1);
    ::setenv("RANK0_PORT", std::to_string(mg.master_port).c_str(), 1);
  }
#else
  throw std::runtime_error("model.multi_gpu is only supported on Linux");
#endif
}

struct TensorParallelGroup::Worker {
  int rank{};
  int fd{-1};
#if defined(__linux__)
  pid_t pid{-1};

  // How this worker died, or an empty string if it is still running. Reaps the process, so the
  // answer is remembered only by the caller. `wait_ms` covers the gap between a worker closing
  // its socket and the kernel making its exit status available.
  std::string Describe(int wait_ms) {
    if (pid <= 0)
      return "already gone";
    for (int elapsed_ms = 0;; elapsed_ms += 50) {
      int status = 0;
      const pid_t reaped = ::waitpid(pid, &status, WNOHANG);
      if (reaped < 0) {
        pid = -1;
        return "already gone";
      }
      if (reaped == pid) {
        pid = -1;
        if (WIFEXITED(status))
          return "exit code " + std::to_string(WEXITSTATUS(status));
        if (WIFSIGNALED(status))
          return "killed by signal " + std::to_string(WTERMSIG(status));
        return "stopped";
      }
      if (elapsed_ms >= wait_ms)
        return {};
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
#endif

  ~Worker() {
#if defined(__linux__)
    if (fd >= 0)
      ::close(fd);
    if (pid > 0) {
      int status = 0;
      // The Shutdown message has already been sent; give the worker a moment to unwind its
      // session before insisting.
      for (int i = 0; i < 100; ++i) {
        pid_t r = ::waitpid(pid, &status, WNOHANG);
        if (r == pid || r < 0)
          return;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      ::kill(pid, SIGKILL);
      ::waitpid(pid, &status, 0);
    }
#endif
  }
};

TensorParallelGroup::TensorParallelGroup(int world_size, std::string log_dir,
                                         std::vector<std::unique_ptr<Worker>> workers)
    : world_size_{world_size}, log_dir_{std::move(log_dir)}, workers_{std::move(workers)} {}

std::unique_ptr<TensorParallelGroup> TensorParallelGroup::Launch(const Config& config) {
  const auto& mg = config.model.multi_gpu;
  if (mg.world_size <= 1 || TensorParallelRank() != 0)
    return nullptr;

#if !defined(__linux__)
  throw std::runtime_error("model.multi_gpu is only supported on Linux");
#else
  const std::string executable = ResolveWorkerExecutable(config);

  std::vector<std::unique_ptr<Worker>> workers;
  for (int rank = 1; rank < mg.world_size; ++rank) {
    int fds[2] = {-1, -1};
    if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) != 0)
      throw std::runtime_error("socketpair failed for rank " + std::to_string(rank) + ": " + std::strerror(errno));

    const int parent_fd = MoveAboveWorkerFd(fds[0]);
    const int child_fd = MoveAboveWorkerFd(fds[1]);

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, child_fd, kWorkerFd);
    if (!mg.log_dir.empty()) {
      const std::string log = mg.log_dir + "/rank" + std::to_string(rank) + ".log";
      posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
    }

    const auto env_strings = BuildWorkerEnvironment(config, rank);
    std::vector<char*> envp;
    envp.reserve(env_strings.size() + 1);
    for (const auto& s : env_strings)
      envp.push_back(const_cast<char*>(s.c_str()));
    envp.push_back(nullptr);

    std::array<char*, 2> argv{const_cast<char*>(executable.c_str()), nullptr};

    pid_t pid = -1;
    const int rc = posix_spawn(&pid, executable.c_str(), &actions, nullptr, argv.data(), envp.data());
    posix_spawn_file_actions_destroy(&actions);
    ::close(child_fd);
    if (rc != 0) {
      ::close(parent_fd);
      throw std::runtime_error("failed to launch " + executable + " for rank " + std::to_string(rank) + ": " +
                               std::strerror(rc));
    }

    auto worker = std::make_unique<Worker>();
    worker->rank = rank;
    worker->fd = parent_fd;
    worker->pid = pid;
    workers.push_back(std::move(worker));
  }

  if (g_log.enabled)
    Log("tensor_parallel", "launched " + std::to_string(mg.world_size - 1) + " rank workers, world_size=" +
                               std::to_string(mg.world_size));

  auto group = std::unique_ptr<TensorParallelGroup>{
      new TensorParallelGroup(mg.world_size, mg.log_dir, std::move(workers))};
  group->WaitForReady(kReadyTimeoutS);
  group->StartWatchdog();
  return group;
#endif
}

void TensorParallelGroup::WaitForReady(int timeout_s) {
#if defined(__linux__)
  std::string errors;
  for (auto& worker : workers_) {
    pollfd poll_fd{worker->fd, POLLIN, 0};
    const int ready = ::poll(&poll_fd, 1, timeout_s * 1000);
    tp::Ack ack{};
    if (ready <= 0) {
      errors += "\n  rank " + std::to_string(worker->rank) + ": no greeting within " + std::to_string(timeout_s) + "s";
    } else if (!tp::RecvAll(worker->fd, &ack, sizeof(ack)) || ack.magic != tp::kMagic) {
      std::string exit = worker->Describe(500);
      errors += "\n  rank " + std::to_string(worker->rank) + ": died during startup (" +
                (exit.empty() ? "closed the control socket" : exit) + ")";
    } else if (ack.status != 0) {
      std::string message(ack.message_bytes, '\0');
      if (ack.message_bytes && !tp::RecvAll(worker->fd, message.data(), message.size()))
        message = "(truncated message)";
      errors += "\n  rank " + std::to_string(worker->rank) + ": " + message;
    }
  }

  if (!errors.empty()) {
    failed_ = true;
    throw std::runtime_error(std::string{"the tensor-parallel workers failed to start:"} + errors +
                             (log_dir_.empty() ? "\nSet model.multi_gpu.log_dir to capture each rank's output."
                                               : "\nSee " + log_dir_ + "/rank*.log for each rank's output."));
  }
#else
  (void)timeout_s;
#endif
}

void TensorParallelGroup::StartWatchdog() {
#if defined(__linux__)
  watchdog_ = std::thread{[this] {
    while (!watchdog_stop_.load(std::memory_order_relaxed)) {
      for (auto& worker : workers_) {
        if (worker->pid <= 0)
          continue;  // already reported
        std::string exit = worker->Describe(0);
        if (exit.empty())
          continue;
        failed_ = true;
        // Not an exception: this process is inside ORT's session creation, blocked in accept()
        // waiting for a rank that is never going to connect, and there is nobody to throw at.
        std::cerr << "onnxruntime-genai: tensor-parallel rank " << worker->rank << " exited (" << exit
                  << ") before joining the collective group. Rank 0 will stay blocked in session creation.";
        if (!log_dir_.empty())
          std::cerr << " See " << log_dir_ << "/rank" << worker->rank << ".log.";
        std::cerr << std::endl;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(kWatchdogPollMs));
    }
  }};
#endif
}

void TensorParallelGroup::StopWatchdog() {
  watchdog_stop_ = true;
  if (watchdog_.joinable())
    watchdog_.join();
}

void TensorParallelGroup::SessionCreated() {
  StopWatchdog();
}

TensorParallelGroup::~TensorParallelGroup() {
  StopWatchdog();
  if (!failed_) {
    try {
      if (pending_)
        Wait();
      Broadcast(static_cast<uint32_t>(tp::Op::Shutdown), 0, {});
      Wait();
    } catch (const std::exception&) {
      // Nothing useful to do while tearing down; the Worker destructor kills what is left.
    }
  }
}

void TensorParallelGroup::Broadcast(uint32_t op, int64_t arg, std::span<const int32_t> payload) {
  if (failed_)
    throw std::runtime_error("the tensor-parallel group is no longer usable after an earlier failure");
  if (pending_)
    throw std::runtime_error("a tensor-parallel message is already in flight");

  tp::Header header{tp::kMagic, op, static_cast<uint32_t>(payload.size()), 0, arg};
  for (auto& worker : workers_) {
    if (!tp::SendAll(worker->fd, &header, sizeof(header)) ||
        (!payload.empty() && !tp::SendAll(worker->fd, payload.data(), payload.size_bytes()))) {
      failed_ = true;
      throw std::runtime_error("rank " + std::to_string(worker->rank) + " went away while sending " +
                               tp::ToString(static_cast<tp::Op>(op)));
    }
  }
  pending_ = true;
}

void TensorParallelGroup::Wait() {
  if (!pending_)
    return;
  pending_ = false;

  std::string errors;
  for (auto& worker : workers_) {
    tp::Ack ack{};
    if (!tp::RecvAll(worker->fd, &ack, sizeof(ack)) || ack.magic != tp::kMagic) {
      failed_ = true;
      errors += "\n  rank " + std::to_string(worker->rank) + ": no reply (the worker process died)";
      continue;
    }
    if (ack.message_bytes > 0) {
      std::string message(ack.message_bytes, '\0');
      if (!tp::RecvAll(worker->fd, message.data(), message.size())) {
        failed_ = true;
        errors += "\n  rank " + std::to_string(worker->rank) + ": truncated error message";
        continue;
      }
      if (ack.status != 0)
        errors += "\n  rank " + std::to_string(worker->rank) + ": " + message;
    }
    if (ack.status != 0)
      failed_ = true;
  }

  if (!errors.empty())
    throw std::runtime_error("tensor-parallel rank failure:" + errors);
}

void TensorParallelGroup::BeginGenerator(int max_length, int batch_size) {
  const std::array<int32_t, 1> payload{batch_size};
  Broadcast(static_cast<uint32_t>(tp::Op::BeginGenerator), max_length, payload);
  Wait();
}

void TensorParallelGroup::EndGenerator() {
  Broadcast(static_cast<uint32_t>(tp::Op::EndGenerator), 0, {});
  Wait();
}

void TensorParallelGroup::SendForward(std::span<const int32_t> tokens) {
  Broadcast(static_cast<uint32_t>(tp::Op::Forward), 0, tokens);
}

void TensorParallelGroup::SendRewind(size_t new_length) {
  Broadcast(static_cast<uint32_t>(tp::Op::Rewind), static_cast<int64_t>(new_length), {});
}

}  // namespace Generators
