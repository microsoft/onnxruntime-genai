// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// One rank of a tensor-parallel group, other than rank 0.
//
// A rank cannot make progress on its own: it blocks in the collective ops of every layer until
// the rest of the group arrives. So this process owns nothing but a Generator over its own
// slice of the model, and replays whatever rank 0 does to it - the same tokens through the same
// forward passes, in the same order. Rank 0 keeps the tokenizer, the search and the sampling,
// so nothing here has to agree with anything: the tokens arrive already decided.

#include "ort_genai.h"
#include "../tp_protocol.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace {

using namespace Generators::tp;

int RankFromEnv() {
  const char* rank = std::getenv("ORTGENAI_TP_RANK");
  return rank ? std::atoi(rank) : -1;
}

[[noreturn]] void Fail(const std::string& message) {
  std::fprintf(stderr, "[tp rank %d] %s\n", RankFromEnv(), message.c_str());
  std::fflush(stderr);
  std::_Exit(1);
}

// True when some socket on this host is in LISTEN state on `port`.
//
// Rank 0 binds its bootstrap socket from deep inside session creation, after it has finished
// loading its weights, and a peer that tries to connect earlier gives up after 40 seconds
// (the retry window is hard-coded in ORT). Waiting for the listening socket to appear before
// touching the session keeps that window from ever opening.
bool Rank0IsListening(int port) {
  for (const char* path : {"/proc/net/tcp", "/proc/net/tcp6"}) {
    std::ifstream file{path};
    std::string line;
    std::getline(file, line);  // header
    while (std::getline(file, line)) {
      std::istringstream fields{line};
      std::string slot, local, remote, state;
      fields >> slot >> local >> remote >> state;
      if (state != "0A")  // TCP_LISTEN
        continue;
      const auto colon = local.rfind(':');
      if (colon == std::string::npos)
        continue;
      if (std::stoi(local.substr(colon + 1), nullptr, 16) == port)
        return true;
    }
  }
  return false;
}

void WaitForRank0(int port, int timeout_s) {
  for (int elapsed_ms = 0; elapsed_ms < timeout_s * 1000; elapsed_ms += 100) {
    if (Rank0IsListening(port))
      return;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  Fail("rank 0 never started listening on port " + std::to_string(port));
}

void Reply(int fd, int32_t status, const std::string& message) {
  Ack ack{kMagic, status, static_cast<uint32_t>(message.size()), 0};
  if (!SendAll(fd, &ack, sizeof(ack)) || (!message.empty() && !SendAll(fd, message.data(), message.size())))
    Fail("rank 0 closed the control socket");
}

// The greeting rank 0 waits for before it opens its own session. Everything that can be checked
// without touching the GPU is checked here, so that a typo in the config turns into an error on
// rank 0 rather than a hang: once rank 0 is inside session creation it is blocked in accept()
// and can no longer be told anything.
[[noreturn]] void FailBeforeReady(int fd, const std::string& message) {
  Reply(fd, 1, message);
  std::_Exit(1);
}

void Greet(int fd, const char* model_path) {
  const std::string config = std::string{model_path} + "/genai_config.json";
  if (!std::ifstream{config})
    FailBeforeReady(fd, "cannot read " + config);
  Reply(fd, 0, {});
}

struct Rank {
  std::unique_ptr<OgaModel> model;
  std::unique_ptr<OgaGeneratorParams> params;
  std::unique_ptr<OgaGenerator> generator;

  void Begin(int64_t max_length, int32_t batch_size) {
    generator.reset();
    params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", static_cast<double>(max_length));
    params->SetSearchOption("batch_size", static_cast<double>(batch_size));
    generator = OgaGenerator::Create(*model, *params);
  }

  void Forward(const std::vector<int32_t>& tokens) {
    if (!generator)
      throw std::runtime_error("Forward before BeginGenerator");
    // AppendTokens is exactly the mirror of what rank 0 does: extend the sequence by these
    // tokens and run one forward pass over them. The sampled logits are discarded here.
    generator->AppendTokens(tokens.data(), tokens.size());
  }
};

}  // namespace

int main() {
  const int rank = RankFromEnv();
  const char* fd_env = std::getenv("ORTGENAI_TP_FD");
  const char* model_path = std::getenv("ORTGENAI_TP_MODEL");
  if (rank < 1 || !fd_env || !model_path)
    Fail("ORTGENAI_TP_RANK, ORTGENAI_TP_FD and ORTGENAI_TP_MODEL must all be set; this executable "
         "is launched by onnxruntime-genai, not run directly");

  const int fd = std::atoi(fd_env);
  Greet(fd, model_path);

  const char* port_env = std::getenv("RANK0_PORT");
  const char* timeout_env = std::getenv("ORTGENAI_TP_STARTUP_TIMEOUT");
  WaitForRank0(port_env ? std::atoi(port_env) : 19555, timeout_env ? std::atoi(timeout_env) : 1800);

  Rank state;
  try {
    // The library reads ORTGENAI_TP_RANK and points the decoder at this rank's graph. Session
    // creation blocks until every rank has joined the NCCL communicator.
    state.model = OgaModel::Create(model_path);
  } catch (const std::exception& e) {
    Fail(std::string{"failed to load the model: "} + e.what());
  }

  for (;;) {
    Header header{};
    if (!RecvAll(fd, &header, sizeof(header)))
      return 0;  // rank 0 exited
    if (header.magic != kMagic)
      Fail("protocol desynchronized");

    std::vector<int32_t> payload(header.count);
    if (header.count && !RecvAll(fd, payload.data(), payload.size() * sizeof(int32_t)))
      Fail("truncated payload");

    try {
      switch (static_cast<Op>(header.op)) {
        case Op::BeginGenerator:
          state.Begin(header.arg, payload.empty() ? 1 : payload[0]);
          break;
        case Op::EndGenerator:
          state.generator.reset();
          break;
        case Op::Forward:
          state.Forward(payload);
          break;
        case Op::Rewind:
          if (state.generator)
            state.generator->RewindTo(static_cast<size_t>(header.arg));
          break;
        case Op::Shutdown:
          Reply(fd, 0, {});
          state.generator.reset();
          state.model.reset();
          return 0;
        default:
          throw std::runtime_error("unknown op " + std::to_string(header.op));
      }
      Reply(fd, 0, {});
    } catch (const std::exception& e) {
      Reply(fd, 1, std::string{ToString(static_cast<Op>(header.op))} + " failed: " + e.what());
    }
  }
}
