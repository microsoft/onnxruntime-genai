// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Stand-in for onnxruntime-genai-tp-worker used by tensor_parallel_test.cpp: it speaks the same
// protocol but loads no model, so the launcher, the environment plumbing, the inherited socket
// and the request/ack loop can be tested without eight GPUs.

#include "models/multi_gpu/tp_protocol.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace Generators::tp;

int main() {
  const char* fd_env = std::getenv("ORTGENAI_TP_FD");
  const char* rank_env = std::getenv("ORTGENAI_TP_RANK");
  if (!fd_env || !rank_env)
    return 2;

  const int fd = std::atoi(fd_env);
  const char* log_path = std::getenv("TP_STUB_LOG");
  const char* fail_on = std::getenv("TP_STUB_FAIL_ON");

  // The greeting rank 0 blocks on before it opens its session.
  if (const char* reason = std::getenv("TP_STUB_FAIL_READY")) {
    const std::string message{reason};
    Ack ack{kMagic, 1, static_cast<uint32_t>(message.size()), 0};
    SendAll(fd, &ack, sizeof(ack));
    SendAll(fd, message.data(), message.size());
    return 1;
  }
  if (std::getenv("TP_STUB_DIE_BEFORE_READY"))
    return 7;
  Ack ready{kMagic, 0, 0, 0};
  if (!SendAll(fd, &ready, sizeof(ready)))
    return 1;

  for (;;) {
    Header header{};
    if (!RecvAll(fd, &header, sizeof(header)) || header.magic != kMagic)
      return 0;

    std::vector<int32_t> payload(header.count);
    if (header.count && !RecvAll(fd, payload.data(), payload.size() * sizeof(int32_t)))
      return 1;

    if (log_path) {
      std::ofstream log{log_path, std::ios::app};
      log << rank_env << ' ' << ToString(static_cast<Op>(header.op)) << ' ' << header.arg;
      for (int32_t token : payload)
        log << ' ' << token;
      log << '\n';
    }

    const bool fail = fail_on && header.op == static_cast<uint32_t>(std::atoi(fail_on));
    const std::string message = fail ? std::string{"stub failure"} : std::string{};
    Ack ack{kMagic, fail ? 1 : 0, static_cast<uint32_t>(message.size()), 0};
    if (!SendAll(fd, &ack, sizeof(ack)))
      return 1;
    if (!message.empty() && !SendAll(fd, message.data(), message.size()))
      return 1;

    if (static_cast<Op>(header.op) == Op::Shutdown)
      return 0;
  }
}
