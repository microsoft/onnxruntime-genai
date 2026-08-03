// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Wire format shared by the tensor-parallel coordinator (rank 0, inside the GenAI shared
// library) and the rank worker executable. Every message is a fixed header optionally followed
// by `count` int32 values, and every message is answered with an Ack. Header-only so the worker
// executable does not depend on any non-public symbol of the shared library.
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

#if defined(__linux__)
#include <cerrno>
#include <unistd.h>
#endif

namespace Generators {
namespace tp {

inline constexpr uint32_t kMagic = 0x30315054;  // "TP01"

enum class Op : uint32_t {
  BeginGenerator = 1,  // arg = max_length, payload = {batch_size}
  EndGenerator = 2,
  Forward = 3,  // payload = the tokens to run this pass over
  Rewind = 4,   // arg = new sequence length
  Shutdown = 5,
};

// Field order chosen so the layout has no padding on any supported ABI.
struct Header {
  uint32_t magic;
  uint32_t op;
  uint32_t count;  // number of int32 payload elements following the header
  uint32_t pad;
  int64_t arg;
};
static_assert(sizeof(Header) == 24, "unexpected Header layout");

struct Ack {
  uint32_t magic;
  int32_t status;  // 0 on success
  uint32_t message_bytes;
  uint32_t pad;
};
static_assert(sizeof(Ack) == 16, "unexpected Ack layout");

// Loop over read()/write() until the whole buffer moves or the peer goes away.
inline bool SendAll(int fd, const void* data, size_t bytes) {
#if defined(__linux__)
  auto* p = static_cast<const uint8_t*>(data);
  while (bytes > 0) {
    ssize_t n = ::write(fd, p, bytes);
    if (n <= 0) {
      if (n < 0 && errno == EINTR) continue;
      return false;
    }
    p += n;
    bytes -= static_cast<size_t>(n);
  }
  return true;
#else
  (void)fd, (void)data, (void)bytes;
  return false;
#endif
}

inline bool RecvAll(int fd, void* data, size_t bytes) {
#if defined(__linux__)
  auto* p = static_cast<uint8_t*>(data);
  while (bytes > 0) {
    ssize_t n = ::read(fd, p, bytes);
    if (n <= 0) {
      if (n < 0 && errno == EINTR) continue;
      return false;
    }
    p += n;
    bytes -= static_cast<size_t>(n);
  }
  return true;
#else
  (void)fd, (void)data, (void)bytes;
  return false;
#endif
}

inline const char* ToString(Op op) {
  switch (op) {
    case Op::BeginGenerator:
      return "BeginGenerator";
    case Op::EndGenerator:
      return "EndGenerator";
    case Op::Forward:
      return "Forward";
    case Op::Rewind:
      return "Rewind";
    case Op::Shutdown:
      return "Shutdown";
  }
  return "?";
}

}  // namespace tp
}  // namespace Generators
