// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Portions of this file consist of AI generated content.

#include "controller_host.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "../generators.h"
#include "../search.h"
#include "model.h"

namespace Generators {

namespace {

DecodeStepHost& Host(OgaDecodeContext* host) {
  return *reinterpret_cast<DecodeStepHost*>(host);
}

// The callbacks below run inside the runtime but may be invoked by controller code that lives across
// a .so boundary, so a thrown C++ exception must never escape. Each callback therefore reports
// failure via its status/sentinel return value instead of propagating an exception.

size_t CbGetSequenceLength(OgaDecodeContext* host) {
  try {
    return Host(host).generator_.TokenCount();
  } catch (...) {
    return 0;
  }
}

int32_t CbGetEosTokenId(OgaDecodeContext* host) {
  try {
    const auto& eos = Host(host).generator_.model_->config_->model.eos_token_id;
    return eos.empty() ? -1 : eos.front();
  } catch (...) {
    return -1;
  }
}

int CbIsDone(OgaDecodeContext* host) {
  try {
    return Host(host).generator_.IsDone() ? 1 : 0;
  } catch (...) {
    return 0;
  }
}

int CbGetTokens(OgaDecodeContext* host, int32_t* out, size_t capacity, size_t* count_out) {
  try {
    auto sequence = Host(host).generator_.GetSequence(0).CopyDeviceToCpu();
    if (count_out) *count_out = sequence.size();
    if (!out || sequence.size() > capacity) return 1;
    std::copy(sequence.begin(), sequence.end(), out);
    return 0;
  } catch (...) {
    return 1;
  }
}

int CbGetLogits(OgaDecodeContext* host, const float** logits_out, size_t* vocab_out) {
  try {
    auto& self = Host(host);
    auto logits = self.generator_.GetLogits();  // computes a forward step if needed; [batch, vocab]
    auto cpu = logits.CopyDeviceToCpu();
    size_t vocab = static_cast<size_t>(self.generator_.model_->config_->model.vocab_size);
    if (vocab == 0 || vocab > cpu.size()) vocab = cpu.size();  // batch 0 occupies the first `vocab`
    self.logits_scratch_.assign(cpu.begin(), cpu.begin() + vocab);
    if (logits_out) *logits_out = self.logits_scratch_.data();
    if (vocab_out) *vocab_out = vocab;
    return 0;
  } catch (...) {
    return 1;
  }
}

int CbGetHiddenStates(OgaDecodeContext* host, const float** hidden_out, size_t* hidden_size_out) {
  try {
    auto& self = Host(host);
    std::array<int64_t, 3> shape{};
    auto hidden = self.generator_.GetHiddenStates(shape);  // [batch, seq, hidden] or empty
    if (hidden.empty()) return 1;
    auto cpu = hidden.CopyDeviceToCpu();
    const int64_t seq = shape[1];
    const int64_t hidden_size = shape[2];
    if (hidden_size <= 0) return 1;
    size_t offset = seq > 0 ? static_cast<size_t>((seq - 1) * hidden_size) : 0;  // batch 0, last position
    if (offset + static_cast<size_t>(hidden_size) > cpu.size()) offset = 0;
    self.hidden_scratch_.assign(cpu.begin() + offset, cpu.begin() + offset + hidden_size);
    if (hidden_out) *hidden_out = self.hidden_scratch_.data();
    if (hidden_size_out) *hidden_size_out = static_cast<size_t>(hidden_size);
    return 0;
  } catch (...) {
    return 1;
  }
}

int CbAppendTokens(OgaDecodeContext* host, const int32_t* tokens, size_t count) {
  try {
    if (count == 0) return 0;
    if (!tokens) return 1;
    Host(host).generator_.AppendAcceptedTokens(cpu_span<const int32_t>(tokens, count));
    return 0;
  } catch (...) {
    return 1;
  }
}

int CbRewindTo(OgaDecodeContext* host, size_t length) {
  try {
    Host(host).generator_.RewindToLength(length);
    return 0;
  } catch (...) {
    return 1;
  }
}

}  // namespace

OgaDecodeStepContext DecodeStepHost::Context() {
  OgaDecodeStepContext ctx{};
  ctx.host = reinterpret_cast<OgaDecodeContext*>(this);
  ctx.GetSequenceLength = &CbGetSequenceLength;
  ctx.GetEosTokenId = &CbGetEosTokenId;
  ctx.IsDone = &CbIsDone;
  ctx.GetTokens = &CbGetTokens;
  ctx.GetLogits = &CbGetLogits;
  ctx.GetHiddenStates = &CbGetHiddenStates;
  ctx.AppendTokens = &CbAppendTokens;
  ctx.RewindTo = &CbRewindTo;
  return ctx;
}

ControllerHook::ControllerHook(OgaDecodeController* self, OgaControllerStepFn step,
                               OgaDecodeControllerDestroyFn destroy, std::shared_ptr<void> keepalive)
    : self_{self}, step_{step}, destroy_{destroy}, keepalive_{std::move(keepalive)} {}

ControllerHook::~ControllerHook() {
  if (destroy_ && self_) destroy_(self_);
}

int ControllerHook::Step(Generator& generator) {
  DecodeStepHost host{generator};
  OgaDecodeStepContext ctx = host.Context();
  int tokens_emitted = 0;
  const int status = step_(self_, &ctx, &tokens_emitted);
  if (status != 0)
    throw std::runtime_error("Decode controller step callback failed with status " + std::to_string(status) + ".");
  return tokens_emitted;
}

std::unique_ptr<ControllerHook> CreateControllerHook(OgaCreateDecodeControllerFn create,
                                                     const std::string& config,
                                                     std::shared_ptr<void> keepalive) {
  if (!create)
    throw std::runtime_error("Decode controller entry point is null.");

  OgaDecodeController* self = nullptr;
  OgaControllerStepFn step = nullptr;
  OgaDecodeControllerDestroyFn destroy = nullptr;
  // The ABI is C (void*), so hand over a borrowed C string; the controller must copy what it retains.
  void* config_arg = config.empty() ? nullptr : const_cast<char*>(config.c_str());
  const int status = create(config_arg, &self, &step, &destroy);
  if (status != 0)
    throw std::runtime_error("Decode controller entry point failed with status " + std::to_string(status) + ".");
  if (!self || !step || !destroy)
    throw std::runtime_error("Decode controller entry point reported success but returned a null handle or callback.");

  return std::make_unique<ControllerHook>(self, step, destroy, std::move(keepalive));
}

}  // namespace Generators
