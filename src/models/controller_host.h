// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Portions of this file consist of AI generated content.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "plugin_api.h"

// Host-side adapter for the controller-plugin escape hatch (issue #2114 §8, PR-E). This binds the
// runtime's C++ Generator step primitives to the stable C `OgaDecodeStepContext` vtable a controller
// plugin calls back into, and wraps a constructed controller so Generator::GenerateNextToken can
// delegate the per-step decode loop to it.
//
// IMPORTANT: this adapter compiles UNCONDITIONALLY (it is just C++ <-> C glue). Only the act of
// LOADING a controller from an external .so is gated behind USE_GENAI_PLUGINS (see plugin_loader.h).
// Keeping the adapter ungated lets an in-tree stub controller exercise the primitive surface in a
// default (plugins-OFF) build without any dynamic loading.

namespace Generators {

struct Generator;

// Binds a Generator to the C-ABI step-primitive vtable. Owns the scratch buffers that pointers
// returned by GetLogits/GetHiddenStates reference; those pointers are valid only until the next
// callback on the same host. Borrows the Generator, which must outlive the host.
class DecodeStepHost {
 public:
  explicit DecodeStepHost(Generator& generator) : generator_{generator} {}

  // Returns a step context whose `host` points at *this and whose callbacks drive `generator_`.
  OgaDecodeStepContext Context();

  Generator& generator_;
  std::vector<float> logits_scratch_;
  std::vector<float> hidden_scratch_;
};

// A constructed controller that owns the per-outer-step decode loop. Created either from a real
// plugin .so (plugin_loader.h, gated by USE_GENAI_PLUGINS) or, for in-tree tests, directly from a
// resolved OgaCreateDecodeControllerFn. Holds an optional keepalive that keeps the plugin module
// loaded for the controller's lifetime.
class ControllerHook {
 public:
  ControllerHook(OgaDecodeController* self, OgaControllerStepFn step,
                 OgaDecodeControllerDestroyFn destroy, std::shared_ptr<void> keepalive);
  ~ControllerHook();

  ControllerHook(const ControllerHook&) = delete;
  ControllerHook& operator=(const ControllerHook&) = delete;

  // Run exactly one controller-owned decode step against `generator`. Returns the number of tokens
  // the controller reports it emitted this step. Throws std::runtime_error on a non-zero status.
  int Step(Generator& generator);

 private:
  OgaDecodeController* self_;
  OgaControllerStepFn step_;
  OgaDecodeControllerDestroyFn destroy_;
  std::shared_ptr<void> keepalive_;  // may be null (in-tree controllers have no module to retain)
};

// Construct a ControllerHook from an already-resolved controller entry point. Shared by the plugin
// loader (which supplies the dlopen keepalive) and by in-tree tests (keepalive null). `config` is the
// opaque controller config string passed verbatim to the entry point.
std::unique_ptr<ControllerHook> CreateControllerHook(OgaCreateDecodeControllerFn create,
                                                     const std::string& config,
                                                     std::shared_ptr<void> keepalive);

}  // namespace Generators
