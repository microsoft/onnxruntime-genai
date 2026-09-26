// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "model.h"

namespace Generators {

// One CPU-only lookup session shared by the Engine target and its block drafter.
// Callers own the output buffers, including their CUDA graph lifetimes.
struct CpuEmbedding {
  struct Workspace {
    Workspace() = default;
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    ~Workspace();
    DeviceSpan<uint8_t> Prepare(Tensor& output);
    void Upload();

   private:
    void Wait();
    DeviceInterface* device_{};
    std::shared_ptr<DeviceBuffer> buffer_;
    size_t capacity_{};
    bool pending_{};
  };

  CpuEmbedding(Model& model, OrtEnv& env);
  void ValidateConsumer(const SessionInfo& info, const std::string& name) const;
  void Run(std::span<const int64_t> ids, Tensor& output, Workspace& workspace) const;

  ONNXTensorElementDataType type_{};
  int64_t hidden_size_{};

 private:
  Config::Model::Embedding config_;
  std::unique_ptr<OrtSession> session_;
  std::unique_ptr<OrtRunOptions> run_options_;
};

}  // namespace Generators
