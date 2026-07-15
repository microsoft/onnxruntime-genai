// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace Generators {

struct Model;

struct Adapter {
  Adapter() = delete;
  Adapter(const Adapter&) = delete;
  Adapter& operator=(const Adapter&) = delete;

  Adapter(const char* adapter_file_path, Ort::Allocator* allocator);

 private:
  // AcquireRef/ReleaseRef/RefCount are intentionally private so that all
  // access to ref_count_ is funneled through Adapters, which holds
  // Adapters::mutex_. Exposing them publicly would make it easy for future
  // call sites to bypass the mutex and reintroduce the data race / TOCTOU
  // window between RefCount() and container erasure in
  // Adapters::UnloadAdapter().
  friend struct Adapters;

  const OrtLoraAdapter* AcquireRef();

  void ReleaseRef();

  int32_t RefCount() const;

  int32_t ref_count_{};
  std::unique_ptr<OrtLoraAdapter> adapter_;
};

struct Adapters : std::enable_shared_from_this<Adapters>, ExternalRefCounted<Adapters> {
  Adapters() = delete;
  Adapters(const Adapters&) = delete;
  Adapters& operator=(const Adapters&) = delete;

  Adapters(const Model* model);

  void LoadAdapter(const char* adapter_file_path, const std::string& adapter_name);

  void UnloadAdapter(const std::string& adapter_name);

  const OrtLoraAdapter* AcquireAdapter(const std::string& adapter_name);

  void ReleaseAdapter(const std::string& adapter_name);

 private:
  const Model* model_;
  // Serializes all access to adapters_ and to per-Adapter ref counts so that
  // load/unload/acquire/release cannot race. Without this, the check-then-erase
  // pattern in UnloadAdapter (and concurrent std::unordered_map mutation) is a
  // use-after-free hazard.
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::unique_ptr<Adapter>> adapters_;
};

}  // namespace Generators
