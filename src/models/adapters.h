// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace Generators {

struct Adapter {
  Adapter() = delete;
  Adapter(const Adapter&) = delete;
  Adapter& operator=(const Adapter&) = delete;

  Adapter(const char* adapter_file_path, Ort::Allocator* allocator);

  const OrtLoraAdapter* AcquireRef();

  void ReleaseRef();

  int32_t RefCount() const;

 private:
  int32_t ref_count_{};
  std::unique_ptr<OrtLoraAdapter> adapter_;
};

struct Adapters : std::enable_shared_from_this<Adapters> {
  Adapters() = delete;
  Adapters(const Adapters&) = delete;
  Adapters& operator=(const Adapters&) = delete;

  Adapters(const Model* model);

  void LoadAdapter(const char* adapter_file_path, const std::string& adapter_name);

  void UnloadAdapter(const std::string& adapter_name);

  const OrtLoraAdapter* AcquireAdapter(const std::string& adapter_name);

  void ReleaseAdapter(const std::string& adapter_name);

  std::shared_ptr<Adapters> external_owner_;

 private:
  const Model* model_;
  std::unordered_map<std::string, std::unique_ptr<Adapter>> adapters_;
};

}  // namespace Generators
