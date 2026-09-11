// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "generator/generators.h"

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>

namespace Generators {

struct Audios;
struct Config;
struct Images;
struct Processor;
struct SessionInfo;
struct Tokenizer;

struct MultiModalProcessor : std::enable_shared_from_this<MultiModalProcessor>, ExternalRefCounted<MultiModalProcessor> {
  MultiModalProcessor(Config& config, const SessionInfo& session_info);

  std::unique_ptr<NamedTensors> Process(const std::string& prompt, const Images* images, const Audios* audios) const;
  std::unique_ptr<NamedTensors> Process(std::span<const char*> prompts, const Images* images, const Audios* audios) const;

  std::shared_ptr<Tokenizer> tokenizer_;
  std::shared_ptr<Processor> processor_;

 private:
  std::unordered_map<std::string, std::function<std::shared_ptr<Processor>(Config&, const SessionInfo&)>> processor_factory_;
};

}  // namespace Generators
