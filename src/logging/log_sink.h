// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string_view>

#include "../filesystem.h"
#include "../logging.h"

namespace Generators {

class LogSink {
 public:
  virtual void Send(const LogCapture& capture) = 0;
  virtual ~LogSink() = default;
};

std::unique_ptr<LogSink> MakeDefaultLogSink();

std::unique_ptr<LogSink> MakeFileLogSink(const fs::path& log_file);

}  // namespace Generators
