// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(__ANDROID__)

#include <memory>

#include "log_sink.h"

namespace Generators {

std::unique_ptr<LogSink> MakeAndroidLogSink();

}  // namespace Generators

#endif  // defined(__ANDROID__)
