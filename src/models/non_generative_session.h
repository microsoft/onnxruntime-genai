// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "../config.h"
#include "../ort_genai_c.h"

#include <string>
#include <string_view>

namespace Generators {

OgaNonGenerativeSession* CreateNonGenerativeSession(const Config& config);
void DestroyNonGenerativeSession(OgaNonGenerativeSession* session);
std::string RunNonGenerativeSessionJson(
    OgaNonGenerativeSession& session, std::string_view request);

}  // namespace Generators
