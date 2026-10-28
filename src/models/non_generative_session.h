// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "../config.h"
#include "../ort_genai_c.h"

OgaNonGenerativeSession* CreateNonGenerativeSession(
    const Generators::Config& config);
