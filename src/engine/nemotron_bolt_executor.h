// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "engine.h"

namespace Generators {

struct NemotronBoltModel;
EngineDependencies CreateNemotronBoltDependencies(std::shared_ptr<NemotronBoltModel> model);

}  // namespace Generators
