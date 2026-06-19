// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>

#include "../filesystem.h"
#include "onnxruntime_api.h"

namespace Generators {

// True when `path` is a directory containing a top-level manifest.json.
bool IsModelPackage(const fs::path& path);

#if ORT_GENAI_HAS_MODEL_PACKAGE
struct PackageLoadResult {
  fs::path package_root;
  fs::path variant_dir;
};

// Opens a model package, resolves an EP, and selects the variant for the package's single
// component. When `explicit_ep` is empty, the EP is auto-detected: the call succeeds when
// the component's variants declare exactly one EP and fails otherwise. The package must
// declare exactly one component.
PackageLoadResult OpenAndSelectVariant(OrtEnv& env,
                                       const fs::path& package_root,
                                       const std::string& explicit_ep);
#endif  // ORT_GENAI_HAS_MODEL_PACKAGE

}  // namespace Generators
