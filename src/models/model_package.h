// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>

#include "../filesystem.h"
#include "onnxruntime_api.h"

// ORT_GENAI_HAS_MODEL_PACKAGE is set when the loaded ONNX Runtime exposes the
// OrtModelPackageApi experimental functions (introduced in API version 28) and the
// experimental header is available on the build's include path. This mirrors the gate
// guarding the OrtModelPackage* RAII wrappers in onnxruntime_api.h. Some Apple/iOS
// toolchains ship the core C API but no experimental header; those builds compile with
// model-package support disabled.
#if !defined(ORT_GENAI_HAS_MODEL_PACKAGE)
#if defined(ORT_API_VERSION) && ORT_API_VERSION >= 28 && ORT_GENAI_HAS_EXPERIMENTAL_C_API
#define ORT_GENAI_HAS_MODEL_PACKAGE 1
#else
#define ORT_GENAI_HAS_MODEL_PACKAGE 0
#endif
#endif

// Internal C++ helpers are not part of the genai DLL's public C API exports on Windows.
// ORT_GENAI_INTERNAL_API (defined in filesystem.h) resolves to dllexport for the genai
// build and dllimport for consumers (e.g. unit_tests).
#define MODEL_PACKAGE_API ORT_GENAI_INTERNAL_API

namespace Generators {

// True when `path` is a directory containing a top-level manifest.json.
MODEL_PACKAGE_API bool IsModelPackage(const fs::path& path);

#if ORT_GENAI_HAS_MODEL_PACKAGE
struct PackageLoadResult {
  fs::path package_root;
  fs::path variant_dir;
};

// Opens a model package, resolves an EP, and selects the variant for the package's single
// component. When `explicit_ep` is empty, the EP is auto-detected: the call succeeds when
// the component's variants declare exactly one EP and fails otherwise. The package must
// declare exactly one component.
MODEL_PACKAGE_API PackageLoadResult OpenAndSelectVariant(OrtEnv& env,
                                                         const fs::path& package_root,
                                                         const std::string& explicit_ep);
#endif  // ORT_GENAI_HAS_MODEL_PACKAGE

}  // namespace Generators
