// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>

#include "../filesystem.h"
#include "onnxruntime_c_api.h"

// ORT_GENAI_HAS_MODEL_PACKAGE is set when the ORT C API version exposes the
// OrtModelPackageApi experimental functions (introduced in API version 28).
#if !defined(ORT_GENAI_HAS_MODEL_PACKAGE)
#if defined(ORT_API_VERSION) && ORT_API_VERSION >= 28
#define ORT_GENAI_HAS_MODEL_PACKAGE 1
#else
#define ORT_GENAI_HAS_MODEL_PACKAGE 0
#endif
#endif

// Internal C++ helpers are not part of the genai DLL's public C API exports on Windows.
// BUILDING_ORT_GENAI_DLL is set privately on the onnxruntime-genai target so consumers
// (e.g. unit_tests) see dllimport for these symbols.
#if defined(_WIN32)
#ifdef BUILDING_ORT_GENAI_DLL
#define MODEL_PACKAGE_API __declspec(dllexport)
#else
#define MODEL_PACKAGE_API __declspec(dllimport)
#endif
#else
#define MODEL_PACKAGE_API
#endif

struct OrtEnv;

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

