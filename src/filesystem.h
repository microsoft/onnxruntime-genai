// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// TODO(baijumeswani): Remove experimental when packaging pipeline can use GCC > 8
#ifdef USE_EXPERIMENTAL_FILESYSTEM
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif
