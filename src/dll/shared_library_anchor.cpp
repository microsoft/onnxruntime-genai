// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace {

// Ensures generators such as Xcode create a link phase for the shared-library target.
[[maybe_unused]] constexpr int kSharedLibraryAnchor = 0;

}  // namespace
