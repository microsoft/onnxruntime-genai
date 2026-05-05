// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Compile-time verification of VisionState type hierarchy.
//
// The primary guard against removing `virtual` from SetExtraInputs is
// the `override` keyword on PixtralVisionState::SetExtraInputs — the
// compiler will reject the build if the base method is not virtual or
// if the signature drifts. These static_asserts document the expected
// type relationships as an additional safety net.

#include <gtest/gtest.h>
#include <type_traits>

#include "models/multi_modal.h"

namespace Generators::test {

// Verify inheritance relationships.
static_assert(std::is_base_of_v<VisionState, PixtralVisionState>,
              "PixtralVisionState must derive from VisionState");
static_assert(std::is_base_of_v<VisionState, QwenVisionState>,
              "QwenVisionState must derive from VisionState");

// Verify polymorphic (has virtual functions — needed for correct dispatch
// through VisionState* base pointers in the factory).
static_assert(std::is_polymorphic_v<VisionState>,
              "VisionState must be polymorphic for factory dispatch");
static_assert(std::is_polymorphic_v<PixtralVisionState>,
              "PixtralVisionState must be polymorphic");

TEST(VisionStateTypeHierarchy, InheritanceAndPolymorphism) {
  // These are compile-time checks (static_asserts above). This test
  // exists so the test runner reports them and the file is linked.
  SUCCEED();
}

}  // namespace Generators::test
