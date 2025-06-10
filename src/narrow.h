// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Generators::narrow_cast(), Generators::narrow(), and Generators::narrowing_error are similar to the GSL
// functions/types of the same name.
// See: https://github.com/isocpp/CppCoreGuidelines/blob/cb0744e931fd9f441649d9a31b6acdbaa789109d/CppCoreGuidelines.md#es49-if-you-must-use-a-cast-use-a-named-cast

#include <stdexcept>  // std::runtime_error
#include <type_traits>
#include <utility>  // std::forward

namespace Generators {

struct narrowing_error : std::runtime_error {
  narrowing_error() : std::runtime_error("narrowing error") {}
};

namespace detail {
[[noreturn]] inline void OnNarrowingError() {
  throw narrowing_error{};
}
}  // namespace detail

template <typename T, typename U>
constexpr T narrow_cast(U&& u) noexcept {
  return static_cast<T>(std::forward<U>(u));
}

// This implementation of Generators::narrow was copied and adapted from:
// https://github.com/microsoft/GSL/blob/a3534567187d2edc428efd3f13466ff75fe5805c/include/gsl/narrow

// narrow() : a checked version of narrow_cast() that throws narrowing_error if the cast changed the value
template <class T, class U, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
constexpr T narrow(U u) {
  constexpr const bool is_different_signedness =
      (std::is_signed<T>::value != std::is_signed<U>::value);

  const T t = narrow_cast<T>(u);  // While this is technically undefined behavior in some cases (i.e., if the source value is of floating-point type
                                  // and cannot fit into the destination integral type), the resultant behavior is benign on the platforms
                                  // that we target (i.e., no hardware trap representations are hit).

  if (static_cast<U>(t) != u || (is_different_signedness && ((t < T{}) != (u < U{})))) {
    detail::OnNarrowingError();
  }

  return t;
}

template <class T, class U, typename std::enable_if<!std::is_arithmetic<T>::value>::type* = nullptr>
constexpr T narrow(U u) {
  const T t = narrow_cast<T>(u);

  if (static_cast<U>(t) != u) {
    detail::OnNarrowingError();
  }

  return t;
}

}  // namespace Generators
