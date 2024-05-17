// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef  _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif

// Windows Header Files:
#include <windows.h>
#define ENABLE_INTSAFE_SIGNED_FUNCTIONS // Only unsigned intsafe math/casts available without this def
#include <intsafe.h>
#include <tchar.h>

using path_type = std::wstring;

[[nodiscard]] inline std::wstring ConvertToW(const UINT codePage, const std::string_view source) {
  // If there's nothing to convert, bail early.
  if (source.empty()) {
    return {};
  }

  int iSource;  // convert to int because Mb2Wc requires it.
  SizeTToInt(source.size(), &iSource);

  // Ask how much space we will need.
  // In certain codepages, Mb2Wc will "successfully" produce zero characters (like in CP50220, where a SHIFT-IN character
  // is consumed but not transformed into anything) without explicitly failing. When it does this, GetLastError will return
  // the last error encountered by the last function that actually did have an error.
  // This is arguably correct (as the documentation says "The function returns 0 if it does not succeed"). There is a
  // difference that we **don't actually care about** between failing and successfully producing zero characters.,
  // Anyway: we need to clear the last error so that we can fail out and IGNORE_BAD_GLE after it inevitably succeed-fails.
  SetLastError(0);
  const auto iTarget = MultiByteToWideChar(codePage, 0, source.data(), iSource, nullptr, 0);

  size_t cchNeeded;
  IntToSizeT(iTarget, &cchNeeded);

  // Allocate ourselves some space
  std::wstring out;
  out.resize(cchNeeded);

  // Attempt conversion for real.
  MultiByteToWideChar(codePage, 0, source.data(), iSource, out.data(), iTarget);

  // Return as a string
  return out;
}

[[nodiscard]] inline std::string ConvertToA(const UINT codepage, const std::wstring_view source) {
  // If there's nothing to convert, bail early.
  if (source.empty()) {
    return {};
  }

  int iSource;  // convert to int because Wc2Mb requires it.
  SizeTToInt(source.size(), &iSource);

  // Ask how much space we will need.
  // clang-format off
#pragma prefast(suppress: __WARNING_W2A_BEST_FIT, "WC_NO_BEST_FIT_CHARS doesn't work in many codepages. Retain old behavior.")
  // clang-format on
  const auto iTarget = WideCharToMultiByte(codepage, 0, source.data(), iSource, nullptr, 0, nullptr, nullptr);

  size_t cchNeeded;
  IntToSizeT(iTarget, &cchNeeded);

  // Allocate ourselves some space
  std::string out;
  out.resize(cchNeeded);

  // Attempt conversion for real.
  // clang-format off
#pragma prefast(suppress: __WARNING_W2A_BEST_FIT, "WC_NO_BEST_FIT_CHARS doesn't work in many codepages. Retain old behavior.")
  // clang-format on
  WideCharToMultiByte(codepage, 0, source.data(), iSource, out.data(), iTarget, nullptr, nullptr);

  // Return as a string
  return out;
}

[[nodiscard]] inline std::wstring acp_to_wide_string(const std::string_view source) {
  int cp = GetACP();
  return ConvertToW(cp, source);
}

[[nodiscard]] inline std::wstring utf8_to_wide_string(const std::string_view source) {
  return ConvertToW(CP_UTF8, source);
}

[[nodiscard]] inline std::string wide_string_to_utf8(const std::wstring_view source) {
  return ConvertToA(CP_UTF8, source);
}

[[nodiscard]] inline path_type concat_file_path(const path_type& a, const path_type& b) {
  if (a.ends_with(L'\\') || a.ends_with(L'/')) {
    return a + b;
  }
  return a + L"\\" + b;
}

#else

using path_type = std::string;

[[nodiscard]] inline path_type concat_file_path(const path_type& a, const path_type& b) {
  if (a.ends_with('/')) {
    return a + b;
  }
  return a + "/" + b;
}


#endif  //  _WIN32
