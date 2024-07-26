// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <string_view>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN  // Exclude rarely-used stuff from Windows headers
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#define ENABLE_INTSAFE_SIGNED_FUNCTIONS  // Only unsigned intsafe math/casts available without this def
#include <intsafe.h>
#include <tchar.h>

inline std::string u16u8(const std::wstring_view& in) {
  // If there's nothing to convert, bail early.
  if (in.empty()) {
    return {};
  }
  int iSource;
  SizeTToInt(in.size(), &iSource);

  const int iTarget = WideCharToMultiByte(CP_UTF8, 0ul, in.data(), iSource, nullptr, 0, nullptr, nullptr);
  size_t cchNeeded;
  IntToSizeT(iTarget, &cchNeeded);

  std::string out{};
  out.resize(cchNeeded);
  WideCharToMultiByte(CP_UTF8, 0ul, in.data(), iSource, out.data(), iTarget, nullptr, nullptr);
  return out;
}

inline std::wstring u8u16(const std::string_view& in) {
  // If there's nothing to convert, bail early.
  if (in.empty()) {
    return {};
  }

  int codePage = CP_UTF8;
  int iSource;  // convert to int because Mb2Wc requires it.
  SizeTToInt(in.size(), &iSource);

  // Ask how much space we will need.
  // In certain codepages, Mb2Wc will "successfully" produce zero characters (like in CP50220, where a SHIFT-IN character
  // is consumed but not transformed into anything) without explicitly failing. When it does this, GetLastError will return
  // the last error encountered by the last function that actually did have an error.
  // This is arguably correct (as the documentation says "The function returns 0 if it does not succeed"). There is a
  // difference that we **don't actually care about** between failing and successfully producing zero characters.,
  // Anyway: we need to clear the last error so that we can fail out and IGNORE_BAD_GLE after it inevitably succeed-fails.
  SetLastError(0);
  const auto iTarget = MultiByteToWideChar(codePage, 0, in.data(), iSource, nullptr, 0);

  size_t cchNeeded;
  IntToSizeT(iTarget, &cchNeeded);

  // Allocate ourselves some space
  std::wstring out;
  out.resize(cchNeeded);

  // Attempt conversion for real.
  MultiByteToWideChar(codePage, 0, in.data(), iSource, out.data(), iTarget);

  // Return as a string
  return out;
}

#endif  // _WIN32