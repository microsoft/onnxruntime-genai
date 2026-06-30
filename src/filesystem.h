// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

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
#include <direct.h>
#endif  // _WIN32

#include <sys/stat.h>

#include <fstream>
#include <stdexcept>
#include <string>

#ifndef _WIN32
#include <limits.h>
#include <unistd.h>
#endif

namespace fs {

class path {
 public:
  path() = default;
  path(const std::string& path) : path_(path) {
#ifdef _WIN32
    wpath_ = to_wstring();
#endif
  };

#ifdef _WIN32
  // Construct from a wide (UTF-16) path, e.g. an ORTCHAR_T path from ONNX Runtime.
  path(const std::wstring& wpath) : path_(to_utf8(wpath)), wpath_(wpath) {};
#endif

  static constexpr char separator =
#ifdef _WIN32
      '\\';
#else
      '/';
#endif

  using ios_base = std::ios_base;
  std::ifstream open(ios_base::openmode mode = ios_base::in) const {
    // if Windows, need to convert the string to UTF-16
#ifdef _WIN32
    return std::ifstream(wpath_, mode);
#else
    return std::ifstream(path_, mode);
#endif  // _WIN32
  }

  std::ofstream open_for_write(ios_base::openmode mode = ios_base::out) const {
    // if Windows, need to convert the string to UTF-16
#ifdef _WIN32
    return std::ofstream(wpath_, mode);
#else
    return std::ofstream(path_, mode);
#endif  // _WIN32
  }

  const std::string& string() const {
    return path_;
  }

  path join(const std::string& path) const {
    return path_ + separator + path;
  }

  path operator/(const std::string& path) const {
    return join(path);
  }

  path operator/(const path& path) {
    return join(path.path_);
  }

#ifdef _WIN32
  const wchar_t* c_str() const {
    return wpath_.c_str();
  }
#else
  const char* c_str() const {
    return path_.c_str();
  }
#endif

  bool is_directory() const {
#ifdef _WIN32
    const int ret = GetFileAttributesW(wpath_.c_str());
    return ret & FILE_ATTRIBUTE_DIRECTORY;
#else
    struct stat info;
    if (stat(path_.c_str(), &info) != 0) {
      return false;
    }
    return (info.st_mode & S_IFDIR) != 0;
#endif  // _WIN32
  }

  bool exists() const {
#ifdef _WIN32
    const int ret = GetFileAttributesW(wpath_.c_str());
    return ret != INVALID_FILE_ATTRIBUTES;
#else
    return std::ifstream(path_).good();
#endif
  }

  bool is_relative() const {
#ifdef _WIN32
    // On Windows, check if path starts with drive letter or UNC path
    if (path_.length() >= 2 && path_[1] == ':') {
      return false;  // Absolute path with drive letter (e.g., "C:\")
    }
    if (path_.length() >= 2 && path_[0] == '\\' && path_[1] == '\\') {
      return false;  // UNC path (e.g., "\\server\share")
    }
    return true;  // Relative path
#else
    // On Unix-like systems, absolute paths start with '/'
    return !path_.empty() && path_[0] != '/';
#endif
  }

  path parent_path() const {
    size_t pos = path_.find_last_of("/\\");
    if (pos == std::string::npos) {
      return path();  // No parent directory found
    }
    return path(path_.substr(0, pos));
  }

 private:
  std::string path_;

#ifdef _WIN32
  std::wstring wpath_;

  std::wstring to_wstring() const {
    // If there's nothing to convert, bail early.
    if (path_.empty()) {
      return {};
    }

    int codePage = CP_UTF8;
    int iSource;  // convert to int because Mb2Wc requires it.
    SizeTToInt(path_.size(), &iSource);

    // Ask how much space we will need.
    // In certain codepages, Mb2Wc will "successfully" produce zero characters (like in CP50220, where a SHIFT-IN character
    // is consumed but not transformed into anything) without explicitly failing. When it does this, GetLastError will return
    // the last error encountered by the last function that actually did have an error.
    // This is arguably correct (as the documentation says "The function returns 0 if it does not succeed"). There is a
    // difference that we **don't actually care about** between failing and successfully producing zero characters.,
    // Anyway: we need to clear the last error so that we can fail out and IGNORE_BAD_GLE after it inevitably succeed-fails.
    SetLastError(0);
    const auto iTarget = MultiByteToWideChar(codePage, 0, path_.data(), iSource, nullptr, 0);

    size_t cchNeeded;
    IntToSizeT(iTarget, &cchNeeded);

    // Allocate ourselves some space
    std::wstring out;
    out.resize(cchNeeded);

    // Attempt conversion for real.
    MultiByteToWideChar(codePage, 0, path_.data(), iSource, out.data(), iTarget);

    // Return as a string
    return out;
  }

  // Convert a wide (UTF-16) path to UTF-8, the inverse of to_wstring().
  static std::string to_utf8(const std::wstring& wpath) {
    if (wpath.empty()) {
      return {};
    }

    int iSource;  // convert to int because Wc2Mb requires it.
    SizeTToInt(wpath.size(), &iSource);

    // Ask how much space we will need.
    SetLastError(0);
    const auto iTarget =
        WideCharToMultiByte(CP_UTF8, 0, wpath.data(), iSource, nullptr, 0, nullptr, nullptr);

    size_t cchNeeded;
    IntToSizeT(iTarget, &cchNeeded);

    // Allocate ourselves some space and convert for real.
    std::string out;
    out.resize(cchNeeded);
    WideCharToMultiByte(CP_UTF8, 0, wpath.data(), iSource, out.data(), iTarget, nullptr, nullptr);

    return out;
  }
#endif  // _WIN32
};

// Namespace-level functions
inline bool exists(const path& p) {
  return p.exists();
}

inline path absolute(const path& p) {
  if (!p.is_relative()) {
    return p;
  }

#ifdef _WIN32
  char cwd_buffer[_MAX_PATH];
  if (!_getcwd(cwd_buffer, sizeof(cwd_buffer))) {
#else
  char cwd_buffer[PATH_MAX];
  if (!getcwd(cwd_buffer, sizeof(cwd_buffer))) {
#endif
    throw std::runtime_error("Failed to get current working directory");
  }

  path cwd_path{cwd_buffer};
  return cwd_path / p;
}

}  // namespace fs
