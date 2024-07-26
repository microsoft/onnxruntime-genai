// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <sys/stat.h>

#include <string>
#include <fstream>

#include "u8u16convert.h"

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
  path(const std::wstring& path) : wpath_(path) {
  };
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

#ifdef _WIN32
  const std::wstring& wstring() const {
    return wpath_;
  }
#endif

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

 private:
  std::string path_;

#ifdef _WIN32
  std::wstring wpath_;

  std::wstring to_wstring() const {
    return u8u16(path_);
  }
#endif  // _WIN32
};

}  // namespace fs
