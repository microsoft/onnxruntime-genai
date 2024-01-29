// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>

namespace tfm {

size_t EncodeUTF8Char(char* buffer, char32_t utf8_char);
std::string EncodeUTF8Char(char32_t utf8_char);
bool ValidateUTF8(const std::string& data);
bool IsDigit(char c);
bool IsHexDigit(char c);
unsigned int HexDigitToInt(char c);
bool IsSurrogate(char32_t c);
bool Unescape(const std::string_view& source, std::string& unescaped, bool is_bytes);
bool UnquoteString(const std::string& str, std::string& unquoted);

inline std::u32string FromUTF8(const std::string_view& utf8) {
  std::u32string ucs32;
  ucs32.reserve(utf8.length() / 2);  // a rough estimation for less memory allocation.
  for (size_t i = 0; i < utf8.size();) {
    char32_t codepoint = 0;
    if ((utf8[i] & 0x80) == 0) {
      codepoint = utf8[i];
      i++;
    } else if ((utf8[i] & 0xE0) == 0xC0) {
      codepoint = ((utf8[i] & 0x1F) << 6) | (utf8[i + 1] & 0x3F);
      i += 2;
    } else if ((utf8[i] & 0xF0) == 0xE0) {
      codepoint = ((utf8[i] & 0x0F) << 12) | ((utf8[i + 1] & 0x3F) << 6) | (utf8[i + 2] & 0x3F);
      i += 3;
    } else {
      codepoint = ((utf8[i] & 0x07) << 18) | ((utf8[i + 1] & 0x3F) << 12) | ((utf8[i + 2] & 0x3F) << 6) | (utf8[i + 3] & 0x3F);
      i += 4;
    }
    ucs32.push_back(codepoint);
  }
  return ucs32;
}

inline std::string ToUTF8(const std::u32string& ucs32) {
  std::string utf8;
  utf8.reserve(ucs32.length() * 4);
  for (char32_t codepoint : ucs32) {
    utf8 += EncodeUTF8Char(codepoint);
  }

  return utf8;
}

}  // namespace tfm
