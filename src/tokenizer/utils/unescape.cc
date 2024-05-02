// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include "unescape.h"

namespace tfm{

size_t EncodeUTF8Char(char* buffer, char32_t utf8_char) {
  if (utf8_char <= 0x7F) {
    *buffer = static_cast<char>(utf8_char);
    return 1;
  } else if (utf8_char <= 0x7FF) {
    buffer[1] = static_cast<char>(0x80 | (utf8_char & 0x3F));
    utf8_char >>= 6;
    buffer[0] = static_cast<char>(0xC0 | utf8_char);
    return 2;
  } else if (utf8_char <= 0xFFFF) {
    buffer[2] = static_cast<char>(0x80 | (utf8_char & 0x3F));
    utf8_char >>= 6;
    buffer[1] = static_cast<char>(0x80 | (utf8_char & 0x3F));
    utf8_char >>= 6;
    buffer[0] = static_cast<char>(0xE0 | utf8_char);
    return 3;
  } else {
    buffer[3] = static_cast<char>(0x80 | (utf8_char & 0x3F));
    utf8_char >>= 6;
    buffer[2] = static_cast<char>(0x80 | (utf8_char & 0x3F));
    utf8_char >>= 6;
    buffer[1] = static_cast<char>(0x80 | (utf8_char & 0x3F));
    utf8_char >>= 6;
    buffer[0] = static_cast<char>(0xF0 | utf8_char);
    return 4;
  }
}

std::string EncodeUTF8Char(char32_t utf8_char) {
  char utf8_buf[5];  // one extra space for zero
  auto clen = EncodeUTF8Char(utf8_buf, utf8_char);
  utf8_buf[clen] = 0;
  return {utf8_buf};
}

 bool ValidateUTF8(const std::string& data) {
    const unsigned char* s = reinterpret_cast<const unsigned char*>(data.c_str());
    const unsigned char* s_end = s + data.size();
    if (*s_end != '\0')
      return false;

    while (*s) {
      if (*s < 0x80)
        /* 0xxxxxxx */
        s++;
      else if ((s[0] & 0xe0) == 0xc0) {
        /* 110XXXXx 10xxxxxx */
        if (s + 1 >= s_end) {
          return false;
        }
        if ((s[1] & 0xc0) != 0x80 ||
            (s[0] & 0xfe) == 0xc0) /* overlong? */
          return false;
        else
          s += 2;
      } else if ((s[0] & 0xf0) == 0xe0) {
        /* 1110XXXX 10Xxxxxx 10xxxxxx */
        if (s + 2 >= s_end) {
          return false;
        }
        if ((s[1] & 0xc0) != 0x80 ||
            (s[2] & 0xc0) != 0x80 ||
            (s[0] == 0xe0 && (s[1] & 0xe0) == 0x80) || /* overlong? */
            (s[0] == 0xed && (s[1] & 0xe0) == 0xa0) || /* surrogate? */
            (s[0] == 0xef && s[1] == 0xbf &&
             (s[2] & 0xfe) == 0xbe)) /* U+FFFE or U+FFFF? */
          return false;
        else
          s += 3;
      } else if ((s[0] & 0xf8) == 0xf0) {
        /* 11110XXX 10XXxxxx 10xxxxxx 10xxxxxx */
        if (s + 3 >= s_end) {
          return false;
        }
        if ((s[1] & 0xc0) != 0x80 ||
            (s[2] & 0xc0) != 0x80 ||
            (s[3] & 0xc0) != 0x80 ||
            (s[0] == 0xf0 && (s[1] & 0xf0) == 0x80) ||    /* overlong? */
            (s[0] == 0xf4 && s[1] > 0x8f) || s[0] > 0xf4) /* > U+10FFFF? */
          return false;
        else
          s += 4;
      } else
        return false;
    }

    return true;
  }


bool IsDigit(char c) { return c >= '0' && c <= '9'; }
bool IsHexDigit(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }

unsigned int HexDigitToInt(char c) {
  unsigned int x = static_cast<unsigned char>(c);
  if (x > '9') {
    x += 9;
  }
  return x & 0xf;
}

bool IsSurrogate(char32_t c) {
  return c >= 0xD800 && c <= 0xDFFF;
}

// Unescape a Python escaped string
bool Unescape(const std::string_view& source, std::string& unescaped, bool is_bytes) {
  // reserve enough space for the worst case, and final size will be calculated at the end.
  unescaped.resize(source.length());
  char* d = unescaped.data();
  const char* p = source.data();
  const char* end = p + source.size();
  const char* last_byte = end - 1;

  while (p == d && p < end && *p != '\\') p++, d++;

  while (p < end) {
    if (*p != '\\') {
      *d++ = *p++;
    } else {
      if (++p > last_byte) {
        return false;
      }
      switch (*p) {
        case 'n':
          *d++ = '\n';
          break;
        case 'r':
          *d++ = '\r';
          break;
        case 't':
          *d++ = '\t';
          break;
        case '\\':
          *d++ = '\\';
          break;
        case '\'':
          *d++ = '\'';
          break;
        case '"':
          *d++ = '\"';
          break;
        case 'x':
        case 'X': {
          if (p >= last_byte) {
            return false;
          } else if (!IsHexDigit(static_cast<unsigned char>(p[1]))) {
            return false;
          }
          unsigned int ch = 0;
          while (p < last_byte &&
                 IsHexDigit(static_cast<unsigned char>(p[1])))
            ch = (ch << 4) + HexDigitToInt(*++p);
          if (ch > 0xFF && !is_bytes) {
            return false;
          }
          if (is_bytes) {
            *d++ = static_cast<char>(ch);
          } else {
            d += EncodeUTF8Char(d, static_cast<char32_t>(ch));
          }
          break;
        }
        case 'u': {
          char32_t rune = 0;
          if (p + 4 >= end) {
            return false;
          }
          for (int i = 0; i < 4; ++i) {
            if (IsHexDigit(static_cast<unsigned char>(p[1]))) {
              rune = (rune << 4) + HexDigitToInt(*++p);
            } else {
              return false;
            }
          }
          if (IsSurrogate(rune)) {
            return false;
          }
          d += EncodeUTF8Char(d, rune);
          break;
        }
        case 'U': {
          char32_t rune = 0;
          if (p + 8 >= end) {
            return false;
          }
          for (int i = 0; i < 8; ++i) {
            if (IsHexDigit(static_cast<unsigned char>(p[1]))) {
              uint32_t newrune = (rune << 4) + HexDigitToInt(*++p);
              if (newrune > 0x10FFFF) {
                return false;
              } else {
                rune = newrune;
              }
            } else {
              return false;
            }
          }
          if (IsSurrogate(rune)) {
            return false;
          }
          d += EncodeUTF8Char(d, rune);
          break;
        }
        default: {
          return false;
        }
      }
      p++;
    }
  }

  unescaped.resize(d - unescaped.data());
  return true;
}

bool UnquoteString(const std::string& str, std::string& unquoted) {
  bool is_bytes = false;
  int idx_0 = 0;
  if (str.front() == 'b') {
    is_bytes = true;
    idx_0 = 1;
  }
  std::string str_view(str.data() + idx_0, str.length() - idx_0);
  if (str_view.length() < 2) {
    return false;
  }

  if ((str_view.front() != '\"' && str_view.front() != '\'') || str_view.back() != str.back()) {
    return false;
  }

  // unescape the string
  return Unescape(std::string_view(str_view.data() + 1, str_view.length() - 2), unquoted, is_bytes);
}

}  // namespace tfm
