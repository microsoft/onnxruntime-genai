// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "json.h"
#include <iostream>
#include <fstream>
#include <cstdarg>

namespace Generators {

LogItems g_log;
static std::ostream* gp_stream{&std::cerr};
static std::unique_ptr<std::ofstream> gp_logfile;

void SetLogBool(std::string_view name, bool value) {
  if (name == "enabled")
    g_log.enabled = value;
  else if (name == "ansi_tags")
    g_log.ansi_tags = value;
  else if (name == "warning")
    g_log.warning = value;
  else if (name == "generate_next_token")
    g_log.generate_next_token = value;
  else if (name == "append_next_tokens")
    g_log.append_next_tokens = value;
  else if (name == "hit_eos")
    g_log.hit_eos = value;
  else if (name == "hit_max_length")
    g_log.hit_max_length = value;
  else if (name == "model_input_values")
    g_log.model_input_values = value;
  else if (name == "model_output_shapes")
    g_log.model_output_shapes = value;
  else if (name == "model_output_values")
    g_log.model_output_values = value;
  else if (name == "model_logits")
    g_log.model_logits = value;
  else if (name == "ort_lib")
    g_log.ort_lib = value;
  else if (name == "value_stats")
    g_log.value_stats = value;
  else
    throw JSON::unknown_value_error{};
}

void SetLogString(std::string_view name, std::string_view value) {
  if (name == "filename") {
    if (value.empty())
      gp_logfile.reset();
    else {
      fs::path filename{std::string(value)};
      gp_logfile = std::make_unique<std::ofstream>(filename.open_for_write());
    }

    if (gp_logfile)
      gp_stream = gp_logfile.get();
    else
      gp_stream = &std::cerr;
  } else
    throw JSON::unknown_value_error{};
}

std::ostream& operator<<(std::ostream& stream, SGR sgr_code) {
  if (g_log.ansi_tags) {
    stream << "\x1b[" << static_cast<int>(sgr_code) << 'm';
  }
  return stream;
}

// #define SGR_EXAMPLE // Uncomment this line to have it display the color example at launch
#ifdef SGR_EXAMPLE
// To show what the ansi tags look like in the terminal, add a call to this function
void SGRExample(std::ostream& stream) {
  stream << "\r\nAnsi SGR Color Test:\r\n"
         << SGR::Fg_Black << "Fg_Black" << SGR::Fg_Red << "Fg_Red" << SGR::Fg_Green << "Fg_Green" << SGR::Fg_Yellow << "Fg_Yellow" << SGR::Fg_Blue << "Fg_Blue" << SGR::Fg_Magenta << "Fg_Magenta" << SGR::Fg_Cyan << "Fg_Cyan" << SGR::Fg_White << "Fg_White" << SGR::Reset << "\r\n"
         << SGR::Bg_Black << "Bg_Black" << SGR::Bg_Red << "Bg_Red" << SGR::Bg_Green << "Bg_Green" << SGR::Bg_Yellow << "Bg_Yellow" << SGR::Bg_Blue << "Bg_Blue" << SGR::Bg_Magenta << "Bg_Magenta" << SGR::Bg_Cyan << "Bg_Cyan" << SGR::Bg_White << "Bg_White" << SGR::Reset << "\r\n"
         << "Styles:\r\n"
         << SGR::Bold << "Bold" << SGR::Reset << ' ' << SGR::Faint << "Faint" << SGR::Reset << ' ' << SGR::Underline << "Underline" << SGR::Reset << std::endl;
}

bool RunExample = (SGRExample(std::cerr), false);
#endif

std::ostream& Log(std::string_view label, std::string_view string) {
  assert(g_log.enabled);

  // Warnings will be yellow, all other labels will be blue
  *gp_stream << SGR::Bold << (label == "warning" ? SGR::Bg_Yellow : SGR::Bg_Blue) << "  " << label << "  " << SGR::Reset << ' ';
  if (!string.empty())
    *gp_stream << string << std::endl;
  return *gp_stream;
}

std::ostream& Log(std::string_view label, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  va_list args_copy;
  va_copy(args_copy, args);
  size_t len = vsnprintf(0, 0, fmt, args_copy);
  if (len <= 0) {
    throw std::runtime_error("Invalid format");
  }
  std::unique_ptr<char[]> buf(new char[len + 1]);
  vsnprintf(buf.get(), len + 1, fmt, args);
  va_end(args);
  return Log(label, std::string(buf.get(), buf.get() + len));
}

}  // namespace Generators
