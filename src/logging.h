// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Generators {

void SetLogBool(std::string_view name, bool value);
void SetLogString(std::string_view name, std::string_view value);

struct LogItems {
  // Special log related entries
  bool enabled{};        // Global on/off for all logging
  bool ansi_tags{true};  // Use ansi SGR color & style tags to make console output easier to read
  bool warning{true};   // warning messages, like options that were set but don't apply

  // Loggable actions, will always have the name below with the log entry
  bool generate_next_token{};
  bool append_next_tokens{};
  bool hit_eos{};  // Only works for CPU non beam search
  bool hit_max_length{};
  bool model_input_values{};   // Dump the input tensor shapes & values before the model runs
  bool model_output_shapes{};  // Before the model runs there are only the output shapes, no values in them. Useful for pre Session::Run debugging
  bool model_output_values{};  // After the model runs the output tensor values can be displayed
  bool model_logits{};         // Same as model_output_values but only for the logits
};

extern LogItems g_log;

// Ansi SGR (Set Graphics Rendition) escape codes to colorize the logs when sent to a console
enum struct SGR : int {
  Reset = 0,
  Bold = 1,
  Faint = 2,
  Underline = 4,

  Fg_Black = 30,
  Fg_Red = 31,
  Fg_Green = 32,
  Fg_Yellow = 33,
  Fg_Blue = 34,
  Fg_Magenta = 35,
  Fg_Cyan = 36,
  Fg_White = 37,

  Bg_Black = 40,
  Bg_Red = 41,
  Bg_Green = 42,
  Bg_Yellow = 43,
  Bg_Blue = 44,
  Bg_Magenta = 45,
  Bg_Cyan = 46,
  Bg_White = 47,
};

std::ostream& operator<<(std::ostream& stream, SGR sgr_code);

std::ostream& Log(std::string_view label, std::string_view text = {});

}  // namespace Generators