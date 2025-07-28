// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
/*
 * General purpose logging for GenAI
 *
 * The two functions to access the log are at the top, SetLogBool and SetLogString.
 * These functions modify the variables in the global LogItems structure.
 *
 * For example, to enable logging, just SetLogBool("enabled", true)
 * To dump model run call input tensors, SetLogBool("model_input_values", true)
 *
 * NOTE: The names in LogItems must match the strings in the APIs and the strings displayed for log entries.
 *       This makes it easy to know what option is displaying which data, and to easily know how to turn options off.
 *
 * Logging to a file is special: SetLogString("filename", "path") as "filename" is not a string in LogItems
 *
 * COLOR: The functions use ANSI SGR terminal codes for color, the 'struct SGR' below makes it easy to add common
 *        options during log options. Just look in the code for examples of how to use it. Note that the colors
 *        may differ in intensity/saturation on different platforms.
 *
 *        "warning" messages will appear in yellow, there is a special case in the Log(...) function for this
 *        There is no red for errors, as errors are exceptions.
 */
namespace Generators {

using CallbackFn = void (*)(const char* string, size_t length);

void SetLogBool(std::string_view name, bool value);
void SetLogString(std::string_view name, std::string_view value);
void SetLogCallback(CallbackFn callback);

struct LogItems {
  // Special log related entries
  bool enabled{};        // Global on/off for all logging
  bool ansi_tags{true};  // Use ansi SGR color & style tags to make console output easier to read
  bool warning{true};    // warning messages, like options that were set but don't apply

  // Loggable actions, will always have the name below with the log entry
  bool generate_next_token{};
  bool append_next_tokens{};
  bool hit_eos{};  // Only works for CPU non beam search
  bool hit_max_length{};
  bool model_input_values{};   // Dump the input tensor shapes & values before the model runs
  bool model_output_shapes{};  // Before the model runs there are only the output shapes, no values in them. Useful for pre Session::Run debugging
  bool model_output_values{};  // After the model runs the output tensor values can be displayed
  bool model_logits{};         // Same as model_output_values but only for the logits
  bool ort_lib{};              // Log the onnxruntime library loading and api calls.
  bool value_stats{true};      // When logging float values, also dump some basic stats about the values (min, max, mean, std dev, and if there are any NaN or Inf values)
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

std::ostream& Log(std::string_view label, const char* fmt, ...);
}  // namespace Generators
