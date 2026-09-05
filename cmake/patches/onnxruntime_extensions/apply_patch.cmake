# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Keeps raw OpenAI tool schemas intact for chat templates that serialize tool
# definitions verbatim.
#
# ort-extensions' NormalizeTools() rewrites tools into the flat Phi-4/Minja shape:
# it unwraps {"type":"function","function":{...}}, flattens "parameters", drops
# "required"/"enum"/"items" and renames the "string" type to "str". Templates that
# render each tool with `tool | tojson` (Qwen 2.5/3.x) then advertise a schema the
# model was never trained on, so it emits argument values that violate the real
# schema (for example target_language "French" instead of the enum value "fr").
#
# The upstream skip heuristic only recognized templates containing the literal
# "tool.function" (Harmony/GPT-OSS). This adds the `tool | tojson` family.

if(NOT DEFINED SOURCE_DIR)
  message(FATAL_ERROR "SOURCE_DIR is required.")
endif()

function(ortgenai_replace_required file_path old_text new_text)
  if(NOT EXISTS "${file_path}")
    message(FATAL_ERROR "ort-extensions source file not found: ${file_path}")
  endif()

  file(READ "${file_path}" contents)
  string(FIND "${contents}" "${new_text}" new_position)
  if(NOT new_position EQUAL -1)
    return()
  endif()

  string(FIND "${contents}" "${old_text}" old_position)
  if(old_position EQUAL -1)
    message(FATAL_ERROR "Expected ort-extensions source text was not found in ${file_path}: ${old_text}")
  endif()

  string(REPLACE "${old_text}" "${new_text}" contents "${contents}")
  file(WRITE "${file_path}" "${contents}")
endfunction()

set(chat_template_cc "${SOURCE_DIR}/shared/api/chat_template.cc")

# std::isspace is used by the helper added below.
ortgenai_replace_required(
  "${chat_template_cc}"
  [=[#include "tokenizer_impl.h"]=]
  [=[#include <cctype>

#include "tokenizer_impl.h"]=])

ortgenai_replace_required(
  "${chat_template_cc}"
  [=[/*
 * This function normalizes quotes in tool-related strings within the input message.]=]
  [=[/*
 * Reports whether a chat template consumes tool definitions in their raw OpenAI shape.
 *
 * NormalizeTools() rewrites tools into the flat Phi-4/Minja shape, which is lossy: it
 * unwraps {"type":"function","function":{...}}, drops "required", "enum", "items" and any
 * nested property schema, and renames the "string" type to "str". That is only safe for
 * templates written against the flat shape. Two families need the raw objects instead:
 *
 *   - Harmony/GPT-OSS templates, which reach into `tool.function` themselves.
 *   - Qwen-style templates, which serialize each tool verbatim with `tool | tojson`.
 *
 * Feeding a normalized tool to the latter silently changes the prompt the model was
 * trained on, so it emits argument values that violate the real schema.
 */
static bool TemplateWantsRawTools(const std::string& tmpl) {
  if (tmpl.find("tool.function") != std::string::npos) {
    return true;
  }

  // Whitespace around a Jinja filter is free-form, so compare without it.
  std::string compact;
  compact.reserve(tmpl.size());
  for (char c : tmpl) {
    if (!std::isspace(static_cast<unsigned char>(c))) {
      compact.push_back(c);
    }
  }

  return compact.find("tool|tojson") != std::string::npos ||
         compact.find("tools|tojson") != std::string::npos;
}

/*
 * This function normalizes quotes in tool-related strings within the input message.]=])

ortgenai_replace_required(
  "${chat_template_cc}"
  [=[    // Determine whether to skip tool normalization based on template content.
    // GPT-OSS/Harmony templates access `tool.function` directly (they expect the raw OpenAI format),
    // so NormalizeTools() would break them by unwrapping the function object.
    // Other templates (Phi-4, Qwen) either use a flat format or access `tool_call.function`.
    bool skip_tool_normalization = false;
    {
      std::string tmpl_str(activated_str);
      // Look for "tool.function" in the template (Harmony/GPT-OSS expects raw OpenAI tool objects).
      skip_tool_normalization = tmpl_str.find("tool.function") != std::string::npos;
    }]=]
  [=[    // Determine whether to skip tool normalization based on template content.
    // Templates that consume tool definitions in their raw OpenAI shape must receive them
    // untouched, because NormalizeTools() unwraps the "function" object and flattens
    // "parameters", which discards "required", "enum" and nested property schemas.
    bool skip_tool_normalization = TemplateWantsRawTools(activated_str);]=])
