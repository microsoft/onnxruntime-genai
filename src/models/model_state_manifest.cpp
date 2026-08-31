// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model_state_manifest.h"

#include <algorithm>
#include <array>
#include <optional>
#include <set>
#include <sstream>
#include <utility>

namespace Generators {
namespace {

using Decoder = Config::Model::Decoder;
using StateGroup = Decoder::StateGroup;
using StateGroupKind = Decoder::StateGroupKind;
using StateUpdateKind = Decoder::StateUpdateKind;

struct StateBinding {
  std::string_view semantic;
  const std::string& input;
  const std::string& output;
};

std::vector<StateBinding> StateBindingsFor(
    const Decoder::Inputs& inputs,
    const Decoder::Outputs& outputs,
    StateGroupKind kind) {
  switch (kind) {
    case StateGroupKind::PagedKeyValue:
      return {
          {"key", inputs.past_key_names, outputs.present_key_names},
          {"value", inputs.past_value_names, outputs.present_value_names},
      };
    case StateGroupKind::FixedConv:
      return {{"state", inputs.past_conv_names, outputs.present_conv_names}};
    case StateGroupKind::FixedRecurrent:
      return {{"state", inputs.past_recurrent_names, outputs.present_recurrent_names}};
    case StateGroupKind::Invalid:
      return {};
  }
  return {};
}

bool IsFixedKind(StateGroupKind kind) {
  return kind == StateGroupKind::FixedConv ||
         kind == StateGroupKind::FixedRecurrent;
}

StateUpdateKind StateUpdateKindFor(StateGroupKind kind) {
  if (kind == StateGroupKind::FixedConv) {
    return StateUpdateKind::CausalConv;
  }
  if (kind == StateGroupKind::FixedRecurrent) {
    return StateUpdateKind::GatedDeltaNet;
  }
  return StateUpdateKind::Invalid;
}

std::string_view StateGroupKindName(StateGroupKind kind) {
  switch (kind) {
    case StateGroupKind::PagedKeyValue:
      return "paged_kv";
    case StateGroupKind::FixedConv:
      return "fixed_conv";
    case StateGroupKind::FixedRecurrent:
      return "fixed_recurrent";
    case StateGroupKind::Invalid:
      return "invalid";
  }
  return "invalid";
}

std::string StateGroupLabel(size_t index, StateGroupKind kind) {
  return "model.decoder.state_groups[" + std::to_string(index) + "] (" +
         std::string{StateGroupKindName(kind)} + ")";
}

void ValidateBindingTemplate(std::string_view group_label,
                             std::string_view semantic,
                             std::string_view direction,
                             const std::string& value) {
  if (value.empty()) {
    throw std::runtime_error(
        std::string{group_label} + " binding '" +
        std::string{semantic} + "' is missing its " + std::string{direction} + " template");
  }

  const auto placeholder = value.find('%');
  if (placeholder == std::string::npos ||
      placeholder + 1 >= value.size() ||
      value[placeholder + 1] != 'd' ||
      value.find('%', placeholder + 2) != std::string::npos) {
    throw std::runtime_error(
        std::string{group_label} + " binding '" +
        std::string{semantic} + "' has malformed " + std::string{direction} +
        " template '" + value + "'; expected exactly one %d");
  }
}

std::string ExpandBinding(const std::string& value, int layer_id) {
  std::string result{value};
  result.replace(result.find("%d"), 2, std::to_string(layer_id));
  return result;
}

bool ShapesCompatible(const std::vector<int64_t>& left,
                      const std::vector<int64_t>& right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (size_t i = 0; i < left.size(); ++i) {
    if (left[i] >= 0 && right[i] >= 0 && left[i] != right[i]) {
      return false;
    }
  }
  return true;
}

std::string ShapeString(const std::vector<int64_t>& shape) {
  std::ostringstream output;
  output << '[';
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << shape[i];
  }
  output << ']';
  return output.str();
}

struct TensorMetadata {
  ONNXTensorElementDataType data_type;
  std::vector<int64_t> shape;
  std::string name;
};

void ValidateCompatiblePair(std::string_view group_label,
                            std::string_view semantic,
                            const TensorMetadata& input,
                            const TensorMetadata& output) {
  if (input.data_type != output.data_type) {
    throw std::runtime_error(
        std::string{group_label} + " binding '" + std::string{semantic} +
        "' has incompatible dtypes for input '" + input.name + "' and output '" +
        output.name + "'");
  }
  if (!ShapesCompatible(input.shape, output.shape)) {
    throw std::runtime_error(
        std::string{group_label} + " binding '" + std::string{semantic} +
        "' has incompatible shapes for input '" + input.name + "' " +
        ShapeString(input.shape) + " and output '" + output.name + "' " +
        ShapeString(output.shape));
  }
}

void ValidatePagedGeometry(std::string_view group_label,
                           const TensorMetadata& tensor,
                           std::optional<TensorMetadata>& reference) {
  if (tensor.shape.size() != 4) {
    throw std::runtime_error(
        std::string{group_label} + " paged binding '" + tensor.name +
        "' must have rank 4, got " + std::to_string(tensor.shape.size()));
  }
  if (!reference) {
    reference = tensor;
    return;
  }
  if (tensor.data_type != reference->data_type ||
      !ShapesCompatible(tensor.shape, reference->shape)) {
    throw std::runtime_error(
        std::string{group_label} + " has incompatible paged geometry between '" +
        reference->name + "' " + ShapeString(reference->shape) +
        " and '" + tensor.name + "' " + ShapeString(tensor.shape));
  }
}

void ValidatePagedTensorGeometry(
    const TensorMetadata& tensor,
    int layer_id,
    size_t expected_block_count,
    bool windowed,
    size_t block_size,
    size_t num_key_value_heads,
    size_t head_size) {
  if (tensor.shape.size() != 4) {
    throw std::runtime_error(
        "Paged cache tensor '" + tensor.name + "' for layer " +
        std::to_string(layer_id) +
        " must have rank 4 [num_blocks, block_size, num_key_value_heads, head_size]; got rank " +
        std::to_string(tensor.shape.size()) + " with shape " +
        ShapeString(tensor.shape) + ".");
  }

  const std::array<size_t, 4> expected{
      expected_block_count,
      block_size,
      num_key_value_heads,
      head_size,
  };
  const std::array<std::string_view, 4> axis_names{
      "num_blocks",
      "block_size",
      "num_key_value_heads",
      "head_size",
  };
  for (size_t axis = 0; axis < expected.size(); ++axis) {
    const int64_t actual = tensor.shape[axis];
    if (actual < 0 ||
        static_cast<uint64_t>(actual) == static_cast<uint64_t>(expected[axis])) {
      continue;
    }

    std::string source;
    if (axis == 0) {
      source = windowed
                   ? " for the windowed paged cache pool."
                   : " for the full-context paged cache pool.";
    } else if (axis == 1) {
      source = " from engine.dynamic_batching.block_size.";
    } else if (axis == 2) {
      source = " from model.decoder.num_key_value_heads.";
    } else {
      source = " from model.decoder.head_size.";
    }
    throw std::runtime_error(
        "Paged cache tensor '" + tensor.name + "' for layer " +
        std::to_string(layer_id) + " has incompatible axis " +
        std::to_string(axis) + " (" + std::string{axis_names[axis]} +
        "): graph dimension " + std::to_string(actual) + "; expected " +
        std::to_string(expected[axis]) + source);
  }
}

bool DimensionsCompatible(int64_t left, int64_t right) {
  return left < 0 || right < 0 || left == right;
}

void ValidateRank(std::string_view group_label,
                  std::string_view role,
                  const TensorMetadata& tensor,
                  size_t expected_rank) {
  if (tensor.shape.size() != expected_rank) {
    throw std::runtime_error(
        std::string{group_label} + " " + std::string{role} + " '" + tensor.name +
        "' must have rank " + std::to_string(expected_rank) + ", got " +
        std::to_string(tensor.shape.size()));
  }
}

void ValidateStateUpdateSession(std::string_view group_label,
                                const StateGroup& group,
                                const Decoder::Inputs& inputs,
                                const Decoder::Outputs& outputs,
                                const ModelStateMetadata& metadata) {
  if (!group.state_update) {
    return;
  }

  const auto& update = *group.state_update;
  const auto update_kind = StateUpdateKindFor(group.kind);
  const auto& capture_count_name = inputs.state_update_capture_count;
  if (!metadata.HasInput(capture_count_name)) {
    throw std::runtime_error(
        std::string{group_label} + " state_update capture_count input was not found: " +
        capture_count_name);
  }
  const TensorMetadata capture_count{
      metadata.GetInputDataType(capture_count_name),
      metadata.GetInputShape(capture_count_name),
      capture_count_name};
  if (capture_count.data_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    throw std::runtime_error(
        std::string{group_label} + " state_update capture_count input '" + capture_count_name +
        "' must have int32 dtype");
  }
  ValidateRank(group_label, "state_update capture_count input", capture_count, 1);

  const auto& active_name = inputs.state_update_active;
  if (!active_name.empty()) {
    if (!metadata.HasInput(active_name)) {
      throw std::runtime_error(
          std::string{group_label} + " state_update active input was not found: " + active_name);
    }
    const TensorMetadata active{
        metadata.GetInputDataType(active_name),
        metadata.GetInputShape(active_name),
        active_name};
    if (active.data_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      throw std::runtime_error(
          std::string{group_label} + " state_update active input '" + active_name +
          "' must have int32 dtype");
    }
    ValidateRank(group_label, "state_update active input", active, 1);
    if (active.shape[0] != 1) {
      throw std::runtime_error(
          std::string{group_label} + " state_update active input '" + active_name +
          "' must have shape [1]");
    }
  }

  const auto get_output = [&](std::string_view role,
                              const std::string& output_template,
                              int layer_id) {
    const auto output_name = ExpandBinding(output_template, layer_id);
    if (!metadata.HasOutput(output_name)) {
      throw std::runtime_error(
          std::string{group_label} + " state_update " + std::string{role} +
          " output was not found: " + output_name);
    }
    return TensorMetadata{
        metadata.GetOutputDataType(output_name),
        metadata.GetOutputShape(output_name),
        output_name};
  };

  const auto state_binding = StateBindingsFor(inputs, outputs, group.kind).front();
  for (const int layer_id : group.layer_ids) {
    const auto state_name = ExpandBinding(state_binding.output, layer_id);
    const TensorMetadata state{
        metadata.GetOutputDataType(state_name),
        metadata.GetOutputShape(state_name),
        state_name};
    ValidateRank(
        group_label, "state output", state,
        update_kind == StateUpdateKind::CausalConv ? 3 : 4);
    if (!DimensionsCompatible(state.shape[0], capture_count.shape[0])) {
      throw std::runtime_error(
          std::string{group_label} + " state_update capture_count input '" + capture_count_name +
          "' has batch dimension incompatible with state output '" + state.name + "'");
    }

    if (update_kind == StateUpdateKind::CausalConv) {
      const auto value = get_output("value", outputs.state_update_conv_value_names, layer_id);
      ValidateRank(group_label, "state_update value output", value, 3);
      if (value.data_type != state.data_type) {
        throw std::runtime_error(
            std::string{group_label} + " state_update value output '" + value.name +
            "' must have the same dtype as state output '" + state.name + "'");
      }
      if (!DimensionsCompatible(state.shape[0], value.shape[0]) ||
          value.shape[1] != update.capacity ||
          !DimensionsCompatible(state.shape[1], value.shape[2])) {
        throw std::runtime_error(
            std::string{group_label} + " state_update value output '" + value.name +
            "' has incompatible batch, capacity, or channel dimensions");
      }
    } else {
      const auto capsule = get_output(
          "capsule", outputs.state_update_recurrent_capsule_names, layer_id);
      ValidateRank(group_label, "state_update capsule output", capsule, 2);
      if (state.data_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          capsule.data_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error(
            std::string{group_label} +
            " gated_delta_net state and state_update capsule must have float dtype");
      }
      if (!DimensionsCompatible(state.shape[0], capsule.shape[0])) {
        throw std::runtime_error(
            std::string{group_label} + " state_update capsule output '" + capsule.name +
            "' has batch dimension incompatible with state output '" + state.name + "'");
      }
      const int64_t expected_width = static_cast<int64_t>(update.capacity) *
                                     (state.shape[1] +
                                      static_cast<int64_t>(update.key_head_count) * state.shape[3] +
                                      state.shape[1] * state.shape[2]);
      if (capsule.shape[1] != expected_width) {
        throw std::runtime_error(
            std::string{group_label} + " state_update capsule output '" + capsule.name +
            "' width must be " + std::to_string(expected_width));
      }
    }
  }
}

}  // namespace

ModelStateManifest::ModelStateManifest(const Decoder& decoder)
    : state_groups_{decoder.state_groups.value_or(std::vector<StateGroup>{})},
      inputs_{decoder.inputs},
      outputs_{decoder.outputs} {
  ValidateConfig(decoder);
}

bool ModelStateManifest::HasStateGroupKind(StateGroupKind kind) const {
  return std::any_of(
      state_groups_.begin(), state_groups_.end(),
      [kind](const StateGroup& group) {
        return group.kind == kind;
      });
}

bool ModelStateManifest::HasFixedStateGroups() const {
  return HasStateGroupKind(StateGroupKind::FixedConv) ||
         HasStateGroupKind(StateGroupKind::FixedRecurrent);
}

void ModelStateManifest::ValidateConfig(const Decoder& decoder) {
  if (!decoder.state_groups) {
    return;
  }
  if (decoder.state_groups->empty()) {
    throw std::runtime_error("model.decoder.state_groups must not be empty");
  }

  std::set<int> paged_layers;
  std::set<std::string> expanded_bindings;
  std::optional<int> fixed_state_update_capacity;
  std::optional<bool> fixed_state_update_enabled;
  size_t fixed_group_count{};
  size_t fixed_state_update_group_count{};

  for (size_t group_index = 0; group_index < decoder.state_groups->size(); ++group_index) {
    const auto& group = (*decoder.state_groups)[group_index];
    const auto group_label = StateGroupLabel(group_index, group.kind);
    if (group.kind == StateGroupKind::Invalid) {
      throw std::runtime_error(group_label + " is missing kind");
    }
    if (group.layer_ids.empty()) {
      throw std::runtime_error(group_label + " must contain at least one layer_id");
    }

    if (IsFixedKind(group.kind)) {
      ++fixed_group_count;
    }

    if (group.state_update) {
      ++fixed_state_update_group_count;
      const auto& update = *group.state_update;
      if (!IsFixedKind(group.kind)) {
        throw std::runtime_error(group_label + " only fixed state groups support state_update");
      }
      if (update.capacity < 1 || update.capacity > Decoder::MaxStateUpdateCapacity) {
        throw std::runtime_error(
            group_label + " state_update capacity must be in [1, " +
            std::to_string(Decoder::MaxStateUpdateCapacity) + "]");
      }
      if (decoder.inputs.state_update_capture_count.empty()) {
        throw std::runtime_error(group_label + " state_update is missing capture_count");
      }
      if (decoder.inputs.state_update_capture_count.find('%') != std::string::npos) {
        throw std::runtime_error(
            group_label + " state_update capture_count must be a graph input name, not a template");
      }
      if (decoder.inputs.state_update_active.find('%') != std::string::npos) {
        throw std::runtime_error(
            group_label + " state_update active must be a graph input name, not a template");
      }
      if (group.kind == StateGroupKind::FixedConv && update.key_head_count != 0) {
        throw std::runtime_error(group_label + " fixed_conv state_update does not use key_head_count");
      }
      if (group.kind == StateGroupKind::FixedRecurrent && update.key_head_count <= 0) {
        throw std::runtime_error(
            group_label + " fixed_recurrent state_update requires key_head_count");
      }

      if (!fixed_state_update_capacity) {
        fixed_state_update_capacity = update.capacity;
      } else if (*fixed_state_update_capacity != update.capacity) {
        throw std::runtime_error(
            "All fixed state_update groups must use the same capacity");
      }
      if (!fixed_state_update_enabled) {
        fixed_state_update_enabled = update.enabled;
      } else if (*fixed_state_update_enabled != update.enabled) {
        throw std::runtime_error(
            "All fixed state_update groups must use the same enabled setting");
      }
    }

    std::set<int> layer_ids;
    for (const int layer_id : group.layer_ids) {
      if (layer_id < 0 || layer_id >= decoder.num_hidden_layers) {
        throw std::runtime_error(
            group_label + " has layer_id " +
            std::to_string(layer_id) + " outside [0, num_hidden_layers)");
      }
      if (!layer_ids.insert(layer_id).second) {
        throw std::runtime_error(
            group_label + " contains duplicate layer_id " +
            std::to_string(layer_id));
      }
      if (group.kind == StateGroupKind::PagedKeyValue &&
          !paged_layers.insert(layer_id).second) {
        throw std::runtime_error(
            group_label + " overlaps another paged_kv group at layer_id " +
            std::to_string(layer_id));
      }
    }

    const auto validate_binding = [&](const StateBinding& binding) {
      ValidateBindingTemplate(group_label, binding.semantic, "input", binding.input);
      ValidateBindingTemplate(group_label, binding.semantic, "output", binding.output);
      for (const int layer_id : group.layer_ids) {
        for (const auto& name : {
                 ExpandBinding(binding.input, layer_id),
                 ExpandBinding(binding.output, layer_id)}) {
          if (!expanded_bindings.insert(name).second) {
            throw std::runtime_error(
                group_label + " resolves more than one binding to '" + name + "'");
          }
        }
      }
    };

    for (const auto& binding : StateBindingsFor(decoder.inputs, decoder.outputs, group.kind)) {
      validate_binding(binding);
    }
    if (group.state_update) {
      const auto& output_template = group.kind == StateGroupKind::FixedConv
                                        ? decoder.outputs.state_update_conv_value_names
                                        : decoder.outputs.state_update_recurrent_capsule_names;
      ValidateBindingTemplate(group_label, "state_update", "output", output_template);
      for (const int layer_id : group.layer_ids) {
        const auto output_name = ExpandBinding(output_template, layer_id);
        if (!expanded_bindings.insert(output_name).second) {
          throw std::runtime_error(
              group_label + " resolves more than one binding to '" + output_name + "'");
        }
      }
    }
  }

  if (fixed_state_update_group_count != 0 &&
      fixed_state_update_group_count != fixed_group_count) {
    throw std::runtime_error(
        "All fixed state groups must declare state_update when any fixed group declares it");
  }
}

void ModelStateManifest::ValidateSession(const ModelStateMetadata& metadata) const {
  for (size_t group_index = 0; group_index < state_groups_.size(); ++group_index) {
    const auto& group = state_groups_[group_index];
    const auto group_label = StateGroupLabel(group_index, group.kind);
    std::optional<TensorMetadata> paged_reference;
    const auto validate_binding = [&](const StateBinding& binding) {
      for (const int layer_id : group.layer_ids) {
        const auto input_name = ExpandBinding(binding.input, layer_id);
        const auto output_name = ExpandBinding(binding.output, layer_id);
        if (!metadata.HasInput(input_name)) {
          throw std::runtime_error(
              group_label + " binding '" + std::string{binding.semantic} +
              "' input was not found: " + input_name);
        }
        if (!metadata.HasOutput(output_name)) {
          throw std::runtime_error(
              group_label + " binding '" + std::string{binding.semantic} +
              "' output was not found: " + output_name);
        }

        TensorMetadata input{
            metadata.GetInputDataType(input_name),
            metadata.GetInputShape(input_name),
            input_name};
        TensorMetadata output{
            metadata.GetOutputDataType(output_name),
            metadata.GetOutputShape(output_name),
            output_name};
        ValidateCompatiblePair(group_label, binding.semantic, input, output);

        if (group.kind == StateGroupKind::PagedKeyValue) {
          ValidatePagedGeometry(group_label, input, paged_reference);
          ValidatePagedGeometry(group_label, output, paged_reference);
        }
      }
    };

    for (const auto& binding : StateBindingsFor(inputs_, outputs_, group.kind)) {
      validate_binding(binding);
    }
    ValidateStateUpdateSession(group_label, group, inputs_, outputs_, metadata);
  }
}

void ModelStateManifest::ValidatePagedCacheGeometry(
    const Decoder& decoder,
    const StateGroup& paged_group,
    const ModelStateMetadata& metadata,
    size_t full_block_count,
    size_t window_block_count,
    const std::set<int>& windowed_layers,
    size_t block_size) {
  for (const auto& binding :
       StateBindingsFor(decoder.inputs, decoder.outputs,
                        StateGroupKind::PagedKeyValue)) {
    for (const int layer_id : paged_group.layer_ids) {
      const auto input_name = ExpandBinding(binding.input, layer_id);
      const auto output_name = ExpandBinding(binding.output, layer_id);
      if (!metadata.HasInput(input_name)) {
        throw std::runtime_error(
            "Paged cache tensor input was not found: " + input_name);
      }
      if (!metadata.HasOutput(output_name)) {
        throw std::runtime_error(
            "Paged cache tensor output was not found: " + output_name);
      }

      const bool windowed = windowed_layers.contains(layer_id);
      const size_t expected_block_count =
          windowed ? window_block_count : full_block_count;
      ValidatePagedTensorGeometry(
          TensorMetadata{
              metadata.GetInputDataType(input_name),
              metadata.GetInputShape(input_name),
              input_name},
          layer_id, expected_block_count, windowed, block_size,
          static_cast<size_t>(decoder.num_key_value_heads),
          static_cast<size_t>(decoder.head_size));
      ValidatePagedTensorGeometry(
          TensorMetadata{
              metadata.GetOutputDataType(output_name),
              metadata.GetOutputShape(output_name),
              output_name},
          layer_id, expected_block_count, windowed, block_size,
          static_cast<size_t>(decoder.num_key_value_heads),
          static_cast<size_t>(decoder.head_size));
    }
  }
}

void ModelStateManifest::ValidateDynamicEngineCompatibility(const Decoder& decoder) {
  ValidateConfig(decoder);

  if (decoder.num_hidden_layers <= 0) {
    throw std::runtime_error(
        "Dynamic batching requires model.decoder.num_hidden_layers to be greater than zero.");
  }
  if (decoder.num_key_value_heads <= 0) {
    throw std::runtime_error(
        "Dynamic batching requires model.decoder.num_key_value_heads to be greater than zero.");
  }
  if (decoder.head_size <= 0) {
    throw std::runtime_error(
        "Dynamic batching requires model.decoder.head_size to be greater than zero.");
  }

  if (!decoder.state_groups) {
    return;
  }

  size_t paged_group_count = 0;
  for (const auto& group : *decoder.state_groups) {
    if (group.kind == StateGroupKind::PagedKeyValue) {
      ++paged_group_count;
    }
  }
  if (paged_group_count != 1) {
    throw std::runtime_error(
        "Dynamic batching requires exactly one paged_kv decoder state group");
  }
}

}  // namespace Generators
