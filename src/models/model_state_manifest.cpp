// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model_state_manifest.h"

#include <algorithm>
#include <optional>
#include <set>
#include <sstream>
#include <utility>

namespace Generators {
namespace {

using Decoder = Config::Model::Decoder;
using StateBinding = Decoder::StateBinding;
using StateGroup = Decoder::StateGroup;
using StateGroupKind = Decoder::StateGroupKind;
using StateUpdateKind = Decoder::StateUpdateKind;

std::string_view StateGroupKindName(StateGroupKind kind) {
  switch (kind) {
    case StateGroupKind::PagedKeyValue:
      return "paged_kv";
    case StateGroupKind::Fixed:
      return "fixed";
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
                                const ModelStateMetadata& metadata) {
  if (!group.state_update) {
    return;
  }

  const auto& update = *group.state_update;
  if (!metadata.HasInput(update.capture_count)) {
    throw std::runtime_error(
        std::string{group_label} + " state_update capture_count input was not found: " +
        update.capture_count);
  }
  const TensorMetadata capture_count{
      metadata.GetInputDataType(update.capture_count),
      metadata.GetInputShape(update.capture_count),
      update.capture_count};
  if (capture_count.data_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    throw std::runtime_error(
        std::string{group_label} + " state_update capture_count input '" + update.capture_count +
        "' must have int32 dtype");
  }
  ValidateRank(group_label, "state_update capture_count input", capture_count, 1);

  if (!update.active.empty()) {
    if (!metadata.HasInput(update.active)) {
      throw std::runtime_error(
          std::string{group_label} + " state_update active input was not found: " + update.active);
    }
    const TensorMetadata active{
        metadata.GetInputDataType(update.active),
        metadata.GetInputShape(update.active),
        update.active};
    if (active.data_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      throw std::runtime_error(
          std::string{group_label} + " state_update active input '" + update.active +
          "' must have int32 dtype");
    }
    ValidateRank(group_label, "state_update active input", active, 1);
    if (active.shape[0] != 1) {
      throw std::runtime_error(
          std::string{group_label} + " state_update active input '" + update.active +
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

  for (const int layer_id : group.layer_ids) {
    const auto state_name = ExpandBinding(group.state->output, layer_id);
    const TensorMetadata state{
        metadata.GetOutputDataType(state_name),
        metadata.GetOutputShape(state_name),
        state_name};
    ValidateRank(
        group_label, "state output", state,
        update.kind == StateUpdateKind::CausalConv ? 3 : 4);
    if (!DimensionsCompatible(state.shape[0], capture_count.shape[0])) {
      throw std::runtime_error(
          std::string{group_label} + " state_update capture_count input '" + update.capture_count +
          "' has batch dimension incompatible with state output '" + state.name + "'");
    }

    if (update.kind == StateUpdateKind::CausalConv) {
      const auto value = get_output("value", update.value, layer_id);
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
      const auto capsule = get_output("capsule", update.capsule, layer_id);
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
    : state_groups_{decoder.state_groups.value_or(std::vector<StateGroup>{})} {
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
  return HasStateGroupKind(StateGroupKind::Fixed);
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
  std::optional<std::pair<std::pair<int, std::string>, std::string>> fixed_state_update_contract;
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

    if (group.kind == StateGroupKind::PagedKeyValue) {
      if (!group.key || !group.value || group.state) {
        throw std::runtime_error(group_label + " requires exactly key and value bindings");
      }
    } else if (!group.state || group.key || group.value) {
      throw std::runtime_error(group_label + " requires exactly one state binding");
    }
    if (group.kind == StateGroupKind::Fixed) {
      ++fixed_group_count;
    }

    if (group.state_update) {
      ++fixed_state_update_group_count;
      const auto& update = *group.state_update;
      if (group.kind != StateGroupKind::Fixed) {
        throw std::runtime_error(group_label + " only fixed state groups support state_update");
      }
      if (update.kind == StateUpdateKind::Invalid) {
        throw std::runtime_error(group_label + " state_update is missing kind");
      }
      if (update.capacity < 1 || update.capacity > 8) {
        throw std::runtime_error(group_label + " state_update capacity must be in [1, 8]");
      }
      if (update.capture_count.empty()) {
        throw std::runtime_error(group_label + " state_update is missing capture_count");
      }
      if (update.capture_count.find('%') != std::string::npos) {
        throw std::runtime_error(
            group_label + " state_update capture_count must be a graph input name, not a template");
      }
      if (update.active.find('%') != std::string::npos) {
        throw std::runtime_error(
            group_label + " state_update active must be a graph input name, not a template");
      }
      if (update.kind == StateUpdateKind::CausalConv) {
        if (update.value.empty() || !update.capsule.empty() || update.key_head_count != 0) {
          throw std::runtime_error(group_label + " causal_conv state_update requires only value");
        }
      } else if (!update.value.empty() || update.capsule.empty() || update.key_head_count <= 0) {
        throw std::runtime_error(
            group_label + " gated_delta_net state_update requires capsule and key_head_count");
      }

      const auto contract = std::pair{std::pair{update.capacity, update.capture_count}, update.active};
      if (!fixed_state_update_contract) {
        fixed_state_update_contract = contract;
      } else if (*fixed_state_update_contract != contract) {
        throw std::runtime_error(
            "All fixed state_update groups must use the same capacity and capture_count input, "
            "and the same active input");
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

    const auto validate_binding = [&](std::string_view semantic, const StateBinding& binding) {
      ValidateBindingTemplate(group_label, semantic, "input", binding.input);
      ValidateBindingTemplate(group_label, semantic, "output", binding.output);
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

    if (group.key) {
      validate_binding("key", *group.key);
    }
    if (group.value) {
      validate_binding("value", *group.value);
    }
    if (group.state) {
      validate_binding("state", *group.state);
    }
    if (group.state_update) {
      const auto& update = *group.state_update;
      const auto validate_update_output = [&](std::string_view semantic,
                                              const std::string& output_template) {
        if (output_template.empty()) {
          return;
        }
        ValidateBindingTemplate(group_label, semantic, "output", output_template);
        for (const int layer_id : group.layer_ids) {
          const auto output_name = ExpandBinding(output_template, layer_id);
          if (!expanded_bindings.insert(output_name).second) {
            throw std::runtime_error(
                group_label + " resolves more than one binding to '" + output_name + "'");
          }
        }
      };
      validate_update_output("state_update.value", update.value);
      validate_update_output("state_update.capsule", update.capsule);
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
    const auto validate_binding = [&](std::string_view semantic, const StateBinding& binding) {
      for (const int layer_id : group.layer_ids) {
        const auto input_name = ExpandBinding(binding.input, layer_id);
        const auto output_name = ExpandBinding(binding.output, layer_id);
        if (!metadata.HasInput(input_name)) {
          throw std::runtime_error(
            group_label + " binding '" + std::string{semantic} +
              "' input was not found: " + input_name);
        }
        if (!metadata.HasOutput(output_name)) {
          throw std::runtime_error(
            group_label + " binding '" + std::string{semantic} +
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
          ValidateCompatiblePair(group_label, semantic, input, output);

        if (group.kind == StateGroupKind::PagedKeyValue) {
          ValidatePagedGeometry(group_label, input, paged_reference);
          ValidatePagedGeometry(group_label, output, paged_reference);
        }
      }
    };

    if (group.key) {
      validate_binding("key", *group.key);
    }
    if (group.value) {
      validate_binding("value", *group.value);
    }
    if (group.state) {
      validate_binding("state", *group.state);
    }
    ValidateStateUpdateSession(group_label, group, metadata);
  }
}

void ModelStateManifest::ValidateDynamicEngineCompatibility(const Decoder& decoder) {
  ValidateConfig(decoder);

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
