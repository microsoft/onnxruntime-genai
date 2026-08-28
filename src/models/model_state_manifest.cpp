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
using StateGroup = Decoder::StateGroup;
using StateGroupKind = Decoder::StateGroupKind;

struct StateBinding {
  std::string_view semantic;
  const std::string& input;
  const std::string& output;
};

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

  for (size_t group_index = 0; group_index < decoder.state_groups->size(); ++group_index) {
    const auto& group = (*decoder.state_groups)[group_index];
    const auto group_label = StateGroupLabel(group_index, group.kind);
    if (group.kind == StateGroupKind::Invalid) {
      throw std::runtime_error(group_label + " is missing kind");
    }
    if (group.layer_ids.empty()) {
      throw std::runtime_error(group_label + " must contain at least one layer_id");
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
