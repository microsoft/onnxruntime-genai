// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
namespace Generators {

void DumpTensor(const Model& model, std::ostream& stream, OrtValue* value, bool dump_value);
void DumpTensors(const Model& model, std::ostream& stream, OrtValue** values, const char** names, size_t count, bool dump_values);

// Outputs the values and (if enabled) the statistics of the values in the given values
void DumpValues(std::ostream& stream, ONNXTensorElementDataType type, const void* values, size_t count);

// Only outputs the values given
template <typename T>
void DumpSpan(std::ostream& stream, std::span<const T> values);
template <typename T>
void DumpSpan(std::ostream& stream, std::span<T> values) { DumpSpan(stream, std::span<const T>{values}); }

}  // namespace Generators