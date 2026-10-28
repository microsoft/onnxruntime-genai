// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
size_t ElementBytes(int type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return 8;
    default:
      throw std::runtime_error("Unsupported calibration tensor type: " + std::to_string(type));
  }
}

void RunSample(Ort::Session& session, const std::string& request) {
  std::ifstream manifest(request);
  manifest.exceptions(std::ios::failbit | std::ios::badbit);
  std::string header;
  int version = 0;
  manifest >> header >> version;
  if (header != "qwen38_calibration_request" || version != 1)
    throw std::runtime_error("Unsupported calibration request: " + request);
  size_t count = 0;
  manifest >> count;
  if (count == 0 || count > 1000) throw std::runtime_error("Invalid calibration input count");
  std::vector<std::string> names(count);
  std::vector<std::vector<uint8_t>> buffers(count);
  std::vector<Ort::Value> values;
  std::vector<const char*> input_names;
  const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  for (size_t i = 0; i < count; ++i) {
    int type = 0;
    size_t rank = 0;
    manifest >> std::quoted(names[i]) >> type >> rank;
    if (rank > 8) throw std::runtime_error("Calibration input rank exceeds eight");
    std::vector<int64_t> shape(rank);
    size_t bytes = ElementBytes(type);
    for (auto& dim : shape) {
      manifest >> dim;
      if (dim <= 0 || static_cast<uint64_t>(dim) > std::numeric_limits<size_t>::max() / bytes)
        throw std::runtime_error("Invalid calibration input dimensions: " + names[i]);
      bytes *= static_cast<size_t>(dim);
    }
    std::string path;
    manifest >> std::quoted(path);
    if (std::filesystem::file_size(path) != bytes)
      throw std::runtime_error("Calibration input byte count mismatch: " + path);
    buffers[i].resize(bytes);
    std::ifstream input(path, std::ios::binary);
    input.exceptions(std::ios::failbit | std::ios::badbit);
    input.read(reinterpret_cast<char*>(buffers[i].data()), static_cast<std::streamsize>(bytes));
    input_names.push_back(names[i].c_str());
    values.emplace_back(Ort::Value::CreateTensor(
        memory, buffers[i].data(), bytes, shape.data(), shape.size(),
        static_cast<ONNXTensorElementDataType>(type)));
  }
  manifest >> count;
  if (count == 0 || count > 1000) throw std::runtime_error("Invalid calibration output count");
  std::vector<std::string> outputs(count), paths(count);
  std::vector<const char*> output_names;
  for (size_t i = 0; i < count; ++i) {
    manifest >> std::quoted(outputs[i]) >> std::quoted(paths[i]);
    output_names.push_back(outputs[i].c_str());
  }
  auto result = session.Run(Ort::RunOptions{nullptr}, input_names.data(), values.data(), values.size(),
                            output_names.data(), output_names.size());
  for (size_t i = 0; i < result.size(); ++i) {
    const auto info = result[i].GetTensorTypeAndShapeInfo();
    const size_t bytes = info.GetElementCount() * ElementBytes(info.GetElementType());
    std::ofstream output(paths[i], std::ios::binary | std::ios::trunc);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.write(static_cast<const char*>(result[i].GetTensorRawData()), static_cast<std::streamsize>(bytes));
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::runtime_error("Expected graph, CUDA provider library, request-list file");
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "qwen38_calibration");
    env.RegisterExecutionProviderLibrary("CUDAExecutionProvider", argv[2]);
    auto devices = env.GetEpDevices();
    std::vector<Ort::ConstEpDevice> selected;
    for (const auto& device : devices) {
      if (std::string(device.EpName()) == "CUDAExecutionProvider") {
        selected.push_back(device);
        break;
      }
    }
    if (selected.empty()) throw std::runtime_error("No CUDA device is available for target calibration");
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    options.DisableMemPattern();
    options.AddConfigEntry("session.use_device_allocator_for_initializers", "1");
    const std::unordered_map<std::string, std::string> provider_options{
        {"arena_extend_strategy", "kSameAsRequested"}};
    options.AppendExecutionProvider_V2(env, selected, provider_options);
    Ort::Session session(env, argv[1], options);
    std::ifstream requests(argv[3]);
    if (!requests) throw std::runtime_error("Cannot open calibration request list");
    requests.exceptions(std::ios::badbit);
    std::string request;
    size_t completed = 0;
    while (requests >> std::quoted(request)) {
      RunSample(session, request);
      std::cout << "Completed calibration sample " << ++completed << std::endl;
    }
    if (!requests.eof() || completed == 0) throw std::runtime_error("Invalid or empty calibration request list");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Target calibration failed: " << error.what() << std::endl;
    return 1;
  }
}
