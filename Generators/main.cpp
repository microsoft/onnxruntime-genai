// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "Generators.h"
#include <iostream>
#include <array>
#include <vector>
#include "onnxruntime_cxx_api_2.h"

#if 0
template <typename T>
static void softmax(T& input) {
  float rowmax = *std::max_element(input.begin(), input.end());
  std::vector<float> y(input.size());
  float sum = 0.0f;
  for (size_t i = 0; i != input.size(); ++i) {
    sum += y[i] = std::exp(input[i] - rowmax);
  }
  for (size_t i = 0; i != input.size(); ++i) {
    input[i] = y[i] / sum;
  }
}

struct Generator {
  Generator() {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
    output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
  }

  std::ptrdiff_t Run() {
    const char* input_names[] = { "Input3" };
    const char* output_names[] = { "Plus214_Output_0" };

    session_.Run(Ort::RunOptions{nullptr}, input_names, & input_tensor_, 1, output_names, & output_tensor_, 1);
    softmax(results_);
    result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
    return result_;
  }

  static constexpr const int width_ = 28;
  static constexpr const int height_ = 28;

  std::array<float, width_* height_> input_image_{};
  std::array<float, 10> results_{};
  int64_t result_{ 0 };

private:
  Ort::Env env;
  Ort::Session session_{env, L"model.onnx", Ort::SessionOptions{nullptr}};

  Ort::Value input_tensor_{nullptr};
  std::array<int64_t, 4> input_shape_{1, 1, width_, height_};

  Ort::Value output_tensor_{nullptr};
  std::array<int64_t, 2> output_shape_{1, 10};
};
#endif

void Test_Lib_GreedySearchTest_GptGreedySearchFp32();
void Test_GreedySearchTest_GptGreedySearchFp32();

void Test_BeamSearchTest_GptBeamSearchFp32();
void Test_Lib_BeamSearchTest_GptBeamSearchFp32();

int main()
{
  Ort::InitApi();

	std::cout << "Generators Utility Library" << std::endl;
  try {
    Test_Lib_GreedySearchTest_GptGreedySearchFp32();
    Test_GreedySearchTest_GptGreedySearchFp32();

    Test_Lib_BeamSearchTest_GptBeamSearchFp32();
    Test_BeamSearchTest_GptBeamSearchFp32();
  }
  catch (const std::exception& e)
  {
    std::cout << "Fatal Exception:" << e.what();
  }
	return 0;
}
