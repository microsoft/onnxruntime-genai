// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "webgpu_cast_kernel.h"

#include <iostream>
#include <stdexcept>

namespace Generators {
namespace WebGPU {

namespace {

const char* kInt32ToInt64Shader = R"(
struct Constants {
  element_count: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<i32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<i32>; // i64 as two i32s
@group(0) @binding(2) var<uniform> constants: Constants;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let index = global_id.x;
  if (index >= constants.element_count) {
    return;
  }
  
  let input_val = input_data[index];
  let output_index = index * 2u;
  
  // Convert int32 to int64 by sign extension
  output_data[output_index] = input_val;  // Low 32 bits
  output_data[output_index + 1u] = select(0i, -1i, input_val < 0);  // High 32 bits (sign extension)
}
)";

const char* kFloat16ToFloat32Shader = R"(
enable f16;

struct Constants {
  element_count: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f16>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> constants: Constants;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let index = global_id.x;
  if (index >= constants.element_count) {
    return;
  }
  
  // Convert f16 to f32 directly
  output_data[index] = f32(input_data[index]);
}
)";

}  // namespace

void CastKernel::CreateInt32ToInt64Pipeline(wgpu::Device device) {
  // Create shader module
  wgpu::ShaderModuleWGSLDescriptor wgsl_desc{};
  wgsl_desc.code = kInt32ToInt64Shader;

  wgpu::ShaderModuleDescriptor shader_desc{};
  shader_desc.nextInChain = &wgsl_desc;
  shader_desc.label = "Int32ToInt64 Cast Shader";

  auto shader_module = device.CreateShaderModule(&shader_desc);

  // Create compute pipeline
  wgpu::ComputePipelineDescriptor pipeline_desc{};
  pipeline_desc.compute.module = shader_module;
  pipeline_desc.compute.entryPoint = "main";
  pipeline_desc.label = "Int32ToInt64 Cast Pipeline";

  int32_to_int64_pipeline_ = device.CreateComputePipeline(&pipeline_desc);

  // Create constants buffer
  wgpu::BufferDescriptor constants_desc{};
  constants_desc.size = 16;  // Align to 16 bytes for uniform buffer
  constants_desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
  constants_desc.label = "Cast Constants Buffer";
  constants_buffer_ = device.CreateBuffer(&constants_desc);
}

void CastKernel::CreateFloat16ToFloat32Pipeline(wgpu::Device device) {
  // Create shader module
  wgpu::ShaderModuleWGSLDescriptor wgsl_desc{};
  wgsl_desc.code = kFloat16ToFloat32Shader;

  wgpu::ShaderModuleDescriptor shader_desc{};
  shader_desc.nextInChain = &wgsl_desc;
  shader_desc.label = "Float16ToFloat32 Cast Shader";

  auto shader_module = device.CreateShaderModule(&shader_desc);

  // Create compute pipeline
  wgpu::ComputePipelineDescriptor pipeline_desc{};
  pipeline_desc.compute.module = shader_module;
  pipeline_desc.compute.entryPoint = "main";
  pipeline_desc.label = "Float16ToFloat32 Cast Pipeline";

  float16_to_float32_pipeline_ = device.CreateComputePipeline(&pipeline_desc);
}

bool CastKernel::CastInt32ToInt64(wgpu::Device device, wgpu::Queue queue, void* input_data, void* output_data, size_t element_count) {
  try {
    // Ensure pipeline exists for this device
    if (!int32_to_int64_pipeline_ || int32_to_int64_pipeline_.Get() == nullptr) {
      CreateInt32ToInt64Pipeline(device);
    }
    WGPUBuffer input_raw = reinterpret_cast<WGPUBuffer>(input_data);
    WGPUBuffer output_raw = reinterpret_cast<WGPUBuffer>(output_data);
    wgpu::Buffer input_buffer(input_raw);
    wgpu::Buffer output_buffer(output_raw);

    // Update constants
    uint32_t constants_data = static_cast<uint32_t>(element_count);
    queue.WriteBuffer(constants_buffer_, 0, &constants_data, sizeof(constants_data));

    // Create bind group using the pipeline's bind group layout
    // Note: We don't cache this because the buffers change each call
    wgpu::BindGroup bind_group;
    {
      // Get bind group layout from the pipeline (required for default pipeline layout)
      auto bind_group_layout = int32_to_int64_pipeline_.GetBindGroupLayout(0);

      // Create bind group
      std::vector<wgpu::BindGroupEntry> bind_entries(3);

      bind_entries[0].binding = 0;
      bind_entries[0].buffer = input_buffer;
      bind_entries[0].size = wgpu::kWholeSize;

      bind_entries[1].binding = 1;
      bind_entries[1].buffer = output_buffer;
      bind_entries[1].size = wgpu::kWholeSize;

      bind_entries[2].binding = 2;
      bind_entries[2].buffer = constants_buffer_;
      bind_entries[2].size = sizeof(uint32_t);

      wgpu::BindGroupDescriptor bind_group_desc{};
      bind_group_desc.layout = bind_group_layout;
      bind_group_desc.entryCount = bind_entries.size();
      bind_group_desc.entries = bind_entries.data();
      bind_group = device.CreateBindGroup(&bind_group_desc);
    }

    // Dispatch compute
    auto encoder = device.CreateCommandEncoder();
    auto compute_pass = encoder.BeginComputePass();

    compute_pass.SetPipeline(int32_to_int64_pipeline_);
    compute_pass.SetBindGroup(0, bind_group);

    uint32_t workgroups = (static_cast<uint32_t>(element_count) + 255) / 256;
    compute_pass.DispatchWorkgroups(workgroups);
    compute_pass.End();

    auto command_buffer = encoder.Finish();
    queue.Submit(1, &command_buffer);

    return true;
  } catch (const std::exception& e) {
    std::cerr << "CastKernel::CastInt32ToInt64 error: " << e.what() << std::endl;
    return false;
  }
}

bool CastKernel::CastFloat16ToFloat32(wgpu::Device device, wgpu::Queue queue, void* input_data, void* output_data, size_t element_count) {
  try {
    // Ensure pipeline exists for this device
    if (!float16_to_float32_pipeline_ || float16_to_float32_pipeline_.Get() == nullptr) {
      CreateFloat16ToFloat32Pipeline(device);
    }
    WGPUBuffer input_raw = reinterpret_cast<WGPUBuffer>(input_data);
    WGPUBuffer output_raw = reinterpret_cast<WGPUBuffer>(output_data);
    wgpu::Buffer input_buffer(input_raw);
    wgpu::Buffer output_buffer(output_raw);

    // Update constants
    uint32_t constants_data = static_cast<uint32_t>(element_count);
    queue.WriteBuffer(constants_buffer_, 0, &constants_data, sizeof(constants_data));

    // Create bind group using the pipeline's bind group layout
    // Note: We don't cache this because the buffers change each call
    wgpu::BindGroup bind_group;
    {
      // Get bind group layout from the pipeline (required for default pipeline layout)
      auto bind_group_layout = float16_to_float32_pipeline_.GetBindGroupLayout(0);

      // Create bind group
      std::vector<wgpu::BindGroupEntry> bind_entries(3);

      bind_entries[0].binding = 0;
      bind_entries[0].buffer = input_buffer;
      bind_entries[0].size = wgpu::kWholeSize;

      bind_entries[1].binding = 1;
      bind_entries[1].buffer = output_buffer;
      bind_entries[1].size = wgpu::kWholeSize;

      bind_entries[2].binding = 2;
      bind_entries[2].buffer = constants_buffer_;
      bind_entries[2].size = sizeof(uint32_t);

      wgpu::BindGroupDescriptor bind_group_desc{};
      bind_group_desc.layout = bind_group_layout;
      bind_group_desc.entryCount = bind_entries.size();
      bind_group_desc.entries = bind_entries.data();
      bind_group = device.CreateBindGroup(&bind_group_desc);
    }

    // Dispatch compute
    auto encoder = device.CreateCommandEncoder();
    auto compute_pass = encoder.BeginComputePass();

    compute_pass.SetPipeline(float16_to_float32_pipeline_);
    compute_pass.SetBindGroup(0, bind_group);

    uint32_t workgroups = (static_cast<uint32_t>(element_count) + 255) / 256;
    compute_pass.DispatchWorkgroups(workgroups);
    compute_pass.End();

    auto command_buffer = encoder.Finish();
    queue.Submit(1, &command_buffer);

    return true;
  } catch (const std::exception& e) {
    std::cerr << "CastKernel::CastFloat16ToFloat32 error: " << e.what() << std::endl;
    return false;
  }
}

}  // namespace WebGPU
}  // namespace Generators
