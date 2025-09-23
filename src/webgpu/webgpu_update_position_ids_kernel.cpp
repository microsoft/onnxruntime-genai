// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "webgpu_update_position_ids_kernel.h"

#ifdef USE_WEBGPU
#include <dawn/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <dawn/webgpu.h>
#include <stdexcept>
#include <cassert>
#include <string>
#include <type_traits>

namespace {

// WGSL compute shader for updating position IDs (continuous decoding only)
const char* kUpdatePositionIdsShaderI64 = R"(
struct Constants {
    total_length: u32,
    new_kv_length: u32,
}

@group(0) @binding(0) var<storage, read_write> positions: array<vec2<u32>>;
@group(0) @binding(1) var<uniform> constants: Constants;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= constants.new_kv_length) {
        return;
    }
    
    // Calculate position IDs for continuous decoding: positions[i] = i + total_length - new_kv_length
    let pos_value = index + constants.total_length - constants.new_kv_length;
    positions[index] = vec2<u32>(pos_value, select(0u, 0xFFFFFFFFu, i32(pos_value) < 0));
}
)";

const char* kUpdatePositionIdsShaderI32 = R"(
struct Constants {
    total_length: u32,
    new_kv_length: u32,
}

@group(0) @binding(0) var<storage, read_write> positions: array<i32>;
@group(0) @binding(1) var<uniform> constants: Constants;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= constants.new_kv_length) {
        return;
    }
    
    // Calculate position IDs for continuous decoding: positions[i] = i + total_length - new_kv_length
    positions[index] = i32(index + constants.total_length - constants.new_kv_length);
}
)";

std::string GetShaderForType(bool is_int64) {
  if (is_int64) {
    return kUpdatePositionIdsShaderI64;
  }
  return kUpdatePositionIdsShaderI32;
}

}  // namespace

template <typename T>
void WebGPUUpdatePositionIdsKernel<T>::InitializePipeline(wgpu::Device device) {
  if (initialized_) {
    return;
  }

  std::string shader_source = GetShaderForType(std::is_same_v<T, int64_t>);

  wgpu::ShaderModuleWGSLDescriptor wgsl_desc;
  wgsl_desc.code = shader_source.c_str();

  wgpu::ShaderModuleDescriptor shader_desc;
  shader_desc.nextInChain = &wgsl_desc;
  shader_desc.label = "UpdatePositionIdsShader";

  wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

  // Create bind group layout: positions buffer (read_write) + constants buffer (uniform)
  wgpu::BindGroupLayoutEntry bind_group_layout_entries[2];

  // positions buffer (read_write)
  bind_group_layout_entries[0].binding = 0;
  bind_group_layout_entries[0].visibility = wgpu::ShaderStage::Compute;
  bind_group_layout_entries[0].buffer.type = wgpu::BufferBindingType::Storage;
  bind_group_layout_entries[0].buffer.hasDynamicOffset = false;

  // constants buffer (uniform)
  bind_group_layout_entries[1].binding = 1;
  bind_group_layout_entries[1].visibility = wgpu::ShaderStage::Compute;
  bind_group_layout_entries[1].buffer.type = wgpu::BufferBindingType::Uniform;
  bind_group_layout_entries[1].buffer.hasDynamicOffset = false;

  wgpu::BindGroupLayoutDescriptor bind_group_layout_desc;
  bind_group_layout_desc.entryCount = 2;
  bind_group_layout_desc.entries = bind_group_layout_entries;

  wgpu::BindGroupLayout bind_group_layout = device.CreateBindGroupLayout(&bind_group_layout_desc);

  // Create pipeline layout
  wgpu::PipelineLayoutDescriptor pipeline_layout_desc;
  pipeline_layout_desc.bindGroupLayoutCount = 1;
  pipeline_layout_desc.bindGroupLayouts = &bind_group_layout;

  wgpu::PipelineLayout pipeline_layout = device.CreatePipelineLayout(&pipeline_layout_desc);

  // Create compute pipeline
  wgpu::ComputePipelineDescriptor pipeline_desc;
  pipeline_desc.layout = pipeline_layout;
  pipeline_desc.compute.module = shader_module;
  pipeline_desc.compute.entryPoint = "main";

  pipeline_ = device.CreateComputePipeline(&pipeline_desc);

  // Create and cache the constants buffer
  wgpu::BufferDescriptor constants_buffer_desc;
  constants_buffer_desc.size = sizeof(Constants);
  constants_buffer_desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
  constants_buffer_ = device.CreateBuffer(&constants_buffer_desc);

  initialized_ = true;
}

template <typename T>
void WebGPUUpdatePositionIdsKernel<T>::UpdatePositionIds(
    wgpu::Device device,
    wgpu::Queue queue,
    T* position_ids,
    int batch_beam_size,
    int total_length,
    int new_kv_length) {
  // Only support batch_beam_size == 1 for graph capture (continuous decoding)
  if (batch_beam_size != 1) {
    throw std::runtime_error("WebGPU UpdatePositionIds only supports batch_beam_size == 1");
  }

  // Initialize the pipeline and cached resources
  InitializePipeline(device);

  // Create constants and upload to cached buffer
  Constants constants;
  constants.total_length = static_cast<uint32_t>(total_length);
  constants.new_kv_length = static_cast<uint32_t>(new_kv_length);

  // Upload constants to cached buffer
  queue.WriteBuffer(constants_buffer_, 0, &constants, sizeof(Constants));

  // Create cached bind group if not already created
  if (!bind_group_initialized_) {
    // Get the existing position IDs buffer (already on GPU)
    // Create a wgpu::Buffer from the existing WGPUBuffer handle without taking ownership
    WGPUBuffer raw_positions_buffer = reinterpret_cast<WGPUBuffer>(position_ids);
    wgpu::Buffer positions_buffer = wgpu::Buffer(raw_positions_buffer);

    // Create bind group for this specific positions buffer and cache it
    wgpu::BindGroupEntry bind_group_entries[2];
    bind_group_entries[0].binding = 0;
    bind_group_entries[0].buffer = positions_buffer;
    bind_group_entries[0].size = positions_buffer.GetSize();

    bind_group_entries[1].binding = 1;
    bind_group_entries[1].buffer = constants_buffer_;
    bind_group_entries[1].size = sizeof(Constants);

    wgpu::BindGroupDescriptor bind_group_desc;
    bind_group_desc.layout = pipeline_.GetBindGroupLayout(0);
    bind_group_desc.entryCount = 2;
    bind_group_desc.entries = bind_group_entries;

    bind_group_ = device.CreateBindGroup(&bind_group_desc);
    bind_group_initialized_ = true;
  }

  uint32_t workgroup_count = (new_kv_length + 255) / 256;

  // Create command encoder and dispatch
  wgpu::CommandEncoderDescriptor encoder_desc;
  wgpu::CommandEncoder encoder = device.CreateCommandEncoder(&encoder_desc);

  wgpu::ComputePassDescriptor compute_pass_desc;
  wgpu::ComputePassEncoder compute_pass = encoder.BeginComputePass(&compute_pass_desc);

  compute_pass.SetPipeline(pipeline_);
  compute_pass.SetBindGroup(0, bind_group_);
  compute_pass.DispatchWorkgroups(workgroup_count);
  compute_pass.End();

  wgpu::CommandBuffer command_buffer = encoder.Finish();
  queue.Submit(1, &command_buffer);
}

template <typename T>
std::string WebGPUUpdatePositionIdsKernel<T>::GetShaderSource() {
  return GetShaderForType(std::is_same_v<T, int64_t>);
}

// Explicit template instantiations
template class WebGPUUpdatePositionIdsKernel<int32_t>;
template class WebGPUUpdatePositionIdsKernel<int64_t>;

#endif  // USE_WEBGPU
