// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "webgpu_update_mask_kernel.h"

#ifdef USE_WEBGPU
#include <dawn/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <dawn/webgpu.h>
#include <stdexcept>
#include <cassert>
#include <string>
#include <type_traits>
#include <cstring>

namespace {

// WGSL compute shader for updating attention masks (static mask handling only)
const char* kUpdateMaskShaderI64 = R"(
struct Constants {
    batch_beam_size: u32,
    new_kv_length: u32,
    total_length: u32,
    max_length: u32,
}

@group(0) @binding(0) var<storage, read_write> mask: array<vec2<u32>>;
@group(0) @binding(1) var<uniform> constants: Constants;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_elements = constants.batch_beam_size * constants.new_kv_length;
    
    if (index >= total_elements) {
        return;
    }
    
    let batch_id = index / constants.new_kv_length;
    let seq_id = (index % constants.new_kv_length) + 1u;
    
    // Update mask_data[batch_id * max_length + total_length - seq_id] = 1
    let mask_index = batch_id * constants.max_length + constants.total_length - seq_id;
    mask[mask_index] = vec2<u32>(1u, 0u);
}
)";

const char* kUpdateMaskShaderI32 = R"(
struct Constants {
    batch_beam_size: u32,
    new_kv_length: u32,
    total_length: u32,
    max_length: u32,
}

@group(0) @binding(0) var<storage, read_write> mask: array<i32>;
@group(0) @binding(1) var<uniform> constants: Constants;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_elements = constants.batch_beam_size * constants.new_kv_length;
    
    if (index >= total_elements) {
        return;
    }
    
    let batch_id = index / constants.new_kv_length;
    let seq_id = (index % constants.new_kv_length) + 1u;
    
    // Update mask_data[batch_id * max_length + total_length - seq_id] = 1
    let mask_index = batch_id * constants.max_length + constants.total_length - seq_id;
    mask[mask_index] = i32(1);
}
)";

std::string GetShaderForType(bool is_int64) {
  if (is_int64) {
    return kUpdateMaskShaderI64;
  }
  return kUpdateMaskShaderI32;
}

}  // namespace

template <typename T>
void WebGPUUpdateMaskKernel<T>::InitializePipeline(wgpu::Device device) {
  if (initialized_) {
    return;
  }

  std::string shader_source = GetShaderForType(std::is_same_v<T, int64_t>);

  wgpu::ShaderModuleWGSLDescriptor wgsl_desc;
  wgsl_desc.code = shader_source.c_str();

  wgpu::ShaderModuleDescriptor shader_desc;
  shader_desc.nextInChain = &wgsl_desc;
  shader_desc.label = "UpdateMaskShader";

  wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

  // Create bind group layout for static shader: mask buffer (read_write) + constants buffer (uniform)
  wgpu::BindGroupLayoutEntry bind_group_layout_entries[2];

  // mask buffer (read_write)
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
void WebGPUUpdateMaskKernel<T>::UpdateMask(
    wgpu::Device device,
    wgpu::Queue queue,
    T* next_mask_data,
    T* mask_data,
    int batch_beam_size,
    int new_kv_length,
    int total_length,
    int max_length,
    bool update_only) {
  // Only handle static mask updates (update_only = true)
  if (!update_only) {
    throw std::runtime_error("Dynamic mask handling not supported in WebGPU implementation");
  }

  // Initialize the pipeline and cached resources
  InitializePipeline(device);

  // Create constants and upload to cached buffer
  Constants constants;
  constants.batch_beam_size = static_cast<uint32_t>(batch_beam_size);
  constants.new_kv_length = static_cast<uint32_t>(new_kv_length);
  constants.total_length = static_cast<uint32_t>(total_length);
  constants.max_length = static_cast<uint32_t>(max_length);

  // Upload constants to cached buffer
  queue.WriteBuffer(constants_buffer_, 0, &constants, sizeof(Constants));

  // Create cached bind group if not already created
  if (!bind_group_initialized_) {
    // Get the existing mask buffer (already on GPU)
    // Create a wgpu::Buffer from the existing WGPUBuffer handle without taking ownership
    WGPUBuffer raw_mask_buffer = reinterpret_cast<WGPUBuffer>(mask_data);
    wgpu::Buffer mask_buffer = wgpu::Buffer(raw_mask_buffer);

    // Create bind group for this specific mask buffer and cache it
    wgpu::BindGroupEntry bind_group_entries[2];
    bind_group_entries[0].binding = 0;
    bind_group_entries[0].buffer = mask_buffer;
    bind_group_entries[0].size = mask_buffer.GetSize();

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

  uint32_t workgroup_count = (batch_beam_size * new_kv_length + 255) / 256;

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
std::string WebGPUUpdateMaskKernel<T>::GetShaderSource() {
  return GetShaderForType(std::is_same_v<T, int64_t>);
}

// Explicit template instantiations
template class WebGPUUpdateMaskKernel<int32_t>;
template class WebGPUUpdateMaskKernel<int64_t>;

#endif  // USE_WEBGPU
