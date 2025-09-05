#include "webgpu_update_mask_kernel.h"

#ifdef USE_WEBGPU

#include <stdexcept>
#include <cassert>
#include <string>
#include <type_traits>
#include <cstring>
#include <dawn/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <dawn/webgpu.h>  // For C API types like WGPUBufferMapAsyncStatus

namespace {

// WGSL compute shader for updating attention masks
const char* kUpdateMaskShaderTemplate = R"(
struct Constants {
    batch_beam_size: u32,
    new_kv_length: u32,
    total_length: u32,
    max_length: u32,
    update_only: u32,
    padding: array<u32, 3>,
}

@group(0) @binding(0) var<storage, read_write> next_mask: array<DATA_TYPE>;
@group(0) @binding(1) var<storage, read> mask: array<DATA_TYPE>;
@group(0) @binding(2) var<uniform> constants: Constants;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_elements = constants.batch_beam_size * constants.max_length;
    
    if (index >= total_elements) {
        return;
    }
    
    let batch_index = index / constants.max_length;
    let seq_index = index % constants.max_length;
    
    if (constants.update_only == 1u) {
        // Static mask update: set ones for new tokens
        if (seq_index < constants.total_length) {
            next_mask[index] = DATA_TYPE(1);
        } else {
            next_mask[index] = DATA_TYPE(0);
        }
    } else {
        // Dynamic mask update: copy existing mask and extend
        if (seq_index < (constants.total_length - constants.new_kv_length)) {
            next_mask[index] = mask[batch_index * constants.max_length + seq_index];
        } else if (seq_index < constants.total_length) {
            next_mask[index] = DATA_TYPE(1);
        } else {
            next_mask[index] = DATA_TYPE(0);
        }
    }
}
)";

std::string GetShaderForType(bool is_int64) {
  std::string shader = kUpdateMaskShaderTemplate;
  // WebGPU doesn't universally support i64, use i32 for both
  std::string data_type = "i32";
  size_t pos = 0;
  while ((pos = shader.find("DATA_TYPE", pos)) != std::string::npos) {
    shader.replace(pos, 9, data_type);
    pos += data_type.length();
  }
  return shader;
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

  // Create bind group layout
  wgpu::BindGroupLayoutEntry bind_group_layout_entries[3];

  // next_mask buffer (read_write)
  bind_group_layout_entries[0].binding = 0;
  bind_group_layout_entries[0].visibility = wgpu::ShaderStage::Compute;
  bind_group_layout_entries[0].buffer.type = wgpu::BufferBindingType::Storage;
  bind_group_layout_entries[0].buffer.hasDynamicOffset = false;

  // mask buffer (read)
  bind_group_layout_entries[1].binding = 1;
  bind_group_layout_entries[1].visibility = wgpu::ShaderStage::Compute;
  bind_group_layout_entries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  bind_group_layout_entries[1].buffer.hasDynamicOffset = false;

  // constants buffer (uniform)
  bind_group_layout_entries[2].binding = 2;
  bind_group_layout_entries[2].visibility = wgpu::ShaderStage::Compute;
  bind_group_layout_entries[2].buffer.type = wgpu::BufferBindingType::Uniform;
  bind_group_layout_entries[2].buffer.hasDynamicOffset = false;

  wgpu::BindGroupLayoutDescriptor bind_group_layout_desc;
  bind_group_layout_desc.entryCount = 3;
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
  InitializePipeline(device);

  // Create constants
  Constants constants;
  constants.batch_beam_size = static_cast<uint32_t>(batch_beam_size);
  constants.new_kv_length = static_cast<uint32_t>(new_kv_length);
  constants.total_length = static_cast<uint32_t>(total_length);
  constants.max_length = static_cast<uint32_t>(max_length);
  constants.update_only = update_only ? 1u : 0u;

  // Calculate buffer sizes
  size_t mask_size = batch_beam_size * max_length * sizeof(T);

  // Create buffers - Note: In real implementation, these buffers should be
  // obtained from ONNX Runtime's WebGPU provider, not created here
  wgpu::BufferDescriptor next_mask_buffer_desc;
  next_mask_buffer_desc.size = mask_size;
  next_mask_buffer_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
  wgpu::Buffer next_mask_buffer = device.CreateBuffer(&next_mask_buffer_desc);

  wgpu::BufferDescriptor mask_buffer_desc;
  mask_buffer_desc.size = mask_size;
  mask_buffer_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
  wgpu::Buffer mask_buffer = device.CreateBuffer(&mask_buffer_desc);

  wgpu::BufferDescriptor constants_buffer_desc;
  constants_buffer_desc.size = sizeof(Constants);
  constants_buffer_desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
  wgpu::Buffer constants_buffer = device.CreateBuffer(&constants_buffer_desc);

  // Upload data to buffers
  queue.WriteBuffer(mask_buffer, 0, mask_data, mask_size);
  queue.WriteBuffer(constants_buffer, 0, &constants, sizeof(Constants));

  // Create bind group
  wgpu::BindGroupEntry bind_group_entries[3];
  bind_group_entries[0].binding = 0;
  bind_group_entries[0].buffer = next_mask_buffer;
  bind_group_entries[0].size = mask_size;

  bind_group_entries[1].binding = 1;
  bind_group_entries[1].buffer = mask_buffer;
  bind_group_entries[1].size = mask_size;

  bind_group_entries[2].binding = 2;
  bind_group_entries[2].buffer = constants_buffer;
  bind_group_entries[2].size = sizeof(Constants);

  wgpu::BindGroupDescriptor bind_group_desc;
  bind_group_desc.layout = pipeline_.GetBindGroupLayout(0);
  bind_group_desc.entryCount = 3;
  bind_group_desc.entries = bind_group_entries;

  wgpu::BindGroup bind_group = device.CreateBindGroup(&bind_group_desc);

  // Create command encoder and dispatch
  wgpu::CommandEncoderDescriptor encoder_desc;
  wgpu::CommandEncoder encoder = device.CreateCommandEncoder(&encoder_desc);

  wgpu::ComputePassDescriptor compute_pass_desc;
  wgpu::ComputePassEncoder compute_pass = encoder.BeginComputePass(&compute_pass_desc);

  compute_pass.SetPipeline(pipeline_);
  compute_pass.SetBindGroup(0, bind_group);

  uint32_t workgroup_count = (batch_beam_size * max_length + 255) / 256;
  compute_pass.DispatchWorkgroups(workgroup_count);

  compute_pass.End();

  // Copy result back to host memory
  wgpu::BufferDescriptor staging_buffer_desc;
  staging_buffer_desc.size = mask_size;
  staging_buffer_desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  wgpu::Buffer staging_buffer = device.CreateBuffer(&staging_buffer_desc);

  encoder.CopyBufferToBuffer(next_mask_buffer, 0, staging_buffer, 0, mask_size);

  wgpu::CommandBuffer command_buffer = encoder.Finish();
  queue.Submit(1, &command_buffer);

  // Map and read result - Note: This is synchronous and blocking
  // In a real implementation, this should be asynchronous
  // For now, we'll use a simpler synchronous approach
  auto future = staging_buffer.MapAsync(wgpu::MapMode::Read, 0, mask_size, wgpu::CallbackMode::WaitAnyOnly,
                                        [](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                                          // Callback for async mapping
                                        });

  // Wait for the mapping to complete
  wgpu::Instance instance = wgpu::CreateInstance();
  instance.WaitAny(future, UINT64_MAX);

  const T* mapped_data = static_cast<const T*>(staging_buffer.GetConstMappedRange());
  std::memcpy(next_mask_data, mapped_data, mask_size);
  staging_buffer.Unmap();
}

template <typename T>
std::string WebGPUUpdateMaskKernel<T>::GetShaderSource() {
  return GetShaderForType(std::is_same_v<T, int64_t>);
}

// Explicit template instantiations
template class WebGPUUpdateMaskKernel<int32_t>;
template class WebGPUUpdateMaskKernel<int64_t>;

#endif  // USE_WEBGPU
