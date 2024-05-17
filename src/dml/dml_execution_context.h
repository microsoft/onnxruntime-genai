// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <d3d12.h>
#include <d3dx12.h>
#include "dml_command_recorder.h"
#include "dml_gpu_event.h"
#include "../models/onnxruntime_api.h"

// Asynchronously performs GPU work, and automatically manages command list recording and submission to queues.
// Work submitted to the DmlExecutionContext is typically recorded onto a command list and may not immediately begin
// execution on the GPU. Call Flush() to force all recorded work to be submitted to the command queue for execution
// on the GPU.
class DmlExecutionContext {
 public:
  // Constructs an DmlExecutionContext that executes on the supplied queue.
  DmlExecutionContext(
      ID3D12Device* d3d12_device,
      IDMLDevice* dml_device,
      ID3D12CommandQueue* queue,
      Ort::Allocator& device_allocator,
      const OrtDmlApi* ort_dml_api);

  // Waits for flushed work, discards unflushed work, and discards associated references to
  // prevent circular references.  Must be the last call on the object before destruction.
  void Close();

  // Queues a CopyBufferRegion (see ID3D12GraphicsCommandList::CopyBufferRegion) for execution. Transition
  // barriers are automatically inserted to transition the source and destination resources to COPY_SOURCE and
  // COPY_DEST if necessary.
  void CopyBufferRegion(
      ID3D12Resource* dst_buffer,
      uint64_t dst_offset,
      D3D12_RESOURCE_STATES dst_state,
      ID3D12Resource* src_buffer,
      uint64_t src_offset,
      D3D12_RESOURCE_STATES src_state,
      uint64_t byte_count);

  void InitializeOperator(
      IDMLCompiledOperator* op,
      const DML_BINDING_DESC& persistent_resource_binding,
      const DML_BINDING_DESC& input_array_binding);

  void ExecuteCommandList(
      ID3D12GraphicsCommandList* command_list,
      _Outptr_ ID3D12Fence** fence,
      _Out_ uint64_t* completion_value);

  void AddUAVBarrier();
  void ResourceBarrier(std::span<const D3D12_RESOURCE_BARRIER> barriers);

  void GetCommandListForRecordingAndInvalidateState(ID3D12GraphicsCommandList** command_list);

  // Forces all queued work to begin executing on the GPU. This method returns immediately and does not wait
  // for the submitted work to complete execution on the GPU.
  void Flush();

  // Returns an event which will become signaled when everything submitted to the execution context thus far has
  // completed execution on the GPU, including work that has yet to be flushed to the queue.
  DmlGpuEvent GetCurrentCompletionEvent();

  // Adds a reference which will be released when queued GPU work is completed
  void QueueReference(IUnknown* object);

  // Release any accumulated references who corresponding GPU fence values have
  // been reached.
  void ReleaseCompletedReferences();

  D3D12_COMMAND_LIST_TYPE GetCommandListTypeForQueue() const;

 private:
  void SetCommandRecorder(DmlCommandRecorder* new_recorder);

  std::shared_ptr<DmlCommandQueue> queue_;

  DmlCommandRecorder* current_recorder_ = nullptr;

  // Up to one of these is active at a time
  DmlCommandRecorder dml_recorder_;

  bool closed_ = false;
};