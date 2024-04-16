#pragma once

// TODO (pavignol): Refactor

#define NOMINMAX
#include <wrl/client.h>
#include <wil/result.h>
#include <d3d12.h>

using Microsoft::WRL::ComPtr;

struct DmlObjects {
  ComPtr<ID3D12Device> d3d12Device;
  ComPtr<ID3D12CommandQueue> commandQueue;
  ComPtr<ID3D12CommandAllocator> commandAllocator;
  ComPtr<ID3D12GraphicsCommandList> commandList;
  ComPtr<ID3D12Resource> upload_buffer;
};

inline DmlObjects CreateDmlObjects() {
  D3D12_COMMAND_QUEUE_DESC commandQueueDescription =
      {
          D3D12_COMMAND_LIST_TYPE_DIRECT,
          0,
          D3D12_COMMAND_QUEUE_FLAG_NONE,
          0,
      };

  DmlObjects dmlObjects;

  THROW_IF_FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dmlObjects.d3d12Device)));
  THROW_IF_FAILED(dmlObjects.d3d12Device->CreateCommandQueue(&commandQueueDescription, IID_PPV_ARGS(&dmlObjects.commandQueue)));
  THROW_IF_FAILED(dmlObjects.d3d12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&dmlObjects.commandAllocator)));
  THROW_IF_FAILED(dmlObjects.d3d12Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, dmlObjects.commandAllocator.Get(), nullptr, IID_PPV_ARGS(&dmlObjects.commandList)));

  return dmlObjects;
}

