#include "generators.h"
#include "search.h"
#include "search_dml.h"
#include <queue>
#include <random>
#include "models/dml/dml_helpers.h"

namespace Generators {

Search_Dml::Search_Dml(const GeneratorParams& params, ID3D12Device* d3d12_device)
    : Search(params),
      d3d12_device_(d3d12_device),
      sequences_{params.input_ids, params.batch_size, params.search.num_beams, params_->search.max_length} {
  auto batch_beam_size = params.BatchBeamSize();
  sequence_lengths_buffer_ = std::make_unique<int32_t[]>(batch_beam_size);
  sequence_lengths_ = cpu_span<int32_t>(sequence_lengths_buffer_.get(), batch_beam_size);

  auto resource_desc = CD3DX12_RESOURCE_DESC::Buffer(batch_beam_size * sizeof(int32_t), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  auto heap_props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
  THROW_IF_FAILED(d3d12_device_->CreateCommittedResource(
      &heap_props,
      D3D12_HEAP_FLAG_NONE,
      &resource_desc,
      D3D12_RESOURCE_STATE_COMMON,
      nullptr,
      IID_PPV_ARGS(eos_meet_buffer_.GetAddressOf())));
}

GreedySearch_Dml::GreedySearch_Dml(const GeneratorParams& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, DmlExecutionContext* execution_context, const OrtDmlApi* ort_dml_api)
    : Search_Dml(params, d3d12_device),
      dml_device_(dml_device),
      execution_context_(execution_context),
      ort_dml_api_(ort_dml_api) {
  auto resource_desc = CD3DX12_RESOURCE_DESC::Buffer(params.batch_size * sizeof(int32_t), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  auto heap_props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
  THROW_IF_FAILED(d3d12_device_->CreateCommittedResource(
      &heap_props,
      D3D12_HEAP_FLAG_NONE,
      &resource_desc,
      D3D12_RESOURCE_STATE_COMMON,
      nullptr,
      IID_PPV_ARGS(next_tokens_buffer_.GetAddressOf())));
}

void Search_Dml::SetLogits(RoamingArray<float> logits_unk) {
  next_token_scores_buffer_ = logits_unk.GetGPU();
}

RoamingArray<int32_t> GreedySearch_Dml::GetNextTokens() {
  return next_tokens_;
}

int Search_Dml::GetSequenceLength() const {
  return sequences_.GetSequenceLength();
}

void GreedySearch_Dml::SelectTop() {
  auto batch_beam_size = params.BatchBeamSize();
  auto vocab_size = params_->vocab_size;

  auto dml_data_type = DmlHelpers::OrtToDmlDataType(in.GetTensorTypeAndShapeInfo()->GetElementType());


  // If the sizes change, we need to recompile the operator and rebuild the command lists. It should only happen
  // once after the very first iteration.
  if (rebind) {
    auto compiled_cast_operator = DmlHelpers::CreateArgMaxOperator(dml_device, element_count, dml_from_type, dml_to_type);

    ComPtr<ID3D12Resource> persistent_resource;
    uint64_t persistent_resource_size = compiled_cast_operator->GetBindingProperties().PersistentResourceSize;

    std::optional<DML_BUFFER_BINDING> persistent_resource_binding;

    if (persistent_resource_size > 0) {
      std::array<int64_t, 1> persistent_resource_shape = {static_cast<int64_t>(persistent_resource_size)};
      auto persistent_tensor = OrtValue::CreateTensor(allocator, persistent_resource_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
      Ort::ThrowOnError(ort_dml_api_->GetD3D12ResourceFromAllocation(&allocator, persistent_tensor->GetTensorMutableRawData(), &persistent_resource));
      persistent_resource_binding = DML_BUFFER_BINDING{persistent_resource.Get(), 0, persistent_resource_size};
    }

    DML_BINDING_DESC persistent_resource_bindingDesc = persistent_resource_binding
                                                           ? DML_BINDING_DESC{DML_BINDING_TYPE_BUFFER, &*persistent_resource_binding}
                                                           : DML_BINDING_DESC{DML_BINDING_TYPE_NONE, nullptr};

    DML_BINDING_DESC input_array_binding_desc = DML_BINDING_DESC{DML_BINDING_TYPE_NONE, nullptr};
    execution_context_->InitializeOperator(compiled_cast_operator.Get(), persistent_resource_bindingDesc, input_array_binding_desc);
    command_list_state = DmlHelpers::BuildReusableCommandList(dml_device, compiled_cast_operator.Get(), persistent_resource.Get(), persistent_resource_binding);
    command_list_state.previousOutput = p_out.get();
  }

  std::array<ID3D12Resource*, 1> input_resources = {next_token_scores_buffer_.Get()};
  std::array<uint64_t, 1> input_sizes = {element_count * DmlHelpers::DataTypeSizeInBytes(dml_from_type)};

  std::array<ID3D12Resource*, 1> output_resources = {next_tokens_buffer_.Get()};
  std::array<uint64_t, 1> output_sizes = {element_count * DmlHelpers::DataTypeSizeInBytes(dml_to_type)};

  DmlHelpers::ExecuteReusableCommandList(execution_context_, command_list_state, allocator, ort_dml_api, input_resources, input_sizes, output_resources, output_sizes, rebind);

  CheckForEOS();
  AppendNextTokensToSequences();
}

void GreedySearch_Dml::SampleTopP(float p, float temperature) {
  THROW_HR(E_NOTIMPL);
}

void GreedySearch_Dml::SampleTopK(int k, float temperature) {
  THROW_HR(E_NOTIMPL);
}

void GreedySearch_Dml::SampleTopKTopP(int k, float p, float temperature) {
  THROW_HR(E_NOTIMPL);
}

void GreedySearch_Dml::CheckForEOS() {
  assert(next_tokens_.size() == eos_meet_.size());
  cuda::Launch_CheckForEOS(next_tokens_.data(), static_cast<int>(next_tokens_.size()), eos_meet_.data(), params_->eos_token_id, params_->pad_token_id, done_cpu_.get(), params_->cuda_stream);
}

void GreedySearch_Dml::AppendNextTokensToSequences() {
  sequences_.AppendNextTokenToSequences(next_tokens_);

  if (sequences_.GetSequenceLength() == params_->search.max_length) {
    *done_cpu_ = true;
  }
}

void Search_Dml::ApplyMinLength(int min_length) {
  if (sequences_.GetSequenceLength() >= min_length)
    return;

  THROW_HR(E_NOTIMPL);
}

void Search_Dml::ApplyRepetitionPenalty(float penalty) {
  if (penalty == 1.0f)
    return;

  THROW_HR(E_NOTIMPL);
}

}  // namespace Generators