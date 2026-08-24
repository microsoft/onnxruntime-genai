// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "model.h"

namespace Generators {

// Manages recurrent state tensors (conv_state + recurrent_state) for hybrid models.
// Auto-discovers recurrent layers by probing session inputs.
struct RecurrentState {
  RecurrentState(State& state);

  void Add();
  void Update();
  void RewindTo(size_t index);

  // Snapshot/restore the recurrent (conv + recurrent) state buffers. Required for
  // speculative decoding (e.g. MTP): the recurrent state cannot be partially rewound
  // (unlike the attention KV cache), so a draft/verify step snapshots the state before
  // a speculative forward and restores it if the draft is rejected. Restore copies back
  // in place so buffer addresses stay stable (required by CUDA-graph replay).
  void Snapshot(size_t position);
  void RestoreSnapshot();

  // Per-position recurrent-state cropping (lossless multi-token MTP). When the model is exported
  // with `state_window=W`, the past/present conv + recurrent state tensors carry a
  // window axis at position 0: slot j holds the state AFTER token (seq_len - W + j) of the
  // forward, right-aligned, so slot W-1 is the state after the last token and is the only slot
  // the ops read back. On a partial-accept MTP step the controller crops the live state to the
  // accepted length by copying slot `position` into slot W-1 -- no full-cost main-model replay
  // forward, and no extra graph outputs to bind or keep alive across CUDA-graph capture.
  bool IsWindowed() const { return state_window_ > 1; }
  int64_t StateWindow() const { return state_window_; }
  void SetForwardLength(int sequence_length);  // Record this step's seq_len (maps position -> slot).
  void CropToPosition(size_t position);        // Copy window slot for `position` -> slot W-1.

  bool IsEmpty() const { return layer_indices_.empty(); }
  int GraphCaptureVariant() const { return graph_buffer_variant_; }

  // ORT captures a CUDA graph by re-running the model inside a single user-visible Run()
  // until the EP reports capture complete (InferenceSession::RunImpl recursion, driven by
  // min_num_runs_before_cuda_graph_capture_). Those extra runs re-feed identical inputs, so
  // idempotent in-place writes such as the KV-cache append are unaffected -- but the
  // recurrent state is an accumulator and gets advanced once per internal run. The fix is to
  // let the capture happen on a throwaway Run, restore the state, and replay once:
  //
  //   if (ShouldFixUpGraphCapture(id)) { SaveForGraphCapture(); Run(); RestoreAfterGraphCapture(id); }
  //   Run();
  //
  // Only needed when inputs alias outputs; the double-buffered path is immune because the
  // extra runs all read the same unchanged `past` buffer.
  bool ShouldFixUpGraphCapture(int graph_id) const;
  void SaveForGraphCapture();
  void RestoreAfterGraphCapture(int graph_id);

 private:
  void ZeroStates(std::vector<std::unique_ptr<OrtValue>>& states);
  void CopyStates(const std::vector<std::unique_ptr<OrtValue>>& src, std::vector<std::unique_ptr<OrtValue>>& dst);
  // Single-kernel version of the CropToPosition copy loop. False if the device has no implementation.
  bool TryBatchedSlotPromote(size_t slot);

  State& state_;
  const Model& model_{state_.model_};

  std::vector<int> layer_indices_;

  // Interleaved as [conv_0, recurrent_0, conv_1, recurrent_1, ...]
  std::vector<std::unique_ptr<OrtValue>> pasts_;
  std::vector<std::unique_ptr<OrtValue>> presents_;
  std::vector<std::unique_ptr<OrtValue>> snapshot_;  // Lazily-allocated copy of the live state for speculative rollback.
  bool snapshot_valid_{false};                       // Whether snapshot_ holds a valid captured state.
  size_t snapshot_position_{};                       // Sequence length represented by snapshot_.

  // Per-position state window. When the model declares a rank-4 conv_state / rank-5
  // recurrent_state, axis 1 is a static window of W per-token states (see IsWindowed above).
  // 1 means the legacy unwindowed layout (a single state, no window axis).
  int64_t state_window_{1};
  int forward_length_{0};  // seq_len of the last SetForwardLength(), needed to map position -> slot.

  // Device-resident {base, slot_bytes} descriptors for the batched CropToPosition kernel, plus the
  // host mirror used to detect a buffer reallocation.
  DeviceSpan<StateSlotDesc> slot_descs_;
  std::vector<StateSlotDesc> slot_descs_cpu_;

  // Mirrors past_present_share_buffer config: true means inputs alias outputs (same allocation,
  // stable handles for graph capture). False uses separate past/present buffers with per-step swap.
  bool share_buffers_{false};
  bool graph_double_buffer_{false};
  int graph_buffer_variant_{0};
  // Graph ids whose capture-time state corruption has already been undone.
  std::vector<int> graph_capture_fixed_up_;
  // Held only across the capture run. These wrap the live buffers and keep the backup in
  // their CPU mirrors, so undoing the capture costs no device memory.
  std::vector<DeviceSpan<uint8_t>> graph_capture_backup_;
  size_t input_index_{~0U};
  size_t output_index_{~0U};

  // Kept alive for state_ const char* pointers
  std::vector<std::string> input_name_strings_;
  std::vector<std::string> output_name_strings_;

  ONNXTensorElementDataType conv_type_{};
  ONNXTensorElementDataType recurrent_type_{};

  std::vector<int64_t> conv_shape_;       // [B, C, K-1], or [B, W, C, K-1] when windowed.
  std::vector<int64_t> recurrent_shape_;  // [B, H_kv, d_k, d_v], or [B, W, ...] when windowed.
};

// Factory: returns nullptr if no recurrent layers are found in the session.
std::unique_ptr<RecurrentState> CreateRecurrentState(State& state);

}  // namespace Generators
