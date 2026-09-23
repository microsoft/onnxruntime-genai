// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "decoder.h"
#include "varlen_decoder_io.h"
#include "../../models/decoder_only.h"

namespace Generators {

struct SimpleDecoder : public Decoder {
  SimpleDecoder(std::shared_ptr<DecoderOnly_Model> model, std::shared_ptr<CacheManager> cache_manager);
  ~SimpleDecoder() override;

  void Decode(ScheduledRequests& scheduled_requests,
              ExecutionContext& context) override;

 private:
  // Why a step ran eagerly, for the graph_capture log. Steady state is one value forever, so only a
  // change is worth reporting.
  enum class GraphFallback {
    kCaptured,
    kDisabled,
    kNonUniformStep,
    kStepTooWide,
    kUncapturableShape,
  };

  // Reports a newly captured shape or a change of fallback reason. Steady state repeats neither, so
  // a healthy run logs nothing after warmup.
  void LogGraphDecision(size_t batch_size, size_t tokens_per_request,
                        const ExecutionContext& context, int annotation_id,
                        GraphFallback fallback, size_t known_shapes);

  std::shared_ptr<DecoderOnly_Model> model_;
  std::shared_ptr<CacheManager> cache_manager_;
  bool has_fixed_state_groups_{};
  size_t position_planes_{};
  GraphFallback last_fallback_{GraphFallback::kCaptured};
  // Allocated only when the model asked for CUDA graphs. Owning it here, rather than in the per-step
  // decoder IO, is what gives the captured graph stable buffer addresses to replay against.
  std::unique_ptr<VarlenGraphBuffers> graph_buffers_;
  CpuEmbedding::Workspace embedding_workspace_;
};

}  // namespace Generators
