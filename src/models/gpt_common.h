#pragma once

namespace Generators {

struct GptModelParams {
  int vocab_size{};
  int head_count{};
  int hidden_size{};
  int layer_count{};
  bool logits_uses_seq_len{};  // Logits shape is [... seq_len, vocab_size ] vs [... 1, vocab_size ]
};

void GetModelParams(GptModelParams& params, OrtSession& session);

}
