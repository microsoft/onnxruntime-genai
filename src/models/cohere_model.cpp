// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "cohere_model.h"

#include <algorithm>
#include <cctype>
#include <map>

namespace Generators {

// --- CohereEncoderState ---

CohereEncoderState::CohereEncoderState(const WhisperModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {}

// Update num_frames_ and (re)create hidden_states_ output for the current
// audio_features_ tensor. Returns the input slot index of the audio features.
void CohereEncoderState::UpdateForCurrentAudio() {
  auto shape_info = audio_features_->GetTensorTypeAndShapeInfo();
  auto shape = shape_info->GetShape();
  int audio_stride = model_.config_->model.encoder.audio_stride;

  if (audio_stride > 0 && shape.size() == 2) {
    // Raw audio input [batch, samples]
    int T_enc = static_cast<int>(shape[1]) / audio_stride + 1;
    num_frames_ = T_enc * 2;  // *2 so GetNumFrames()/2 == T_enc for CrossCache
  } else if (audio_stride > 0 && shape.size() == 3) {
    // Mel input [batch, n_mels, T_mel]
    int T_mel = static_cast<int>(shape[2]);
    int T_enc = (T_mel - 1) / 8 + 1;
    num_frames_ = T_enc * 2;
  } else {
    throw std::runtime_error("Cohere Transcribe requires audio_stride > 0 in config. "
                             "Got audio_stride=" + std::to_string(audio_stride) +
                             " with input rank=" + std::to_string(shape.size()));
  }

  if (model_.session_info_.HasOutput(model_.config_->model.encoder.outputs.hidden_states)) {
    auto hidden_states_shape = std::array<int64_t, 3>{params_->BatchBeamSize(), GetNumFrames() / 2, model_.config_->model.encoder.hidden_size};
    hidden_states_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), hidden_states_shape,
                                            shape_info->GetElementType());
  }
}

void CohereEncoderState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  const std::string& audio_name = model_.config_->model.encoder.inputs.audio_features;

  // Add audio features (CPU -> encoder device).
  for (const auto& [name, value] : extra_inputs) {
    if (name == audio_name) {
      audio_features_ = model_.ExpandInputs(value->ort_tensor_, params_->search.num_beams);
      break;
    }
  }
  if (audio_features_ == nullptr) {
    throw std::runtime_error("audio_features must be provided via SetInputs");
  }

  input_names_.push_back(audio_name.c_str());
  inputs_.push_back(audio_features_.get());

  UpdateForCurrentAudio();

  // Add mel_length input if provided. Route through ExpandInputs so it ends
  // up on the same device as the encoder (CUDA copies CPU->device here).
  for (const auto& [name, value] : extra_inputs) {
    if (name == "mel_length") {
      mel_length_ = model_.ExpandInputs(value->ort_tensor_, params_->search.num_beams);
      input_names_.push_back("mel_length");
      inputs_.push_back(mel_length_.get());
      break;
    }
  }

  // Add encoder hidden states output if the model has it
  if (hidden_states_) {
    outputs_.push_back(hidden_states_.get());
    output_names_.push_back(model_.config_->model.encoder.outputs.hidden_states.c_str());
  }
}

void CohereEncoderState::SetChunkAudioFeatures(std::shared_ptr<Tensor> audio_features_tensor, std::shared_ptr<Tensor> mel_length_tensor) {
  // Replace audio features input. Copy CPU -> encoder device via ExpandInputs.
  audio_features_ = model_.ExpandInputs(audio_features_tensor->ort_tensor_, params_->search.num_beams);

  UpdateForCurrentAudio();

  const std::string& audio_name = model_.config_->model.encoder.inputs.audio_features;
  for (size_t i = 0; i < input_names_.size(); ++i) {
    if (std::string(input_names_[i]) == audio_name) {
      inputs_[i] = audio_features_.get();
      break;
    }
  }

  // Replace mel_length (CPU -> device)
  if (mel_length_tensor) {
    mel_length_ = model_.ExpandInputs(mel_length_tensor->ort_tensor_, params_->search.num_beams);
    for (size_t i = 0; i < input_names_.size(); ++i) {
      if (std::string(input_names_[i]) == "mel_length") {
        inputs_[i] = mel_length_.get();
        break;
      }
    }
  }

  // Update hidden states output pointer if applicable
  if (hidden_states_) {
    for (size_t i = 0; i < output_names_.size(); ++i) {
      if (std::string(output_names_[i]) == model_.config_->model.encoder.outputs.hidden_states) {
        outputs_[i] = hidden_states_.get();
        break;
      }
    }
  }

  // Mark encoder as needing to run again
  first_run_ = true;
}

DeviceSpan<float> CohereEncoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.encoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.encoder.run_options.value());
  }
  State::Run(*model_.session_encoder_);
  return {};
}

// --- CohereState ---

CohereState::CohereState(const WhisperModel& model, const GeneratorParams& params, DeviceSpan<int32_t> sequence_lengths)
    : State{params, model},
      model_{model} {
  encoder_state_ = std::make_unique<CohereEncoderState>(model, params);
  // decoder_state_ and cross_cache_ created in SetExtraInputs after num_frames is known
}

void CohereState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  // Extract and store chunk data before passing to encoder
  std::vector<ExtraInput> encoder_inputs;
  std::map<int, std::shared_ptr<Tensor>> indexed_mels;
  std::map<int, std::shared_ptr<Tensor>> indexed_mel_lengths;

  for (const auto& [name, tensor] : extra_inputs) {
    if (name == "cohere_chunk_count") {
      total_chunks_ = static_cast<int>(tensor->ort_tensor_->GetTensorData<int64_t>()[0]);
      current_chunk_ = 0;
    } else if (name.substr(0, 13) == "cohere_chunk_" && name.find("mel_length") == std::string::npos) {
      // Parse index from cohere_chunk_N
      int idx = std::stoi(name.substr(13));
      indexed_mels[idx] = tensor;
    } else if (name.substr(0, 24) == "cohere_chunk_mel_length_") {
      int idx = std::stoi(name.substr(24));
      indexed_mel_lengths[idx] = tensor;
    } else {
      encoder_inputs.push_back({name, tensor});
    }
  }

  // Store chunks in sorted order
  for (auto& [idx, tensor] : indexed_mels)
    chunk_mels_.push_back(tensor);
  for (auto& [idx, tensor] : indexed_mel_lengths)
    chunk_mel_lengths_.push_back(tensor);

  encoder_state_->SetExtraInputs(encoder_inputs);

  // Now that num_frames is known, create the decoder
  decoder_state_ = std::make_unique<WhisperDecoderState>(model_, *params_, encoder_state_->GetNumFrames());

  // Create cross cache
  if (encoder_state_->HasCrossKVCacheOutputs()) {
    cross_cache_ = std::make_unique<CrossCache>(*this, encoder_state_->GetNumFrames() / 2);
    encoder_state_->AddCrossCache(cross_cache_);
    decoder_state_->AddCrossCache(cross_cache_);
    transpose_k_cache_buffer_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), cross_cache_->GetShape(), cross_cache_->GetType());
  }

  if (!encoder_state_->HasCrossKVCacheOutputs()) {
    decoder_state_->inputs_.push_back(encoder_state_->GetHiddenStates());
    decoder_state_->input_names_.push_back(model_.config_->model.decoder.inputs.encoder_hidden_states.c_str());
  }
}

bool CohereState::AdvanceToNextChunk() {
  if (!HasMoreChunks()) return false;

  size_t chunk_idx = current_chunk_;
  auto& next_mel = chunk_mels_[chunk_idx];
  auto& next_mel_length = chunk_mel_lengths_[chunk_idx];

  encoder_state_->SetChunkAudioFeatures(next_mel, next_mel_length);

  int new_num_frames = encoder_state_->GetNumFrames();

  encoder_state_->outputs_.clear();
  encoder_state_->output_names_.clear();

  decoder_state_ = std::make_unique<WhisperDecoderState>(model_, *params_, new_num_frames);

  if (encoder_state_->HasCrossKVCacheOutputs()) {
    cross_cache_ = std::make_unique<CrossCache>(*this, new_num_frames / 2);
    encoder_state_->AddCrossCache(cross_cache_);
    decoder_state_->AddCrossCache(cross_cache_);
    transpose_k_cache_buffer_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), cross_cache_->GetShape(), cross_cache_->GetType());
  }

  if (!encoder_state_->HasCrossKVCacheOutputs()) {
    decoder_state_->inputs_.push_back(encoder_state_->GetHiddenStates());
    decoder_state_->input_names_.push_back(model_.config_->model.decoder.inputs.encoder_hidden_states.c_str());
  }

  // Reset state for new chunk
  first_run_ = true;
  current_chunk_++;
  return true;
}

void CohereState::SaveChunkTokens(const int32_t* tokens, size_t count) {
  completed_chunk_tokens_.emplace_back(tokens, tokens + count);
}

static std::string StripAsciiWhitespace(const std::string& s) {
  // Mirrors Python str.strip(): strips ASCII whitespace (space, \t, \n, \r, \v, \f).
  static const char* ws = " \t\n\r\v\f";
  const size_t start = s.find_first_not_of(ws);
  if (start == std::string::npos) return {};
  const size_t end = s.find_last_not_of(ws);
  return s.substr(start, end - start + 1);
}

// --- Helpers for overlap-aware chunk text merging --------------------------
//
// The processor emits OVERLAPPING audio chunks; adjacent transcripts share
// the same words at their seam. We dedup by finding the longest run of
// matching normalized words between the tail of `prev` and the head of `next`
// and dropping it from `next`. We also strip stray sentence-end punctuation
// from the prev tail when the continuation starts with a lowercase word, so
// "...UK." + "come from..." becomes "...UK come from..." not "...UK. come...".

static std::string ToLowerAscii(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) out.push_back(static_cast<char>(std::tolower(c)));
  return out;
}

static std::string NormalizeWord(const std::string& w) {
  // Lowercase and strip leading/trailing non-alphanumeric codepoints for matching.
  // Must handle non-ASCII punctuation like Spanish `¿`, `¡`, quotes `«»`, etc.,
  // so e.g. "¿de" matches "de" across a chunk seam.
  auto is_word_byte = [](unsigned char c) {
    // ASCII alphanumeric, or any UTF-8 continuation/start byte (>= 0x80)
    // — letters with diacritics like é, ó, ñ live in the 0x80+ range and
    // should be kept; only ASCII punctuation/symbols are stripped.
    return std::isalnum(c) != 0 || c >= 0x80;
  };
  // Strip leading non-word ASCII chars.
  size_t a = 0;
  while (a < w.size() && !is_word_byte(static_cast<unsigned char>(w[a]))) ++a;
  // Strip trailing non-word ASCII chars.
  size_t b = w.size();
  while (b > a && !is_word_byte(static_cast<unsigned char>(w[b - 1]))) --b;
  if (a >= b) return {};
  // Now strip non-ASCII punctuation codepoints (¿ ¡ « » “ ” ‘ ’ — …) at the ends.
  // These all start with bytes 0xC2/0xC3/0xE2 in UTF-8; we explicitly list the
  // sequences we care about for matching.
  static const char* kStripPrefixes[] = {
      "\xC2\xBF",  // ¿
      "\xC2\xA1",  // ¡
      "\xC2\xAB",  // «
      "\xE2\x80\x9C",  // “
      "\xE2\x80\x98",  // ‘
  };
  static const char* kStripSuffixes[] = {
      "\xC2\xBB",  // »
      "\xE2\x80\x9D",  // ”
      "\xE2\x80\x99",  // ’
      "\xE2\x80\xA6",  // …
      "\xE2\x80\x94",  // —
      "\xE2\x80\x93",  // –
  };
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto* p : kStripPrefixes) {
      size_t plen = std::strlen(p);
      if (b - a >= plen && std::memcmp(w.data() + a, p, plen) == 0) {
        a += plen;
        changed = true;
        break;
      }
    }
    for (auto* s : kStripSuffixes) {
      size_t slen = std::strlen(s);
      if (b - a >= slen && std::memcmp(w.data() + b - slen, s, slen) == 0) {
        b -= slen;
        changed = true;
        break;
      }
    }
  }
  if (a >= b) return {};
  return ToLowerAscii(w.substr(a, b - a));
}

static std::vector<std::string> SplitOnWhitespace(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  const size_t n = s.size();
  while (i < n) {
    while (i < n && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    size_t j = i;
    while (j < n && !std::isspace(static_cast<unsigned char>(s[j]))) ++j;
    if (j > i) out.emplace_back(s.substr(i, j - i));
    i = j;
  }
  return out;
}

static bool StartsWithLowerAscii(const std::string& w) {
  // First non-punct alpha char check.
  for (unsigned char c : w) {
    if (std::isalpha(c)) return std::islower(c) != 0;
    if (std::isalnum(c)) return false;
  }
  return false;
}

static std::string RStripSentencePunct(std::string w) {
  while (!w.empty()) {
    char c = w.back();
    if (c == '.' || c == '!' || c == '?') {
      w.pop_back();
    } else {
      break;
    }
  }
  return w;
}

// Append `next` onto `prev` with overlap dedup. Returns merged text.
static std::string MergeWithOverlapDedup(const std::string& prev, const std::string& next,
                                         size_t min_match = 3, size_t max_lookback = 25) {
  if (prev.empty()) return next;
  if (next.empty()) return prev;
  auto pw = SplitOnWhitespace(prev);
  auto nw = SplitOnWhitespace(next);
  if (pw.empty()) return next;
  if (nw.empty()) return prev;

  std::vector<std::string> pn(pw.size()), nn(nw.size());
  for (size_t i = 0; i < pw.size(); ++i) pn[i] = NormalizeWord(pw[i]);
  for (size_t i = 0; i < nw.size(); ++i) nn[i] = NormalizeWord(nw[i]);

  size_t best_k = 0;
  size_t upper = std::min({max_lookback, pw.size(), nw.size()});
  for (size_t k = upper; k >= min_match; --k) {
    bool ok = true;
    for (size_t i = 0; i < k; ++i) {
      if (pn[pn.size() - k + i] != nn[i]) { ok = false; break; }
    }
    if (ok) { best_k = k; break; }
    if (k == 0) break;
  }
  if (best_k == 0 && !pn.back().empty() && pn.back() == nn.front()) {
    best_k = 1;
  }

  // Strip terminal sentence punctuation from prev's last word if the
  // continuation begins mid-sentence (lowercase).
  if (best_k < nw.size() && StartsWithLowerAscii(nw[best_k])) {
    pw.back() = RStripSentencePunct(pw.back());
  }

  std::string out;
  for (size_t i = 0; i < pw.size(); ++i) {
    if (i) out += ' ';
    out += pw[i];
  }
  for (size_t i = best_k; i < nw.size(); ++i) {
    out += ' ';
    out += nw[i];
  }
  return out;
}

std::string CohereState::GetJoinedChunkText(const Tokenizer& tokenizer, const std::string& separator) const {
  // Decode each chunk, strip whitespace, then merge with overlap-aware word
  // dedup at the seams (chunks are emitted with audio overlap by the processor).
  // `separator` is honored only as the inter-chunk fallback when no overlap is
  // detected (e.g. zh/ja/ko/vi want "" — but those are not handled here yet).
  (void)separator;
  std::string merged;
  for (const auto& chunk : completed_chunk_tokens_) {
    if (chunk.empty()) continue;
    auto raw = tokenizer.Decode(std::span<const int32_t>(chunk.data(), chunk.size()));
    auto stripped = StripAsciiWhitespace(raw);
    if (stripped.empty()) continue;
    if (merged.empty()) {
      merged = std::move(stripped);
    } else {
      merged = MergeWithOverlapDedup(merged, stripped);
    }
  }
  return merged;
}

// Tail-words count to hold back for potential revision by the next chunk.
// Must be >= MergeWithOverlapDedup's max_lookback so any future overlap match
// can still be discovered against pending_text_.
static constexpr size_t kPendingTailWords = 25;

void CohereState::CommitChunkText(const std::string& chunk_text, const std::vector<int32_t>& chunk_tokens,
                                  bool is_final, const Tokenizer& tokenizer) {
  std::string stripped = StripAsciiWhitespace(chunk_text);

  if (pending_text_.empty() && pending_tokens_.empty()) {
    // First chunk or no overlap — just use chunk tokens directly.
    if (is_final) {
      // Only chunk: commit everything, no roundtrip.
      for (auto t : chunk_tokens) committed_tokens_.push_back(t);
      return;
    }
    // Not final: hold tail words back for dedup with next chunk.
    auto words = SplitOnWhitespace(stripped);
    if (words.size() <= kPendingTailWords) {
      pending_text_ = stripped;
      pending_tokens_ = chunk_tokens;
      return;
    }
    // Commit the first N words by finding token boundary via decode length matching.
    size_t commit_words = words.size() - kPendingTailWords;
    // Build commit text and pending text
    std::string commit_text;
    for (size_t i = 0; i < commit_words; ++i) {
      if (i) commit_text += ' ';
      commit_text += words[i];
    }
    // Find how many tokens produce the commit_text prefix.
    // Decode tokens one by one and count characters to find the split point.
    size_t split_tok = chunk_tokens.size();
    for (size_t t = 1; t <= chunk_tokens.size(); ++t) {
      auto partial = tokenizer.Decode(std::span<const int32_t>(chunk_tokens.data(), t));
      auto partial_stripped = StripAsciiWhitespace(partial);
      auto partial_words = SplitOnWhitespace(partial_stripped);
      if (partial_words.size() >= commit_words) {
        // Check the first commit_words match
        bool match = true;
        for (size_t w = 0; w < commit_words && match; ++w) {
          if (partial_words[w] != words[w]) match = false;
        }
        if (match) {
          // If partial has exactly commit_words, this token completes the commit portion
          // but we want to include tokens that contribute to commit words only.
          // Find the minimal t where we have commit_words complete words.
          split_tok = t;
          break;
        }
      }
    }
    for (size_t i = 0; i < split_tok; ++i)
      committed_tokens_.push_back(chunk_tokens[i]);
    pending_tokens_.assign(chunk_tokens.begin() + split_tok, chunk_tokens.end());
    std::string ptext;
    for (size_t i = commit_words; i < words.size(); ++i) {
      if (i > commit_words) ptext += ' ';
      ptext += words[i];
    }
    pending_text_ = std::move(ptext);
    return;
  }

  // We have pending text/tokens from previous chunk. Merge via text-level dedup.
  std::string merged;
  if (stripped.empty()) {
    merged = pending_text_;
  } else {
    merged = MergeWithOverlapDedup(pending_text_, stripped);
  }

  // The pending_tokens_ cover pending_text_. The chunk_tokens cover stripped.
  // After dedup, some words from the head of stripped are dropped (overlap).
  // We need to figure out how many tokens from chunk_tokens to skip.
  auto merged_words = SplitOnWhitespace(merged);
  auto pending_words = SplitOnWhitespace(pending_text_);
  auto chunk_words = SplitOnWhitespace(stripped);

  // Find how many words from chunk were consumed by the overlap dedup.
  // merged = pending_words + chunk_words[overlap_skip:]
  // So: merged.size() = pending_words.size() + chunk_words.size() - overlap_skip
  size_t overlap_skip = 0;
  if (!chunk_words.empty() && merged_words.size() < pending_words.size() + chunk_words.size()) {
    overlap_skip = pending_words.size() + chunk_words.size() - merged_words.size();
  }

  // Find token index in chunk_tokens that corresponds to skipping overlap_skip words.
  size_t tok_skip = 0;
  if (overlap_skip > 0 && overlap_skip < chunk_words.size()) {
    for (size_t t = 1; t <= chunk_tokens.size(); ++t) {
      auto partial = tokenizer.Decode(std::span<const int32_t>(chunk_tokens.data(), t));
      auto partial_words = SplitOnWhitespace(StripAsciiWhitespace(partial));
      if (partial_words.size() > overlap_skip) {
        tok_skip = t - 1;
        // The t-1'th token still produced the last overlap word; t starts the new content
        // But actually we need the first token AFTER the overlap words are complete
        // Re-check: if partial_words.size() > overlap_skip, the word at index overlap_skip
        // has started, which means tokens 0..t-1 include it. We want to start from the
        // token that first introduces word at overlap_skip.
        // Let's find the exact boundary:
        for (size_t t2 = t; t2 >= 1; --t2) {
          auto p2 = tokenizer.Decode(std::span<const int32_t>(chunk_tokens.data(), t2 - 1));
          auto pw2 = SplitOnWhitespace(StripAsciiWhitespace(p2));
          if (pw2.size() <= overlap_skip) {
            tok_skip = t2 - 1;
            break;
          }
        }
        break;
      }
    }
    if (overlap_skip >= chunk_words.size()) {
      tok_skip = chunk_tokens.size();  // all chunk words were overlap
    }
  } else if (overlap_skip >= chunk_words.size()) {
    tok_skip = chunk_tokens.size();
  }

  // Now: commit pending_tokens_ + chunk_tokens[tok_skip:], but hold back tail if not final.
  std::vector<int32_t> all_new_tokens;
  all_new_tokens.insert(all_new_tokens.end(), pending_tokens_.begin(), pending_tokens_.end());
  all_new_tokens.insert(all_new_tokens.end(), chunk_tokens.begin() + tok_skip, chunk_tokens.end());

  if (is_final) {
    // Commit everything.
    for (auto t : all_new_tokens) committed_tokens_.push_back(t);
    pending_tokens_.clear();
    pending_text_.clear();
    return;
  }

  // Hold back tail words.
  // Decode all_new_tokens to get the full merged text and find the split.
  auto full_text = tokenizer.Decode(std::span<const int32_t>(all_new_tokens.data(), all_new_tokens.size()));
  auto full_words = SplitOnWhitespace(StripAsciiWhitespace(full_text));
  if (full_words.size() <= kPendingTailWords) {
    pending_tokens_ = std::move(all_new_tokens);
    std::string ptext;
    for (size_t i = 0; i < full_words.size(); ++i) {
      if (i) ptext += ' ';
      ptext += full_words[i];
    }
    pending_text_ = std::move(ptext);
    return;
  }

  size_t commit_words = full_words.size() - kPendingTailWords;
  // Find token split point.
  size_t split_tok = all_new_tokens.size();
  for (size_t t = 1; t <= all_new_tokens.size(); ++t) {
    auto partial = tokenizer.Decode(std::span<const int32_t>(all_new_tokens.data(), t));
    auto partial_words = SplitOnWhitespace(StripAsciiWhitespace(partial));
    if (partial_words.size() >= commit_words) {
      split_tok = t;
      break;
    }
  }
  for (size_t i = 0; i < split_tok; ++i)
    committed_tokens_.push_back(all_new_tokens[i]);
  pending_tokens_.assign(all_new_tokens.begin() + split_tok, all_new_tokens.end());
  std::string ptext;
  for (size_t i = commit_words; i < full_words.size(); ++i) {
    if (i > commit_words) ptext += ' ';
    ptext += full_words[i];
  }
  pending_text_ = std::move(ptext);
}

DeviceSpan<int32_t> CohereState::GetCommittedSpan() const {
  size_t n = std::min(streamed_count_, committed_tokens_.size());
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  return cpu->WrapMemory<int32_t>(
      std::span<int32_t>(const_cast<int32_t*>(committed_tokens_.data()), n));
}

DeviceSpan<float> CohereState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (encoder_state_->IsFirstRun()) {
    encoder_state_->Run(current_length, next_tokens, next_indices);
    decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length, first_run_);
    return decoder_state_->Run(current_length, next_tokens, next_indices);
  }

  decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length, first_run_);
  auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);
  first_run_ = false;
  return logits;
}

OrtValue* CohereState::GetInput(const char* name) {
  for (size_t i = 0; i < encoder_state_->input_names_.size(); i++) {
    if (std::strcmp(encoder_state_->input_names_[i], name) == 0)
      return encoder_state_->inputs_[i];
  }
  for (size_t i = 0; i < decoder_state_->input_names_.size(); i++) {
    if (std::strcmp(decoder_state_->input_names_[i], name) == 0)
      return decoder_state_->inputs_[i];
  }
  return State::GetInput(name);
}

OrtValue* CohereState::GetOutput(const char* name) {
  for (size_t i = 0; i < encoder_state_->output_names_.size(); i++) {
    if (std::strcmp(encoder_state_->output_names_[i], name) == 0)
      return encoder_state_->outputs_[i];
  }
  for (size_t i = 0; i < decoder_state_->output_names_.size(); i++) {
    if (std::strcmp(decoder_state_->output_names_[i], name) == 0)
      return decoder_state_->outputs_[i];
  }
  return State::GetOutput(name);
}

// --- CohereModel ---

std::unique_ptr<State> CohereModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<CohereState>(*this, params, sequence_lengths);
}

}  // namespace Generators
