# ONNX Runtime GenAI: Nemotron-Speech ASR Support

## Overview

This document describes the implemented support for **NVIDIA Nemotron-Speech-Streaming-En-0.6b** (FastConformer-CacheAware-RNNT, 600M parameters) in ONNX Runtime GenAI.

**Status: ✅ Phase 1 Complete — Basic (non-streaming) inference working end-to-end.**

## Architecture

### Model: `nemotron_asr`

```
NemotronModel (Model)
├── session_encoder_    // audio_signal[B,128,T] + length[B] → outputs[B,1024,T'] + encoded_lengths[B]
├── session_decoder_    // targets[B,L] + target_length_orig[B] → decoder_output[B,640,2] + LSTM states
├── session_joint_      // encoder_output[B,T',1024] + decoder_output[B,L,640] → joint_output[B,T',L,1025]
└── CreateState() → NemotronState

NemotronState (State)
├── RunEncoder()              // Runs encoder once, transposes output [B,1024,T'] → [B,T',1024]
├── RunDecoder(token_id)      // Runs prediction network for one token
├── RunJoint(enc, dec)        // Runs joint network, returns argmax token
├── GreedyDecode()            // Full RNNT greedy decode loop across all encoder frames
└── Run()                     // Emits decoded tokens one-by-one via logits for generator framework
```

### Model Comparison: Whisper vs Nemotron

| Aspect | Whisper | Nemotron-Speech |
|--------|---------|-----------------|
| Architecture | Encoder-Decoder (Transformer) | Encoder-Decoder (FastConformer + RNNT) |
| Encoder | Transformer encoder | FastConformer (24 layers, cache-aware) |
| Decoder | Transformer decoder (autoregressive) | RNNT prediction network (LSTM) + joint network |
| Decoding | Beam search on logits | RNNT greedy decode (per encoder frame) |
| Streaming | No (fixed 30s chunks) | Yes (configurable 80ms–1.12s chunks) — *not yet implemented* |
| Cross-attention | Yes | No (RNNT uses joint network) |
| Vocab size | Model-dependent | 1025 (1024 BPE + 1 blank) |
| Tokenizer | Whisper tokenizer | T5Tokenizer (Unigram/SentencePiece) |

## Implementation Details

### Files Created

| File | Lines | Purpose |
|------|-------|---------|
| [`src/models/nemotron.h`](../src/models/nemotron.h) | ~100 | `NemotronModel` + `NemotronState` declarations |
| [`src/models/nemotron.cpp`](../src/models/nemotron.cpp) | ~340 | Full RNNT inference: encoder, decoder, joint, greedy decode |

### Files Modified

| File | Change |
|------|--------|
| [`src/models/model_type.h`](../src/models/model_type.h) | Added `nemotron_asr` to `ALM` array; added `IsNemotronASR()` helper |
| [`src/models/model.cpp`](../src/models/model.cpp) | Added `NemotronModel` dispatch in `CreateModel()` (before `WhisperModel`) |

### Model Type Registration

In [`src/models/model_type.h`](../src/models/model_type.h):

```cpp
inline static bool IsALM(const std::string& model_type) {
  static constexpr std::array<std::string_view, 2> ALM = {"nemotron_asr", "whisper"};
  return std::find(ALM.begin(), ALM.end(), model_type) != ALM.end();
}

inline static bool IsNemotronASR(const std::string& model_type) {
  return model_type == "nemotron_asr";
}
```

### Model Factory Dispatch

In [`src/models/model.cpp`](../src/models/model.cpp) `CreateModel()`, Nemotron is dispatched before the generic ALM/Whisper path:

```cpp
if (ModelType::IsNemotronASR(config->model.type))
  return std::make_shared<NemotronModel>(std::move(config), ort_env);
if (ModelType::IsALM(config->model.type))
  return std::make_shared<WhisperModel>(std::move(config), ort_env);
```

### NemotronModel Class

```cpp
struct NemotronModel : Model {
  NemotronModel(std::unique_ptr<Config> config, OrtEnv& ort_env);
  std::unique_ptr<State> CreateState(...) const override;

  std::unique_ptr<OrtSession> session_encoder_;   // FastConformer encoder
  std::unique_ptr<OrtSession> session_decoder_;   // RNNT prediction network
  std::unique_ptr<OrtSession> session_joint_;     // Joint network

  int encoder_hidden_size_{1024};
  int decoder_hidden_size_{640};
  int vocab_size_{1025};
  int blank_id_{1024};  // RNNT blank = last token
};
```

### RNNT Greedy Decode Algorithm

The core RNNT decode loop in `NemotronState::GreedyDecode()`:

```
for each encoder frame t = 0..T':
    for sym = 0..max_symbols_per_step:
        run decoder(current_token) → decoder_hidden
        run joint(encoder_frame[t], decoder_hidden) → logits[1025]
        next_token = argmax(logits)
        if next_token == blank:
            break  // advance to next encoder frame
        else:
            emit(next_token)
            current_token = next_token
```

### Token Emission Strategy

RNNT decodes all tokens in a single `Run()` call (non-autoregressive at the framework level). To integrate with the generator's autoregressive `generate_next_token()` loop:

1. **First `Run()` call**: Runs encoder + full RNNT greedy decode, stores all decoded tokens
2. **Each subsequent `Run()` call**: Returns logits with the next decoded token having score `10.0` (all others `-100.0`)
3. **After all tokens emitted**: Returns logits with blank (1024) as highest → triggers EOS detection

This allows the standard `while not generator.is_done(): generator.generate_next_token()` pattern to work.

### Config Schema

[`genai_config.json`](../tools/nemotron_export/onnx_models/genai_config.json):

```json
{
  "model": {
    "type": "nemotron_asr",
    "vocab_size": 1025,
    "bos_token_id": 0,
    "eos_token_id": 1024,
    "pad_token_id": 1024,
    "encoder": {
      "filename": "encoder.onnx",
      "hidden_size": 1024,
      "num_hidden_layers": 24,
      "inputs": { "audio_features": "audio_signal", "encoder_input_lengths": "length" },
      "outputs": { "hidden_states": "outputs", "encoder_output_lengths": "encoded_lengths" }
    },
    "decoder": {
      "filename": "decoder.onnx",
      "hidden_size": 640,
      "num_hidden_layers": 2,
      "pipeline": {
        "joint": {
          "filename": "joint.onnx",
          "inputs": ["encoder_output", "decoder_output"],
          "outputs": ["joint_output"]
        }
      }
    }
  },
  "search": { "max_length": 8192, "num_beams": 1 }
}
```

### Tokenizer

The tokenizer uses **T5Tokenizer / Unigram** format (not the default BPE/GPT2 path):

- [`tokenizer.json`](../tools/nemotron_export/onnx_models/tokenizer.json): HuggingFace Unigram model with `[[token, score], ...]` vocab pairs and `Metaspace` decoder (`▁` → space)
- [`tokenizer_config.json`](../tools/nemotron_export/onnx_models/tokenizer_config.json): `tokenizer_class: "T5Tokenizer"` to route through ORT Extensions' SentencePiece/Unigram path

Created by [`tools/nemotron_export/create_ort_tokenizer.py`](../tools/nemotron_export/create_ort_tokenizer.py) which converts the original NeMo `vocab.txt` to HuggingFace format.

## Python API Usage

```python
import numpy as np
import onnxruntime_genai as og

# Load model
model = og.Model("path/to/onnx_models")
tokenizer = og.Tokenizer(model)

# Prepare audio mel-spectrogram features externally
# (e.g., using NeMo's FilterbankFeatures or torchaudio MelSpectrogram)
# Shape: [batch=1, mel_bins=128, time_frames=T]
audio_features = compute_mel_spectrogram(wav_file)  # np.float32

# Create generator
params = og.GeneratorParams(model)
params.set_search_options(max_length=512, batch_size=1)
generator = og.Generator(model, params)

# Set inputs via NamedTensors (same pattern as Whisper)
inputs = og.NamedTensors()
inputs["audio_signal"] = audio_features
inputs["input_ids"] = np.array([[0]], dtype=np.int32)  # dummy BOS to trigger inference
generator.set_inputs(inputs)

# Generate (standard loop)
while not generator.is_done():
    generator.generate_next_token()

# Decode (skip first token = dummy BOS, filter out blank=1024)
tokens = list(generator.get_sequence(0))
text_ids = np.array([t for t in tokens[1:] if t != 1024], dtype=np.int32)
transcription = tokenizer.decode(text_ids)
print(transcription)
```

## Verified Test Results

### E2E Test ([`tools/nemotron_export/test_e2e.py`](../tools/nemotron_export/test_e2e.py))

| Test | Result |
|------|--------|
| Model loading (`og.Model`) | ✅ Pass |
| Tokenizer load + decode (T5Tokenizer/Unigram) | ✅ Pass |
| Token decode: `▁the` → `"the"`, `▁and` → `"and"` | ✅ Pass |
| Batch decode: `[34, 3, 23]` → `"I a m"` | ✅ Pass |
| Generator + `set_inputs` + `generate_next_token` loop | ✅ Pass |
| EOS detection via blank token (1024) | ✅ Pass |
| Random audio → all blanks (correct) | ✅ Pass |
| Raw ONNX baseline matches OG output | ✅ Pass |

### Real Speech Test ([`tools/nemotron_export/test_real_speech.py`](../tools/nemotron_export/test_real_speech.py))

Tested with `test/test_models/audios/jfk.flac` (JFK inaugural address, 11s):

| Path | Tokens | Transcription |
|------|--------|---------------|
| Raw ONNX Runtime | 18 | `"rough for you ask what what what what what what."` |
| onnxruntime-genai | 18 | `"rough for you ask what what what what what what."` |
| **Token match** | **100%** | Both paths produce identical tokens |

The partial recognition and repetition are expected — see Known Limitations below.

## Build

```bash
python build.py --config Release --skip_tests --skip_examples
```

New source files are picked up automatically via CMake GLOB in `src/models/`.

## Known Limitations

### 1. Stateless Decoder (Accuracy Impact)

The ONNX decoder model was exported **without LSTM state I/O**. The decoder only accepts `targets` and `target_length_orig` as inputs — LSTM hidden/cell states (`getitem_1`, `getitem_2`) are outputs only, not fed back in. Each `RunDecoder()` call starts with fresh LSTM state.

**Impact**: The prediction network has no memory of previously emitted tokens, causing repetition artifacts (e.g., `"what what what..."` instead of `"what your country can do for you"`).

**Fix**: Re-export the decoder with `getitem_1`/`getitem_2` as both inputs and outputs, and carry state between `RunDecoder()` calls in the C++ code.

### 2. No Cache-Aware Streaming

The encoder processes the entire audio in one shot. For real-time streaming, the FastConformer's cache-aware architecture requires re-exporting the encoder with convolution and attention caches exposed as I/O.

**Fix**: Re-export the encoder in streaming mode with explicit cache tensors per layer, and implement chunk-based `ProcessChunk()` in C++.

### 3. CPU Only

Not yet tested or built with `--use_cuda`. For real-time performance on long audio, GPU acceleration is needed.

### 4. External Audio Preprocessing

Audio → mel-spectrogram conversion must be done externally (e.g., NeMo `FilterbankFeatures`, or torchaudio `MelSpectrogram`). There is no built-in audio processor in the C++ pipeline. The `NemotronProcessor` was removed during development due to `OrtxFeatureExtractor` compatibility issues.

**Preprocessing requirements**: 16kHz sample rate, 128 mel bins, 25ms window, 10ms hop, 512-point FFT, per-feature normalization.

## Implementation Phases

### Phase 1: Basic Inference ✅ Complete
- [x] Add `nemotron_asr` to `IsALM()` and `IsNemotronASR()`
- [x] Create `NemotronModel` class (3-session: encoder, decoder, joint)
- [x] Create `NemotronState` with RNNT greedy decode
- [x] Export tokenizer in T5Tokenizer/Unigram format
- [x] Token emission via logits for generator framework compatibility
- [x] Test with dummy audio (all blanks — correct)
- [x] Test with real speech (JFK audio — tokens match raw ONNX baseline)

### Phase 2: Stateful Decoder — Not Started
- [ ] Re-export decoder ONNX with LSTM state I/O (`getitem_1`/`getitem_2` as inputs)
- [ ] Carry LSTM hidden/cell state between `RunDecoder()` calls
- [ ] Validate transcription accuracy improves (no more repetition)

### Phase 3: Cache-Aware Streaming — Not Started
- [ ] Re-export encoder in streaming mode with per-layer conv/attention caches
- [ ] Implement chunk-based `ProcessChunk()` API
- [ ] Add `ResetStreamingState()` method
- [ ] Test streaming inference with configurable chunk sizes

### Phase 4: Optimization — Not Started
- [ ] GPU/CUDA support (`--use_cuda`)
- [ ] Built-in audio preprocessor (mel spectrogram in C++)
- [ ] RNNT beam search
- [ ] Batched inference support

## References

- Source: [`src/models/nemotron.h`](../src/models/nemotron.h), [`src/models/nemotron.cpp`](../src/models/nemotron.cpp)
- Tests: [`tools/nemotron_export/test_e2e.py`](../tools/nemotron_export/test_e2e.py), [`tools/nemotron_export/test_real_speech.py`](../tools/nemotron_export/test_real_speech.py)
- Tokenizer export: [`tools/nemotron_export/create_ort_tokenizer.py`](../tools/nemotron_export/create_ort_tokenizer.py)
- Config: [`tools/nemotron_export/onnx_models/genai_config.json`](../tools/nemotron_export/onnx_models/genai_config.json)
- Whisper reference: [`src/models/whisper.h`](../src/models/whisper.h), [`src/models/whisper.cpp`](../src/models/whisper.cpp)
- [Nemotron-Speech Model Card](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
