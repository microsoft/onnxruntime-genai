# GenAI architecture

GenAI is divided into a few main sections:

* Models - Manages the input/output OrtValues for an OrtSession for each model type supported
  * DecoderOnly
  * MultiModal
  * Pipelined
  * Whisper (in progress)
* Scoring - Takes the logits and figures out the next token
  * Greedy Search - TopK/TopP/Temperature
  * Beam Search
* IO Processing
  * Tokenizing/Detokenizing
  * Audio & Image Encoding & Decoding
* Providers - Provider device specific code to handle device memory management & provider accelerated versions of scoring
  * CPU
  * Cuda - Accelerated scoring and model IO handling
  * Dml - Accelerated model IO handling
  * WebGPU - Device memory KV cache storage
  * QNN - Device memory allocation, but memory is CPU accessible
* Generator - Ties the model & scoring together and holds the runtime state

```mermaid
classDiagram
  direction RL
  namespace Models {
    class Config{
      genai_config.json
    }
    class Model{
      Config
      OrtSessionOptions
      SessionInfo
      DeviceInterface
    }
    class Adapters{
    }
    class State{
    }
    class DecoderOnly_State{
      DefaultInputIDs
      Logits
      DefaultKeyValueCache
      DefaultPositionInputs
      ExtraInputs
    }
    class IntermediatePipelineState{
      id
    }
    class DecoderOnlyPipelineState{
      InputIDs
      Logits
      OrtValues
      KeyValueCache
    }
    class VisionState{
      MultiModalFeatures
    }
    class SpeechState{
      MultiModalFeatures
    }
    class EmbeddingState{
      MultiModalFeatures image_features
      MultiModalFeatures audio_features
      Embeddings input_embeds
    }
    class DecoderState{
      DefaultPositionInputs
      DefaultKeyValueCache
      Logits
      Embeddings
    }
    class MultiModalPipelineState{
      VisionState
      SpeechState
      EmbeddingState
      DecoderState
      Adapters
    }
    class WhisperState{
      DefaultInputIDs
      Logits
      DefaultKeyValueCache
      CrossCache
    }
  }
  DecoderOnly_State <|-- State
  IntermediatePipelineState <|-- State
  DecoderOnlyPipelineState <|-- State
  VisionState <|-- State
  SpeechState <|-- State
  EmbeddingState <|-- State
  DecoderState <|-- State
  DecoderOnly_State <|-- State
  MultiModalPipelineState <|-- State
  WhisperState <|-- State
  namespace IO Processing {
    class Tokenizer{
    }
    class TokenizerStream{
    }
    class MultiModalProcessor{
    }
  }
  Model --> Tokenizer
  Model --> MultiModalProcessor
  TokenizerStream ..> Tokenizer
  namespace CPUScoring {
    class Search_CPU{
    }
    class GreedySearch_CPU{
       RandomNumberGenerator
       next_tokens[]
       +SelectTop()
       +SampleTopK()
       +SampleTopP()
       +SampleTopKTopP()
    }
    class BeamSearch_CPU{
       +SelectTop()
    }
  }
  namespace CudaScoring {
    class Search_Cuda{
    }
    class GreedySearch_Cuda{
       +SelectTop()
       +SampleTopK()
       +SampleTopP()
       +SampleTopKTopP()
    }
    class BeamSearch_Cuda{
       +SelectTop()
    }
  }
  class Search{
    Sequences
    GeneratorPArams
  }
  class GeneratorParams{
  }
  GeneratorParams --> Config
  class Sequences {
    int32 tokens[]
  }
  class Generator{
    State
    Search
  }
  style Generator stroke:#000,stroke-width:4px
  Search --* Sequences
  Search --> GeneratorParams

  Search_CPU <|-- Search
  GreedySearch_CPU <|-- Search_CPU
  BeamSearch_CPU <|-- Search_CPU
  Search_Cuda <|-- Search
  GreedySearch_Cuda <|-- Search_Cuda
  BeamSearch_Cuda <|-- Search_Cuda
  Model --o Config
  State --> Model
  Generator --* State
  Generator --* Search

  namespace Providers {
    class DeviceInterface
    class CpuInterface
    class CudaInterface
    class DmlInterface
    class WebGPUInterface
    class QNNInterface
  }
  CpuInterface <|-- DeviceInterface
  CudaInterface <|-- DeviceInterface
  DmlInterface <|-- DeviceInterface
  WebGPUInterface <|-- DeviceInterface
  QNNInterface <|-- DeviceInterface
  Model --* DeviceInterface
```

# C API Objects
```mermaid
classDiagram
  direction RL
  namespace Utility Types {
    class OgaStringArray {
    }
    class OgaSequences {
    }
    class OgaTensor {
    }
    class OgaNamedTensors {
    }
  }
  namespace Models {
    class OgaConfig {
    }
    class OgaModel {
    }
    class OgaAdapters {
    }
  }
  OgaModel --o OgaConfig
  namespace IO Processing {
    class OgaTokenizer {
    }
    class OgaTokenizerStream {
    }
    class OgaMultiModalProcessor {
    }
    class OgaImages {
    }
    class OgaAudios {
    }
  }
  OgaTokenizerStream --> OgaTokenizer
  OgaImages --> OgaMultiModalProcessor
  OgaAudios --> OgaMultiModalProcessor
  namespace Generation {
    class OgaGeneratorParams {
    }
    class OgaGenerator {
    }
  }
  OgaModel --> OgaGeneratorParams
  OgaGeneratorParams --> OgaGenerator
```
