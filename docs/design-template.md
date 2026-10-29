# Vision Pipeline

## 1. Problem statement: Background and motivation

Gemma-4-E2B-it is supported through the MultiModalLanguageModal pipeline which creates single audio, vision, embedding and decoder sessions. Our goal is to have decoder/vision models on NPU (qnn/ov). NPU compilation generally requires static shapes, but the pooler-projector part of the vision model is dynamic shaped as it extracts the exact image patch count. 

Therefore, we split the vision model into 2 parts (static shapes encoder, dynamic shaped pooler-projector) which needs to run in OGA.


## 2. Impementation ask: What do we need?

We now need 2-stage vision model support in the MultiModalLanguageModel pipeline (single-session decoder in multimodal models) and Qwen_2_5_VL_Pipeline (pipelined-decoder in multimodal/vl models).

Rather than hardcoding a 2 stage vision pipeline, we could create a generic vision pipeline that can be reused by other classes.

## 3. Current architecture: What is already there?


## 4. Gap analysis


## 5. Design options


