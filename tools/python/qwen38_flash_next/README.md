# Qwen3.8-Flash-Next (`qwen38_flash_graph`) Package Builder

`tools/python/qwen38_flash_next/build_package.py` procedurally constructs self-contained ONNX Runtime GenAI model packages for **Qwen3.8-Flash-Next** (`model.type: "qwen38_flash_graph"`) + **DFlash2** (`S=8`, 7-draft speculative decoding) directly from the two Hugging Face checkpoints (`Qwen3.8-Flash-Next` and `Qwen3.8-Flash-Next-DFlash2`), without depending on any pre-exported `.onnx` model.

Every generated package supports:
- **Single unified `model.onnx` across all context lengths (`32K` / `128K` / `256K`)**: cache capacity dimensions (`kv_capacity` and `state_capacity`) are symbolic and `contrib_const_dsa_total_seq` is set to `[262152]`, so a single `model.onnx` serves `32K` (`num_blocks: 142`), `128K` (`num_blocks: 525`), and `256K` (`num_blocks: 1037`) without duplicate `model_128k.onnx` / `model_256k.onnx` files.
- **Pure GPU execution** up to **256K (`262,144`) context length** on H200 / high-VRAM GPUs (`gpu-128k`, `gpu-256k` -> `model.onnx`).
- **QMoE CPU/GPU hybrid offload** on 20–36.5 GB consumer GPUs (`consumer-gpu-24g`, `consumer-gpu-24g-host-64g`, `consumer-gpu-30g-ram-48g-32k`, `consumer-gpu-32g` -> `model.onnx` + `expert-profile.bin`).
- **Authoritative protocol boundary token IDs** (`bot_token_id: 248058`, `eot_token_id: 248059`, `bor_token_id: 248068`, `eor_token_id: 248069`) and **Engine dynamic batching** (`max_batch_size: 1`, `max_scheduled_tokens: 8192`).

---

## 1. Building Predefined RC5 Packages from the Hugging Face Checkpoints

```bash
# Build all 4 core RC5 packages directly from /data/tlwu/Qwen3.8-Flash-Next and /data/tlwu/Qwen3.8-Flash-Next-DFlash2:
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /data/tlwu/Qwen3.8-Flash-Next \
  --dflash-dir /data/tlwu/Qwen3.8-Flash-Next-DFlash2 \
  --models-root /home/tianlei/models \
  --link-root /data/tlwu \
  --presets rc5_all

# Or build a single predefined preset (e.g., 2b375_x02_y14_z32_dp23_b64):
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /data/tlwu/Qwen3.8-Flash-Next \
  --dflash-dir /data/tlwu/Qwen3.8-Flash-Next-DFlash2 \
  --preset 2b375_x02_y14_z32_dp23_b64 \
  --out /data/tlwu/qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x02_y14_z32_dp23_b64
```

| Preset Name | Alias | `(x INT4, y INT3, z INT2)` | KL Mode | Avg Bits | QMoE Size | 128K Peak VRAM (`num_blocks=525`) | 256K Peak VRAM (`num_blocks=1037`) | Min VRAM (`gpu-128k` / `gpu-256k`) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `qwen_flash_next_int4_int8_rc5_contrib_dflash2` | `all_int4` | `(48, 0, 0)` | `rank` | `4.000` | `63.28 GiB` | `75.42 GiB` (`80.99 GB`) | `77.90 GiB` (`83.65 GB`) | `83 GB` / `87 GB` |
| `qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top16_44_33_b64` | `top16` | `(16, 32, 0)` | `rank` | `3.333` | `51.56 GiB` | `64.79 GiB` (`69.57 GB`) | `67.63 GiB` (`72.62 GB`) | `72 GB` / `76 GB` |
| `qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top08_44_33_b64` | `top08` | `(8, 40, 0)` | `rank` | `3.167` | `48.63 GiB` | `61.87 GiB` (`66.43 GB`) | `64.71 GiB` (`69.48 GB`) | `69 GB` / `73 GB` |
| `qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x02_y14_z32_dp23_b64` | `2b375` | `(2, 14, 32)` | `dp23` | `2.375` | `37.06 GiB` | `50.30 GiB` (`54.01 GB`) | `52.00 GiB` (`55.84 GB`) | `57 GB` / `61 GB` |

---

## 2. Building Custom `(x INT4 + y INT3 + z INT2)` QMoE Partitions Sorted by KL Similarity

```bash
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /data/tlwu/Qwen3.8-Flash-Next \
  --dflash-dir /data/tlwu/Qwen3.8-Flash-Next-DFlash2 \
  -x 2 -y 14 -z 32 --kl-mode dp23 \
  --out /home/tianlei/models/qwen_flash_next_custom_2_14_32
```

---

## 3. Exporting Zero-Copy GGUF Model Packages (`build_package_gguf.py`)

`tools/python/qwen38_flash_next/build_package_gguf.py` exports `unsloth/Qwen3.8-Flash-Next-GGUF` quantization variants (`UD-IQ1_S`, `UD-IQ2_XS`, `UD-IQ3_XXS`) and the `IQ4_NL` n-gram (`engram`) GGUF directly to `qwen38_flash_graph` packages using `GatherPacked`, `MatMulPacked`, and `MoEPacked`, sharing dynamic cache surgery (`ensure_unified_dynamic_cache_model`), protocol boundary tokens, and config migration helpers with `build_package.py`:

```bash
python tools/python/qwen38_flash_next/build_package_gguf.py \
  --gguf-root /data/tlwu/qwen_flash_next_ggufs \
  --base-pkg /data/tlwu/qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x02_y14_z32_dp23_b64 \
  --out-root /data/tlwu \
  --variants UD-IQ1_S,UD-IQ2_XS,UD-IQ3_XXS
```

| GGUF Variant | Total GPU Weights | 32K Peak VRAM (`num_blocks=142`, min `141`) | 64K Peak VRAM (`num_blocks=270`, min `269`) | 128K Peak VRAM (`num_blocks=525`) | 256K Peak VRAM (`num_blocks=1037`) | `runtime_profiles` Min VRAM (`gpu-32k` / `gpu-64k` / `gpu-128k` / `gpu-256k`) |
|---|---:|---:|---:|---:|---:|---:|
| **`UD-IQ1_S`** | `35.66 GiB` (`38.29 GB`) | `41.39 GiB` (`44.45 GB`) | `41.74 GiB` (`44.82 GB`) | `42.85 GiB` (`46.00 GB`) | `45.54 GiB` (`48.90 GB`) | `47 GB` / `48 GB` / `49 GB` / `52 GB` |
| **`UD-IQ2_XS`** | `44.46 GiB` (`47.74 GB`) | `50.14 GiB` (`53.84 GB`) | `50.49 GiB` (`54.21 GB`) | `51.59 GiB` (`55.39 GB`) | `54.28 GiB` (`58.28 GB`) | `56 GB` / `57 GB` / `58 GB` / `61 GB` |
| **`UD-IQ3_XXS`** | `51.73 GiB` (`55.55 GB`) | `57.44 GiB` (`61.68 GB`) | `57.79 GiB` (`62.05 GB`) | `58.89 GiB` (`63.24 GB`) | `61.59 GiB` (`66.13 GB`) | `64 GB` / `65 GB` / `66 GB` / `69 GB` |

---

## 4. Component Precision Options (`precision_options.py`)

Both `build_package.py` and `build_package_gguf.py` accept `--precision-profile {default,fp32,fp32_fp16kv,fp32_fp16kv_v2}` as well as individual per-component flags (`--kv-cache-dtype`, `--rotary-cache-dtype`, `--qsa-qkv-dtype`, `--indexer-dtype`, `--gdn-gate-dtype`, `--gdn-precision`, `--ple-engram`, `--ple-norm-dtype`, `--small-weights-source`):

```bash
# Build rc5_top16_fp32_fp16kv_v2, rc5_all_int4_fp32_fp16kv_v2, and rc5_2b375_fp32_fp16kv_v2
# (native INT8 PLE + bit-exact RC5-calibrated FP16 V & FP16 KV cache + FP32 rotary cache + FP32 GDN/indexer/PLE norms):
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /data/tlwu/Qwen3.8-Flash-Next \
  --dflash-dir /data/tlwu/Qwen3.8-Flash-Next-DFlash2 \
  --preset top16_44_33_b64,qwen_flash_next_int4_int8_rc5_contrib_dflash2,2b375_x02_y14_z32_dp23_b64 \
  --precision-profile fp32_fp16kv_v2 \
  --out /data/tlwu/qwen_flash_next_rc5_v2

# Build rc5_all_int4_fp32_fp16kv (FP16 KV cache + FP32 rotary cache + FP32 GDN/indexer/PLE norms + Q8_0 engram):
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /data/tlwu/Qwen3.8-Flash-Next \
  --dflash-dir /data/tlwu/Qwen3.8-Flash-Next-DFlash2 \
  --preset qwen_flash_next_int4_int8_rc5_contrib_dflash2 \
  --precision-profile fp32_fp16kv \
  --out /data/tlwu/qwen_flash_next_rc5_all_int4_fp32_fp16kv

# Build rc5_2b375 with BF16 KV cache + FP32 GDN/indexer/PLE norms:
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /data/tlwu/Qwen3.8-Flash-Next \
  --dflash-dir /data/tlwu/Qwen3.8-Flash-Next-DFlash2 \
  --preset 2b375_x02_y14_z32_dp23_b64 \
  --precision-profile fp32 \
  --out /data/tlwu/qwen_flash_next_rc5_2b375

# Build GGUF UD-IQ3_XXS and UD-IQ2_XS with Q8_0 engram, FP32 long-context path, and FP16 KV cache:
python tools/python/qwen38_flash_next/build_package_gguf.py \
  --gguf-root /data/tlwu/qwen_flash_next_ggufs \
  --base-pkg /data/tlwu/qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top16_44_33_b64_unified \
  --out-root /data/tlwu \
  --variants UD-IQ3_XXS,UD-IQ2_XS \
  --precision-profile fp32_fp16kv \
  --pkg-suffix _q8_engram_fp32_fp16kv
```

