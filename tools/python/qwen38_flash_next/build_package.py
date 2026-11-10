#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Unified Qwen3.8-Flash-Next RC5 (`qwen38_flash_graph`) package builder with KL-similarity `(x INT4 + y INT3 + z INT2)`
QMoE layer partitioning, predefined RC5 configurations, context-tier graphs, and unified GPU + QMoE offload support.

Key capabilities:
1. Predefined RC5 configurations (`--preset` / `--presets`) reproducing our shipping and candidate RC5 packages:
   - `qwen_flash_next_int4_int8_rc5_contrib_dflash2` (`all_int4`: `x=48, y=0, z=0`, `4.000` avg bits)
   - `qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top16_44_33_b64` (`top16_44_33_b64`: `x=16, y=32, z=0`, `3.333` avg bits)
   - `qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top08_44_33_b64` (`top08_44_33_b64`: `x=8, y=40, z=0`, `3.167` avg bits)
   - `qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x02_y14_z32_dp23_b64` (`2b375_x02_y14_z32_dp23_b64`: `x=2, y=14, z=32`, `2.375` avg bits)
   - Additional sub-3-bit champions (`3b0_x06_y36_z06_dp3_b64`, `2b8_x04_y30_z14_dp3_b64`, `2b5_x02_y20_z26_dp3_b64`,
     `2b375_x00_y18_z30_dp23_b64`, `2b25_x00_y12_z36_dp2_b64`, `2b125_x00_y06_z42_dp2_b64`, `2b0_x00_y00_z48_b64`).
2. Arbitrary `(x INT4 + y INT3 + z INT2)` QMoE layer counts (`-x / --x-int4`, `-y / --y-int3`, `-z / --z-int2` or `--qmoe-xyz X,Y,Z`)
   using 48-layer KL similarity / divergence profiles (`--kl-mode {auto,rank,dp23,dp3,dp2,dp4}` or `--kl-study-json`) to sort and
   partition QMoE layers.
3. Single-package support for BOTH pure GPU execution and QMoE CPU/GPU hybrid offload:
   - Standard `com.microsoft::QMoE` nodes with `router_weights` (input 14) cleared (`swiglu_fusion=1`, `normalize_routing_weights=1`).
   - Built-in `runtime_profiles` in `genai_config.json` (`consumer-gpu-24g`, `consumer-gpu-24g-host-64g`,
     `consumer-gpu-30g-ram-48g-32k`, `consumer-gpu-30g-ram-48g-32k-full`, `consumer-gpu-32g`) plus `expert-profile.bin`
     (STRP v1 popularity profile), so datacenter GPUs (`> 36.5 GB` VRAM) run full-HBM GPU mode automatically while 20-36.5 GB
     GPUs (or `ORTGENAI_ENABLE_RUNTIME_PROFILE` / `ORT_QMOE_DEVICE_BUDGET_MB`) run QMoE hybrid offload on the exact same model folder.
   - Optional explicit offload flags (`--offload-mode`, `--offload-placement`, `--device-expert-bytes`, `--host-expert-bytes`,
     `--target-resident-slots`, `--total-resident-slots`, `--uniform-profile-slots`) to stamp `session.moe.offload.*` entries directly
     onto `decoder.session_options`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import graph_builder as gb
import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from package_config import migrate_qwen38_graph_config
from package_io import (
    ALIGN,
    CONTEXT_TIERS,
    HFCheckpointReader,
    align_up,
    build_context_tier_model,
    ensure_unified_dynamic_cache_model,
    make_ext_init,
)
from package_paths import reject_output_overlap
from precision_options import add_precision_cli_args, apply_precision_config_to_package, resolve_precision_config

PLE_INT8_SCALE = 0.000703105776
STRP_MAGIC = 0x50525453
EXPERT_PROFILE_NAME = "expert-profile.bin"

DEFAULT_HF_DIR = Path("/data/tlwu/Qwen3.8-Flash-Next")
DEFAULT_DFLASH_DIR = Path("/data/tlwu/Qwen3.8-Flash-Next-DFlash2")
PROTOCOL_BOUNDARY_TOKEN_IDS = {
    "bot_token_id": 248058,
    "eot_token_id": 248059,
    "bor_token_id": 248068,
    "eor_token_id": 248069,
}
DEFAULT_ENGINE_CONFIG = {
    "dynamic_batching": {
        "max_batch_size": 1,
        "max_scheduled_tokens": 8192,
    }
}
DEFAULT_MAX_CONTEXT_LENGTH = 262144

OFFLOAD_MODES = ("required", "auto")
OFFLOAD_PLACEMENTS = ("adaptive", "static")
OFFLOAD_OPTION_PREFIX = "session.moe.offload."
ROUTER_WEIGHTS_INPUT = 14
UNSUPPORTED_OFFLOAD_INPUTS = (8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20)
OFFLOAD_WEIGHT_INPUTS = (2, 3, 5, 6)
NVFP4_GLOBAL_SCALE_INPUTS = (15, 16)

REMOVED_OPTIONS = {
    "--migrate-shared-onnx": "it rewrites legacy 26-input QMoEOffload graphs, and the runtime has no QMoEOffload operator",
    "--out-onnx": "it belongs to --migrate-shared-onnx",
    "--strip-phase": "it belongs to --migrate-shared-onnx",
    "--strip-stats": "it belongs to --migrate-shared-onnx",
    "--target-spare-slots": "spare slots are the runtime's decision; choose --offload-placement static (none) or adaptive",
    "--dense-offload-qmoe": "it rewrote dense/model.onnx to the QMoEOffload operator, which the runtime does not have",
    "--dense-stream-min-rows": "it set attributes of the QMoEOffload operator, which the runtime does not have",
    "--dense-stream-wave": "it set attributes of the QMoEOffload operator, which the runtime does not have",
}

# fmt: off
# 48-layer single-layer KL sensitivity ranking (most sensitive -> least sensitive when quantized from INT4):
RANKED_SENSITIVE_LAYERS: tuple[int, ...] = (
    27, 2, 0, 29, 26, 7, 3, 1, 4, 13, 10, 5, 12, 8, 9, 6,
    24, 25, 15, 14, 28, 30, 16, 18, 11, 33, 39, 32, 19, 34, 17, 31,
    21, 35, 23, 38, 22, 36, 46, 37, 40, 20, 43, 41, 45, 42, 44, 47,
)

# Single-layer KL divergence from the 48-layer INT4 B32 reference (`dp4` costs):
KL_SINGLE_INT3_B64: tuple[float, ...] = (
    0.028866, 0.030570, 0.028318, 0.023054, 0.026862, 0.022529, 0.020582, 0.022105,
    0.019310, 0.019404, 0.021704, 0.019183, 0.021730, 0.020985, 0.019793, 0.018960,
    0.017359, 0.016832, 0.016958, 0.013846, 0.011894, 0.014480, 0.017076, 0.012894,
    0.017265, 0.017447, 0.020004, 0.018149, 0.016618, 0.019304, 0.012021, 0.013757,
    0.013711, 0.013395, 0.011802, 0.011763, 0.008537, 0.011010, 0.008003, 0.011320,
    0.009495, 0.008055, 0.006624, 0.007957, 0.005410, 0.006203, 0.007243, 0.005695,
)
KL_SINGLE_INT3_B32: tuple[float, ...] = (
    0.027516, 0.024475, 0.027448, 0.024496, 0.026893, 0.023450, 0.022847, 0.023520,
    0.021115, 0.021440, 0.020802, 0.018795, 0.017193, 0.017938, 0.019460, 0.018783,
    0.019131, 0.016324, 0.014028, 0.013046, 0.012338, 0.013672, 0.013993, 0.012015,
    0.017650, 0.016461, 0.022144, 0.016902, 0.016120, 0.017241, 0.011874, 0.011799,
    0.013701, 0.015294, 0.009938, 0.013118, 0.007840, 0.008704, 0.008480, 0.009060,
    0.008969, 0.007876, 0.005626, 0.007647, 0.005008, 0.006181, 0.005567, 0.004616,
)
KL_SINGLE_INT2_B64: tuple[float, ...] = (
    0.039752, 0.033853, 0.038601, 0.036583, 0.037931, 0.034520, 0.031094, 0.036606,
    0.029413, 0.028467, 0.026995, 0.027690, 0.032025, 0.032707, 0.023039, 0.028972,
    0.028127, 0.024258, 0.027429, 0.022017, 0.017673, 0.022908, 0.023868, 0.024692,
    0.030882, 0.030974, 0.049393, 0.035554, 0.035097, 0.039092, 0.026996, 0.029208,
    0.036323, 0.032546, 0.025411, 0.035768, 0.021500, 0.022826, 0.027229, 0.031573,
    0.026394, 0.018789, 0.015224, 0.019687, 0.015085, 0.019096, 0.020224, 0.016387,
)
KL_SINGLE_INT2_B32: tuple[float, ...] = (
    0.040398, 0.035937, 0.037597, 0.035473, 0.039172, 0.034419, 0.033251, 0.033174,
    0.033350, 0.025448, 0.030627, 0.029146, 0.033270, 0.034246, 0.025803, 0.029305,
    0.025752, 0.023418, 0.025483, 0.021536, 0.016844, 0.020613, 0.020959, 0.019756,
    0.029752, 0.026798, 0.047265, 0.041332, 0.031265, 0.040648, 0.026096, 0.025828,
    0.033288, 0.032242, 0.026598, 0.024712, 0.019854, 0.019966, 0.021842, 0.031362,
    0.023854, 0.016365, 0.014876, 0.021694, 0.013051, 0.016450, 0.017165, 0.013978,
)

# Leave-one-out marginal KL divergence around the 48-layer INT3 B64 operating point (`dp3` costs):
KL_MARGINAL_3_TO_4_B32: tuple[float, ...] = (
    0.000300, 0.004624, -0.008380, 0.002116, 0.000186, 0.007002, 0.002262, -0.001474,
    0.002075, -0.003592, 0.001541, 0.009280, 0.003710, -0.000268, -0.001032, 0.004456,
    0.001615, 0.007907, 0.001097, 0.003729, -0.001537, 0.000029, -0.001627, 0.000035,
    -0.004738, -0.001599, -0.008474, -0.000410, -0.006690, -0.001602, 0.000936, -0.001186,
    -0.008341, 0.005931, -0.002989, -0.000156, 0.001497, -0.003304, -0.001906, -0.001786,
    -0.006153, -0.003547, 0.003667, -0.002245, -0.000832, -0.002344, -0.005103, -0.003521,
)
KL_MARGINAL_3_TO_2_B64: tuple[float, ...] = (
    0.009146, 0.017674, -0.002724, 0.003918, 0.005188, 0.010480, 0.009935, 0.004969,
    0.009991, 0.003478, 0.001439, 0.010214, 0.004181, 0.004464, 0.004890, 0.008212,
    0.000349, 0.010999, 0.007341, 0.005896, 0.008713, 0.006066, 0.007797, -0.003160,
    0.005535, 0.006267, 0.009829, 0.006775, 0.000286, 0.005219, -0.003642, 0.014451,
    0.007196, 0.008306, -0.006550, 0.003950, 0.002819, 0.001326, 0.009345, 0.016399,
    0.005220, 0.001449, 0.004430, 0.012631, 0.004531, 0.004288, 0.003628, 0.008012,
)
KL_MARGINAL_3_TO_2_B32: tuple[float, ...] = (
    0.010971, 0.019289, -0.003361, 0.004921, 0.006431, 0.007745, 0.009927, 0.008646,
    0.009898, 0.003422, 0.004595, 0.003697, 0.009629, 0.001783, -0.000220, 0.008056,
    -0.000882, 0.009417, 0.007587, 0.000238, 0.010749, 0.006242, 0.009013, 0.003900,
    0.005026, 0.004870, 0.007510, 0.024638, -0.001891, -0.008198, -0.007719, 0.008475,
    0.000528, 0.017072, -0.004817, 0.003560, -0.000195, 0.005643, 0.009662, 0.013832,
    0.002531, 0.000998, 0.004843, 0.009135, 0.004891, 0.003500, 0.007689, 0.005026,
)

# Leave-one-out marginal KL divergence around the 48-layer INT2 B64 operating point (`dp2` costs):
KL_MARGINAL_2_TO_4_B32: tuple[float, ...] = (
    0.004306, -0.013611, -0.023139, -0.015036, -0.013226, -0.005766, 0.004573, -0.000153,
    -0.007325, 0.013717, 0.002070, -0.011494, -0.007123, -0.009117, -0.013639, -0.014568,
    -0.011514, -0.010834, -0.015460, -0.009908, -0.011202, -0.008942, -0.010111, -0.003048,
    -0.002426, -0.007756, 0.007416, -0.013861, -0.010414, -0.008787, 0.004431, -0.018548,
    -0.019673, -0.016938, -0.003975, -0.005994, -0.008064, -0.000160, -0.009612, -0.014834,
    -0.012496, -0.010521, -0.009896, -0.015797, -0.006055, -0.009760, -0.018907, -0.001969,
)
KL_MARGINAL_2_TO_3_B64: tuple[float, ...] = (
    -0.007191, -0.017342, -0.024796, -0.007254, -0.009527, -0.011190, 0.009505, 0.003002,
    -0.004019, -0.001079, -0.002250, -0.011888, -0.016027, -0.007590, -0.012036, -0.004232,
    -0.000075, -0.006703, -0.016159, -0.007826, -0.005135, -0.007415, -0.006811, -0.012299,
    0.005627, 0.000136, -0.002390, -0.006925, -0.012550, -0.015572, -0.004620, -0.017152,
    -0.008360, -0.011281, -0.005421, -0.011415, -0.005411, -0.002616, -0.011037, -0.014974,
    -0.010674, -0.014591, -0.006155, -0.017069, -0.003306, -0.009796, -0.017849, 0.002595,
)

# 48-layer KL sensitivity & marginal KLD tables calibrated under RC6 (`FP32` GDN + `FP32` Indexer + `FP32` RoPE + `FP16` QKV):
RANKED_SENSITIVE_LAYERS_FP32: tuple[int, ...] = (
    2, 0, 26, 29, 1, 3, 4, 5, 18, 7, 33, 27, 28, 6, 32, 24,
    13, 31, 25, 11, 12, 16, 8, 35, 9, 15, 39, 10, 17, 38, 21, 40,
    22, 34, 30, 14, 23, 37, 19, 36, 43, 20, 45, 46, 41, 47, 44, 42,
)
KL_SINGLE_INT3_B64_FP32: tuple[float, ...] = (
    0.027692, 0.028703, 0.029138, 0.029547, 0.026133, 0.023059, 0.023210, 0.024261,
    0.021041, 0.020363, 0.021272, 0.021121, 0.023560, 0.023146, 0.020733, 0.019927,
    0.017724, 0.015676, 0.017743, 0.016256, 0.014247, 0.017840, 0.015696, 0.012967,
    0.017232, 0.015576, 0.022109, 0.017202, 0.016940, 0.018528, 0.013656, 0.012994,
    0.014050, 0.012823, 0.012386, 0.010907, 0.007680, 0.008635, 0.008073, 0.010263,
    0.009127, 0.007202, 0.005429, 0.007693, 0.005351, 0.006146, 0.006480, 0.005261,
)
KL_SINGLE_INT2_B64_FP32: tuple[float, ...] = (
    0.046458, 0.040769, 0.048297, 0.037694, 0.036964, 0.036858, 0.033493, 0.035954,
    0.030056, 0.029861, 0.029002, 0.031766, 0.031679, 0.032990, 0.023629, 0.029711,
    0.030898, 0.026841, 0.036629, 0.022619, 0.019384, 0.026241, 0.025430, 0.023262,
    0.033134, 0.032007, 0.045066, 0.035375, 0.034005, 0.041019, 0.025217, 0.032845,
    0.033189, 0.035806, 0.025384, 0.030011, 0.020658, 0.023001, 0.026547, 0.029282,
    0.026107, 0.017042, 0.014351, 0.020398, 0.016141, 0.018097, 0.017258, 0.016553,
)
KL_MARGINAL_3_TO_4_B32_FP32: tuple[float, ...] = (
    -0.001082, 0.002509, -0.008695, -0.003464, -0.002465, 0.000495, 0.001242, -0.009925,
    -0.006139, -0.003182, -0.000415, 0.001087, -0.008322, 0.002535, 0.004893, -0.000546,
    -0.002623, -0.001225, 0.002060, 0.000734, -0.003837, -0.001299, -0.004318, -0.005482,
    -0.004667, -0.003222, -0.007339, -0.003979, -0.009607, -0.002974, -0.004736, -0.006016,
    -0.017202, -0.003304, -0.008831, -0.000563, -0.000165, -0.005135, -0.004452, -0.003498,
    -0.003940, -0.002250, 0.000697, -0.000956, -0.003959, -0.001424, -0.007386, -0.002351,
)
KL_MARGINAL_3_TO_2_B64_FP32: tuple[float, ...] = (
    0.004477, 0.008201, -0.004845, -0.001998, 0.006715, 0.001596, 0.002387, 0.002264,
    -0.003408, -0.001531, 0.000476, 0.002445, -0.000426, 0.002849, 0.000864, -0.000286,
    -0.001221, -0.001059, 0.004727, 0.001312, 0.003829, 0.008355, -0.002165, -0.005352,
    0.000508, 0.002249, 0.008942, 0.004227, -0.002616, 0.003998, -0.000127, 0.012762,
    -0.000151, 0.009434, -0.014157, 0.007192, -0.003653, -0.000775, 0.009549, 0.015009,
    -0.001402, -0.000475, 0.008329, 0.015382, 0.003168, 0.008955, -0.002685, 0.011496,
)
KL_MARGINAL_2_TO_4_B32_FP32: tuple[float, ...] = (
    -0.010118, -0.005520, -0.017916, -0.013477, -0.009596, -0.012830, -0.003022, -0.004245,
    0.004246, -0.002784, -0.006533, -0.009797, -0.002877, -0.019545, -0.013639, -0.012764,
    -0.009014, -0.013910, -0.013295, -0.003081, -0.009441, -0.014177, -0.010013, -0.004412,
    0.002884, -0.006168, 0.000975, -0.008598, -0.018508, 0.005214, 0.003914, -0.025724,
    -0.018006, -0.010042, 0.001261, -0.006572, -0.008549, -0.000295, -0.009103, -0.020638,
    -0.003599, 0.003690, -0.005880, -0.007881, -0.006379, -0.009662, -0.017038, -0.001225,
)
KL_MARGINAL_2_TO_3_B64_FP32: tuple[float, ...] = (
    -0.005311, -0.017914, -0.013309, -0.008003, -0.014046, -0.010135, -0.000239, -0.007429,
    -0.001982, -0.003070, -0.005477, -0.003494, -0.010099, -0.005185, -0.012612, -0.003594,
    -0.010729, -0.010745, -0.009427, -0.000449, -0.010707, -0.021646, -0.013371, -0.004596,
    0.008825, -0.006280, -0.002779, 0.000128, -0.013641, -0.007890, -0.001974, -0.021655,
    -0.010321, -0.014047, -0.001405, -0.010786, -0.005514, -0.000363, -0.012100, -0.020676,
    -0.006096, -0.000937, -0.007267, -0.009861, -0.002339, -0.008566, -0.014504, 0.003505,
)
# fmt: on

KL_MODES = (
    "auto",
    "rank",
    "dp23",
    "dp3",
    "dp2",
    "dp4",
    "auto_fp32",
    "rank_fp32",
    "dp23_fp32",
    "dp3_fp32",
    "dp2_fp32",
    "dp4_fp32",
)

# Predefined RC5/RC6 model configurations (`--preset` / `--presets`):
RC5_CORE_PRESETS: tuple[str, ...] = (
    "qwen_flash_next_int4_int8_rc5_contrib_dflash2",
    "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top16_44_33_b64",
    "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top08_44_33_b64",
    "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x02_y14_z32_dp23_b64",
)
RC6_CORE_PRESETS: tuple[str, ...] = (
    "qwen_flash_next_rc6_all_int4_fp32",
    "qwen_flash_next_rc6_top16_fp32",
    "qwen_flash_next_rc6_2b375_fp32",
)

PRESETS: dict[str, dict] = {
    "qwen_flash_next_int4_int8_rc5_contrib_dflash2": {
        "pkg_name": "qwen_flash_next_int4_int8_rc5_contrib_dflash2",
        "x": 48,
        "y": 0,
        "z": 0,
        "bs4": 32,
        "bs3": 64,
        "bs2": 64,
        "kl_mode": "rank",
    },
    "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top16_44_33_b64": {
        "pkg_name": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top16_44_33_b64",
        "x": 16,
        "y": 32,
        "z": 0,
        "bs4": 32,
        "bs3": 64,
        "bs2": 64,
        "kl_mode": "rank",
    },
    "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top08_44_33_b64": {
        "pkg_name": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top08_44_33_b64",
        "x": 8,
        "y": 40,
        "z": 0,
        "bs4": 32,
        "bs3": 64,
        "bs2": 64,
        "kl_mode": "rank",
    },
    "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x02_y14_z32_dp23_b64": {
        "pkg_name": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x02_y14_z32_dp23_b64",
        "x": 2,
        "y": 14,
        "z": 32,
        "bs4": 32,
        "bs3": 64,
        "bs2": 64,
        "kl_mode": "dp23",
    },
    "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x00_y18_z30_dp23_b64": {
        "pkg_name": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x00_y18_z30_dp23_b64",
        "x": 0,
        "y": 18,
        "z": 30,
        "bs4": 32,
        "bs3": 64,
        "bs2": 64,
        "kl_mode": "dp23",
    },
    "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_3b0_x06_y36_z06_dp3_b64": {
        "pkg_name": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_3b0_x06_y36_z06_dp3_b64",
        "x": 6,
        "y": 36,
        "z": 6,
        "bs4": 32,
        "bs3": 64,
        "bs2": 64,
        "kl_mode": "dp3",
    },
    "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b8_x04_y30_z14_dp3_b64": {
        "pkg_name": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b8_x04_y30_z14_dp3_b64",
        "x": 4,
        "y": 30,
        "z": 14,
        "bs4": 32,
        "bs3": 64,
        "bs2": 64,
        "kl_mode": "dp3",
    },
    "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b5_x02_y20_z26_dp3_b64": {
        "pkg_name": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b5_x02_y20_z26_dp3_b64",
        "x": 2,
        "y": 20,
        "z": 26,
        "bs4": 32,
        "bs3": 64,
        "bs2": 64,
        "kl_mode": "dp3",
    },
    "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b25_x00_y12_z36_dp2_b64": {
        "pkg_name": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b25_x00_y12_z36_dp2_b64",
        "x": 0,
        "y": 12,
        "z": 36,
        "bs4": 32,
        "bs3": 64,
        "bs2": 64,
        "kl_mode": "dp2",
    },
    "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b125_x00_y06_z42_dp2_b64": {
        "pkg_name": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b125_x00_y06_z42_dp2_b64",
        "x": 0,
        "y": 6,
        "z": 42,
        "bs4": 32,
        "bs3": 64,
        "bs2": 64,
        "kl_mode": "dp2",
    },
    "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b0_x00_y00_z48_b64": {
        "pkg_name": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b0_x00_y00_z48_b64",
        "x": 0,
        "y": 0,
        "z": 48,
        "bs4": 32,
        "bs3": 64,
        "bs2": 64,
        "kl_mode": "dp2",
    },
    "qwen_flash_next_rc6_all_int4_fp32": {
        "pkg_name": "qwen_flash_next_rc6_all_int4_fp32",
        "x": 48,
        "y": 0,
        "z": 0,
        "bs4": 32,
        "bs3": 64,
        "bs2": 64,
        "kl_mode": "rank_fp32",
        "precision_profile": "fp32_fp16kv_v2",
        "consolidate_data_files": True,
    },
    "qwen_flash_next_rc6_top16_fp32": {
        "pkg_name": "qwen_flash_next_rc6_top16_fp32",
        "x": 16,
        "y": 32,
        "z": 0,
        "bs4": 32,
        "bs3": 64,
        "bs2": 64,
        "kl_mode": "rank_fp32",
        "precision_profile": "fp32_fp16kv_v2",
        "consolidate_data_files": True,
    },
    "qwen_flash_next_rc6_2b375_fp32": {
        "pkg_name": "qwen_flash_next_rc6_2b375_fp32",
        "x": 2,
        "y": 14,
        "z": 32,
        "bs4": 32,
        "bs3": 64,
        "bs2": 64,
        "kl_mode": "dp23_fp32",
        "precision_profile": "fp32_fp16kv_v2",
        "consolidate_data_files": True,
    },
}

PRESET_ALIASES: dict[str, str] = {
    "all_int4": "qwen_flash_next_int4_int8_rc5_contrib_dflash2",
    "rc5_all_int4": "qwen_flash_next_int4_int8_rc5_contrib_dflash2",
    "top16_44_33_b64": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top16_44_33_b64",
    "top16": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top16_44_33_b64",
    "top08_44_33_b64": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top08_44_33_b64",
    "top08": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top08_44_33_b64",
    "2b375_x02_y14_z32_dp23_b64": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x02_y14_z32_dp23_b64",
    "2b375": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x02_y14_z32_dp23_b64",
    "2b375_x00_y18_z30_dp23_b64": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x00_y18_z30_dp23_b64",
    "3b0_x06_y36_z06_dp3_b64": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_3b0_x06_y36_z06_dp3_b64",
    "2b8_x04_y30_z14_dp3_b64": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b8_x04_y30_z14_dp3_b64",
    "2b5_x02_y20_z26_dp3_b64": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b5_x02_y20_z26_dp3_b64",
    "2b25_x00_y12_z36_dp2_b64": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b25_x00_y12_z36_dp2_b64",
    "2b125_x00_y06_z42_dp2_b64": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b125_x00_y06_z42_dp2_b64",
    "2b0_x00_y00_z48_b64": "qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b0_x00_y00_z48_b64",
    "rc6_all_int4": "qwen_flash_next_rc6_all_int4_fp32",
    "rc6_all_int4_fp32": "qwen_flash_next_rc6_all_int4_fp32",
    "qwen_flash_next_rc6_all_int4": "qwen_flash_next_rc6_all_int4_fp32",
    "rc6_all_int4_fp32_fp16kv_v2": "qwen_flash_next_rc6_all_int4_fp32",
    "qwen_flash_next_rc6_all_int4_fp32_fp16kv_v2": "qwen_flash_next_rc6_all_int4_fp32",
    "rc6_top16": "qwen_flash_next_rc6_top16_fp32",
    "rc6_top16_fp32": "qwen_flash_next_rc6_top16_fp32",
    "qwen_flash_next_rc6_top16": "qwen_flash_next_rc6_top16_fp32",
    "rc6_top16_fp32_fp16kv_v2": "qwen_flash_next_rc6_top16_fp32",
    "qwen_flash_next_rc6_top16_fp32_fp16kv_v2": "qwen_flash_next_rc6_top16_fp32",
    "rc6_2b375": "qwen_flash_next_rc6_2b375_fp32",
    "rc6_2b375_fp32": "qwen_flash_next_rc6_2b375_fp32",
    "qwen_flash_next_rc6_2b375": "qwen_flash_next_rc6_2b375_fp32",
    "rc6_2b375_fp32_fp16kv_v2": "qwen_flash_next_rc6_2b375_fp32",
    "qwen_flash_next_rc6_2b375_fp32_fp16kv_v2": "qwen_flash_next_rc6_2b375_fp32",
}

# Exact block counts at block_size=256 with 13 reserved DFlash2 blocks (dflash2_max_snapshots=1):
#   32K  (33,024 tokens) = 129 + 13 = 142 blocks (kv_capacity = 33,032)
#   128K (131,072 tokens) = 512 + 13 = 525 blocks (kv_capacity = 131,080)
#   256K (262,144 tokens) = 1024 + 13 = 1037 blocks (kv_capacity = 262,152)
DFLASH2_RESERVED_BLOCKS = 13
NUM_BLOCKS_32K = 142
NUM_BLOCKS_128K = 525
NUM_BLOCKS_256K = 1037

INT3_DEQUANT_SESSION_OPTIONS = {
    "ep.cuda.qmoe_int_dequant_max_scratch_bytes": "2147483648",
    "ep.cuda.qmoe_row_tile_size": "8192",
}
RC5_PREFILL_CONFIG = {"chunk": 8192, "warmup_lengths": [8192, 2048, 512, 128, 37]}


def resolve_preset_spec(name: str) -> dict:
    canonical = PRESET_ALIASES.get(name, name)
    if canonical not in PRESETS:
        known = sorted(set(PRESETS) | set(PRESET_ALIASES))
        raise ValueError(f"Unknown preset {name!r}; choose from {known}")
    return dict(PRESETS[canonical])


def solve_optimal_partition_dp(
    cost_4: list[float] | tuple[float, ...],
    cost_3: list[float] | tuple[float, ...],
    cost_2: list[float] | tuple[float, ...],
    x: int,
    y: int,
    z: int,
) -> tuple[list[int], list[int], list[int]]:
    """Find exact partition (X, Y, Z) of {0..n-1} with |X|=x, |Y|=y, |Z|=z
    minimizing sum_{i in X} cost_4[i] + sum_{i in Y} cost_3[i] + sum_{i in Z} cost_2[i].
    """
    n = len(cost_4)
    if len(cost_3) != n or len(cost_2) != n:
        raise ValueError(f"cost arrays must have the same length, got {n}, {len(cost_3)}, {len(cost_2)}")
    if x < 0 or y < 0 or z < 0 or x + y + z != n:
        raise ValueError(f"Invalid QMoE layer counts (x={x}, y={y}, z={z}) for {n} layers (must sum to {n})")

    inf = 1e30
    dp = np.full((n + 1, x + 1, z + 1), inf, dtype=np.float64)
    choice = np.zeros((n + 1, x + 1, z + 1), dtype=np.int8)
    dp[0, 0, 0] = 0.0

    for i in range(n):
        c4, c3, c2 = float(cost_4[i]), float(cost_3[i]), float(cost_2[i])
        for cx in range(min(i, x) + 1):
            for cz in range(min(i - cx, z) + 1):
                cur = dp[i, cx, cz]
                if cur >= inf * 0.5:
                    continue
                cy = i - cx - cz
                if cy + 1 <= y and cur + c3 < dp[i + 1, cx, cz]:
                    dp[i + 1, cx, cz] = cur + c3
                    choice[i + 1, cx, cz] = 3
                if cx + 1 <= x and cur + c4 < dp[i + 1, cx + 1, cz]:
                    dp[i + 1, cx + 1, cz] = cur + c4
                    choice[i + 1, cx + 1, cz] = 4
                if cz + 1 <= z and cur + c2 < dp[i + 1, cx, cz + 1]:
                    dp[i + 1, cx, cz + 1] = cur + c2
                    choice[i + 1, cx, cz + 1] = 2

    layers_4: list[int] = []
    layers_3: list[int] = []
    layers_2: list[int] = []
    cx, cz = x, z
    for i in range(n, 0, -1):
        ch = int(choice[i, cx, cz])
        layer = i - 1
        if ch == 4:
            layers_4.append(layer)
            cx -= 1
        elif ch == 2:
            layers_2.append(layer)
            cz -= 1
        else:
            layers_3.append(layer)
    return sorted(layers_4), sorted(layers_3), sorted(layers_2)


def _load_kl_tables(
    num_layers: int = 48,
    study_json: Path | None = None,
    *,
    fp32_gdn: bool = False,
) -> dict[str, list[float]]:
    if study_json is not None:
        study = json.loads(Path(study_json).read_text())
        if "s4_to_3" in study:
            c3_b64 = [float(study["s4_to_3"][i]["kl_mean"]) for i in range(num_layers)]
            c3_b32 = list(c3_b64)
            c2_b64 = [float(study["s4_to_2"][i]["kl_mean"]) for i in range(num_layers)]
            c2_b32 = list(c2_b64)
            kl3_ref = float(study["uniform_results"]["uniform_int3_b64"]["kl_mean"])
            kl2_ref = float(study["uniform_results"]["uniform_int2_b64"]["kl_mean"])
            m3_4 = [float(study["s3_to_4"][i]["kl_mean"]) - kl3_ref for i in range(num_layers)]
            m3_2_b64 = [float(study["s3_to_2"][i]["kl_mean"]) - kl3_ref for i in range(num_layers)]
            m3_2_b32 = list(m3_2_b64)
            m2_4 = [float(study["s2_to_4"][i]["kl_mean"]) - kl2_ref for i in range(num_layers)]
            m2_3 = [float(study["s2_to_3"][i]["kl_mean"]) - kl2_ref for i in range(num_layers)]
            return {
                "c3_b64": c3_b64,
                "c3_b32": c3_b32,
                "c2_b64": c2_b64,
                "c2_b32": c2_b32,
                "m3_4": m3_4,
                "m3_2_b64": m3_2_b64,
                "m3_2_b32": m3_2_b32,
                "m2_4": m2_4,
                "m2_3": m2_3,
            }
        c3_b64 = [float(study["single_from_int4"]["int3_b64"][i]["kl_mean"]) for i in range(num_layers)]
        c3_b32 = [float(study["single_from_int4"]["int3_b32"][i]["kl_mean"]) for i in range(num_layers)]
        c2_b64 = [float(study["single_from_int4"]["int2_b64"][i]["kl_mean"]) for i in range(num_layers)]
        c2_b32 = [float(study["single_from_int4"]["int2_b32"][i]["kl_mean"]) for i in range(num_layers)]
        kl3_ref = float(study["uniform_results"]["uniform_int3_b64"]["kl_mean"])
        m3_4 = [float(study["single_from_int3_b64"]["to_int4_b32"][i]["kl_mean"]) - kl3_ref for i in range(num_layers)]
        m3_2_b64 = [
            float(study["single_from_int3_b64"]["to_int2_b64"][i]["kl_mean"]) - kl3_ref for i in range(num_layers)
        ]
        m3_2_b32 = [
            float(study["single_from_int3_b64"]["to_int2_b32"][i]["kl_mean"]) - kl3_ref for i in range(num_layers)
        ]
        low_path = Path(study_json).parent / "qmoe_2b0_2b5_kl_study_results.json"
        if "single_from_int2_b64" in study:
            s2 = study["single_from_int2_b64"]
            kl2_ref = float(study["uniform_results"]["uniform_int2_b64"]["kl_mean"])
            m2_4 = [float(s2["to_int4_b32"][i]["kl_mean"]) - kl2_ref for i in range(num_layers)]
            m2_3 = [float(s2["to_int3_b64"][i]["kl_mean"]) - kl2_ref for i in range(num_layers)]
        elif low_path.exists():
            low_study = json.loads(low_path.read_text())
            s2 = low_study["single_from_int2_b64"]
            kl2_ref = float(study["uniform_results"]["uniform_int2_b64"]["kl_mean"])
            m2_4 = [float(s2["to_int4_b32"][i]["kl_mean"]) - kl2_ref for i in range(num_layers)]
            m2_3 = [float(s2["to_int3_b64"][i]["kl_mean"]) - kl2_ref for i in range(num_layers)]
        else:
            m2_4 = list(KL_MARGINAL_2_TO_4_B32[:num_layers])
            m2_3 = list(KL_MARGINAL_2_TO_3_B64[:num_layers])
        return {
            "c3_b64": c3_b64,
            "c3_b32": c3_b32,
            "c2_b64": c2_b64,
            "c2_b32": c2_b32,
            "m3_4": m3_4,
            "m3_2_b64": m3_2_b64,
            "m3_2_b32": m3_2_b32,
            "m2_4": m2_4,
            "m2_3": m2_3,
        }

    def _fit(seq: tuple[float, ...]) -> list[float]:
        if num_layers <= len(seq):
            return list(seq[:num_layers])
        return [seq[i % len(seq)] for i in range(num_layers)]

    if fp32_gdn:
        return {
            "c3_b64": _fit(KL_SINGLE_INT3_B64_FP32),
            "c3_b32": _fit(KL_SINGLE_INT3_B64_FP32),
            "c2_b64": _fit(KL_SINGLE_INT2_B64_FP32),
            "c2_b32": _fit(KL_SINGLE_INT2_B64_FP32),
            "m3_4": _fit(KL_MARGINAL_3_TO_4_B32_FP32),
            "m3_2_b64": _fit(KL_MARGINAL_3_TO_2_B64_FP32),
            "m3_2_b32": _fit(KL_MARGINAL_3_TO_2_B64_FP32),
            "m2_4": _fit(KL_MARGINAL_2_TO_4_B32_FP32),
            "m2_3": _fit(KL_MARGINAL_2_TO_3_B64_FP32),
        }

    return {
        "c3_b64": _fit(KL_SINGLE_INT3_B64),
        "c3_b32": _fit(KL_SINGLE_INT3_B32),
        "c2_b64": _fit(KL_SINGLE_INT2_B64),
        "c2_b32": _fit(KL_SINGLE_INT2_B32),
        "m3_4": _fit(KL_MARGINAL_3_TO_4_B32),
        "m3_2_b64": _fit(KL_MARGINAL_3_TO_2_B64),
        "m3_2_b32": _fit(KL_MARGINAL_3_TO_2_B32),
        "m2_4": _fit(KL_MARGINAL_2_TO_4_B32),
        "m2_3": _fit(KL_MARGINAL_2_TO_3_B64),
    }


def select_qmoe_layers_by_kl(
    x: int,
    y: int,
    z: int,
    *,
    kl_mode: str = "auto",
    bs3: int = 64,
    bs2: int = 64,
    num_layers: int = 48,
    study_json: Path | None = None,
) -> tuple[list[int], list[int], list[int], str]:
    """Partition `num_layers` QMoE layers into `x` INT4, `y` INT3, and `z` INT2 layers using KL similarity/divergence.

    Returns `(int4_layers, int3_layers, int2_layers, resolved_kl_mode)`.
    """
    if kl_mode not in KL_MODES:
        raise ValueError(f"Unknown kl_mode {kl_mode!r}; expected one of {KL_MODES}")
    if x < 0 or y < 0 or z < 0 or x + y + z != num_layers:
        raise ValueError(
            f"QMoE layer counts must be non-negative and sum to {num_layers}, got x={x}, y={y}, z={z} (sum={x + y + z})"
        )

    use_fp32_gdn = kl_mode.endswith("_fp32")
    resolved = kl_mode
    if resolved in ("auto", "auto_fp32"):
        sfx = "_fp32" if use_fp32_gdn else ""
        if z == 0:
            resolved = "rank" + sfx
        elif z < max(1, int(0.6 * num_layers)):
            resolved = "dp3" + sfx
        elif z < max(2, int(0.82 * num_layers)):
            resolved = "dp23" + sfx
        else:
            resolved = "dp2" + sfx

    base_mode = resolved[:-5] if resolved.endswith("_fp32") else resolved
    if base_mode == "rank":
        ref_ranked = RANKED_SENSITIVE_LAYERS_FP32 if use_fp32_gdn else RANKED_SENSITIVE_LAYERS
        if num_layers == len(ref_ranked) and study_json is None:
            ranked = list(ref_ranked)
        else:
            tables = _load_kl_tables(num_layers=num_layers, study_json=study_json, fp32_gdn=use_fp32_gdn)
            key = "c2_b64" if z > 0 else ("c3_b32" if bs3 == 32 else "c3_b64")
            ranked = [i for i, _ in sorted(enumerate(tables[key]), key=lambda t: t[1], reverse=True)]
        layers_4 = sorted(ranked[:x])
        layers_3 = sorted(ranked[x : x + y])
        layers_2 = sorted(ranked[x + y :])
        return layers_4, layers_3, layers_2, resolved

    tables = _load_kl_tables(num_layers=num_layers, study_json=study_json, fp32_gdn=use_fp32_gdn)
    zeros = [0.0] * num_layers
    if base_mode == "dp4":
        c3 = tables["c3_b32"] if bs3 == 32 else tables["c3_b64"]
        c2 = tables["c2_b32"] if bs2 == 32 else tables["c2_b64"]
        layers_4, layers_3, layers_2 = solve_optimal_partition_dp(zeros, c3, c2, x, y, z)
    elif base_mode == "dp3":
        m3_2 = tables["m3_2_b32"] if bs2 == 32 else tables["m3_2_b64"]
        layers_4, layers_3, layers_2 = solve_optimal_partition_dp(tables["m3_4"], zeros, m3_2, x, y, z)
    elif base_mode == "dp2":
        layers_4, layers_3, layers_2 = solve_optimal_partition_dp(tables["m2_4"], tables["m2_3"], zeros, x, y, z)
    elif base_mode == "dp23":
        m3_4 = tables["m3_4"]
        m3_2 = tables["m3_2_b32"] if bs2 == 32 else tables["m3_2_b64"]
        m2_4 = tables["m2_4"]
        m2_3 = tables["m2_3"]
        if use_fp32_gdn and (y + z) > 0:
            w2 = float(z) / float(y + z)
            w3 = float(y) / float(y + z)
        else:
            w2 = 0.5
            w3 = 0.5
        b23_4 = [w3 * m3_4[i] + w2 * (m2_4[i] - m2_3[i]) for i in range(num_layers)]
        b23_2 = [w3 * m3_2[i] + w2 * (0.0 - m2_3[i]) for i in range(num_layers)]
        layers_4, layers_3, layers_2 = solve_optimal_partition_dp(b23_4, zeros, b23_2, x, y, z)
    else:
        raise ValueError(f"Unhandled kl_mode: {resolved}")
    return layers_4, layers_3, layers_2, resolved


def compute_size_info(
    x: int,
    y: int,
    z: int,
    bs4: int = 32,
    bs3: int = 64,
    bs2: int = 64,
    num_layers: int = 48,
) -> dict:
    if x + y + z != num_layers:
        raise ValueError(f"x + y + z must equal {num_layers}, got {x + y + z}")
    num_experts = 512
    fc1_n, fc1_k = 1280, 2560
    fc2_n, fc2_k = 2560, 640
    weights_per_layer = num_experts * (fc1_n * fc1_k + fc2_n * fc2_k)
    w_bytes = weights_per_layer * (4 * x + 3 * y + 2 * z) // 8
    s_elems_base = num_experts * (fc1_n * fc1_k + fc2_n * fc2_k)
    s_bytes = x * (s_elems_base // bs4) * 2 + y * (s_elems_base // bs3) * 2 + z * (s_elems_base // bs2) * 2
    qmoe_bytes = w_bytes + s_bytes
    non_qmoe_bytes = 58_016_268_944
    total_bytes = qmoe_bytes + non_qmoe_bytes
    int3_b32_qmoe_bytes = num_layers * (weights_per_layer * 3 // 8 + (s_elems_base // 32) * 2)
    int3_b64_qmoe_bytes = num_layers * (weights_per_layer * 3 // 8 + (s_elems_base // 64) * 2)
    return {
        "x_int4": x,
        "y_int3": y,
        "z_int2": z,
        "bit_sum": 4 * x + 3 * y + 2 * z,
        "avg_bits": round((4 * x + 3 * y + 2 * z) / float(num_layers), 4),
        "bs4": bs4,
        "bs3": bs3,
        "bs2": bs2,
        "qmoe_weights_gib": round(w_bytes / 2**30, 3),
        "qmoe_scales_gib": round(s_bytes / 2**30, 3),
        "qmoe_total_gib": round(qmoe_bytes / 2**30, 3),
        "total_pkg_gib": round(total_bytes / 2**30, 3),
        "saved_gib_vs_int3_b32": round((int3_b32_qmoe_bytes - qmoe_bytes) / 2**30, 3),
        "saved_gib_vs_int3_b64": round((int3_b64_qmoe_bytes - qmoe_bytes) / 2**30, 3),
    }


# ---------------------------------------------------------------------------
# Weight Quantization, SM80 Prepacking, PLE Shards & External Data Writers
# ---------------------------------------------------------------------------


def _preprocess_weights_for_mixed_gemm_torch(tensor: torch.Tensor, bits: int, sm: int = 80) -> torch.Tensor:
    bits_a = 16
    bits_b = 4 if bits == 4 else 8
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)

    permutation_map = {
        "16_8": [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15],
        "16_4": [
            0,
            1,
            8,
            9,
            16,
            17,
            24,
            25,
            2,
            3,
            10,
            11,
            18,
            19,
            26,
            27,
            4,
            5,
            12,
            13,
            20,
            21,
            28,
            29,
            6,
            7,
            14,
            15,
            22,
            23,
            30,
            31,
        ],
    }
    mma_shape_n = 8
    b_rows_per_mma = 8 * 16 // bits_b

    num_experts, num_rows, num_cols = tensor.shape[0], tensor.shape[1], tensor.shape[2]
    if num_rows % b_rows_per_mma != 0 or num_cols % mma_shape_n != 0:
        raise ValueError(f"Incompatible shape ({num_rows}, {num_cols}) for bits={bits_b}")

    if sm < 100:
        pmap = permutation_map[f"{bits_a}_{bits_b}"]
        row_idx = [(r // b_rows_per_mma) * b_rows_per_mma + pmap[r % b_rows_per_mma] for r in range(num_rows)]
        tensor = tensor[:, row_idx, :]

    original_shape = tensor.shape
    if bits_b == 4:
        u = tensor.view(torch.uint8)
        high = (u >> 4).permute(0, 2, 1).unsqueeze(2)
        low = ((u << 4) >> 4).permute(0, 2, 1).unsqueeze(2)
        merged = torch.cat([low, high], dim=2).reshape(u.shape[0], -1, u.shape[1])
        merged = merged[:, :, 0::2] + merged[:, :, 1::2] * 16
        tensor = merged.view(torch.int8).reshape(original_shape)
    else:
        tensor = tensor.permute(0, 2, 1).reshape(original_shape)

    interleave = bits_a // bits_b
    if interleave > 1 and sm < 90:
        rows_per_tile = 128 * 8 // bits_a
        elts_in_int32 = 32 // bits_b
        tensor = tensor.reshape(
            num_experts, -1, interleave, num_rows // rows_per_tile, rows_per_tile * 4 // elts_in_int32
        )
        tensor = tensor.permute(0, 1, 3, 2, 4).reshape(original_shape)

    if bits_b == 8:
        t = tensor.to(torch.int16)
        t += -256 * (t > 127).to(torch.int16) + 128
        t = t.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(original_shape)
        tensor = t.to(torch.uint8).view(torch.int8)
    else:
        u = tensor.view(torch.uint8)
        high = (u >> 4).unsqueeze(-1)
        low = ((u << 4) >> 4).unsqueeze(-1)
        merged = torch.cat([low, high], dim=-1).reshape(u.shape[0], u.shape[1], -1)
        merged = merged.reshape(-1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]].reshape(merged.shape)
        merged = merged.to(torch.int16)
        merged += -16 * (merged > 7).to(torch.int16) + 8
        merged = merged[:, :, 0::2] + merged[:, :, 1::2] * 16
        tensor = merged.to(torch.uint8).view(torch.int8)

    return tensor.contiguous()


@torch.no_grad()
def fast_quant_prepack_batch(
    weights_bf16: torch.Tensor,
    bits: int,
    block_size: int = 32,
    is_qmoe: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact SM80 CutlassFpAIntB / QMoE blockwise symmetric quantizer + prepacker."""
    orig_shape = weights_bf16.shape
    if weights_bf16.ndim == 2:
        weights_bf16 = weights_bf16.unsqueeze(0)
    num_experts, rows, cols = weights_bf16.shape
    if cols % block_size != 0 or cols % 64 != 0:
        raise ValueError(f"Invalid K={cols} for block_size={block_size}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = weights_bf16.to(device=dev, dtype=torch.bfloat16)

    qmax = float((1 << (bits - 1)) - 1)
    qmin = -qmax if is_qmoe else -float(1 << (bits - 1))

    w_f32 = w.float().reshape(num_experts, rows, cols // block_size, block_size)
    amax = w_f32.abs().amax(dim=-1)
    scales_bf16 = (amax / qmax).to(torch.bfloat16)
    scales_f32 = scales_bf16.float().unsqueeze(-1)
    q = torch.where(
        scales_f32 != 0,
        torch.round(w_f32 / scales_f32),
        torch.zeros_like(w_f32),
    )
    q_signed = q.clamp(qmin, qmax).to(torch.int16).reshape(num_experts, rows, cols)

    signed_t = q_signed.permute(0, 2, 1).contiguous()
    if bits == 4:
        packed_t = (
            ((signed_t[:, :, 0::2] & 0x0F) | ((signed_t[:, :, 1::2] & 0x0F) << 4)).to(torch.uint8).view(torch.int8)
        )
    else:
        packed_t = signed_t.to(torch.uint8).view(torch.int8)

    out_b = _preprocess_weights_for_mixed_gemm_torch(packed_t, bits=bits, sm=80).view(torch.uint8)
    packed_cols = rows * bits // 8

    if is_qmoe:
        out_b = out_b.reshape(num_experts, cols, packed_cols)
        return out_b.cpu(), scales_bf16.cpu()
    out_b = out_b.reshape(num_experts, rows, cols // 32, 32 * bits // 8)
    if len(orig_shape) == 2:
        out_b = out_b.squeeze(0)
        scales_bf16 = scales_bf16.squeeze(0)
    return out_b.cpu(), scales_bf16.cpu()


@torch.no_grad()
def prepack_qmoe_sm80_torch(q_signed: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack signed integer tensor `[E, N, K]` (`q_signed` in `[-2^(b-1), 2^(b-1)-1]`)
    into the SM80 CUTLASS `fpA_intB` layout `[E, K, N // (8 // bits)]` `uint8`.
    """
    if bits not in (2, 4):
        raise ValueError(f"prepack_qmoe_sm80_torch supports bits in (2, 4), got {bits}")
    num_experts, rows, cols = q_signed.shape
    pack_factor = 8 // bits
    if rows % 64 != 0 or cols % 64 != 0:
        raise ValueError(f"N ({rows}) and K ({cols}) must be multiples of 64")

    w_ekn = q_signed.permute(0, 2, 1).contiguous()
    # fmt: off
    if bits == 2:
        row_perm = torch.tensor(
            [
                0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57,
                2, 3, 10, 11, 18, 19, 26, 27, 34, 35, 42, 43, 50, 51, 58, 59,
                4, 5, 12, 13, 20, 21, 28, 29, 36, 37, 44, 45, 52, 53, 60, 61,
                6, 7, 14, 15, 22, 23, 30, 31, 38, 39, 46, 47, 54, 55, 62, 63,
            ],
            device=q_signed.device,
            dtype=torch.long,
        )
    else:
        row_perm = torch.tensor(
            [
                0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27,
                4, 5, 12, 13, 20, 21, 28, 29, 6, 7, 14, 15, 22, 23, 30, 31,
                32, 33, 40, 41, 48, 49, 56, 57, 34, 35, 42, 43, 50, 51, 58, 59,
                36, 37, 44, 45, 52, 53, 60, 61, 38, 39, 46, 47, 54, 55, 62, 63,
            ],
            device=q_signed.device,
            dtype=torch.long,
        )
    # fmt: on

    base = torch.arange(cols // 64, device=q_signed.device, dtype=torch.long).unsqueeze(1) * 64
    row_idx = (base + row_perm.unsqueeze(0)).reshape(-1)
    w_perm = w_ekn.index_select(1, row_idx)
    w_enk = w_perm.permute(0, 2, 1).contiguous()

    if bits == 2:
        u = (w_enk.to(torch.int16) + 2).to(torch.uint8)
        u16 = u.reshape(-1, 16)[:, [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]].reshape(
            num_experts, rows, cols // 4, 4
        )
        packed = (u16[..., 0] | (u16[..., 1] << 2) | (u16[..., 2] << 4) | (u16[..., 3] << 6)).reshape(
            num_experts, cols, rows // 4
        )
        interleave = 8
    else:
        u = (w_enk.to(torch.int16) + 8).to(torch.uint8)
        u8 = u.reshape(-1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]].reshape(num_experts, rows, cols // 2, 2)
        packed = (u8[..., 0] | (u8[..., 1] << 4)).reshape(num_experts, cols, rows // 2)
        interleave = 4

    rows_per_tile = 64
    bytes_per_tile = rows_per_tile // pack_factor
    packed = packed.reshape(num_experts, rows // interleave, interleave, cols // rows_per_tile, bytes_per_tile)
    return packed.permute(0, 1, 3, 2, 4).contiguous().reshape(num_experts, cols, rows // pack_factor)


@torch.no_grad()
def unpack_raw_int3_to_signed(packed_u8: torch.Tensor, num_experts: int, rows: int, cols: int) -> torch.Tensor:
    """Unpack raw INT3 `[E, N, K * 3 // 8]` `uint8` into signed `int16` `[E, N, K]` in `[-4, +3]`."""
    b = packed_u8.reshape(num_experts, rows, cols // 8, 3).to(torch.int32)
    w24 = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    shifts = torch.tensor([0, 3, 6, 9, 12, 15, 18, 21], device=packed_u8.device, dtype=torch.int32)
    codes = (w24.unsqueeze(-1) >> shifts) & 0x7
    return (codes - 4).to(torch.int16).reshape(num_experts, rows, cols)


@torch.no_grad()
def unpack_sm80_int2_to_signed(packed_u8: torch.Tensor, num_experts: int, rows: int, cols: int) -> torch.Tensor:
    """Exact inverse of `prepack_qmoe_sm80_torch(q_signed, bits=2)`."""
    pack_factor = 4
    interleave = 8
    rows_per_tile = 64
    bytes_per_tile = rows_per_tile // pack_factor

    p = packed_u8.reshape(num_experts, rows // interleave, cols // rows_per_tile, interleave, bytes_per_tile)
    p = p.permute(0, 1, 3, 2, 4).contiguous().reshape(num_experts, rows, cols // 4)
    u16_perm = torch.stack([p & 0x3, (p >> 2) & 0x3, (p >> 4) & 0x3, (p >> 6) & 0x3], dim=-1).reshape(-1, 16)
    inv_col = torch.tensor(
        [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15], device=packed_u8.device, dtype=torch.long
    )
    u = u16_perm[:, inv_col].reshape(num_experts, rows, cols)
    w_perm = (u.to(torch.int16) - 2).permute(0, 2, 1).contiguous()
    # fmt: off
    row_perm = torch.tensor(
        [
            0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57,
            2, 3, 10, 11, 18, 19, 26, 27, 34, 35, 42, 43, 50, 51, 58, 59,
            4, 5, 12, 13, 20, 21, 28, 29, 36, 37, 44, 45, 52, 53, 60, 61,
            6, 7, 14, 15, 22, 23, 30, 31, 38, 39, 46, 47, 54, 55, 62, 63,
        ],
        device=packed_u8.device,
        dtype=torch.long,
    )
    # fmt: on
    inv_row_perm = torch.argsort(row_perm)
    base = torch.arange(cols // 64, device=packed_u8.device, dtype=torch.long).unsqueeze(1) * 64
    inv_row_idx = (base + inv_row_perm.unsqueeze(0)).reshape(-1)
    return w_perm.index_select(1, inv_row_idx).permute(0, 2, 1).contiguous()


@torch.no_grad()
def quantize_and_pack_raw_qmoe(
    w_bf16: torch.Tensor,
    bits: int,
    block_size: int = 32,
    mode: str = "mse",
    sm80_prepack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `w_bf16` `[E, N, K]` into `(packed_weights, scales_bf16)`."""
    num_experts, rows, cols = w_bf16.shape
    if cols % block_size != 0 or cols % 8 != 0:
        raise ValueError(f"K ({cols}) must be divisible by block_size ({block_size}) and 8")
    wf = w_bf16.float().reshape(num_experts, rows, cols // block_size, block_size)

    if mode == "sym_rtn":
        qmax = float((1 << (bits - 1)) - 1)
        qmin = -qmax
        amax = wf.abs().amax(dim=-1)
        scales_bf16 = (amax / qmax).to(torch.bfloat16)
    elif mode in ("asym_rtn", "mse"):
        qmin = -float(1 << (bits - 1))
        qmax = float((1 << (bits - 1)) - 1)
        neg_peak = (-wf).clamp(min=0.0).amax(dim=-1)
        pos_peak = wf.clamp(min=0.0).amax(dim=-1)
        raw_scale = torch.maximum(neg_peak / (-qmin), pos_peak / qmax)
        scales_bf16 = raw_scale.to(torch.bfloat16)
        if mode == "mse":
            best_scales = scales_bf16.clone()
            s_f32 = best_scales.float().unsqueeze(-1)
            q_init = torch.where(s_f32 > 0, torch.round(wf / s_f32), torch.zeros_like(wf)).clamp(qmin, qmax)
            best_err = ((wf - q_init * s_f32) ** 2).sum(dim=-1)
            for ratio in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.02, 1.05):
                cand_s = (raw_scale * ratio).to(torch.bfloat16)
                cs_f32 = cand_s.float().unsqueeze(-1)
                q_c = torch.where(cs_f32 > 0, torch.round(wf / cs_f32), torch.zeros_like(wf)).clamp(qmin, qmax)
                err = ((wf - q_c * cs_f32) ** 2).sum(dim=-1)
                improved = err < best_err
                best_err = torch.where(improved, err, best_err)
                best_scales = torch.where(improved, cand_s, best_scales)
            scales_bf16 = best_scales
    else:
        raise ValueError(f"Unknown mode: {mode}")

    s_f32 = scales_bf16.float().unsqueeze(-1)
    q = torch.where(s_f32 > 0, torch.round(wf / s_f32), torch.zeros_like(wf)).clamp(qmin, qmax)
    if sm80_prepack:
        q_signed = q.to(torch.int8).reshape(num_experts, rows, cols)
        packed = prepack_qmoe_sm80_torch(q_signed, bits=bits)
        return packed.cpu(), scales_bf16.contiguous().cpu()

    zero_offset = 1 << (bits - 1)
    codes = (q.to(torch.int16) + zero_offset).to(torch.uint8).reshape(num_experts, rows, cols)
    if bits == 2:
        c4 = codes.reshape(num_experts, rows, cols // 4, 4)
        packed = (c4[..., 0] | (c4[..., 1] << 2) | (c4[..., 2] << 4) | (c4[..., 3] << 6)).contiguous()
    elif bits == 4:
        c2 = codes.reshape(num_experts, rows, cols // 2, 2)
        packed = (c2[..., 0] | (c2[..., 1] << 4)).contiguous()
    elif bits == 3:
        c8 = codes.reshape(num_experts, rows, cols // 8, 8).to(torch.int32)
        w24 = (
            c8[..., 0]
            | (c8[..., 1] << 3)
            | (c8[..., 2] << 6)
            | (c8[..., 3] << 9)
            | (c8[..., 4] << 12)
            | (c8[..., 5] << 15)
            | (c8[..., 6] << 18)
            | (c8[..., 7] << 21)
        )
        b0 = (w24 & 0xFF).to(torch.uint8)
        b1 = ((w24 >> 8) & 0xFF).to(torch.uint8)
        b2 = ((w24 >> 16) & 0xFF).to(torch.uint8)
        packed = torch.stack([b0, b1, b2], dim=-1).reshape(num_experts, rows, cols * 3 // 8).contiguous()
    else:
        raise ValueError(f"Unsupported bits={bits}")

    return packed.cpu(), scales_bf16.contiguous().cpu()


def build_ple_int8(src_v12d_dir: Path, hf_dir: Path, out_dir: Path) -> dict:
    """Quantize FP8 PLE safetensors shards into 10 INT8 files (`ple/ple_int8_00..09.data`) and write `ple_table.txt`."""
    t0 = time.time()
    ple_dir = out_dir / "ple"
    ple_dir.mkdir(parents=True, exist_ok=True)

    src_lines = (src_v12d_dir / "ple_table.txt").read_text().splitlines()
    new_lines: list[str] = []
    shard_entries: list[tuple[str, int, int]] = []
    for line in src_lines:
        if line.startswith("scale "):
            new_lines.append(f"scale -{PLE_INT8_SCALE:.12f}")
        elif line.startswith("shard "):
            parts = line.split()
            shard_entries.append((parts[1], int(parts[2]), int(parts[3])))
        else:
            new_lines.append(line)

    if len(shard_entries) != 128:
        raise ValueError(f"Expected 128 PLE shards, got {len(shard_entries)}")

    total_bytes = 0
    file_ranges = [
        *((file_idx, file_idx * 13, (file_idx + 1) * 13) for file_idx in range(8)),
        *((8 + j, 104 + j * 12, 104 + (j + 1) * 12) for j in range(2)),
    ]
    for file_idx, start_idx, end_idx in file_ranges:
        rel_name = f"ple/ple_int8_{file_idx:02d}.data"
        cur_pos = 0
        with open(out_dir / rel_name, "wb") as cur_f:
            for idx in range(start_idx, end_idx):
                rel_path, offset, rows = shard_entries[idx]
                src_file = hf_dir / Path(rel_path).name
                if not src_file.exists():
                    src_file = src_v12d_dir / rel_path
                nbytes = rows * 160
                with open(src_file, "rb") as sf:
                    sf.seek(offset)
                    raw_fp8 = sf.read(nbytes)
                if len(raw_fp8) != nbytes:
                    raise RuntimeError(f"Short PLE read on {src_file}: got {len(raw_fp8)}, expected {nbytes}")
                u8 = torch.from_numpy(np.frombuffer(raw_fp8, dtype=np.uint8).copy())
                f32 = u8.view(torch.float8_e4m3fn).float()
                q_i8 = torch.clamp(torch.round(f32 / PLE_INT8_SCALE), -127.0, 127.0).to(torch.int8)
                out_bytes = q_i8.numpy().tobytes()

                aligned_off = align_up(cur_pos, ALIGN)
                if aligned_off > cur_pos:
                    cur_f.write(b"\x00" * (aligned_off - cur_pos))
                    cur_pos = aligned_off
                cur_f.write(out_bytes)
                new_lines.append(f"shard {rel_name} {cur_pos} {rows}")
                cur_pos += len(out_bytes)
                total_bytes += len(out_bytes)

    (out_dir / "ple_table.txt").write_text("\n".join(new_lines) + "\n")
    return {"shards": len(shard_entries), "files": 10, "bytes": total_bytes, "elapsed_s": round(time.time() - t0, 2)}


class ContribWeightWriter:
    """Appends converted tensors to `<out>/contrib_weights.data` at 4096-byte aligned offsets.

    With `reuse_data` the file starts as a hard link of an existing `contrib_weights.data` and new tensors are appended
    after its end in a private copy, so the reused package is never modified.
    """

    def __init__(self, src_dir: Path, out_dir: Path, reuse_data: Path | None = None) -> None:
        self.src_dir = src_dir
        self.out_dir = out_dir
        self.data_file_name = "contrib_weights.data"
        self.data_path = out_dir / self.data_file_name
        self.fh = None
        self.pos = 0
        self.closed = False
        self.cache: dict[tuple, dict] = {}
        self.data_path.unlink(missing_ok=True)
        if reuse_data is None:
            self.fh = open(self.data_path, "wb")  # noqa: SIM115
        else:
            os.link(reuse_data, self.data_path)
            self.pos = self.data_path.stat().st_size

    def close(self) -> None:
        self.closed = True
        if self.fh is not None:
            self.fh.close()
            self.fh = None
        if self.pos == 0 and self.data_path.exists():
            self.data_path.unlink()

    def _writable(self):
        if self.closed:
            raise ValueError("write to a closed ContribWeightWriter")
        if self.fh is None:
            private = self.data_path.with_name(self.data_file_name + ".tmp")
            shutil.copyfile(self.data_path, private)
            os.replace(private, self.data_path)
            self.fh = open(self.data_path, "ab")  # noqa: SIM115
            if self.fh.tell() != self.pos:
                raise RuntimeError(f"{self.data_path} has {self.fh.tell()} bytes, expected the reused {self.pos}")
        return self.fh

    def write_bytes(self, raw: bytes, dtype: int, dims: list[int]) -> dict:
        fh = self._writable()
        pad = align_up(self.pos) - self.pos
        if pad:
            fh.write(b"\x00" * pad)
            self.pos += pad
        off = self.pos
        fh.write(raw)
        self.pos += len(raw)
        return {
            "location": self.data_file_name,
            "offset": off,
            "length": len(raw),
            "dtype": dtype,
            "dims": list(dims),
        }


# ---------------------------------------------------------------------------
# RC5 Graph Surgery (SpeculativePrefixCommit, RouterGateUpProj, QMoE, Tiers)
# ---------------------------------------------------------------------------

SPC_NODE_NAME = "fold/SpeculativePrefixCommit"
SPC_OP_TYPE = "SpeculativePrefixCommit"
SPC_DOMAIN = "com.microsoft"
SPC_TAIL_DONE = "fold.tail_done"
SPC_CTL_IN = "fold.ctl"
SPC_CTX_COMMIT_IN = "fold.ctx_commit"
SPC_CTL_OUT = "fold.ctl_out"
SPC_CTX_COMMIT_OUT = "fold.ctx_commit_out"
SPC_REMOVED_GRAPH_INPUTS = ("fold.ctx_b",)
SPC_MANIFEST_VALUE = {
    "contract": "rc4 (ORT PR #33474)",
    "inputs": [
        "next_tokens",
        "input_ids",
        "validation.aggregate",
        SPC_CTL_IN,
        SPC_CTX_COMMIT_IN,
        "fold.ctx_a",
        "fold.ptrs",
    ],
    "outputs": [SPC_TAIL_DONE, SPC_CTL_OUT, SPC_CTX_COMMIT_OUT],
}


def ensure_rc5_speculative_prefix_commit(model: onnx.ModelProto, head_rows: int = 8) -> bool:
    """Ensure the graph uses the 7-input / 3-output `com.microsoft::SpeculativePrefixCommit` contract.
    Returns True if the node was upgraded, False if it was already on the RC4/RC5 contract or absent.
    """
    g = model.graph
    spc_nodes = [
        (idx, n)
        for idx, n in enumerate(g.node)
        if (n.op_type == SPC_OP_TYPE and n.domain == SPC_DOMAIN) or n.name == "fold/FoldMid2"
    ]
    if not spc_nodes:
        return False
    idx, old = spc_nodes[0]
    if (
        old.op_type == SPC_OP_TYPE
        and len(old.input) == 7
        and len(old.output) == 3
        and list(old.output) == [SPC_TAIL_DONE, SPC_CTL_OUT, SPC_CTX_COMMIT_OUT]
    ):
        return False

    attrs = {a.name: helper.get_attribute_value(a) for a in old.attribute}
    rows = int(attrs.get("head_rows", head_rows))
    src_inputs = list(old.input)[:8]
    keep_inputs = [src_inputs[i] for i in (0, 1, 2, 3, 4, 5, 7)]
    new_node = helper.make_node(
        SPC_OP_TYPE,
        keep_inputs,
        [SPC_TAIL_DONE, SPC_CTL_OUT, SPC_CTX_COMMIT_OUT],
        name=SPC_NODE_NAME,
        domain=SPC_DOMAIN,
        head_rows=rows,
    )
    g.node[idx].CopyFrom(new_node)

    in_by_name = {i.name: i for i in g.input}
    kept_in = [i for i in g.input if i.name not in SPC_REMOVED_GRAPH_INPUTS]
    if len(kept_in) != len(g.input):
        del g.input[:]
        g.input.extend(kept_in)

    outs = list(g.output)
    out_names = [o.name for o in outs]
    if SPC_TAIL_DONE in out_names:
        new_outs = []
        for out_name, in_name in ((SPC_CTL_OUT, SPC_CTL_IN), (SPC_CTX_COMMIT_OUT, SPC_CTX_COMMIT_IN)):
            if out_name not in out_names and in_name in in_by_name:
                vi = onnx.ValueInfoProto()
                vi.CopyFrom(in_by_name[in_name])
                vi.name = out_name
                new_outs.append(vi)
        if new_outs:
            pos = out_names.index(SPC_TAIL_DONE) + 1
            outs[pos:pos] = new_outs
            del g.output[:]
            g.output.extend(outs)
    return True


def fuse_router_gate_up_proj(model: onnx.ModelProto) -> int:
    """Fuse `MatMul + Split + SwiGLU` of each MoE layer's router/shared-expert prefix into `com.microsoft::RouterGateUpProj`.
    Idempotent: returns the number of newly fused layers (0 if already fused).
    """
    graph = model.graph
    consts = {i.name: i for i in graph.initializer}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for n in graph.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)
    producers = {o: n for n in graph.node for o in n.output}

    replaced = 0
    drop: set[int] = set()
    fused_at: dict[int, onnx.NodeProto] = {}
    for split in graph.node:
        if split.op_type != "Split" or not split.name.endswith("SplitRouterSwishMul/Split"):
            continue
        matmul = producers.get(split.input[0])
        if matmul is None or matmul.op_type != "MatMul" or len(consumers.get(split.input[0], [])) != 1:
            continue
        sizes_init = consts.get(split.input[1]) if len(split.input) > 1 else None
        if sizes_init is None:
            continue
        sizes = numpy_helper.to_array(sizes_init).tolist()
        if len(sizes) != 2 or sizes[1] % 2:
            continue
        router, _gate_up = sizes
        gate_up_users = consumers.get(split.output[1], [])
        if len(gate_up_users) != 1 or gate_up_users[0].op_type != "SwiGLU":
            continue
        swiglu = gate_up_users[0]
        if len(swiglu.input) > 1 and swiglu.input[1]:
            continue
        sw_attrs = {a.name: helper.get_attribute_value(a) for a in swiglu.attribute}
        fused = helper.make_node(
            "RouterGateUpProj",
            [matmul.input[0], matmul.input[1]],
            [split.output[0], swiglu.output[0]],
            name=matmul.name.replace("MatMulBf16", "RouterGateUpProj"),
            domain="com.microsoft",
            router_size=int(router),
            activation="swiglu",
            limit=float(sw_attrs.get("limit", 0.0)),
            activation_alpha=float(sw_attrs.get("activation_alpha", 1.0)),
            activation_beta=float(sw_attrs.get("activation_beta", 0.0)),
        )
        fused_at[id(matmul)] = fused
        drop.update([id(split), id(swiglu)])
        replaced += 1

    if replaced:
        nodes = [fused_at.get(id(n), n) for n in graph.node if id(n) not in drop]
        del graph.node[:]
        graph.node.extend(nodes)
    return replaced


# ---------------------------------------------------------------------------
# QMoE Offload Contract Checks, Popularity Profile & Runtime Profiles
# ---------------------------------------------------------------------------


def qmoe_nodes(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    return [n for n in model.graph.node if n.domain == "com.microsoft" and n.op_type == "QMoE"]


def qmoe_attribute(node: onnx.NodeProto, name: str, default):
    for attribute in node.attribute:
        if attribute.name == name:
            value = helper.get_attribute_value(attribute)
            return value.decode() if isinstance(value, bytes) else value
    return default


def node_label(name: str, ordinal: int) -> str:
    return repr(name) if name else f"<unnamed #{ordinal}>"


def parse_layer_index(name: str) -> int | None:
    """Mirrors `MoeOffloadSessionPolicy::TryExtractLayerIndexFromName`: digits after 'layers.', 'layer.' or 'blocks.'."""
    for prefix in ("layers.", "layer.", "blocks."):
        search_from = 0
        while search_from < len(name):
            position = name.find(prefix, search_from)
            if position < 0:
                break
            end = position + len(prefix)
            digits_end = end
            while digits_end < len(name) and name[digits_end] in "0123456789":
                digits_end += 1
            if digits_end > end:
                value = int(name[end:digits_end])
                if value > 2**31 - 1:
                    return None
                following = name[digits_end] if digits_end < len(name) else ""
                if not following or not (following.isascii() and following.isalnum()):
                    return value
            search_from = end
    return None


def qmoe_offload_problems(node: onnx.NodeProto, constants: dict[str, onnx.TensorProto]) -> list[str]:
    """List any reasons the ORT QMoE offload backend rejects `node` (empty list = eligible for offload)."""
    problems: list[str] = []

    def slot(index: int) -> str:
        return node.input[index] if index < len(node.input) else ""

    quant_type = qmoe_attribute(node, "quant_type", "int")
    bits = qmoe_attribute(node, "expert_weight_bits", 4)
    fc1_bits = qmoe_attribute(node, "fc1_expert_weight_bits", bits)
    fc2_bits = qmoe_attribute(node, "fc2_expert_weight_bits", bits)
    block_size = qmoe_attribute(node, "block_size", -1)
    prepacked = qmoe_attribute(node, "weights_prepacked", -1)
    nvfp4_row_major = quant_type == "nvfp4" and prepacked == 1
    int4_prepacked = quant_type == "int" and bits == fc1_bits == fc2_bits == 4 and prepacked != 0 and block_size == 32
    int3_raw = (
        quant_type == "int" and bits == fc1_bits == fc2_bits == 3 and prepacked == 0 and block_size in (32, 64, 128)
    )
    int2_prepacked = (
        quant_type == "int" and bits == fc1_bits == fc2_bits == 2 and prepacked != 0 and block_size in (32, 64, 128)
    )
    if quant_type == "nvfp4" and block_size not in (-1, 16):
        problems.append(f"block_size={block_size}: quant_type='nvfp4' takes block_size 16")
    if not (nvfp4_row_major or int4_prepacked or int3_raw or int2_prepacked):
        problems.append(
            f"quant_type={quant_type!r} expert_weight_bits={bits} block_size={block_size} weights_prepacked={prepacked} "
            "is not NVFP4 with weights_prepacked=1, INT4 with block_size=32 and prepacked weights, raw INT3, "
            "or prepacked INT2"
        )
    elif not (bits == 4 or int3_raw or int2_prepacked):
        problems.append(f"expert_weight_bits={bits} must be 4 for NVFP4")

    activation = qmoe_attribute(node, "activation_type", "relu")
    swiglu_fusion = qmoe_attribute(node, "swiglu_fusion", 0)
    if activation != "swiglu" or swiglu_fusion not in (0, 1):
        problems.append(
            f"activation_type={activation!r} swiglu_fusion={swiglu_fusion}: only swiglu with swiglu_fusion 0 or 1 is offloaded"
        )
    if qmoe_attribute(node, "use_sparse_mixer", 0) == 1:
        problems.append("use_sparse_mixer=1 is not offloaded")

    unsupported = [index for index in UNSUPPORTED_OFFLOAD_INPUTS if slot(index)]
    if unsupported:
        problems.append(f"inputs {unsupported} are not supported by the offload backend")

    not_constant = [index for index in OFFLOAD_WEIGHT_INPUTS if slot(index) not in constants]
    if not_constant:
        problems.append(f"inputs {not_constant} (expert weights and scales) must be constant graph initializers")
    else:
        for index in (2, 5):
            rank = len(constants[slot(index)].dims)
            if rank != 3:
                problems.append(f"input {index} has rank {rank}, expected 3 (experts x rows x columns)")
    if quant_type == "nvfp4":
        missing = [index for index in NVFP4_GLOBAL_SCALE_INPUTS if slot(index) not in constants]
        if missing:
            problems.append(
                f"NVFP4 needs the constant global scales {list(NVFP4_GLOBAL_SCALE_INPUTS)}, missing {missing}"
            )
    return problems


def prepare_qmoe_for_offload(model: onnx.ModelProto) -> dict:
    """Clear optional `router_weights` (input 14) on all `com.microsoft::QMoE` nodes and validate offload eligibility."""
    graph_inputs = {value.name for value in model.graph.input}
    constants = {
        init.name: init for init in model.graph.initializer if model.ir_version < 4 or init.name not in graph_inputs
    }
    nodes = qmoe_nodes(model)
    cleared = 0
    for node in nodes:
        if len(node.input) > ROUTER_WEIGHTS_INPUT and node.input[ROUTER_WEIGHTS_INPUT]:
            node.input[ROUTER_WEIGHTS_INPUT] = ""
            cleared += 1
        while len(node.input) > 7 and not node.input[-1]:
            node.input.pop()
    rejected = {}
    for ordinal, node in enumerate(nodes):
        problems = qmoe_offload_problems(node, constants)
        if problems:
            rejected[ordinal] = problems
    return {"nodes": [node.name for node in nodes], "router_weights_cleared": cleared, "rejected": rejected}


def load_expert_profile_plan(
    profile_path: Path,
    num_layers: int = 48,
    num_experts: int = 512,
    total_slots: int = 960,
    min_slots_per_layer: int = 4,
    uniform: bool = False,
) -> dict[int, list[int]]:
    """Load an STRP v1 `expert-profile.bin` and pick the hottest experts of every layer, `total_slots` in total."""
    if num_layers <= 0 or num_experts <= 0:
        raise ValueError(f"num_layers and num_experts must be positive, got {num_layers} and {num_experts}")
    if total_slots < 0 or min_slots_per_layer < 0:
        raise ValueError(
            f"total_slots and min_slots_per_layer must not be negative: {total_slots}, {min_slots_per_layer}"
        )

    data = profile_path.read_bytes()
    if len(data) < 24:
        raise ValueError(f"{profile_path} is {len(data)} bytes, shorter than the 24-byte STRP header")
    magic, version, hdr_layers, hdr_experts, _slots, n_ranked = struct.unpack_from("<IIIIII", data, 0)
    if magic != STRP_MAGIC or version != 1:
        raise ValueError(f"Invalid STRP header in {profile_path}: magic=0x{magic:08x}, version={version}")
    if hdr_layers != num_layers or hdr_experts != num_experts:
        raise ValueError(
            f"Profile {profile_path} is for ({hdr_layers} layers, {hdr_experts} experts), "
            f"expected ({num_layers}, {num_experts})"
        )
    # Each ranked entry is a (layer, expert) pair of uint16 values, as read by the QMoE offload planner.
    expected_bytes = 24 + n_ranked * 4
    if len(data) < expected_bytes:
        raise ValueError(
            f"Profile {profile_path} is truncated: {len(data)} bytes, header declares {n_ranked} entries ({expected_bytes} bytes)"
        )

    entries = np.frombuffer(data, dtype=np.uint16, count=n_ranked * 2, offset=24).reshape(n_ranked, 2)
    by_layer: dict[int, list[int]] = {i: [] for i in range(num_layers)}
    seen_in_layer: dict[int, set[int]] = {i: set() for i in range(num_layers)}
    ordered_unique: list[tuple[int, int]] = []
    for row in entries:
        lay, exp = int(row[0]), int(row[1])
        if not (0 <= lay < num_layers and 0 <= exp < num_experts):
            raise ValueError(f"Profile {profile_path} has out-of-range (layer={lay}, expert={exp})")
        if exp not in seen_in_layer[lay]:
            seen_in_layer[lay].add(exp)
            by_layer[lay].append(exp)
            ordered_unique.append((lay, exp))

    if total_slots > len(ordered_unique):
        raise ValueError(
            f"total_slots ({total_slots}) exceeds the {len(ordered_unique)} unique ranked experts in {profile_path}"
        )

    if uniform:
        base_per_layer = total_slots // num_layers
        rem = total_slots % num_layers
        for i in range(num_layers):
            if len(by_layer[i]) < base_per_layer:
                raise ValueError(
                    f"layer {i} only ranks {len(by_layer[i])} experts in {profile_path}, fewer than the uniform share {base_per_layer}"
                )
        rank_of: dict[tuple[int, int], int] = {pair: idx for idx, pair in enumerate(ordered_unique)}
        candidates = [
            (rank_of[(i, by_layer[i][base_per_layer])], i)
            for i in range(num_layers)
            if len(by_layer[i]) > base_per_layer
        ]
        if len(candidates) < rem:
            raise ValueError(
                f"only {len(candidates)} layers have an expert beyond the uniform share {base_per_layer}, cannot place {rem} remainder slots"
            )
        candidates.sort()
        extra_layers = {i for _, i in candidates[:rem]}
        plan = {i: sorted(by_layer[i][: base_per_layer + (1 if i in extra_layers else 0)]) for i in range(num_layers)}
        return plan

    selected: dict[int, set[int]] = {i: set() for i in range(num_layers)}
    floor = min(min_slots_per_layer, total_slots // num_layers)
    for i in range(num_layers):
        for exp in by_layer[i][:floor]:
            selected[i].add(exp)

    cur_count = sum(len(s) for s in selected.values())
    for lay, exp in ordered_unique:
        if cur_count >= total_slots:
            break
        if exp not in selected[lay]:
            selected[lay].add(exp)
            cur_count += 1
    if cur_count != total_slots:
        raise ValueError(f"Profile {profile_path} filled {cur_count} slots, expected {total_slots}")
    return {i: sorted(selected[i]) for i in range(num_layers)}


def format_node_overrides(caps: dict[str, int]) -> str:
    entries = []
    for name, cap in caps.items():
        if not name or name != name.strip(" \t\n\v\f\r") or ";" in name or ":" in name:
            raise ValueError(f"node name {name!r} cannot appear in session.moe.offload.node_overrides")
        if not 0 <= cap <= 2**31 - 1:
            raise ValueError(f"resident slot cap {cap} of node {name!r} is outside 0..{2**31 - 1}")
        entries.append(f"{name}:max_resident_slots={cap}")
    return ";".join(entries)


def offload_session_options(
    mode: str,
    placement: str = "adaptive",
    device_expert_bytes: str | None = None,
    host_expert_bytes: str | None = None,
    popularity_file: str | None = None,
    node_overrides: str | None = None,
) -> dict[str, str]:
    if mode not in OFFLOAD_MODES:
        raise ValueError(f"offload mode {mode!r} is not one of {OFFLOAD_MODES}")
    if placement not in OFFLOAD_PLACEMENTS:
        raise ValueError(f"offload placement {placement!r} is not one of {OFFLOAD_PLACEMENTS}")
    options = {"mode": mode, "placement": placement, "cpu_execution": "auto"}
    optional = {
        "device_expert_bytes": device_expert_bytes,
        "host_expert_bytes": host_expert_bytes,
        "popularity_file": popularity_file,
        "node_overrides": node_overrides,
    }
    options.update({key: value for key, value in optional.items() if value})
    return {OFFLOAD_OPTION_PREFIX + key: value for key, value in options.items()}


def estimate_peak_vram_and_thresholds(
    x_int4: int = 48,
    y_int3: int = 0,
    z_int2: int = 0,
    bs4: int = 32,
    bs3: int = 64,
    bs2: int = 64,
    num_experts: int = 512,
    hidden_size: int = 2560,
    inter_size: int = 640,
) -> dict:
    """Compute per-model QMoE size, empirically calibrated 128K/256K peak VRAM, and runtime-profile VRAM thresholds.

    Calibrated against exact H200 `cudaMemGetInfo` / `nvidia-smi` peak memory measurements across all 4 RC5 models:
      - `2b375`    (2/14/32, QMoE=37.06 GiB): 128K (num_blocks=525) peak = 50.304 GiB (54.013 GB / 51,511 MiB),
                                              256K (num_blocks=1037) peak = 52.003 GiB (55.838 GB / 53,251 MiB)
      - `top08`    (8/40/0,  QMoE=48.63 GiB): 128K (num_blocks=525) peak = 61.872 GiB (66.434 GB / 63,357 MiB),
                                              256K (num_blocks=1037) peak = 64.708 GiB (69.480 GB / 66,261 MiB)
      - `top16`    (16/32/0, QMoE=51.56 GiB): 128K (num_blocks=525) peak = 64.792 GiB (69.570 GB / 66,347 MiB),
                                              256K (num_blocks=1037) peak = 67.630 GiB (72.617 GB / 69,253 MiB)
      - `all_int4` (48/0/0,  QMoE=63.28 GiB): 128K (num_blocks=525) peak = 75.423 GiB (80.985 GB / 77,233 MiB),
                                              256K (num_blocks=1037) peak = 77.901 GiB (83.646 GB / 79,771 MiB)
    """
    elements_per_layer = num_experts * ((2 * inter_size) * hidden_size + hidden_size * inter_size)
    w_bytes = elements_per_layer * (4 * x_int4 + 3 * y_int3 + 2 * z_int2) // 8
    s_bytes = (
        elements_per_layer
        * 2
        * ((x_int4 // bs4 if bs4 else 0) + (y_int3 // bs3 if bs3 else 0) + (z_int2 // bs2 if bs2 else 0))
    )
    # Exact byte formula using integer division per layer
    l4_b = elements_per_layer * 4 // 8 + (elements_per_layer // bs4) * 2
    l3_b = elements_per_layer * 3 // 8 + (elements_per_layer // bs3) * 2
    l2_b = elements_per_layer * 2 // 8 + (elements_per_layer // bs2) * 2
    qmoe_bytes = x_int4 * l4_b + y_int3 * l3_b + z_int2 * l2_b
    _ = (w_bytes, s_bytes)

    overhead_128k = 14_220_000_000 if y_int3 > 0 else 13_037_000_000
    if y_int3 > 0 and z_int2 > 0:
        overhead_256k = 16_045_000_000
    elif y_int3 > 0:
        overhead_256k = 17_260_000_000
    else:
        overhead_256k = 15_698_000_000

    peak_128k_bytes = qmoe_bytes + overhead_128k
    peak_256k_bytes = qmoe_bytes + overhead_256k

    gb = 1_000_000_000
    min_vram_128k = ((peak_128k_bytes + 2 * gb + gb - 1) // gb) * gb
    min_vram_256k = max(min_vram_128k + 4 * gb, ((peak_256k_bytes + 2_500_000_000 + gb - 1) // gb) * gb)

    return {
        "qmoe_bytes": qmoe_bytes,
        "qmoe_gib": round(qmoe_bytes / 2**30, 3),
        "peak_128k_bytes": peak_128k_bytes,
        "peak_128k_gib": round(peak_128k_bytes / 2**30, 3),
        "peak_128k_gb": round(peak_128k_bytes / 1e9, 3),
        "peak_256k_bytes": peak_256k_bytes,
        "peak_256k_gib": round(peak_256k_bytes / 2**30, 3),
        "peak_256k_gb": round(peak_256k_bytes / 1e9, 3),
        "min_vram_128k_bytes": min_vram_128k,
        "min_vram_256k_bytes": min_vram_256k,
        "num_blocks_32k": NUM_BLOCKS_32K,
        "num_blocks_128k": NUM_BLOCKS_128K,
        "num_blocks_256k": NUM_BLOCKS_256K,
    }


def default_runtime_profiles(
    decoder_filename: str = "model.onnx",
    *,
    x_int4: int = 48,
    y_int3: int = 0,
    z_int2: int = 0,
    bs4: int = 32,
    bs3: int = 64,
    bs2: int = 64,
) -> list[dict]:
    """Hardware-matched `runtime_profiles` sizing `num_blocks` and VRAM thresholds to the model's footprint."""
    vram_info = estimate_peak_vram_and_thresholds(x_int4, y_int3, z_int2, bs4=bs4, bs3=bs3, bs2=bs2)
    min_128k = vram_info["min_vram_128k_bytes"]
    min_256k = vram_info["min_vram_256k_bytes"]
    return [
        {
            "id": "consumer-gpu-24g",
            "eligibility": {
                "minimum_total_device_memory_bytes": 21474836480,
                "maximum_total_device_memory_bytes": 26843545600,
                "minimum_effective_host_memory_bytes": 68719476736,
                "is_integrated": False,
            },
            "overlay": {
                "model": {
                    "decoder": {
                        "filename": decoder_filename,
                        "session_options": {
                            "session.moe.offload.mode": "required",
                            "session.moe.offload.placement": "adaptive",
                            "session.moe.offload.cpu_execution": "auto",
                            "session.moe.offload.device_expert_bytes": "10351538176",
                            "session.moe.offload.popularity_file": EXPERT_PROFILE_NAME,
                        },
                    }
                },
                "engine": {
                    "dynamic_batching": {
                        "num_blocks": NUM_BLOCKS_32K,
                        "max_batch_size": 1,
                        "max_scheduled_tokens": 8192,
                    }
                },
            },
        },
        {
            "id": "consumer-gpu-24g-host-64g",
            "eligibility": {
                "minimum_total_device_memory_bytes": 21474836480,
                "maximum_total_device_memory_bytes": 26843545600,
                "maximum_effective_host_memory_bytes": 68719476735,
                "is_integrated": False,
            },
            "overlay": {
                "model": {
                    "decoder": {
                        "filename": decoder_filename,
                        "session_options": {
                            "session.moe.offload.mode": "required",
                            "session.moe.offload.placement": "adaptive",
                            "session.moe.offload.cpu_execution": "auto",
                            "session.moe.offload.device_expert_bytes": "10351538176",
                            "session.moe.offload.host_expert_bytes": "17179869184",
                            "session.moe.offload.popularity_file": EXPERT_PROFILE_NAME,
                        },
                    }
                },
                "engine": {
                    "dynamic_batching": {
                        "num_blocks": NUM_BLOCKS_32K,
                        "max_batch_size": 1,
                        "max_scheduled_tokens": 8192,
                    }
                },
            },
        },
        {
            "id": "consumer-gpu-30g-ram-48g-32k",
            "eligibility": {
                "minimum_total_device_memory_bytes": 26843545601,
                "maximum_total_device_memory_bytes": 32212254720,
                "maximum_effective_host_memory_bytes": 68719476735,
                "is_integrated": False,
            },
            "overlay": {
                "model": {
                    "decoder": {
                        "filename": decoder_filename,
                        "session_options": {
                            "session.moe.offload.mode": "required",
                            "session.moe.offload.placement": "adaptive",
                            "session.moe.offload.cpu_execution": "auto",
                            "session.moe.offload.device_expert_bytes": "18289511424",
                            "session.moe.offload.host_expert_bytes": "41339060224",
                            "session.moe.offload.popularity_file": EXPERT_PROFILE_NAME,
                        },
                    }
                },
                "engine": {
                    "dynamic_batching": {
                        "num_blocks": NUM_BLOCKS_32K,
                        "max_batch_size": 1,
                        "max_scheduled_tokens": 8192,
                    }
                },
            },
        },
        {
            "id": "consumer-gpu-30g-ram-48g-32k-full",
            "eligibility": {
                "minimum_total_device_memory_bytes": 1,
                "maximum_total_device_memory_bytes": 2,
                "is_integrated": False,
            },
            "overlay": {
                "model": {
                    "decoder": {
                        "filename": decoder_filename,
                        "session_options": {
                            "session.moe.offload.mode": "required",
                            "session.moe.offload.placement": "adaptive",
                            "session.moe.offload.cpu_execution": "auto",
                            "session.moe.offload.device_expert_bytes": "20132659200",
                            "session.moe.offload.host_expert_bytes": "45097156608",
                            "session.moe.offload.popularity_file": EXPERT_PROFILE_NAME,
                        },
                    }
                },
                "engine": {
                    "dynamic_batching": {
                        "num_blocks": NUM_BLOCKS_32K,
                        "max_batch_size": 1,
                        "max_scheduled_tokens": 8192,
                    }
                },
            },
        },
        {
            "id": "consumer-gpu-32g",
            "eligibility": {
                "minimum_total_device_memory_bytes": 26843545601,
                "maximum_total_device_memory_bytes": 36507222016,
                "minimum_effective_host_memory_bytes": 68719476736,
                "is_integrated": False,
            },
            "overlay": {
                "model": {
                    "decoder": {
                        "filename": decoder_filename,
                        "session_options": {
                            "session.moe.offload.mode": "required",
                            "session.moe.offload.placement": "adaptive",
                            "session.moe.offload.cpu_execution": "auto",
                            "session.moe.offload.device_expert_bytes": "19229035520",
                            "session.moe.offload.popularity_file": EXPERT_PROFILE_NAME,
                        },
                    }
                },
                "engine": {
                    "dynamic_batching": {
                        "num_blocks": NUM_BLOCKS_32K,
                        "max_batch_size": 1,
                        "max_scheduled_tokens": 8192,
                    }
                },
            },
        },
        {
            "id": "gpu-128k",
            "eligibility": {
                "minimum_total_device_memory_bytes": min_128k,
                "maximum_total_device_memory_bytes": min_256k - 1,
                "is_integrated": False,
            },
            "overlay": {
                "model": {
                    "decoder": {
                        "filename": decoder_filename,
                        "session_options": {
                            "session.moe.offload.mode": "disabled",
                        },
                    }
                },
                "engine": {
                    "dynamic_batching": {
                        "num_blocks": NUM_BLOCKS_128K,
                        "max_batch_size": 1,
                        "max_scheduled_tokens": 8192,
                    }
                },
            },
        },
        {
            "id": "gpu-256k",
            "eligibility": {
                "minimum_total_device_memory_bytes": min_256k,
                "is_integrated": False,
            },
            "overlay": {
                "model": {
                    "decoder": {
                        "filename": decoder_filename,
                        "session_options": {
                            "session.moe.offload.mode": "disabled",
                        },
                    }
                },
                "engine": {
                    "dynamic_batching": {
                        "num_blocks": NUM_BLOCKS_256K,
                        "max_batch_size": 1,
                        "max_scheduled_tokens": 8192,
                    }
                },
            },
        },
    ]


def default_base_genai_config() -> dict:
    """Create the canonical Qwen3.8-Flash-Next RC5 `genai_config.json` structure from scratch."""
    return {
        "model": {
            "bos_token_id": 248044,
            "eos_token_id": [248046, 248044],
            "pad_token_id": 248044,
            **PROTOCOL_BOUNDARY_TOKEN_IDS,
            "type": "qwen38_flash_graph",
            "vocab_size": 248320,
            "context_length": DEFAULT_MAX_CONTEXT_LENGTH,
            "decoder": {
                "filename": "model.onnx",
                "hidden_size": 2560,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "num_hidden_layers": 48,
                "head_size": 256,
                "session_options": {
                    "ep.cuda.fpa_intb_gemm": "1",
                    "ep.cuda.fpa_intb_gemv_wave_aware": "1",
                    "ep.cuda.fpa_intb_gemv_paired_k": "1",
                    "session.use_device_allocator_for_initializers": "1",
                    "ep.cuda.matmul_nbits_m_chunk_size": "2048",
                    "ep.cuda.enable_gemm_auto_tune": "1",
                    "ep.cuda.gemm_tactic_cache_to_model": "1",
                    "ep.cuda.fpa_intb_profile_m": "1,2,4,8,16,32,64,128,256,512,1024,2048",
                    "provider_options": [
                        {
                            "cuda": {
                                "enable_cuda_graph": "1",
                                "arena_extend_strategy": "kSameAsRequested",
                            }
                        }
                    ],
                },
            },
            "qwen38_graph": {
                "plane_library": "provider/libqwen38_plane.so",
                "ple_table": "ple_table.txt",
                "kv_capacity": 33032,
                "warmup": True,
                "mtp": {
                    "enabled": True,
                    "prefill_head": False,
                },
                "fold": {
                    "enabled": True,
                },
                "prefill": {
                    "weights_pool": "weights_pool.txt",
                    "enabled": True,
                    "chunk": 8192,
                    "warmup_lengths": [8192, 2048, 512, 128, 37],
                },
            },
            "embedding": {
                "filename": "embedding.onnx",
                "session_options": {
                    "intra_op_num_threads": 1,
                },
                "inputs": {
                    "input_ids": "input_ids",
                },
                "outputs": {
                    "inputs_embeds": "inputs_embeds",
                },
            },
            "dflash2": {
                "filename": "dflash2.onnx",
                "session_options": {
                    "ep.cuda.fpa_intb_gemm": "1",
                    "ep.cuda.fpa_intb_gemv_wave_aware": "1",
                    "ep.cuda.fpa_intb_gemv_paired_k": "1",
                    "session.use_device_allocator_for_initializers": "1",
                    "ep.cuda.matmul_nbits_m_chunk_size": "512",
                    "ep.cuda.enable_gemm_auto_tune": "1",
                    "ep.cuda.gemm_tactic_cache_to_model": "1",
                    "ep.cuda.fpa_intb_profile_m": "1,2,4,8,16,32,64,128,256,512",
                    "provider_options": [
                        {
                            "cuda": {
                                "enable_cuda_graph": "1",
                            }
                        }
                    ],
                },
                "num_hidden_layers": 5,
                "num_key_value_heads": 8,
                "head_size": 128,
                "block_size": 8,
                "num_draft_tokens": 7,
                "selector_top_k": 16,
                "mask_token_id": 248077,
                "sliding_window": 2048,
                "main_aux_hidden_states": "aux_hidden_states",
                "inputs": {
                    "aux_hidden_states": "aux_hidden_states",
                    "input_ids": "input_ids",
                    "qkv_row_map": "qkv_row_map",
                    "block_row_index": "block_row_index",
                    "cumulative_sequence_lengths": "cumulative_sequence_lengths",
                    "past_sequence_lengths": "past_sequence_lengths",
                    "block_table": "block_table",
                    "attention_metadata": "attention_metadata",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                },
                "outputs": {
                    "candidate_ids": "draft_candidate_ids",
                    "scores": "draft_scores",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
            },
        },
        "engine": json.loads(json.dumps(DEFAULT_ENGINE_CONFIG)),
        "search": {
            "do_sample": False,
            "top_k": 1,
            "temperature": 1.0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "max_length": DEFAULT_MAX_CONTEXT_LENGTH,
            "min_length": 0,
            "past_present_share_buffer": False,
        },
        "runtime_profiles": default_runtime_profiles("model.onnx"),
    }


# ---------------------------------------------------------------------------
# QMoE Layer Rewriting, Bank Quantization & Context-Tier Graphs
# ---------------------------------------------------------------------------


def _infer_qmoe_geometry(model: onnx.ModelProto) -> tuple[int, int, int]:
    """Infer `(num_experts, hidden_size, inter_size)` from the first QMoE node's initializers."""
    inits = {t.name: t for t in model.graph.initializer}
    for n in qmoe_nodes(model):
        w1 = inits.get(n.input[2]) if len(n.input) > 2 else None
        s1 = inits.get(n.input[3]) if len(n.input) > 3 else None
        s2 = inits.get(n.input[6]) if len(n.input) > 6 else None
        if w1 is not None and s1 is not None and s2 is not None and len(s1.dims) == 3 and len(s2.dims) == 3:
            num_experts = int(s1.dims[0])
            fc1_n = int(s1.dims[1])
            hidden_size = int(s2.dims[1])
            return num_experts, hidden_size, fc1_n // 2
    return 512, 2560, 640


def ensure_qmoe_int4_bank(
    pool_dir: Path,
    hf_dir: Path | None = None,
    num_layers: int = 48,
    num_experts: int = 512,
    hidden_size: int = 2560,
    inter_size: int = 640,
    bs4: int = 32,
) -> None:
    """Verify that `qmoe_int4.data` and `qmoe_scales.data` exist in `pool_dir`, quantizing from `hf_dir` if missing."""
    w_p = pool_dir / "qmoe_int4.data"
    s_p = pool_dir / "qmoe_scales.data"
    fc1_w_sz = num_experts * hidden_size * (2 * inter_size // 2)
    fc2_w_sz = num_experts * inter_size * (hidden_size // 2)
    fc1_s_sz = num_experts * (2 * inter_size) * (hidden_size // bs4) * 2
    fc2_s_sz = num_experts * hidden_size * (inter_size // bs4) * 2
    exp_w_sz = num_layers * (fc1_w_sz + fc2_w_sz)
    exp_s_sz = num_layers * (fc1_s_sz + fc2_s_sz)
    if w_p.exists() and s_p.exists() and w_p.stat().st_size >= exp_w_sz and s_p.stat().st_size >= exp_s_sz:
        return
    if hf_dir is None or not hf_dir.exists():
        raise FileNotFoundError(
            f"Missing qmoe_int4.data / qmoe_scales.data in {pool_dir}. Pass --hf-dir to quantize them from the checkpoint."
        )

    pool_dir.mkdir(parents=True, exist_ok=True)
    hf_reader = HFCheckpointReader(hf_dir)
    n_gpus = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 0
    wave_size = min(8, max(1, n_gpus)) if n_gpus > 0 else 1

    with open(w_p, "wb") as fw, open(s_p, "wb") as fs:
        for start in range(0, num_layers, wave_size):
            layers = list(range(start, min(num_layers, start + wave_size)))

            def _quant_int4_layer(l_idx: int) -> tuple[bytes, bytes, bytes, bytes]:
                dev = torch.device(f"cuda:{l_idx % n_gpus}") if n_gpus > 0 else torch.device("cpu")
                pfx = f"model.layers.{l_idx}.mlp"
                raw_gu = hf_reader.read_bf16_tensor(f"{pfx}.experts.gate_up_proj", device="cpu").to(dev)
                half = raw_gu.shape[1] // 2
                gu_dev = torch.stack([raw_gu[:, :half, :], raw_gu[:, half:, :]], dim=2).reshape_as(raw_gu).contiguous()
                del raw_gu
                dn_dev = hf_reader.read_bf16_tensor(f"{pfx}.experts.down_proj", device="cpu").to(dev).contiguous()
                w1, s1 = fast_quant_prepack_batch(gu_dev, bits=4, block_size=bs4, is_qmoe=True)
                w2, s2 = fast_quant_prepack_batch(dn_dev, bits=4, block_size=bs4, is_qmoe=True)
                return (
                    w1.view(torch.uint8).numpy().tobytes(),
                    w2.view(torch.uint8).numpy().tobytes(),
                    s1.view(torch.uint16).numpy().tobytes(),
                    s2.view(torch.uint16).numpy().tobytes(),
                )

            with ThreadPoolExecutor(max_workers=len(layers)) as ex:
                wave_res = list(ex.map(_quant_int4_layer, layers))
            for w1b, w2b, s1b, s2b in wave_res:
                fw.write(w1b)
                fw.write(w2b)
                fs.write(s1b)
                fs.write(s2b)


def ensure_qmoe_banks(
    pool_dir: Path,
    needed_specs: list[tuple[str, int, int, str, bool, str, str, list[int], list[int]]],
    hf_dir: Path | None = None,
    num_layers: int = 48,
) -> None:
    """Verify that all required low-bit QMoE bank files exist in `pool_dir`, quantizing from `hf_dir` if missing."""
    need_build = []
    for spec in needed_specs:
        _proj, _bits, _bs, _mode, _sm80, w_fn, s_fn, w_sh, s_sh = spec
        w_p = pool_dir / w_fn
        s_p = pool_dir / s_fn
        exp_w_sz = num_layers * int(np.prod(w_sh))
        exp_s_sz = num_layers * int(np.prod(s_sh)) * 2
        if not (w_p.exists() and s_p.exists() and w_p.stat().st_size == exp_w_sz and s_p.stat().st_size == exp_s_sz):
            need_build.append(spec)

    if not need_build:
        return
    if hf_dir is None or not hf_dir.exists():
        missing_names = sorted({spec[5] for spec in need_build} | {spec[6] for spec in need_build})
        raise FileNotFoundError(
            f"Missing QMoE bank files in {pool_dir}: {missing_names}. Pass --hf-dir to quantize them from the checkpoint."
        )

    pool_dir.mkdir(parents=True, exist_ok=True)
    hf_reader = HFCheckpointReader(hf_dir)
    n_gpus = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 0
    wave_size = min(8, max(1, n_gpus)) if n_gpus > 0 else 1

    w_handles: dict[str, object] = {}
    s_handles: dict[str, object] = {}
    try:
        for spec in need_build:
            _, _, _, _, _, w_fn, s_fn, _, _ = spec
            if w_fn not in w_handles:
                w_handles[w_fn] = open(pool_dir / w_fn, "wb")  # noqa: SIM115
            if s_fn not in s_handles:
                s_handles[s_fn] = open(pool_dir / s_fn, "wb")  # noqa: SIM115

        for start in range(0, num_layers, wave_size):
            layers = list(range(start, min(num_layers, start + wave_size)))

            def _quantize_layer(l_idx: int) -> dict[tuple[str, str], tuple[bytes, bytes]]:
                dev = torch.device(f"cuda:{l_idx % n_gpus}") if n_gpus > 0 else torch.device("cpu")
                pfx = f"model.layers.{l_idx}.mlp"
                need_fc1 = any(s[0] == "fc1" for s in need_build)
                need_fc2 = any(s[0] == "fc2" for s in need_build)
                gu_dev = None
                dn_dev = None
                if need_fc1:
                    raw_gu = hf_reader.read_bf16_tensor(f"{pfx}.experts.gate_up_proj", device="cpu").to(dev)
                    half = raw_gu.shape[1] // 2
                    gu_dev = (
                        torch.stack([raw_gu[:, :half, :], raw_gu[:, half:, :]], dim=2).reshape_as(raw_gu).contiguous()
                    )
                    del raw_gu
                if need_fc2:
                    dn_dev = hf_reader.read_bf16_tensor(f"{pfx}.experts.down_proj", device="cpu").to(dev).contiguous()

                out_bytes: dict[tuple[str, str], tuple[bytes, bytes]] = {}
                for proj, bits, bs, mode, sm80, w_fn, s_fn, _, _ in need_build:
                    src_t = gu_dev if proj == "fc1" else dn_dev
                    pw, ps = quantize_and_pack_raw_qmoe(src_t, bits=bits, block_size=bs, mode=mode, sm80_prepack=sm80)
                    out_bytes[(w_fn, s_fn)] = (
                        pw.numpy().tobytes(),
                        ps.view(torch.uint16).numpy().tobytes(),
                    )
                return out_bytes

            with ThreadPoolExecutor(max_workers=len(layers)) as ex:
                wave_res = list(ex.map(_quantize_layer, layers))

            for l_res in wave_res:
                for (w_fn, s_fn), (wb, sb) in l_res.items():
                    w_handles[w_fn].write(wb)
                    s_handles[s_fn].write(sb)
    finally:
        for fh in list(w_handles.values()) + list(s_handles.values()):
            fh.close()


def rewrite_qmoe_layers(
    model: onnx.ModelProto,
    int4_layers: list[int],
    int3_layers: list[int],
    int2_layers: list[int],
    bs4: int = 32,
    bs3: int = 64,
    bs2: int = 64,
) -> tuple[dict[int, dict], set[str]]:
    """Rewrite `com.microsoft::QMoE` nodes and external initializers in `model` in place for `(X, Y, Z)`.

    Returns `(layer_recipe, needed_bank_filenames)`.
    """
    x_set = set(int4_layers)
    y_set = set(int3_layers)
    z_set = set(int2_layers)
    nodes = qmoe_nodes(model)
    num_layers = len(nodes)
    if len(x_set) + len(y_set) + len(z_set) != num_layers or (x_set | y_set | z_set) != set(range(num_layers)):
        raise ValueError(
            f"(int4, int3, int2) layers must partition 0..{num_layers - 1}; "
            f"got sizes ({len(x_set)}, {len(y_set)}, {len(z_set)})"
        )

    num_experts, hidden_size, inter_size = _infer_qmoe_geometry(model)
    fc1_n, fc1_k = 2 * inter_size, hidden_size
    fc2_n, fc2_k = hidden_size, inter_size

    layer_recipe: dict[int, dict] = {}
    needed_banks: set[str] = set()
    for l_idx in range(num_layers):
        if l_idx in x_set:
            if bs4 != 32:
                raise ValueError(f"INT4 QMoE layers require bs4=32, got {bs4}")
            layer_recipe[l_idx] = {
                "base_int4": True,
                "sm80_prepack": True,
                "bits": 4,
                "bs": bs4,
                "fc1_w_file": "qmoe_int4.data",
                "fc1_s_file": "qmoe_scales.data",
                "fc2_w_file": "qmoe_int4.data",
                "fc2_s_file": "qmoe_scales.data",
            }
            needed_banks.update(("qmoe_int4.data", "qmoe_scales.data"))
        elif l_idx in y_set:
            w1 = f"qmoe_fc1_int3_b{bs3}.data"
            s1 = f"qmoe_fc1_int3_b{bs3}_scales.data"
            w2 = f"qmoe_fc2_int3_b{bs3}.data"
            s2 = f"qmoe_fc2_int3_b{bs3}_scales.data"
            layer_recipe[l_idx] = {
                "base_int4": False,
                "sm80_prepack": False,
                "bits": 3,
                "bs": bs3,
                "fc1_w_file": w1,
                "fc1_s_file": s1,
                "fc2_w_file": w2,
                "fc2_s_file": s2,
            }
            needed_banks.update((w1, s1, w2, s2))
        else:
            w1 = f"qmoe_fc1_int2_b{bs2}_sm80.data"
            s1 = f"qmoe_fc1_int2_b{bs2}_scales.data"
            w2 = f"qmoe_fc2_int2_b{bs2}_sm80.data"
            s2 = f"qmoe_fc2_int2_b{bs2}_scales.data"
            layer_recipe[l_idx] = {
                "base_int4": False,
                "sm80_prepack": True,
                "bits": 2,
                "bs": bs2,
                "fc1_w_file": w1,
                "fc1_s_file": s1,
                "fc2_w_file": w2,
                "fc2_s_file": s2,
            }
            needed_banks.update((w1, s1, w2, s2))

    g = model.graph
    inits_by_name = {t.name: t for t in g.initializer}
    new_nodes: list[onnx.NodeProto] = []
    qmoe_ordinal = 0
    for n in g.node:
        if n.domain == "com.microsoft" and n.op_type == "QMoE":
            l_idx = parse_layer_index(n.name)
            if l_idx is None and len(n.input) > 2:
                l_idx = parse_layer_index(n.input[2])
            if l_idx is None:
                l_idx = qmoe_ordinal
            qmoe_ordinal += 1
            cfg = layer_recipe[l_idx]
            b = cfg["bits"]
            bs = cfg["bs"]
            sm80_pre = bool(cfg["sm80_prepack"])
            pfx = f"model.layers.{l_idx}.mlp"
            w1_name = n.input[2] if len(n.input) > 2 and n.input[2] else f"{pfx}.fc1_experts_weights"
            s1_name = n.input[3] if len(n.input) > 3 and n.input[3] else f"{pfx}.fc1_scales"
            w2_name = n.input[5] if len(n.input) > 5 and n.input[5] else f"{pfx}.fc2_experts_weights"
            s2_name = n.input[6] if len(n.input) > 6 and n.input[6] else f"{pfx}.fc2_scales"
            node_inputs = [n.input[0], n.input[1], w1_name, s1_name, "", w2_name, s2_name]
            k_val = int(qmoe_attribute(n, "k", 10))

            if sm80_pre:
                fc1_w_dims = [num_experts, fc1_k, fc1_n // (8 // b)]
                fc2_w_dims = [num_experts, fc2_k, fc2_n // (8 // b)]
            else:
                fc1_w_dims = [num_experts, fc1_n, fc1_k * b // 8]
                fc2_w_dims = [num_experts, fc2_n, fc2_k * b // 8]
            fc1_s_dims = [num_experts, fc1_n, fc1_k // bs]
            fc2_s_dims = [num_experts, fc2_n, fc2_k // bs]

            fc1_w_sz = int(np.prod(fc1_w_dims))
            fc1_s_sz = int(np.prod(fc1_s_dims)) * 2
            fc2_w_sz = int(np.prod(fc2_w_dims))
            fc2_s_sz = int(np.prod(fc2_s_dims)) * 2

            if cfg["base_int4"]:
                w_layer_stride = fc1_w_sz + fc2_w_sz
                s_layer_stride = fc1_s_sz + fc2_s_sz
                inits_by_name[w1_name] = make_ext_init(
                    w1_name, "qmoe_int4.data", l_idx * w_layer_stride, fc1_w_sz, TensorProto.UINT8, fc1_w_dims
                )
                inits_by_name[w2_name] = make_ext_init(
                    w2_name,
                    "qmoe_int4.data",
                    l_idx * w_layer_stride + fc1_w_sz,
                    fc2_w_sz,
                    TensorProto.UINT8,
                    fc2_w_dims,
                )
                inits_by_name[s1_name] = make_ext_init(
                    s1_name, "qmoe_scales.data", l_idx * s_layer_stride, fc1_s_sz, TensorProto.BFLOAT16, fc1_s_dims
                )
                inits_by_name[s2_name] = make_ext_init(
                    s2_name,
                    "qmoe_scales.data",
                    l_idx * s_layer_stride + fc1_s_sz,
                    fc2_s_sz,
                    TensorProto.BFLOAT16,
                    fc2_s_dims,
                )
                new_nodes.append(
                    helper.make_node(
                        "QMoE",
                        node_inputs,
                        list(n.output),
                        name=n.name,
                        domain="com.microsoft",
                        activation_type="swiglu",
                        normalize_routing_weights=1,
                        k=k_val,
                        expert_weight_bits=4,
                        block_size=bs,
                        swiglu_fusion=1,
                        weights_prepacked=1,
                    )
                )
            else:
                inits_by_name[w1_name] = make_ext_init(
                    w1_name, cfg["fc1_w_file"], l_idx * fc1_w_sz, fc1_w_sz, TensorProto.UINT8, fc1_w_dims
                )
                inits_by_name[s1_name] = make_ext_init(
                    s1_name, cfg["fc1_s_file"], l_idx * fc1_s_sz, fc1_s_sz, TensorProto.BFLOAT16, fc1_s_dims
                )
                inits_by_name[w2_name] = make_ext_init(
                    w2_name, cfg["fc2_w_file"], l_idx * fc2_w_sz, fc2_w_sz, TensorProto.UINT8, fc2_w_dims
                )
                inits_by_name[s2_name] = make_ext_init(
                    s2_name, cfg["fc2_s_file"], l_idx * fc2_s_sz, fc2_s_sz, TensorProto.BFLOAT16, fc2_s_dims
                )
                new_nodes.append(
                    helper.make_node(
                        "QMoE",
                        node_inputs,
                        list(n.output),
                        name=n.name,
                        domain="com.microsoft",
                        activation_type="swiglu",
                        normalize_routing_weights=1,
                        k=k_val,
                        expert_weight_bits=b,
                        fc1_expert_weight_bits=b,
                        fc2_expert_weight_bits=b,
                        fc3_expert_weight_bits=b,
                        block_size=bs,
                        swiglu_fusion=1,
                        weights_prepacked=1 if sm80_pre else 0,
                        quant_type="int",
                    )
                )
        else:
            new_nodes.append(n)

    del g.node[:]
    g.node.extend(new_nodes)
    used_inputs = {inp for n in g.node for inp in n.input if inp}
    kept_inits = [t for name, t in inits_by_name.items() if name in used_inputs]
    del g.initializer[:]
    g.initializer.extend(kept_inits)
    del g.value_info[:]
    return layer_recipe, needed_banks


def compute_package_referenced_size(pkg_dir: Path) -> dict:
    spans: set[tuple[str, int, int]] = set()
    by_file: dict[str, int] = {}
    onnx_files = [f for f in ("model.onnx", "dflash2.onnx", "embedding.onnx") if (pkg_dir / f).exists()]
    for mname in onnx_files:
        m = onnx.load(str(pkg_dir / mname), load_external_data=False)
        for t in m.graph.initializer:
            ext = {e.key: e.value for e in t.external_data}
            if "location" in ext:
                loc = ext["location"]
                off = int(ext.get("offset", 0))
                ln = int(ext.get("length", 0))
                if not ln and (pkg_dir / loc).exists():
                    ln = (pkg_dir / loc).stat().st_size - off
                key = (loc, off, ln)
                if key not in spans:
                    spans.add(key)
                    by_file[loc] = by_file.get(loc, 0) + ln
    ple_bytes = sum(p.stat().st_size for p in (pkg_dir / "ple").glob("*.data")) if (pkg_dir / "ple").exists() else 0
    onnx_bytes = sum((pkg_dir / f).stat().st_size for f in onnx_files)
    qmoe_w_bytes = sum(
        v
        for k, v in by_file.items()
        if k.startswith("qmoe_") and not k.endswith("_scales.data") and k != "qmoe_scales.data"
    )
    qmoe_s_bytes = sum(
        v
        for k, v in by_file.items()
        if k.startswith("qmoe_") and (k.endswith("_scales.data") or k == "qmoe_scales.data")
    )
    total_bytes = sum(by_file.values()) + ple_bytes + onnx_bytes
    return {
        "total_bytes": total_bytes,
        "total_mib": round(total_bytes / 2**20, 2),
        "total_gib": round(total_bytes / 2**30, 3),
        "total_gb": round(total_bytes / 1e9, 3),
        "qmoe_weights_gib": round(qmoe_w_bytes / 2**30, 3),
        "qmoe_scales_gib": round(qmoe_s_bytes / 2**30, 3),
        "qmoe_total_gib": round((qmoe_w_bytes + qmoe_s_bytes) / 2**30, 3),
        "by_file_mib": {k: round(v / 2**20, 2) for k, v in sorted(by_file.items())},
    }


def configure_rc5_genai_config(
    config: dict,
    *,
    has_int3_layers: bool,
    prefill_chunk: int | None = None,
    stop_token_ids: list[int] | None = None,
    include_runtime_profiles: bool = True,
    offload_entries: dict[str, str] | None = None,
    max_context_length: int | None = DEFAULT_MAX_CONTEXT_LENGTH,
    x_int4: int = 48,
    y_int3: int = 0,
    z_int2: int = 0,
    bs4: int = 32,
    bs3: int = 64,
    bs2: int = 64,
) -> list[str]:
    """Update `genai_config.json` in place for RC5 GPU + QMoE offload compatibility and return migration notes."""
    model_sec = config.setdefault("model", {})
    for token_key, token_val in PROTOCOL_BOUNDARY_TOKEN_IDS.items():
        if model_sec.get(token_key) != token_val:
            model_sec[token_key] = token_val

    if max_context_length is not None and int(model_sec.get("context_length", 0)) < max_context_length:
        model_sec["context_length"] = max_context_length

    qwen = model_sec.get("qwen38_graph")
    changes: list[str] = []
    if isinstance(qwen, dict):
        changes.extend(migrate_qwen38_graph_config(qwen))
        prefill = qwen.setdefault("prefill", {})
        if prefill.get("enabled", True):
            chunk = RC5_PREFILL_CONFIG["chunk"] if prefill_chunk is None else prefill_chunk
            kv_cap = int(qwen.get("kv_capacity", 33032))
            if chunk <= 0 or chunk > kv_cap:
                raise ValueError(f"--prefill-chunk ({chunk}) must be in 1..{kv_cap}")
            prefill["chunk"] = chunk
            prefill["warmup_lengths"] = [w for w in RC5_PREFILL_CONFIG["warmup_lengths"] if w <= chunk]
        elif prefill_chunk is not None:
            if prefill_chunk <= 0 or prefill_chunk % 2 != 0:
                raise ValueError(
                    f"--prefill-chunk ({prefill_chunk}) must be a positive even integer when prefill.enabled is false"
                )
            prefill["chunk"] = prefill_chunk

    if stop_token_ids is not None:
        model_sec["eos_token_id"] = list(stop_token_ids)

    dec_opts = model_sec.setdefault("decoder", {}).setdefault("session_options", {})
    prov_opts = dec_opts.get("provider_options")
    if isinstance(prov_opts, list):
        for entry in prov_opts:
            if isinstance(entry, dict) and isinstance(entry.get("cuda"), dict):
                cuda_opts = entry["cuda"]
                if "min_num_runs_before_cuda_graph_capture" in cuda_opts:
                    del cuda_opts["min_num_runs_before_cuda_graph_capture"]
                    changes.append(
                        "dropped decoder.session_options.provider_options.cuda.min_num_runs_before_cuda_graph_capture"
                    )
                cuda_opts.setdefault("enable_cuda_graph", "1")
                cuda_opts.setdefault("arena_extend_strategy", "kSameAsRequested")

    if has_int3_layers:
        dec_opts.update(INT3_DEQUANT_SESSION_OPTIONS)
    else:
        for k in INT3_DEQUANT_SESSION_OPTIONS:
            dec_opts.pop(k, None)

    if offload_entries:
        dec_opts.update(offload_entries)

    engine_sec = config.setdefault("engine", {})
    dyn_batch = engine_sec.setdefault("dynamic_batching", {})
    dyn_batch.setdefault("max_batch_size", DEFAULT_ENGINE_CONFIG["dynamic_batching"]["max_batch_size"])
    dyn_batch.setdefault("max_scheduled_tokens", DEFAULT_ENGINE_CONFIG["dynamic_batching"]["max_scheduled_tokens"])

    search_sec = config.setdefault("search", {})
    if max_context_length is not None and int(search_sec.get("max_length", 0)) < max_context_length:
        search_sec["max_length"] = max_context_length

    if include_runtime_profiles:
        dec_file = model_sec.get("decoder", {}).get("filename", "model.onnx")
        eff_y = y_int3 if (x_int4 != 48 or y_int3 != 0 or z_int2 != 0) else (1 if has_int3_layers else 0)
        eff_x = x_int4 if (x_int4 != 48 or y_int3 != 0 or z_int2 != 0) else (47 if has_int3_layers else 48)
        config["runtime_profiles"] = default_runtime_profiles(
            dec_file, x_int4=eff_x, y_int3=eff_y, z_int2=z_int2, bs4=bs4, bs3=bs3, bs2=bs2
        )
    return changes


def link_file(src: Path, dst: Path, copy: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if src.is_symlink():
        src = src.resolve()
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


DEFAULT_ORT_LIB_DIRS: tuple[Path, ...] = (
    Path("/home/tianlei/git/ort-qwen-gguf/build/cu130/Release"),
    Path("/home/tianlei/git/ort-gguf-bolt/build/cu130/Release"),
    Path("/home/tianlei/git/onnxruntime-qmoe-shared-direct/build/Linux/Release"),
)
DEFAULT_PLANE_LIB_CANDIDATES: tuple[Path, ...] = ()


def _prepare_procedural_base_assets(
    out_pkg: Path,
    hf_dir: Path,
    dflash_dir: Path,
    *,
    reuse_banks_from: Path | None = None,
    kv_scale_file: Path | None = None,
    expert_profile: Path | None = None,
    copy: bool = False,
) -> tuple[onnx.ModelProto, Path | None]:
    """Procedurally build or link non-QMoE `.data` / `ple/` banks and construct `embedding.onnx`, `dflash2.onnx`,
    and the initial 48-layer `model.onnx` `ModelProto` directly from `hf_dir` and `dflash_dir`.
    """
    non_qmoe_files = (
        "embedding.onnx.data",
        "dflash2.onnx.data",
        "norm_and_rope.data",
        "attn_consts.data",
        "contrib_weights.data",
        "matmul_int8.data",
        "matmul_scales.data",
        "ple_table.txt",
        "weights_pool.txt",
    )
    can_reuse = (
        reuse_banks_from is not None
        and reuse_banks_from.is_dir()
        and all((reuse_banks_from / fn).is_file() for fn in non_qmoe_files)
        and (reuse_banks_from / "ple").is_dir()
    )

    if can_reuse:
        bank_src = reuse_banks_from.resolve()
        for fn in non_qmoe_files:
            if fn == "weights_pool.txt":
                shutil.copy2(bank_src / fn, out_pkg / fn)
            else:
                link_file(bank_src / fn, out_pkg / fn, copy=copy)
        for tsv in bank_src.glob("*.tsv"):
            if tsv.is_file():
                shutil.copy2(tsv, out_pkg / tsv.name)
        for p in sorted((bank_src / "ple").iterdir()):
            if p.is_file():
                link_file(p, out_pkg / "ple" / p.name, copy=copy)
        for tok_fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
            src_tok = (hf_dir / tok_fn) if (hf_dir is not None and (hf_dir / tok_fn).is_file()) else (bank_src / tok_fn)
            if src_tok.is_file():
                link_file(src_tok, out_pkg / tok_fn, copy=copy)

        tensor_locations, embedded_weights = gb.load_bank_metadata_from_dir(bank_src)
        gb.save_bank_metadata_to_dir(out_pkg, tensor_locations, embedded_weights)

        emb_proto = gb.build_embedding_model_proto()
        onnx.save_model(emb_proto, str(out_pkg / "embedding.onnx"), save_as_external_data=False)

        base_kernels_u16: dict[str, np.ndarray] = {}
        if dflash_dir is not None and (dflash_dir / "model.safetensors").is_file():
            df_loader = gb.CheckpointTensorLoader(dflash_dir)
            for lid in range(5):
                for kind in ("attention_conv", "mlp_conv"):
                    bk = df_loader.read_tensor(f"layers.{lid}.{kind}.base_kernel").to(torch.bfloat16).contiguous()
                    base_kernels_u16[f"/dflash2/layers.{lid}/{kind}/base_kernel_all_1x4x160x16"] = (
                        bk.reshape(1, 1, 4, 160, 16).view(torch.uint16).numpy().copy()
                    )
        else:
            with open(out_pkg / "dflash2.onnx.data", "rb") as f_df:
                for lid in range(5):
                    for kind in ("attention_conv", "mlp_conv"):
                        pfx = f"/dflash2/layers.{lid}/{kind}"
                        parts = []
                        for sfx in (
                            "prepare.base_kernel.0",
                            "prepare.base_kernel.1",
                            "finish.base_kernel.0",
                            "finish.base_kernel.1",
                        ):
                            meta = tensor_locations[f"{pfx}/{sfx}"]
                            f_df.seek(int(meta["offset"]))
                            raw = f_df.read(int(meta["length"]))
                            parts.append(np.frombuffer(raw, dtype=np.uint16).reshape(160, 16).copy())
                        base_kernels_u16[f"{pfx}/base_kernel_all_1x4x160x16"] = np.stack(parts, axis=0).reshape(
                            1, 1, 4, 160, 16
                        )
        dflash_proto = gb.build_dflash2_model_proto(tensor_locations, base_kernels_u16)
        onnx.save_model(dflash_proto, str(out_pkg / "dflash2.onnx"), save_as_external_data=False)

        prof_path: Path | None = None
        if expert_profile is not None and expert_profile.is_file():
            prof_path = expert_profile.resolve()
            link_file(prof_path, out_pkg / EXPERT_PROFILE_NAME, copy=copy)
        elif (bank_src / EXPERT_PROFILE_NAME).is_file():
            link_file(bank_src / EXPERT_PROFILE_NAME, out_pkg / EXPERT_PROFILE_NAME, copy=copy)
            prof_path = out_pkg / EXPERT_PROFILE_NAME
        elif hf_dir is not None and hf_dir.is_dir():
            gb.build_expert_profile_from_hf(
                gb.CheckpointTensorLoader(hf_dir), out_pkg / EXPERT_PROFILE_NAME, list(KL_SINGLE_INT3_B64)
            )
            prof_path = out_pkg / EXPERT_PROFILE_NAME
    else:
        if hf_dir is None or not hf_dir.is_dir():
            raise FileNotFoundError(f"Valid --hf-dir checkpoint directory is required, got {hf_dir}")
        if dflash_dir is None or not dflash_dir.is_dir():
            raise FileNotFoundError(f"Valid --dflash-dir checkpoint directory is required, got {dflash_dir}")

        hf_loader = gb.CheckpointTensorLoader(hf_dir)
        emb_locs = gb.build_embedding_from_hf(hf_loader, out_pkg)
        gb.build_ple_int8_from_hf(hf_loader, out_pkg)
        target_locs, embedded_weights = gb.build_target_non_qmoe_from_hf(hf_loader, out_pkg, fast_quant_prepack_batch)
        tensor_locations = {**emb_locs, **target_locs}
        dflash_locs = gb.build_dflash2_from_hf(dflash_dir, out_pkg, tensor_locations, fast_quant_prepack_batch)
        tensor_locations.update(dflash_locs)
        gb.write_weights_pool_txt(out_pkg, tensor_locations)
        gb.save_bank_metadata_to_dir(out_pkg, tensor_locations, embedded_weights)

        for tok_fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
            src_tok = hf_dir / tok_fn
            if src_tok.is_file():
                link_file(src_tok, out_pkg / tok_fn, copy=copy)

        if expert_profile is not None and expert_profile.is_file():
            link_file(expert_profile.resolve(), out_pkg / EXPERT_PROFILE_NAME, copy=copy)
        else:
            gb.build_expert_profile_from_hf(hf_loader, out_pkg / EXPERT_PROFILE_NAME, list(KL_SINGLE_INT3_B64))
        prof_path = out_pkg / EXPERT_PROFILE_NAME

    # Seed dummy QMoE INT4 locations so build_target_model_proto populates all 48 QMoE nodes before rewrite_qmoe_layers
    for l_idx in range(48):
        pfx = f"model.layers.{l_idx}.mlp"
        tensor_locations.setdefault(
            f"{pfx}.fc1_experts_weights",
            {
                "location": "qmoe_int4.data",
                "offset": l_idx * 1258291200,
                "length": 838860800,
                "dtype": TensorProto.UINT8,
                "dims": [512, 2560, 640],
            },
        )
        tensor_locations.setdefault(
            f"{pfx}.fc2_experts_weights",
            {
                "location": "qmoe_int4.data",
                "offset": l_idx * 1258291200 + 838860800,
                "length": 419430400,
                "dtype": TensorProto.UINT8,
                "dims": [512, 640, 1280],
            },
        )
        tensor_locations.setdefault(
            f"{pfx}.fc1_scales",
            {
                "location": "qmoe_scales.data",
                "offset": l_idx * 157286400,
                "length": 104857600,
                "dtype": TensorProto.BFLOAT16,
                "dims": [512, 1280, 80],
            },
        )
        tensor_locations.setdefault(
            f"{pfx}.fc2_scales",
            {
                "location": "qmoe_scales.data",
                "offset": l_idx * 157286400 + 104857600,
                "length": 52428800,
                "dtype": TensorProto.BFLOAT16,
                "dims": [512, 2560, 20],
            },
        )

    kv_scales = gb.load_calibrated_kv_scales(kv_scale_file)
    model = gb.build_target_model_proto(tensor_locations, embedded_weights, kv_scales=kv_scales)
    return model, prof_path


def build_rc5_package(
    out_pkg: Path,
    base_pkg: Path | None = None,
    *,
    x_int4: int,
    y_int3: int,
    z_int2: int,
    bs4: int = 32,
    bs3: int = 64,
    bs2: int = 64,
    kl_mode: str = "auto",
    study_json: Path | None = None,
    int4_layers: list[int] | None = None,
    int3_layers: list[int] | None = None,
    int2_layers: list[int] | None = None,
    pool_dir: Path | None = None,
    hf_dir: Path | None = None,
    dflash_dir: Path | None = None,
    reuse_banks_from: Path | None = None,
    kv_scale_file: Path | None = None,
    expert_profile: Path | None = None,
    provider_lib: Path | None = None,
    ort_lib_dir: Path | None = None,
    plane_lib: Path | None = None,
    context_tiers: tuple[str, ...] = (),
    prefill_chunk: int | None = None,
    stop_token_ids: list[int] | None = None,
    offload_mode: str | None = None,
    offload_placement: str = "adaptive",
    device_expert_bytes: str | None = None,
    host_expert_bytes: str | None = None,
    target_resident_slots: int | None = None,
    total_resident_slots: int | None = None,
    min_slots_per_layer: int | None = None,
    uniform_profile_slots: bool = False,
    link_root: Path | None = None,
    copy: bool = False,
    extra_meta: dict | None = None,
) -> dict:
    """Build one self-contained Qwen3.8-Flash-Next RC5 `(x INT4 + y INT3 + z INT2)` package supporting GPU & QMoE offload."""
    t0 = time.time()
    base_pkg_resolved = base_pkg.resolve() if base_pkg is not None else None
    reuse_resolved = reuse_banks_from.resolve() if reuse_banks_from is not None else None
    reject_output_overlap(
        out_pkg,
        {
            "--base-pkg": base_pkg,
            "--reuse-banks-from": reuse_banks_from,
            "--pool-dir": pool_dir,
            "--hf-dir": hf_dir,
            "--dflash-dir": dflash_dir,
            "--kv-scale-file": kv_scale_file,
            "--expert-profile": expert_profile,
            "--provider-lib": provider_lib,
            "--ort-lib-dir": ort_lib_dir,
            "--plane-lib": plane_lib,
            "--kl-study-json": study_json,
        },
    )

    if out_pkg.exists() or out_pkg.is_symlink():
        if out_pkg.is_symlink() or out_pkg.is_file():
            out_pkg.unlink()
        else:
            shutil.rmtree(out_pkg)
    out_pkg.mkdir(parents=True, exist_ok=True)
    for subdir in ("provider", "ple"):
        if base_pkg_resolved is None or (base_pkg_resolved / subdir).exists() or subdir == "provider":
            (out_pkg / subdir).mkdir(exist_ok=True)

    procedural_prof: Path | None = None
    if base_pkg_resolved is not None:
        src_model_path = base_pkg_resolved / "model.onnx"
        if not src_model_path.is_file():
            raise FileNotFoundError(f"Missing target model.onnx in --base-pkg: {src_model_path}")
        if not (base_pkg_resolved / "genai_config.json").is_file():
            raise FileNotFoundError(
                f"Missing genai_config.json in --base-pkg: {base_pkg_resolved / 'genai_config.json'}"
            )
        model = onnx.load(str(src_model_path), load_external_data=False)
        for n in model.graph.node:
            if n.op_type == "QMoEOffload":
                raise ValueError(
                    f"{src_model_path} contains legacy QMoEOffload nodes; rebuild with standard com.microsoft::QMoE nodes"
                )
        ensure_rc5_speculative_prefix_commit(model)
        fused_routers = fuse_router_gate_up_proj(model)
        ensure_unified_dynamic_cache_model(model)
    else:
        eff_hf_dir = hf_dir or DEFAULT_HF_DIR
        eff_dflash_dir = dflash_dir or DEFAULT_DFLASH_DIR
        model, procedural_prof = _prepare_procedural_base_assets(
            out_pkg,
            eff_hf_dir,
            eff_dflash_dir,
            reuse_banks_from=reuse_resolved,
            kv_scale_file=kv_scale_file,
            expert_profile=expert_profile,
            copy=copy,
        )
        fused_routers = 48

    nodes = qmoe_nodes(model)
    num_layers = len(nodes)
    if num_layers == 0:
        raise ValueError("Target model has no com.microsoft::QMoE nodes")

    if int4_layers is not None and int3_layers is not None and int2_layers is not None:
        layers_4, layers_3, layers_2 = sorted(int4_layers), sorted(int3_layers), sorted(int2_layers)
        resolved_kl_mode = kl_mode
    else:
        layers_4, layers_3, layers_2, resolved_kl_mode = select_qmoe_layers_by_kl(
            x_int4,
            y_int3,
            z_int2,
            kl_mode=kl_mode,
            bs3=bs3,
            bs2=bs2,
            num_layers=num_layers,
            study_json=study_json,
        )

    num_experts, hidden_size, inter_size = _infer_qmoe_geometry(model)
    fc1_n, fc1_k = 2 * inter_size, hidden_size
    fc2_n, fc2_k = hidden_size, inter_size

    bank_search_dirs = [d for d in (base_pkg_resolved, reuse_resolved) if d is not None]
    default_pool_parent = (base_pkg_resolved or reuse_resolved or out_pkg.resolve()).parent
    effective_pool_dir = (pool_dir or (default_pool_parent / "_qmoe_lowbit_shared_banks")).resolve()
    eff_hf_for_qmoe = hf_dir or (DEFAULT_HF_DIR if base_pkg_resolved is None else None)

    if layers_4:
        has_int4_bank = any(
            (d / "qmoe_int4.data").exists() and (d / "qmoe_scales.data").exists()
            for d in [*bank_search_dirs, effective_pool_dir]
        )
        if not has_int4_bank:
            ensure_qmoe_int4_bank(
                effective_pool_dir,
                hf_dir=eff_hf_for_qmoe,
                num_layers=num_layers,
                num_experts=num_experts,
                hidden_size=hidden_size,
                inter_size=inter_size,
                bs4=bs4,
            )

    needed_bank_specs: list[tuple[str, int, int, str, bool, str, str, list[int], list[int]]] = []
    if layers_3:
        needed_bank_specs.extend(
            [
                (
                    "fc1",
                    3,
                    bs3,
                    "mse",
                    False,
                    f"qmoe_fc1_int3_b{bs3}.data",
                    f"qmoe_fc1_int3_b{bs3}_scales.data",
                    [num_experts, fc1_n, fc1_k * 3 // 8],
                    [num_experts, fc1_n, fc1_k // bs3],
                ),
                (
                    "fc2",
                    3,
                    bs3,
                    "mse",
                    False,
                    f"qmoe_fc2_int3_b{bs3}.data",
                    f"qmoe_fc2_int3_b{bs3}_scales.data",
                    [num_experts, fc2_n, fc2_k * 3 // 8],
                    [num_experts, fc2_n, fc2_k // bs3],
                ),
            ]
        )
    if layers_2:
        needed_bank_specs.extend(
            [
                (
                    "fc1",
                    2,
                    bs2,
                    "mse",
                    True,
                    f"qmoe_fc1_int2_b{bs2}_sm80.data",
                    f"qmoe_fc1_int2_b{bs2}_scales.data",
                    [num_experts, fc1_k, fc1_n // 4],
                    [num_experts, fc1_n, fc1_k // bs2],
                ),
                (
                    "fc2",
                    2,
                    bs2,
                    "mse",
                    True,
                    f"qmoe_fc2_int2_b{bs2}_sm80.data",
                    f"qmoe_fc2_int2_b{bs2}_scales.data",
                    [num_experts, fc2_k, fc2_n // 4],
                    [num_experts, fc2_n, fc2_k // bs2],
                ),
            ]
        )

    if needed_bank_specs:
        missing_in_base = [
            spec
            for spec in needed_bank_specs
            if not any((d / spec[5]).exists() and (d / spec[6]).exists() for d in bank_search_dirs)
        ]
        if missing_in_base:
            ensure_qmoe_banks(effective_pool_dir, missing_in_base, hf_dir=eff_hf_for_qmoe, num_layers=num_layers)

    _layer_recipe, needed_banks = rewrite_qmoe_layers(model, layers_4, layers_3, layers_2, bs4=bs4, bs3=bs3, bs2=bs2)
    offload_check = prepare_qmoe_for_offload(model)
    if offload_check["rejected"] and offload_mode == "required":
        first = next(iter(offload_check["rejected"]))
        reasons = "; ".join(offload_check["rejected"][first])
        raise ValueError(
            f"{len(offload_check['rejected'])} of {num_layers} QMoE nodes cannot be offloaded: "
            f"first, {node_label(offload_check['nodes'][first], first)}: {reasons}"
        )

    resolved_profile: Path | None = None
    if expert_profile is not None:
        if not expert_profile.is_file():
            raise FileNotFoundError(f"--expert-profile not found: {expert_profile}")
        resolved_profile = expert_profile.resolve()
    elif procedural_prof is not None and procedural_prof.is_file():
        resolved_profile = procedural_prof.resolve()
    elif base_pkg_resolved is not None and (base_pkg_resolved / EXPERT_PROFILE_NAME).is_file():
        resolved_profile = (base_pkg_resolved / EXPERT_PROFILE_NAME).resolve()

    explicit_offload_entries: dict[str, str] | None = None
    if offload_mode is not None or target_resident_slots is not None or total_resident_slots is not None:
        eff_mode = offload_mode or "required"
        node_overrides_str: str | None = None
        if total_resident_slots is not None or target_resident_slots is not None:
            if resolved_profile is not None and (
                total_resident_slots is not None or uniform_profile_slots or min_slots_per_layer is not None
            ):
                tot = total_resident_slots if total_resident_slots is not None else target_resident_slots * num_layers
                plan = load_expert_profile_plan(
                    resolved_profile,
                    num_layers=num_layers,
                    num_experts=num_experts,
                    total_slots=tot,
                    min_slots_per_layer=4 if min_slots_per_layer is None else min_slots_per_layer,
                    uniform=uniform_profile_slots,
                )
                caps = {offload_check["nodes"][i]: len(plan[i]) for i in range(num_layers)}
            else:
                if target_resident_slots is None:
                    raise ValueError("--total-resident-slots requires --expert-profile")
                caps = dict.fromkeys(offload_check["nodes"], target_resident_slots)
            node_overrides_str = format_node_overrides(caps)
        explicit_offload_entries = offload_session_options(
            mode=eff_mode,
            placement=offload_placement,
            device_expert_bytes=device_expert_bytes,
            host_expert_bytes=host_expert_bytes,
            popularity_file=EXPERT_PROFILE_NAME if resolved_profile is not None else None,
            node_overrides=node_overrides_str,
        )

    if base_pkg_resolved is not None and (base_pkg_resolved / "genai_config.json").is_file():
        config = json.loads((base_pkg_resolved / "genai_config.json").read_text())
    else:
        config = default_base_genai_config()

    configure_rc5_genai_config(
        config,
        has_int3_layers=len(layers_3) > 0,
        prefill_chunk=prefill_chunk,
        stop_token_ids=stop_token_ids,
        include_runtime_profiles=True,
        offload_entries=explicit_offload_entries,
        x_int4=len(layers_4),
        y_int3=len(layers_3),
        z_int2=len(layers_2),
        bs4=bs4,
        bs3=bs3,
        bs2=bs2,
    )

    if base_pkg_resolved is not None:
        skip_root = {
            "model.onnx",
            "model_2k.onnx",
            "model_8k.onnx",
            "model_16k.onnx",
            "model_32k.onnx",
            "model_64k.onnx",
            "model_128k.onnx",
            "model_256k.onnx",
            "model_dynamic_ctx.onnx",
            "model_int8kv.onnx",
            "package_manifest.json",
            "genai_config.json",
        }
        qmoe_data_files = {
            p.name
            for p in base_pkg_resolved.iterdir()
            if p.is_file() and (p.name.startswith("qmoe_") and p.name.endswith(".data"))
        }
        skip_root.update(qmoe_data_files - needed_banks)

        copied_suffixes = (".tsv",)
        copied_names = ("weights_pool.txt",)

        for p in sorted(base_pkg_resolved.iterdir()):
            if p.is_file() and p.name not in skip_root and not p.name.endswith(".lock"):
                if p.name in copied_names or p.name.endswith(copied_suffixes):
                    shutil.copy2(p, out_pkg / p.name)
                else:
                    link_file(p, out_pkg / p.name, copy=copy)

        if (base_pkg_resolved / "ple").is_dir():
            for p in sorted((base_pkg_resolved / "ple").iterdir()):
                if p.is_file():
                    link_file(p, out_pkg / "ple" / p.name, copy=copy)

    src_prov_dir = (
        (base_pkg_resolved / "provider")
        if (base_pkg_resolved is not None and (base_pkg_resolved / "provider").is_dir())
        else (
            reuse_resolved / "provider"
            if (reuse_resolved is not None and (reuse_resolved / "provider").is_dir())
            else None
        )
    )
    if src_prov_dir is not None and src_prov_dir.is_dir():
        for p in sorted(src_prov_dir.iterdir()):
            if p.is_file():
                shutil.copy2(p, out_pkg / "provider" / p.name)
    elif ort_lib_dir is None:
        for cand_dir in DEFAULT_ORT_LIB_DIRS:
            if cand_dir.is_dir() and (cand_dir / "libonnxruntime_providers_cuda.so").exists():
                ort_lib_dir = cand_dir
                break

    if ort_lib_dir is not None and ort_lib_dir.is_dir():
        for name in (
            "libonnxruntime.so",
            "libonnxruntime.so.1",
            "libonnxruntime.so.1.32.0",
            "libonnxruntime_providers_cuda.so",
            "libonnxruntime_providers_shared.so",
        ):
            p = ort_lib_dir / name
            if p.exists():
                shutil.copy2(p.resolve(), out_pkg / "provider" / name)
    if provider_lib is not None and provider_lib.is_file():
        shutil.copy2(provider_lib.resolve(), out_pkg / "provider" / provider_lib.name)
    if plane_lib is not None and plane_lib.is_file():
        shutil.copy2(plane_lib.resolve(), out_pkg / "provider" / "libqwen38_plane.so")
    elif not (out_pkg / "provider" / "libqwen38_plane.so").exists():
        for cand_plane in DEFAULT_PLANE_LIB_CANDIDATES:
            if cand_plane.is_file():
                shutil.copy2(cand_plane.resolve(), out_pkg / "provider" / "libqwen38_plane.so")
                break

    if resolved_profile is not None and not (out_pkg / EXPERT_PROFILE_NAME).exists():
        link_file(resolved_profile, out_pkg / EXPERT_PROFILE_NAME, copy=copy)

    for b_fn in sorted(needed_banks):
        dst_b = out_pkg / b_fn
        if dst_b.exists():
            continue
        src_b: Path | None = None
        for cand_dir in (*bank_search_dirs, effective_pool_dir):
            if (cand_dir / b_fn).exists():
                src_b = cand_dir / b_fn
                break
        if src_b is None:
            raise FileNotFoundError(
                f"Required QMoE bank file {b_fn} not found in {bank_search_dirs} or {effective_pool_dir}"
            )
        link_file(src_b, dst_b, copy=copy)

    onnx.save_model(model, str(out_pkg / "model.onnx"), save_as_external_data=False)

    built_tiers: list[str] = ["unified_32k_128k_256k"]
    for tier in context_tiers:
        t_norm = tier.strip().lower()
        if not t_norm or t_norm in ("unified", "all"):
            continue
        if t_norm not in CONTEXT_TIERS:
            raise ValueError(f"Unknown context tier {tier!r}; choose from {sorted(CONTEXT_TIERS)}")
        kv_cap, st_cap = CONTEXT_TIERS[t_norm]
        m_tier = build_context_tier_model(model, kv_cap, st_cap)
        onnx.save_model(m_tier, str(out_pkg / f"model_{t_norm}.onnx"), save_as_external_data=False)
        built_tiers.append(t_norm)

    (out_pkg / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")

    sz_info = compute_package_referenced_size(out_pkg)
    ref_manifest = (
        json.loads((base_pkg_resolved / "package_manifest.json").read_text())
        if (base_pkg_resolved is not None and (base_pkg_resolved / "package_manifest.json").is_file())
        else {}
    )
    vram_info = estimate_peak_vram_and_thresholds(
        len(layers_4), len(layers_3), len(layers_2), bs4=bs4, bs3=bs3, bs2=bs2
    )
    manifest = {
        "variant": out_pkg.name,
        "qmoe_recipe": {
            "x_int4": len(layers_4),
            "y_int3": len(layers_3),
            "z_int2": len(layers_2),
            "bit_sum": 4 * len(layers_4) + 3 * len(layers_3) + 2 * len(layers_2),
            "avg_bits": round((4 * len(layers_4) + 3 * len(layers_3) + 2 * len(layers_2)) / float(num_layers), 4),
            "bs4": bs4,
            "bs3": bs3,
            "bs2": bs2,
            "kl_mode": resolved_kl_mode,
            "int4_layers": layers_4,
            "int3_layers": layers_3,
            "int2_layers": layers_2,
            **(extra_meta or {}),
        },
        "size_breakdown": sz_info,
        "vram_requirements": vram_info,
        "context_tiers": built_tiers,
        "supports_gpu_and_qmoe_offload": True,
        "router_weights_cleared": offload_check["router_weights_cleared"],
        "speculative_prefix_commit": ref_manifest.get("speculative_prefix_commit", SPC_MANIFEST_VALUE),
        "router_gate_up_proj": ref_manifest.get(
            "router_gate_up_proj",
            {
                "op": "com.microsoft::RouterGateUpProj",
                "activation": "swiglu",
                "fused_layers": fused_routers or 48,
                "ort_prs": ["#33650", "#32126"],
            },
        ),
        "prefill_config_in_rc5": dict(
            config.get("model", {}).get("qwen38_graph", {}).get("prefill", RC5_PREFILL_CONFIG)
        ),
    }
    if len(layers_3) > 0:
        manifest["decoder_session_options_added_in_rc5"] = dict(INT3_DEQUANT_SESSION_OPTIONS)
    if "ort_provider" in ref_manifest:
        manifest["ort_provider"] = ref_manifest["ort_provider"]

    (out_pkg / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if link_root is not None and link_root.resolve() != out_pkg.parent.resolve():
        link_root.mkdir(parents=True, exist_ok=True)
        link_path = link_root / out_pkg.name
        if link_path.is_symlink() or link_path.exists():
            if link_path.is_symlink():
                link_path.unlink()
        if not link_path.exists():
            link_path.symlink_to(out_pkg.resolve())

    print(
        f"Built {out_pkg.name} (x={len(layers_4)}, y={len(layers_3)}, z={len(layers_2)}, kl_mode={resolved_kl_mode}, "
        f"avg={(4 * len(layers_4) + 3 * len(layers_3) + 2 * len(layers_2)) / float(num_layers):.3f}b): "
        f"Total={sz_info['total_gib']:.3f} GiB ({sz_info['total_gb']:.3f} GB), "
        f"QMoE={sz_info['qmoe_total_gib']:.3f} GiB (W={sz_info['qmoe_weights_gib']:.3f}, S={sz_info['qmoe_scales_gib']:.3f}) "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )
    return manifest


def budget_bytes(text: str) -> str:
    value = text.strip()
    if value.lower() == "auto":
        return "auto"
    if re.fullmatch(r"[0-9]+", value) and int(value) <= 2**64 - 1:
        return str(int(value))
    raise argparse.ArgumentTypeError(f"expected 'auto' or a non-negative byte count, got {text!r}")


def parse_xyz_tuple(text: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 3 or not all(re.fullmatch(r"[0-9]+", p) for p in parts):
        raise argparse.ArgumentTypeError(f"--qmoe-xyz expects 'X,Y,Z' non-negative integers, got {text!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in REMOVED_OPTIONS:
        ap.add_argument(flag, nargs="?", const="", default=None, help=argparse.SUPPRESS)

    ap.add_argument(
        "--base-pkg",
        "--from-package",
        "--src",
        dest="base_pkg",
        type=Path,
        default=None,
        help="Source/base Qwen3.8-Flash-Next RC5 (or RC4/RC2) package folder",
    )
    ap.add_argument("--out", type=Path, default=None, help="Output package directory (for single-package builds)")
    ap.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="Parent directory where named preset packages are written when --out is omitted",
    )
    ap.add_argument(
        "--link-root",
        type=Path,
        default=None,
        help="Optional directory (e.g. /data/tlwu) where symlinks to built packages are created",
    )
    ap.add_argument(
        "--name-suffix",
        type=str,
        default="",
        help="Optional suffix appended to preset package folder names (e.g. '_new' for side-by-side testing)",
    )
    ap.add_argument(
        "--preset",
        "--recipe",
        dest="preset",
        type=str,
        default=None,
        help="Predefined RC5 configuration name (e.g. all_int4, top16_44_33_b64, top08_44_33_b64, 2b375_x02_y14_z32_dp23_b64, or 'rc5_all')",
    )
    ap.add_argument(
        "--presets",
        type=str,
        default=None,
        help="Comma-separated list of presets to build into --models-root (or 'rc5_all' for the 4 core RC5 models)",
    )
    ap.add_argument("-x", "--x-int4", type=int, default=None, help="Number of INT4 QMoE layers")
    ap.add_argument("-y", "--y-int3", type=int, default=None, help="Number of INT3 QMoE layers")
    ap.add_argument("-z", "--z-int2", type=int, default=None, help="Number of INT2 QMoE layers")
    ap.add_argument("--qmoe-xyz", type=parse_xyz_tuple, default=None, help="QMoE layer counts as 'X,Y,Z'")
    ap.add_argument("--bs4", type=int, default=32, help="Block size for INT4 QMoE layers (default: 32)")
    ap.add_argument(
        "--bs3", type=int, default=64, choices=(32, 64, 128), help="Block size for INT3 QMoE layers (default: 64)"
    )
    ap.add_argument(
        "--bs2", type=int, default=64, choices=(32, 64, 128), help="Block size for INT2 QMoE layers (default: 64)"
    )
    ap.add_argument(
        "--kl-mode",
        choices=KL_MODES,
        default="auto",
        help="KL-similarity layer sorting/partitioning mode: auto (default), rank, dp23, dp3, dp2, or dp4",
    )
    ap.add_argument(
        "--kl-study-json", type=Path, default=None, help="Optional JSON file with custom 48-layer KL study results"
    )
    ap.add_argument(
        "--pool-dir", type=Path, default=None, help="Directory holding shared low-bit QMoE bank .data files"
    )
    ap.add_argument(
        "--hf-dir",
        type=Path,
        default=None,
        help=f"Target Hugging Face checkpoint directory (default: {DEFAULT_HF_DIR})",
    )
    ap.add_argument(
        "--dflash-dir",
        type=Path,
        default=None,
        help=f"DFlash2 Hugging Face checkpoint directory (default: {DEFAULT_DFLASH_DIR})",
    )
    ap.add_argument(
        "--reuse-banks-from",
        type=Path,
        default=None,
        help="Optional directory of previously quantized binary .data / ple/ banks to hard-link across multi-preset builds",
    )
    ap.add_argument(
        "--kv-scale-file",
        type=Path,
        default=None,
        help="Optional calibrated KV scale JSON file (defaults to embedded calibrated Qwen3.8-Flash-Next scales)",
    )
    ap.add_argument("--expert-profile", type=Path, default=None, help="STRP v1 expert-profile.bin file")
    ap.add_argument("--provider-lib", type=Path, default=None, help="Path to libonnxruntime_providers_cuda.so")
    ap.add_argument("--ort-lib-dir", type=Path, default=None, help="Directory containing ORT shared libraries")
    ap.add_argument("--plane-lib", type=Path, default=None, help="Path to libqwen38_plane.so")
    ap.add_argument(
        "--context-tiers",
        type=str,
        default="",
        help="Optional comma-separated fixed-capacity tier ONNX graphs to emit alongside unified model.onnx (default: none, since unified model.onnx serves 32k/128k/256k)",
    )
    ap.add_argument("--prefill-chunk", type=int, default=None, help="Override model.qwen38_graph.prefill.chunk")
    ap.add_argument("--prefill-chunk-tokens", type=int, default=None, help="Alias for --prefill-chunk")
    ap.add_argument("--stop-token-id", type=int, default=None, help="Single stop token ID override")
    ap.add_argument("--stop-token-ids", type=str, default=None, help="Comma-separated stop token IDs override")
    ap.add_argument(
        "--offload-mode", choices=OFFLOAD_MODES, default=None, help="Optional explicit session.moe.offload.mode"
    )
    ap.add_argument(
        "--offload-placement", choices=OFFLOAD_PLACEMENTS, default="adaptive", help="session.moe.offload.placement"
    )
    ap.add_argument(
        "--device-expert-bytes", type=budget_bytes, default=None, help="session.moe.offload.device_expert_bytes"
    )
    ap.add_argument(
        "--host-expert-bytes", type=budget_bytes, default=None, help="session.moe.offload.host_expert_bytes"
    )
    ap.add_argument("--target-resident-slots", type=int, default=None, help="Per-layer resident expert slot cap")
    ap.add_argument(
        "--total-resident-slots", type=int, default=None, help="Total resident expert slots across all layers"
    )
    ap.add_argument(
        "--min-slots-per-layer",
        type=int,
        default=None,
        help="Minimum slots per layer when spreading --total-resident-slots",
    )
    ap.add_argument(
        "--uniform-profile-slots", action="store_true", help="Spread --total-resident-slots uniformly across layers"
    )
    ap.add_argument("--copy", action="store_true", help="Copy weight files instead of hard-linking")
    add_precision_cli_args(ap)
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for flag, reason in REMOVED_OPTIONS.items():
        if getattr(args, flag.lstrip("-").replace("-", "_")) is not None:
            parser.error(f"{flag} was removed: {reason}")

    if (
        args.prefill_chunk is not None
        and args.prefill_chunk_tokens is not None
        and args.prefill_chunk != args.prefill_chunk_tokens
    ):
        parser.error(
            f"--prefill-chunk ({args.prefill_chunk}) and --prefill-chunk-tokens ({args.prefill_chunk_tokens}) disagree"
        )
    prefill_chunk = args.prefill_chunk if args.prefill_chunk is not None else args.prefill_chunk_tokens

    stop_ids: list[int] | None = None
    if args.stop_token_ids is not None:
        stop_ids = [int(x.strip()) for x in args.stop_token_ids.split(",") if x.strip()]
    elif args.stop_token_id is not None:
        stop_ids = [args.stop_token_id]

    if args.base_pkg is None and args.reuse_banks_from is None:
        eff_hf = args.hf_dir or DEFAULT_HF_DIR
        eff_df = args.dflash_dir or DEFAULT_DFLASH_DIR
        if not eff_hf.exists() or not eff_df.exists():
            parser.error(
                "Provide --hf-dir and --dflash-dir (or --base-pkg / --from-package) to build the ONNX model package"
            )

    tiers = tuple(t.strip() for t in args.context_tiers.split(",") if t.strip())

    build_jobs: list[tuple[Path, dict]] = []
    preset_list_raw = args.presets or args.preset
    if preset_list_raw is not None:
        names = [s.strip() for s in preset_list_raw.split(",") if s.strip()]
        expanded: list[str] = []
        for nm in names:
            if nm in ("rc5_all", "rc5", "all_rc5"):
                expanded.extend(RC5_CORE_PRESETS)
            elif nm in ("rc6_all", "rc6", "all_rc6"):
                expanded.extend(RC6_CORE_PRESETS)
            elif nm == "all":
                expanded.extend(PRESETS.keys())
            else:
                expanded.append(nm)
        for nm in expanded:
            spec = resolve_preset_spec(nm)
            pkg_name = spec["pkg_name"] + args.name_suffix
            if args.out is not None and len(expanded) == 1:
                out_dir = args.out
            elif args.models_root is not None:
                out_dir = args.models_root / pkg_name
            else:
                parser.error("--out or --models-root is required when building presets")
            build_jobs.append((out_dir, spec))
    else:
        if args.qmoe_xyz is not None:
            x, y, z = args.qmoe_xyz
        elif args.x_int4 is not None or args.y_int3 is not None or args.z_int2 is not None:
            if args.x_int4 is None or args.y_int3 is None or args.z_int2 is None:
                parser.error(
                    "Specify all three of --x-int4 (-x), --y-int3 (-y), and --z-int2 (-z), or use --qmoe-xyz / --preset"
                )
            x, y, z = args.x_int4, args.y_int3, args.z_int2
        else:
            x, y, z = 48, 0, 0

        spec = {
            "x": x,
            "y": y,
            "z": z,
            "bs4": args.bs4,
            "bs3": args.bs3,
            "bs2": args.bs2,
            "kl_mode": args.kl_mode,
        }
        if args.out is not None:
            out_dir = args.out
        elif args.models_root is not None:
            avg_b = (4 * x + 3 * y + 2 * z) / 48.0
            b_tag = f"{avg_b:.3f}".rstrip("0").rstrip(".").replace(".", "b")
            pkg_name = (
                f"qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_{b_tag}_x{x:02d}_y{y:02d}_z{z:02d}_b{args.bs3}"
                f"{args.name_suffix}"
            )
            out_dir = args.models_root / pkg_name
        else:
            parser.error("--out or --models-root is required")
        build_jobs.append((out_dir, spec))

    try:
        base_pcfg = resolve_precision_config(args)
        job_pcfgs = []
        for _out_dir, spec in build_jobs:
            pcfg = base_pcfg
            if not base_pcfg.is_non_default() and "precision_profile" in spec:
                spec_args = argparse.Namespace(**vars(args))
                spec_args.precision_profile = spec["precision_profile"]
                if spec.get("consolidate_data_files"):
                    spec_args.consolidate_data_files = True
                pcfg = resolve_precision_config(spec_args)
            job_pcfgs.append(pcfg)
    except ValueError as e:
        parser.error(str(e))

    first_built_out: Path | None = None
    for (out_dir, spec), pcfg in zip(build_jobs, job_pcfgs, strict=True):
        eff_reuse = args.reuse_banks_from or first_built_out
        eff_kl_mode = spec.get("kl_mode", args.kl_mode)
        if eff_kl_mode == "auto" and pcfg.gdn_precision == "fp32":
            eff_kl_mode = "auto_fp32"
        build_rc5_package(
            out_pkg=out_dir,
            base_pkg=args.base_pkg,
            x_int4=spec["x"],
            y_int3=spec["y"],
            z_int2=spec["z"],
            bs4=spec.get("bs4", args.bs4),
            bs3=spec.get("bs3", args.bs3),
            bs2=spec.get("bs2", args.bs2),
            kl_mode=eff_kl_mode,
            study_json=args.kl_study_json,
            pool_dir=args.pool_dir,
            hf_dir=args.hf_dir,
            dflash_dir=args.dflash_dir,
            reuse_banks_from=eff_reuse,
            kv_scale_file=args.kv_scale_file,
            expert_profile=args.expert_profile,
            provider_lib=args.provider_lib,
            ort_lib_dir=args.ort_lib_dir,
            plane_lib=args.plane_lib,
            context_tiers=tiers,
            prefill_chunk=prefill_chunk,
            stop_token_ids=stop_ids,
            offload_mode=args.offload_mode,
            offload_placement=args.offload_placement,
            device_expert_bytes=args.device_expert_bytes,
            host_expert_bytes=args.host_expert_bytes,
            target_resident_slots=args.target_resident_slots,
            total_resident_slots=args.total_resident_slots,
            min_slots_per_layer=args.min_slots_per_layer,
            uniform_profile_slots=args.uniform_profile_slots,
            link_root=args.link_root,
            copy=args.copy,
        )
        if pcfg.is_non_default():
            apply_precision_config_to_package(out_dir, pcfg, hf_dir=args.hf_dir or DEFAULT_HF_DIR)
        if first_built_out is None:
            first_built_out = out_dir
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
