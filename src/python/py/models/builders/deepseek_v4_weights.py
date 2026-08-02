# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Adapter from the DeepSeek-V4-Flash checkpoint to the builder's state dict.

The checkpoint stores the dense projections as block-scaled fp8 (E4M3 weight
`[N, K]` + E8M0 scale `[N/128, K/128]`) and the routed experts as fp4 packed
along the input dim (`[N, K/2]` uint8 + E8M0 scale `[N, K/32]`).  This class
serves the builder's key names, materializing one tensor at a time, and does the
two format conversions the graph needs:

* fp8 -> the per-row `[N, ceil(K/128)]` fp32 scales that
  `com.microsoft.MatMulBlockQuantizedFp8Weight` expects (the weight itself is
  used as-is);
* fp4 -> QMoE's `[E, K, N/2]` layout, with gate and up interleaved into fc1.

``expert_range`` limits which routed experts are read at all, so a per-rank
export never touches the other ranks' share of the 137 GiB of experts.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import torch

from .deepseek_v4 import pack_for_qmoe
from .safetensors_store import SafeTensorStore, e8m0_to_float

FP8_BLOCK = 128

_FFN_RENAMES = {
    "gate_weight": "gate.weight",
    "gate_bias": "gate.bias",
    "tid2eid": "gate.tid2eid",
}
_FFN_KEY = re.compile(r"(layers\.\d+\.ffn)\.(.+)")
_SHARED_KEY = re.compile(r"sw(\d)\.weight")
_EXPERT_KEY = re.compile(r"layers\.(\d+)\.ffn\.(fc1_q|fc1_s|fc2_q|fc2_s)")


class DSV4Weights(Mapping):
    """The builder's state dict, backed by the checkpoint.

    ``expert_range`` is the half-open range of routed experts this rank owns.
    """

    # Experts are narrowed here, so the builder must not shard them again.
    pre_sharded = True

    def __init__(self, model_dir, expert_range, device="cpu"):
        self.store = SafeTensorStore(model_dir)
        self.expert_range = expert_range
        self.device = device
        self._layer = None
        self._experts: dict[str, torch.Tensor] = {}

    # -- Mapping ---------------------------------------------------------- #

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __contains__(self, key):
        return self._resolve(key) is not None or _EXPERT_KEY.fullmatch(key) is not None

    def __getitem__(self, key):
        m = _EXPERT_KEY.fullmatch(key)
        if m is not None:
            return self._expert_tensors(int(m.group(1)))[m.group(2)]
        ckpt_key = self._resolve(key)
        if ckpt_key is None:
            raise KeyError(key)
        if self._scale_key(ckpt_key) is not None:
            return self.store.dequant(ckpt_key, device=self.device, block=FP8_BLOCK)
        return self.store[ckpt_key]

    # -- name mapping ----------------------------------------------------- #

    def _resolve(self, key):
        """Builder key -> checkpoint key, or ``None`` if there is no such weight."""
        if key in self.store:
            return key
        m = _FFN_KEY.fullmatch(key)
        if m is None:
            return None
        base, tail = m.groups()
        if tail in _FFN_RENAMES:
            renamed = f"{base}.{_FFN_RENAMES[tail]}"
        else:
            shared = _SHARED_KEY.fullmatch(tail)
            if shared is None:
                return None
            renamed = f"{base}.shared_experts.w{shared.group(1)}.weight"
        return renamed if renamed in self.store else None

    def _scale_key(self, ckpt_key):
        if not ckpt_key.endswith("weight"):
            return None
        key = ckpt_key[: -len("weight")] + "scale"
        return key if key in self.store else None

    # -- block-scaled fp8 ------------------------------------------------- #

    def qweight(self, key):
        """``(fp8 weight [N, K], fp32 scales [N, ceil(K/128)])`` or ``None``.

        The checkpoint scales one 128x128 tile at a time; the operator wants one
        scale per row and K-block, so the rows are just expanded out.
        """
        ckpt_key = self._resolve(key)
        if ckpt_key is None:
            return None
        scale_key = self._scale_key(ckpt_key)
        if scale_key is None:
            return None
        w = self.store[ckpt_key]
        if w.dtype != torch.float8_e4m3fn:
            return None
        s = e8m0_to_float(self.store[scale_key])
        return w, s.repeat_interleave(FP8_BLOCK, 0)[: w.shape[0]].contiguous()

    # -- fp4 experts ------------------------------------------------------ #

    def _expert_tensors(self, layer):
        if self._layer != layer:
            self._layer = layer
            self._experts = self._pack_experts(layer)
        return self._experts

    def _pack_experts(self, layer):
        prefix = f"layers.{layer}.ffn.experts"
        lo, hi = self.expert_range

        def load(which):
            """-> (blocks [E, N, K/32, 16], scales [E, N, K/32]).

            The checkpoint's `[N, K/2]` fp4 layout -- two codes per byte along K,
            even K in the low nibble -- is already what `pack_for_qmoe` consumes;
            only the trailing dim has to be split into 32-value blocks.
            """
            blocks, scales = [], []
            for e in range(lo, hi):
                q = self.store[f"{prefix}.{e}.{which}.weight"].to(self.device).view(torch.uint8)
                s = self.store[f"{prefix}.{e}.{which}.scale"].to(self.device).view(torch.uint8)
                n, half_k = q.shape
                blocks.append(q.view(n, half_k // 16, 16))
                scales.append(s)
            return torch.stack(blocks), torch.stack(scales)

        b1, s1 = load("w1")
        b3, s3 = load("w3")
        e, n, kb, _ = b1.shape
        # fc1 = [gate, up] interleaved along the output dim (swiglu_fusion=1).
        fc1_b = torch.stack([b1, b3], dim=2).reshape(e, 2 * n, kb, 16)
        fc1_s = torch.stack([s1, s3], dim=2).reshape(e, 2 * n, s1.shape[-1])
        del b1, b3, s1, s3
        b2, s2 = load("w2")
        return {
            "fc1_q": pack_for_qmoe(fc1_b),
            "fc1_s": fc1_s,
            "fc2_q": pack_for_qmoe(b2),
            "fc2_s": s2,
        }
