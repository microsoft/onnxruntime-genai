# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""ONNX local-function fallbacks for ``com.microsoft`` contrib ops.

When ``onnxruntime < 1.26`` is installed the native kernel for some contrib
ops may not be registered.  :class:`LocalFunctionsMixin` provides class
methods that build ONNX local-function bodies using only standard opset-21
primitives, and registers them into the ONNX model whenever the installed ORT
version requires the fallback.

Usage::

    class Model(LocalFunctionsMixin, ...):
        ...
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir

# Sentinel used as the "end-of-tensor" value for ONNX Slice nodes.  We pick
# 2**62 rather than sys.maxsize (2**63-1) to stay safely within the signed
# INT64 range that ONNX Slice accepts on all platforms.
_ONNX_LARGE_SLICE_END: int = 2**62


class LocalFunctionsMixin:
    """Mixin that adds ONNX local-function helpers to model-builder classes.

    Concrete subclasses are expected to expose ``self.model`` (an
    :class:`onnx_ir.Model`) and ``self.io_dtype`` (an
    :class:`onnx_ir.DataType`) — both of which are set by
    :class:`~modelbuilder.builders.base.Model`.
    """

    # ------------------------------------------------------------------
    # ORT version detection
    # ------------------------------------------------------------------

    @staticmethod
    def _ort_version() -> tuple[int, ...]:
        """Return the installed onnxruntime version as an integer tuple.

        For example ``(1, 24, 4)``.

        Returns ``(99, 99, 0)`` when onnxruntime is not installed **or** when
        the version string cannot be parsed (e.g. a dev build with a
        non-numeric suffix).  Both cases are treated as "current enough" so
        that no local-function fallback is added.
        """
        try:
            import onnxruntime

            return tuple(int(x) for x in onnxruntime.__version__.split(".")[:3])
        except (ImportError, ValueError):
            return (99, 99, 0)

    # ------------------------------------------------------------------
    # CausalConvWithState local function
    # ------------------------------------------------------------------

    @classmethod
    def _make_causal_conv_local_function(cls, K: int, io_dtype: ir.DataType) -> ir.Function:
        """Build an ONNX local function that implements ``com.microsoft:CausalConvWithState``.

        The function body uses only standard ONNX opset-21 primitives so that
        runtimes without the native ``CausalConvWithState`` kernel (e.g. ORT < 1.26)
        can fall back to this definition automatically.

        The convolution kernel width *K* is unrolled at function-build time.
        The channel count *C* is inferred dynamically from the weight shape, so the
        same function definition handles any *C* value.

        Formal inputs
        -------------
        ``X``           – input tensor ``[B, C, S]``
        ``W``           – depthwise-conv weight ``[C, K]``  (squeezed from ``[C,1,K]``)
        ``bias``        – conv bias ``[C]``
        ``past_state``  – causal-conv carry state ``[B, C, K-1]``

        Formal outputs
        --------------
        ``Y``              – SiLU-activated output ``[B, C, S]``
        ``present_state``  – updated carry state ``[B, C, K-1]``
        """

        def mkv(name):
            return ir.Value(name=name)

        # Formal inputs / outputs
        X = mkv("X")
        X.dtype = io_dtype
        W = mkv("W")
        W.dtype = io_dtype
        bias = mkv("bias")
        bias.dtype = io_dtype
        past = mkv("past_state")
        past.dtype = io_dtype
        Y = mkv("Y")
        Y.dtype = io_dtype
        present = mkv("present_state")
        present.dtype = io_dtype

        nodes: list[ir.Node] = []

        def ci(name: str, data) -> ir.Value:
            """Create an INT64 Constant node; return its output Value."""
            t = ir.tensor(np.array(data, dtype=np.int64), name=name)
            v = mkv(name)
            v.dtype = ir.DataType.INT64
            nodes.append(ir.node("Constant", inputs=[], outputs=[v], attributes={"value": t}))
            return v

        # ---- 1. padded = Concat([past_state, X], axis=2) → [B, C, S+K-1] ----
        padded = mkv("padded")
        padded.dtype = io_dtype
        nodes.append(ir.node("Concat", inputs=[past, X], outputs=[padded], attributes={"axis": 2}))

        # ---- 2. S = Shape(X)[2] stored as a 1-D tensor for Slice inputs ----
        shape_x = mkv("shape_x")
        shape_x.dtype = ir.DataType.INT64
        nodes.append(ir.node("Shape", inputs=[X], outputs=[shape_x]))

        idx2 = ci("idx2", 2)  # scalar index 2
        s_scalar = mkv("s_scalar")
        s_scalar.dtype = ir.DataType.INT64
        nodes.append(ir.node("Gather", inputs=[shape_x, idx2], outputs=[s_scalar], attributes={"axis": 0}))

        idx0 = ci("idx0", [0])  # 1-D index [0] for Unsqueeze
        s_1d = mkv("s_1d")
        s_1d.dtype = ir.DataType.INT64
        nodes.append(ir.node("Unsqueeze", inputs=[s_scalar, idx0], outputs=[s_1d]))

        axes2 = ci("axes2", [2])  # slice along axis 2
        axes1 = ci("axes1", [1])  # slice along axis 1
        large_end = ci("large_end", [_ONNX_LARGE_SLICE_END])  # effectively +∞ for end-of-tensor slice
        one_1d = ci("one_1d", [1])  # constant [1] for shape construction

        # ---- 3. Dynamic reshape shape [1, C, 1] built from bias dimensions ----
        # Shape(bias) → 1-element int64 vector [C]; concat with [1] on each side.
        shape_bias = mkv("shape_bias")
        shape_bias.dtype = ir.DataType.INT64
        nodes.append(ir.node("Shape", inputs=[bias], outputs=[shape_bias]))

        w_shp = mkv("w_shp")  # [1, C, 1] as a shape vector
        w_shp.dtype = ir.DataType.INT64
        nodes.append(ir.node("Concat", inputs=[one_1d, shape_bias, one_1d], outputs=[w_shp], attributes={"axis": 0}))

        # Reshape bias from [C] to [1, C, 1] for broadcasting
        bias_r = mkv("bias_r")
        bias_r.dtype = io_dtype
        nodes.append(ir.node("Reshape", inputs=[bias, w_shp], outputs=[bias_r]))

        # ---- 4. Unrolled depthwise conv: contrib_k = W[:,k] * padded[:,:, k:k+S] ----
        contribs: list[ir.Value] = []
        for k in range(K):
            starts_k = ci(f"starts_{k}", [k])

            # ends_k = s_1d + k  (dynamic)
            if k == 0:
                ends_k = s_1d
            else:
                k_off = ci(f"k_offset_{k}", [k])
                ends_k = mkv(f"ends_{k}")
                ends_k.dtype = ir.DataType.INT64
                nodes.append(ir.node("Add", inputs=[s_1d, k_off], outputs=[ends_k]))

            # slice_k = padded[:, :, k : k+S]  → [B, C, S]
            slice_k = mkv(f"slice_{k}")
            slice_k.dtype = io_dtype
            nodes.append(ir.node("Slice", inputs=[padded, starts_k, ends_k, axes2], outputs=[slice_k]))

            # W_k = W[:, k:k+1]  → [C, 1]
            ws = ci(f"ws_{k}", [k])
            we = ci(f"we_{k}", [k + 1])
            wk_sl = mkv(f"wk_sl_{k}")
            wk_sl.dtype = io_dtype
            nodes.append(ir.node("Slice", inputs=[W, ws, we, axes1], outputs=[wk_sl]))

            # Reshape W_k from [C, 1] to [1, C, 1] for broadcast multiplication
            wk_r = mkv(f"wk_r_{k}")
            wk_r.dtype = io_dtype
            nodes.append(ir.node("Reshape", inputs=[wk_sl, w_shp], outputs=[wk_r]))

            contrib = mkv(f"contrib_{k}")
            contrib.dtype = io_dtype
            nodes.append(ir.node("Mul", inputs=[wk_r, slice_k], outputs=[contrib]))
            contribs.append(contrib)

        # Sum all K contributions
        total: ir.Value = contribs[0]
        for i in range(1, K):
            new_total = mkv(f"sum_{i}")
            new_total.dtype = io_dtype
            nodes.append(ir.node("Add", inputs=[total, contribs[i]], outputs=[new_total]))
            total = new_total

        # Add bias (broadcast [1, C, 1] + [B, C, S] → [B, C, S])
        pre_silu = mkv("pre_silu")
        pre_silu.dtype = io_dtype
        nodes.append(ir.node("Add", inputs=[total, bias_r], outputs=[pre_silu]))

        # SiLU activation: Y = x * sigmoid(x)
        sig = mkv("sig")
        sig.dtype = io_dtype
        nodes.append(ir.node("Sigmoid", inputs=[pre_silu], outputs=[sig]))
        nodes.append(ir.node("Mul", inputs=[pre_silu, sig], outputs=[Y]))

        # ---- 5. present_state = padded[:, :, S:] ----
        nodes.append(ir.node("Slice", inputs=[padded, s_1d, large_end, axes2], outputs=[present]))

        body = ir.Graph(
            inputs=(X, W, bias, past), outputs=(Y, present), nodes=nodes, opset_imports={"": 21}, name="CausalConvWithState_body"
        )
        return ir.Function("com.microsoft", "CausalConvWithState", "", graph=body, attributes={})

    def _register_causal_conv_local_function(self, K: int) -> None:
        """Register the ``CausalConvWithState`` local function if ORT < 1.26.

        The function is added to ``self.model.functions`` at most once per
        model (guarded by the function key lookup).

        Args:
            K: Convolution kernel width.  Derived from ``present_conv_shape[-1] + 1``.
        """
        if self._ort_version() < (1, 26):
            func_key = ("com.microsoft", "CausalConvWithState", "")
            if func_key not in self.model.functions:
                func = self._make_causal_conv_local_function(K, self.io_dtype)
                self.model.functions[func_key] = func
