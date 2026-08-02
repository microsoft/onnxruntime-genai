"""Check the checkpoint adapter and the fp8 operator against reference math.

Two things can silently go wrong when re-laying-out a quantized checkpoint: the
nibble order of the fp4 experts and the axis the fp8 block scales run along.
Both are pinned here against `inference/convert.py`, the reference code shipped
with the weights.  The last section runs a real projection through
`com.microsoft.MatMulBlockQuantizedFp8Weight`, which also proves the operator is
present in the installed ONNX Runtime build.

    python check_weights.py --ckpt /path/to/DeepSeek-V4-Flash-0731
"""

import argparse
import os
import sys

import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "..", "src", "python", "py", "models")
sys.path.insert(0, os.path.abspath(_MODELS))

from builders.deepseek_v4 import mxfp4_dequantize, pack_for_qmoe  # noqa: E402
from builders.deepseek_v4_weights import DSV4Weights  # noqa: E402


def check(tag, got, want, tol=0.0):
    d = (got.float() - want.float()).abs().max().item()
    print(f"  {tag:28s} max|d|={d:.3e} {'OK' if d <= tol else 'FAIL'}")
    return d <= tol


def unpack_qmoe(packed, n):
    """QMoE `[E, K, N/2]` -> `[E, N, K]` fp4 codes."""
    e, k, _ = packed.shape
    codes = torch.stack([packed & 0x0F, packed >> 4], dim=-1).reshape(e, k, n)
    return codes.permute(0, 2, 1)


def check_adapter(a, w, cast_e2m1fn_to_e4m3fn):
    ok = True

    # -- block-scaled fp8: the operator reads scale[n, k // 128] --
    for key in ("attn.wq_a.weight", "attn.wkv.weight", "ffn.sw2.weight"):
        full = f"layers.{a.layer}.{key}"
        q, s = w.qweight(full)
        n, k = q.shape
        assert s.shape == (n, k // 128), s.shape
        got = q.float() * s.repeat_interleave(128, 1)[:, :k]
        ok &= check(key, got, w.store.dequant(w._resolve(full), dtype=torch.float32))

    # -- fp4 experts: nibble order and the QMoE transpose --
    p = f"layers.{a.layer}.ffn.experts"
    for which in ("w1", "w2"):
        want = []
        for e in range(a.experts):
            f8, s8 = cast_e2m1fn_to_e4m3fn(w.store[f"{p}.{e}.{which}.weight"],
                                           w.store[f"{p}.{e}.{which}.scale"])
            blk = torch.exp2(s8.view(torch.uint8).float() - 127.0)
            want.append(f8.float() * blk.repeat_interleave(128, 0).repeat_interleave(128, 1))
        want = torch.stack(want)

        n, half_k = w.store[f"{p}.0.{which}.weight"].shape
        blocks = torch.stack([
            w.store[f"{p}.{e}.{which}.weight"].view(torch.uint8).view(n, half_k // 16, 16)
            for e in range(a.experts)
        ])
        scales = torch.stack([w.store[f"{p}.{e}.{which}.scale"].view(torch.uint8)
                              for e in range(a.experts)])
        ok &= check(f"{which} mxfp4_dequantize", mxfp4_dequantize(blocks, scales), want)

        # The same codes after the round trip into QMoE's layout.
        codes = unpack_qmoe(pack_for_qmoe(blocks), n)
        ref = torch.stack([blocks & 0x0F, blocks >> 4], dim=-1).reshape(blocks.shape[0], n, -1)
        ok &= check(f"{which} pack_for_qmoe", codes.float(), ref.float())

    # -- fc1 is [gate, up] interleaved along the output dim --
    t = w[f"layers.{a.layer}.ffn.fc1_q"]
    n1, half_k1 = w.store[f"{p}.0.w1.weight"].shape
    k1 = half_k1 * 2
    assert t.shape == (a.experts, k1, n1), (t.shape, (a.experts, k1, n1))
    assert w[f"layers.{a.layer}.ffn.fc1_s"].shape == (a.experts, 2 * n1, k1 // 32)
    fc1 = unpack_qmoe(t, 2 * n1)
    for i, which in enumerate(("w1", "w3")):
        q = w.store[f"{p}.0.{which}.weight"].view(torch.uint8)
        ref = torch.stack([q & 0x0F, q >> 4], dim=-1).reshape(n1, k1)
        ok &= check(f"fc1 slot {which}", fc1[0, i::2].float(), ref.float())

    # -- renamed keys resolve --
    for key in ("gate_weight", "gate_bias", "sw1.weight"):
        full = f"layers.{a.layer}.ffn.{key}"
        assert full in w and w[full].numel() > 0, full
    print("  renamed keys                 OK")
    return ok


def check_fp8_op(a, w):
    key = f"layers.{a.layer}.attn.wq_a.weight"
    q, s = w.qweight(key)
    n, k = q.shape

    nodes = [
        helper.make_node("Cast", ["A"], ["Ab"], to=TensorProto.BFLOAT16),
        helper.make_node("MatMulBlockQuantizedFp8Weight", ["Ab", "B", "b_scale"], ["Yb"],
                         domain="com.microsoft", block_size=128),
        helper.make_node("Cast", ["Yb"], ["Y"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(
        nodes, "fp8_probe",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, ["M", k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["M", n])],
        initializer=[
            helper.make_tensor("B", TensorProto.FLOAT8E4M3FN, [n, k],
                               q.view(torch.uint8).numpy().tobytes(), raw=True),
            helper.make_tensor("b_scale", TensorProto.FLOAT, list(s.shape),
                               s.numpy().tobytes(), raw=True),
        ])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)])
    model.ir_version = 10
    path = os.path.join(a.tmp_dir, "fp8_probe.onnx")
    onnx.save(model, path)

    torch.manual_seed(0)
    x = torch.randn(17, k, dtype=torch.bfloat16)
    ref = x.float() @ w.store.dequant(key, dtype=torch.float32).T
    sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider"])
    got = torch.from_numpy(sess.run(None, {"A": x.float().numpy()})[0])

    d = (got - ref).abs().max().item()
    rel = d / ref.abs().max().item()
    ok = rel < 2e-2
    print(f"  MatMulBlockQuantizedFp8Weight max|d|={d:.3e} rel={rel:.3e} "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--tmp-dir", default="/tmp")
    a = ap.parse_args()
    a.ckpt = os.path.abspath(os.path.expanduser(a.ckpt))

    # `inference/convert.py` ships with the checkpoint and is the authority on
    # both quantization layouts.
    sys.path.insert(0, os.path.join(a.ckpt, "inference"))
    from convert import cast_e2m1fn_to_e4m3fn

    w = DSV4Weights(a.ckpt, expert_range=(0, a.experts))
    ok = check_adapter(a, w, cast_e2m1fn_to_e4m3fn)
    ok &= check_fp8_op(a, w)
    print("WEIGHT CHECK PASS" if ok else "WEIGHT CHECK FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
