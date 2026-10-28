# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import tempfile
import unittest
from pathlib import Path

import torch
from builders.qwen3_8_flash.quantization import (
    SafetensorsCheckpoint,
    dequantize_nvfp4,
    prepack_int2,
    quantize_expert,
    quantize_int2,
)
from safetensors.torch import save_file


class Nvfp4Int2Tests(unittest.TestCase):
    def test_nvfp4_nibble_order(self):
        packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8)
        scales = torch.ones((1, 1)).to(torch.float8_e4m3fn)
        expected = torch.tensor([[0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]])
        torch.testing.assert_close(dequantize_nvfp4(packed, scales, torch.tensor(1.0)), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            dequantize_nvfp4(packed, scales.view(torch.uint8), torch.tensor(1.0)), expected, rtol=0, atol=0
        )

    def test_nvfp4_rounds_combined_scale_before_product(self):
        packed = torch.full((1, 8), 0x77, dtype=torch.uint8)
        scales = torch.tensor([[1.5]]).to(torch.float8_e4m3fn)
        global_scale = torch.tensor(0.1)
        expected = ((scales.float() * global_scale).to(torch.bfloat16).float() * 6).to(torch.bfloat16).float()
        direct = (scales.float() * global_scale * 6).to(torch.bfloat16).float()
        self.assertFalse(torch.equal(expected, direct))
        torch.testing.assert_close(
            dequantize_nvfp4(packed, scales, global_scale), expected.expand(1, 16), rtol=0, atol=0
        )

    def test_nvfp4_zero_scale_is_valid(self):
        actual = dequantize_nvfp4(
            torch.full((1, 8), 0x77, dtype=torch.uint8),
            torch.zeros((1, 1)).to(torch.float8_e4m3fn),
            torch.tensor(1.0),
        )
        self.assertEqual(torch.count_nonzero(actual).item(), 0)

    def test_nvfp4_rejects_invalid_inputs(self):
        packed = torch.zeros((1, 8), dtype=torch.uint8)
        scales = torch.ones((1, 1)).to(torch.float8_e4m3fn)
        for bad in (torch.tensor(0.0), torch.tensor(-1.0), torch.tensor(float("nan")), torch.tensor(float("inf"))):
            with self.subTest(global_scale=bad), self.assertRaisesRegex(ValueError, "finite and positive"):
                dequantize_nvfp4(packed, scales, bad)
        for bad in (torch.tensor([[-1.0]]), torch.tensor([[float("nan")]])):
            with self.subTest(scales=bad), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                dequantize_nvfp4(packed, bad.to(torch.float8_e4m3fn), torch.tensor(1.0))
        with self.assertRaisesRegex(ValueError, "shape"):
            dequantize_nvfp4(packed, torch.ones((1, 2)).to(torch.float8_e4m3fn), torch.tensor(1.0))
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            dequantize_nvfp4(packed[:, :7], scales, torch.tensor(1.0))

    def test_int2_code_order_and_both_scale_orientations(self):
        for sign in (1, -1):
            weights = torch.tensor([[-2.0, -1.0, 0.0, 1.0]]).repeat(3, 16) * sign
            result = quantize_int2(weights)
            self.assertEqual(result.packed.shape, (3, 16))
            self.assertEqual(result.packed.dtype, torch.uint8)
            self.assertTrue(torch.all(result.packed == 0xE4))
            torch.testing.assert_close(result.scales, torch.full((3, 1), sign, dtype=torch.bfloat16))
            self.assertEqual(result.squared_error, 0)
            self.assertEqual(result.squared_norm, float(weights.square().sum()))

    def test_int2_zero_and_orientation_tie(self):
        result = quantize_int2(torch.zeros((2, 128)))
        self.assertTrue(torch.all(result.packed == 0xAA))
        self.assertTrue(torch.all(result.scales > 0))
        self.assertTrue(torch.isfinite(result.scales).all())
        self.assertEqual(result.squared_error, 0)
        result = quantize_int2(torch.ones((1, 64)))
        self.assertTrue(torch.all(result.scales == 1))
        self.assertTrue(torch.all(result.packed == 0xFF))

    def test_int2_error_matches_decoded_bytes(self):
        weights = torch.randn((5, 192), generator=torch.Generator().manual_seed(1234))
        result = quantize_int2(weights)
        codes = torch.stack([(result.packed >> shift) & 3 for shift in (0, 2, 4, 6)], dim=-1).float() - 2
        decoded = codes.reshape(5, 3, 64) * result.scales.float().unsqueeze(-1)
        self.assertEqual(result.scales.shape, (5, 3))
        self.assertEqual(result.scales.dtype, torch.bfloat16)
        self.assertEqual(result.squared_error, float((weights.reshape(5, 3, 64) - decoded).square().sum()))
        self.assertEqual(result.squared_norm, float(weights.square().sum()))

    def test_int2_rejects_invalid_inputs(self):
        for bad in (
            torch.zeros((0, 64)),
            torch.zeros((1, 0)),
            torch.zeros((1, 63)),
            torch.ones((1, 64), dtype=torch.int32),
        ):
            with self.subTest(shape=bad.shape, dtype=bad.dtype), self.assertRaises(ValueError):
                quantize_int2(bad)
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                quantize_int2(torch.full((1, 64), value))
        with self.assertRaisesRegex(ValueError, "projection"):
            quantize_expert(None, 0, 0, "other")
        with self.assertRaisesRegex(ValueError, "index"):
            quantize_expert(None, 48, 0, "gate_proj")

    def test_prepack_int2_scalar_layout_reference(self):
        generator = torch.Generator().manual_seed(1234)
        for n, k in ((8, 64), (16, 128), (24, 192)):
            packed = torch.randint(0, 256, (n, k // 4), dtype=torch.uint8, generator=generator)
            before = packed.clone()
            output = torch.zeros(n * k // 4, dtype=torch.uint8)
            for row in range(n):
                for col in range(k):
                    source_col = col // 64 * 64 + col % 16 // 2 * 8 + col % 2 + col % 64 // 16 * 2
                    code = (int(packed[row, source_col // 4]) >> (2 * (source_col % 4))) & 3
                    index = (row // 8 * (k // 64) + col // 64) * 512 + row % 8 * 64 + col % 64
                    index = index // 16 * 16 + index % 16 // 2 + index % 2 * 8
                    output[index // 4] |= code << (index % 4 * 2)
            torch.testing.assert_close(prepack_int2(packed), output.reshape_as(packed), rtol=0, atol=0)
            torch.testing.assert_close(packed, before, rtol=0, atol=0)

    def test_prepack_int2_rejects_invalid_shapes(self):
        for packed in (
            torch.zeros(8, 16),
            torch.zeros((8, 16, 1), dtype=torch.uint8),
            torch.zeros((7, 16), dtype=torch.uint8),
            torch.zeros((8, 15), dtype=torch.uint8),
            torch.zeros((0, 16), dtype=torch.uint8),
        ):
            with self.subTest(shape=packed.shape, dtype=packed.dtype), self.assertRaises(ValueError):
                prepack_int2(packed)

    def test_checkpoint_indexed_and_single_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = torch.arange(8)
            save_file({"first": expected}, root / "first.safetensors")
            with SafetensorsCheckpoint(root) as checkpoint:
                torch.testing.assert_close(checkpoint.tensor("first"), expected)
                with self.assertRaisesRegex(KeyError, "Missing"):
                    checkpoint.tensor("missing")
            with self.assertRaisesRegex(RuntimeError, "closed"):
                checkpoint.tensor("first")
            save_file({"second": expected + 1}, root / "second.safetensors")
            with self.assertRaisesRegex(ValueError, "index or exactly one"):
                SafetensorsCheckpoint(root)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"first": "first.safetensors", "second": "second.safetensors"}})
            )
            with SafetensorsCheckpoint(root) as checkpoint:
                torch.testing.assert_close(checkpoint.tensor("first"), expected)
                torch.testing.assert_close(checkpoint.tensor("second"), expected + 1)

    def test_checkpoint_rejects_shard_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            save_file({"outside": torch.zeros(1)}, root / "outside.safetensors")
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"outside": "../outside.safetensors"}})
            )
            with self.assertRaisesRegex(ValueError, "escapes checkpoint"):
                SafetensorsCheckpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
