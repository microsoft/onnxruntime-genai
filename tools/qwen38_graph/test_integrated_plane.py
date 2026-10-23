# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ctypes
import os
import unittest

import numpy as np

from test_plane_kernels import PlaneFunctions, load_plane_api


class IntegratedPlaneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = load_plane_api(os.environ.get("QWEN_GENAI_CUDA_LIBRARY"))

    def setUp(self):
        self.buffers = []
        self.rng = np.random.default_rng(1234)

    def tearDown(self):
        self.assertEqual(self.api.qg_stream_sync(None), 0)
        for pointer in self.buffers:
            self.assertEqual(self.api.qg_free_(pointer), 0)

    def device(self, array):
        array = np.ascontiguousarray(array)
        pointer = ctypes.c_void_p()
        self.assertEqual(self.api.qg_malloc_(ctypes.byref(pointer), array.nbytes), 0)
        self.buffers.append(pointer)
        self.assertEqual(self.api.qg_memcpy_h2d(pointer, array.ctypes.data, array.nbytes, None), 0)
        self.assertEqual(self.api.qg_stream_sync(None), 0)
        return pointer

    def read(self, pointer, shape, dtype):
        array = np.empty(shape, dtype=dtype)
        self.assertEqual(self.api.qg_memcpy_d2h(array.ctypes.data, pointer, array.nbytes, None), 0)
        self.assertEqual(self.api.qg_stream_sync(None), 0)
        return array

    def test_version_and_size_must_match(self):
        self.assertFalse(self.api._getter(0, ctypes.sizeof(PlaneFunctions)))
        self.assertFalse(self.api._getter(2, ctypes.sizeof(PlaneFunctions)))
        self.assertFalse(self.api._getter(1, ctypes.sizeof(PlaneFunctions) - 1))
        self.assertGreater(self.api.mtp_sizeof_params(), 0)

    def test_lattice_selection_and_invalid_inputs(self):
        for drafts in (1, 2, 3):
            for top_k in (1, 4, 16):
                candidates = np.arange(drafts * top_k, dtype=np.int32).reshape(drafts, top_k)
                scores = self.rng.standard_normal((drafts, top_k, top_k)).astype(np.float32)
                expected = np.full(16, -7, dtype=np.int64)
                control = self.device(expected)
                previous = 0
                for step, slot in enumerate((2, 8, 9)[:drafts]):
                    previous = int(scores[step, previous].argmax())
                    expected[slot] = candidates[step, previous]
                self.assertEqual(self.api.qg_dflash_select(
                    None, self.device(candidates), self.device(scores), control, drafts, top_k), 0)
                np.testing.assert_array_equal(self.read(control, 16, np.int64), expected)
        self.assertNotEqual(self.api.qg_dflash_select(None, None, None, None, 0, 0), 0)

    def test_bf16_argmax_and_tie_breaking(self):
        for length in (1, 257, 4096, 151936):
            logits = self.rng.standard_normal(length).astype(np.float32)
            logits[0] = 10
            logits[-1] = 10
            bf16 = (logits.view(np.uint32) >> 16).astype(np.uint16)
            blocks = (length + 255) // 256
            values = self.device(np.zeros(blocks, dtype=np.float32))
            indices = self.device(np.zeros(blocks, dtype=np.int32))
            output = self.device(np.zeros(1, dtype=np.int32))
            self.api.mtp_argmax_part(None, self.device(bf16), length, values, indices, blocks)
            self.assertEqual(self.api.qg_argmax_finish(
                None, values, indices, blocks, output), 0)
            self.assertEqual(self.read(output, 1, np.int32).item(), int(logits.argmax()))
        self.assertNotEqual(self.api.qg_argmax_finish(None, None, None, 0, None), 0)

    def test_transpose(self):
        src = self.rng.standard_normal((2, 128, 128)).astype(np.float32)
        dst = self.device(np.zeros_like(src))
        self.api.mtp_transpose_f32(None, self.device(src), dst, 128, 2)
        np.testing.assert_array_equal(self.read(dst, src.shape, np.float32), src.swapaxes(-1, -2))


if __name__ == "__main__":
    unittest.main()
