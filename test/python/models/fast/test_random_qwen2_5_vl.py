# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np
from model_builder_test_case import ModelBuilderTestCase, hide_stdout, requires_cuda, requires_transformers, run_session_or_io_binding

QWEN2_5_VL_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


@requires_transformers("5")  # text_config missing
class TestRandomQwen25VL(ModelBuilderTestCase):
    def common_fast_qwen25vl_random_weights(self, precision, provider):
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLTextConfig

        num_hidden_layers = 1

        # Minimal Qwen2.5-VL text config matching the architecture but with
        # small dimensions to keep the test fast and completely offline.
        text_config = Qwen2_5_VLTextConfig(
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32000,
            # mrope_section must satisfy: sum(mrope_section) * 2 == head_size
            # head_size = hidden_size // num_attention_heads = 512 // 8 = 64
            # so sum(mrope_section) must be 32: [8, 12, 12] sums to 32.
            rope_scaling={"type": "mrope", "mrope_section": [8, 12, 12]},
        )
        # Use a minimal vision config to avoid allocating the default 7B-scale
        # vision encoder (depth=32, hidden_size=3584) which causes OOM in CI.
        # Text-only inference never invokes the vision encoder, so its exact
        # dimensions don't affect correctness.
        vision_config = {
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "out_hidden_size": 64,
            "fullatt_block_indexes": [0],
        }
        config = Qwen2_5_VLConfig(text_config=text_config, vision_config=vision_config)
        config.architectures = ["Qwen2_5_VLForConditionalGeneration"]

        model = Qwen2_5_VLForConditionalGeneration(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()

        head_size = text_config.hidden_size // text_config.num_attention_heads
        self.run_vl_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=QWEN2_5_VL_MODEL_NAME,
            basename=f"test_discrepancies_qwen25vl_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=text_config.num_key_value_heads,
            head_size=head_size,
            vocab_size=text_config.vocab_size,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
            pt_mode="inputs_embeds",
        )

    @hide_stdout()
    def test_fast_discrepancy_qwen25vl_fp32_cpu(self):
        self.common_fast_qwen25vl_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_qwen25vl_fp16_cpu(self):
        self.common_fast_qwen25vl_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_qwen25vl_int4_cpu(self):
        self.common_fast_qwen25vl_random_weights("int4", "cpu")

    @unittest.skip("fails due to incorrect model")
    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen25vl_fp32_cuda(self):
        self.common_fast_qwen25vl_random_weights("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen25vl_fp16_cuda(self):
        self.common_fast_qwen25vl_random_weights("fp16", "cuda")

    @unittest.skip(
        "Could not find an implementation for MatMul(13) node with name '/model/layers.0/attn/q_proj/MatMul'"
    )
    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen25vl_bf16_cuda(self):
        self.common_fast_qwen25vl_random_weights("bf16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
