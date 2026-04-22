# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest

from model_builder_test_case import ModelBuilderTestCase, hide_stdout, requires_cuda

SMOLLM3_MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"


class TestSmolLM3(ModelBuilderTestCase):
    def common_fast_smollm3_random_weights(self, precision, provider):
        from transformers import AutoModelForCausalLM
        from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config

        # num_hidden_layers=4 is required so that both rope and no-rope
        # layers are exercised (no_rope_layers=[1,1,1,0] by default).
        num_hidden_layers = 4
        config = SmolLM3Config(
            architectures=["SmolLM3ForCausalLM"],
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            vocab_size=128256,
            pad_token_id=None,
        )

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=SMOLLM3_MODEL_NAME,
            basename=f"test_discrepancies_smollm3_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.hidden_size // config.num_attention_heads,
            vocab_size=config.vocab_size,
        )

    def common_smollm3_greedy_generation(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config

        # num_hidden_layers=4 is required so that both rope and no-rope
        # layers are exercised (no_rope_layers=[1,1,1,0] by default).
        num_hidden_layers = 4
        config = SmolLM3Config(
            architectures=["SmolLM3ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            vocab_size=128256,
            pad_token_id=None,
        )

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_greedy_generation_test(
            model=model,
            tokenizer=tokenizer,
            model_name=SMOLLM3_MODEL_NAME,
            basename=f"test_generation_smollm3_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.hidden_size // config.num_attention_heads,
            vocab_size=config.vocab_size,
            eos_token_id=config.eos_token_id,
        )

    @hide_stdout()
    def test_smollm3_fp32_cpu_greedy_generation(self):
        self.common_smollm3_greedy_generation("fp32", "cpu")

    @unittest.skip("issue")
    @hide_stdout()
    def test_smollm3_fp16_cpu_greedy_generation(self):
        self.common_smollm3_greedy_generation("fp16", "cpu")

    @unittest.skip("fails due to incorrect model")
    @hide_stdout()
    def test_smollm3_fp32_cuda_greedy_generation(self):
        self.common_smollm3_greedy_generation("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_smollm3_fp16_cuda_greedy_generation(self):
        self.common_smollm3_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_smollm3_bf16_cuda_greedy_generation(self):
        self.common_smollm3_greedy_generation("bf16", "cuda")

    @hide_stdout()
    def test_fast_discrepancy_smollm3_fp32_cpu(self):
        self.common_fast_smollm3_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_smollm3_fp16_cpu(self):
        self.common_fast_smollm3_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_smollm3_int4_cpu(self):
        self.common_fast_smollm3_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_smollm3_fp16_cuda(self):
        self.common_fast_smollm3_random_weights("fp16", "cuda")

    @hide_stdout()
    def test_smollm3_fp32_cpu_genai_generate(self):
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config

        from models.builder import create_model

        prefix = "test_smollm3_fp32_cpu_genai_generate"
        num_hidden_layers = 4
        config = SmolLM3Config(
            architectures=["SmolLM3ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            vocab_size=128256,
            pad_token_id=None,
        )

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=SMOLLM3_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
        )

        self.run_genai_generation_test(output_dir, model, config.vocab_size, config.eos_token_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
