# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest

from model_builder_test_case import ModelBuilderTestCase, hide_stdout, requires_cuda, requires_transformers

ERNIE_MODEL_NAME = "baidu/ERNIE-4.5-0.3B-PT"


@requires_transformers("5")
class TestErnie4_5(ModelBuilderTestCase):
    def common_fast_ernie4_5_random_weights(self, precision, provider):
        from transformers import AutoModelForCausalLM, Ernie4_5Config

        num_hidden_layers = 1

        # Minimal Ernie4_5 config matching the Ernie4_5ForCausalLM architecture
        # but with small dimensions to keep the test fast and completely offline.
        # head_dim=64 matches hidden_size // num_attention_heads = 512 // 8.
        config = Ernie4_5Config(
            architectures=["Ernie4_5ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-05,
            rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
            use_bias=False,
            vocab_size=32000,
        )

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=ERNIE_MODEL_NAME,
            basename=f"test_discrepancies_ernie4_5_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
            kind="fast",
        )

    def common_ernie4_5_greedy_generation(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM, Ernie4_5Config

        num_hidden_layers = 1

        config = Ernie4_5Config(
            architectures=["Ernie4_5ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-05,
            rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
            use_bias=False,
            vocab_size=32000,
        )

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_greedy_generation_test(
            model=model,
            tokenizer=tokenizer,
            model_name=ERNIE_MODEL_NAME,
            basename=f"test_generation_ernie4_5_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
            eos_token_id=config.eos_token_id,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
        )

    @hide_stdout()
    def test_ernie4_5_fp32_cpu_greedy_generation(self):
        self.common_ernie4_5_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_ernie4_5_fp16_cpu_greedy_generation(self):
        self.common_ernie4_5_greedy_generation("fp16", "cpu")

    @unittest.skip("fails due to incorrect model")
    @hide_stdout()
    def test_ernie4_5_fp32_cuda_greedy_generation(self):
        self.common_ernie4_5_greedy_generation("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_ernie4_5_fp16_cuda_greedy_generation(self):
        self.common_ernie4_5_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_ernie4_5_bf16_cuda_greedy_generation(self):
        self.common_ernie4_5_greedy_generation("bf16", "cuda")

    @hide_stdout()
    def test_fast_discrepancy_ernie4_5_fp32_cpu(self):
        self.common_fast_ernie4_5_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_ernie4_5_fp16_cpu(self):
        self.common_fast_ernie4_5_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_ernie4_5_int4_cpu(self):
        self.common_fast_ernie4_5_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_ernie4_5_fp16_cuda(self):
        self.common_fast_ernie4_5_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_ernie4_5_bf16_cuda(self):
        self.common_fast_ernie4_5_random_weights("bf16", "cuda")

    @hide_stdout()
    def test_ernie4_5_fp32_cpu_genai_generate(self):
        import torch
        from transformers import AutoModelForCausalLM, Ernie4_5Config

        from models.builder import create_model

        prefix = "test_ernie4_5_fp32_cpu_genai_generate"
        num_hidden_layers = 1
        config = Ernie4_5Config(
            architectures=["Ernie4_5ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-05,
            rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
            use_bias=False,
            vocab_size=32000,
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
            model_name=ERNIE_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        self.run_genai_generation_test(output_dir, model, config.vocab_size, config.eos_token_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
