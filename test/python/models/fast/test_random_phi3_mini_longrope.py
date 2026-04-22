# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest

from model_builder_test_case import ModelBuilderTestCase, hide_stdout, requires_cuda

MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"


class TestRandomPhi3MiniLongRoPE(ModelBuilderTestCase):
    def common_fast_phi3_mini_longrope_random_weights(self, precision, provider):
        from transformers import Phi3Config, Phi3ForCausalLM

        num_hidden_layers = 1

        # Minimal Phi3Config for the LongRoPE variant: max_position_embeddings must
        # differ from original_max_position_embeddings so that create_model selects
        # Phi3MiniLongRoPEModel.
        # head_size = hidden_size // num_attention_heads = 512 // 8 = 64
        # => rotary_dim_half = head_size // 2 = 32, so short/long factors need length 32.
        head_size = 64
        config = Phi3Config(
            architectures=["Phi3ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=8192,
            original_max_position_embeddings=4096,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_scaling={"type": "longrope", "short_factor": [1.0] * (head_size // 2), "long_factor": [1.0] * (head_size // 2)},
            vocab_size=32064,
        )

        model = Phi3ForCausalLM(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_discrepancies_phi3_mini_longrope_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=head_size,
            vocab_size=config.vocab_size,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
            atol={"fp16": 3e-2, "bf16": 2e-2, "fp32": 2e-3, "int4": 0.5},
            rtol={"fp16": 10, "bf16": 10, "fp32": 1e-4, "int4": 10000},
            kind="fast",
        )

    def common_phi3_mini_longrope_greedy_generation(self, precision, provider):
        import torch
        from transformers import Phi3Config, Phi3ForCausalLM

        num_hidden_layers = 1
        head_size = 64
        config = Phi3Config(
            architectures=["Phi3ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=8192,
            original_max_position_embeddings=4096,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_scaling={"type": "longrope", "short_factor": [1.0] * (head_size // 2), "long_factor": [1.0] * (head_size // 2)},
            vocab_size=32064,
        )

        torch.manual_seed(42)
        model = Phi3ForCausalLM(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_greedy_generation_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_generation_phi3_mini_longrope_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=head_size,
            vocab_size=config.vocab_size,
            eos_token_id=config.eos_token_id,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
        )

    @hide_stdout()
    def test_fast_discrepancy_phi3_mini_longrope_fp32_cpu(self):
        self.common_fast_phi3_mini_longrope_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_phi3_mini_longrope_fp16_cpu(self):
        self.common_fast_phi3_mini_longrope_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_phi3_mini_longrope_int4_cpu(self):
        self.common_fast_phi3_mini_longrope_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_phi3_mini_longrope_fp16_cuda(self):
        self.common_fast_phi3_mini_longrope_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_phi3_mini_longrope_bf16_cuda(self):
        self.common_fast_phi3_mini_longrope_random_weights("bf16", "cuda")

    @hide_stdout()
    def test_phi3_mini_longrope_fp32_cpu_greedy_generation(self):
        self.common_phi3_mini_longrope_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_phi3_mini_longrope_fp16_cpu_greedy_generation(self):
        self.common_phi3_mini_longrope_greedy_generation("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_phi3_mini_longrope_fp16_cuda_greedy_generation(self):
        self.common_phi3_mini_longrope_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_phi3_mini_longrope_bf16_cuda_greedy_generation(self):
        self.common_phi3_mini_longrope_greedy_generation("bf16", "cuda")

    @hide_stdout()
    def test_phi3_mini_longrope_fp32_cpu_genai_generate(self):
        import torch
        from transformers import Phi3Config, Phi3ForCausalLM

        from models.builder import create_model

        prefix = "test_phi3_mini_longrope_fp32_cpu_genai_generate"
        num_hidden_layers = 1
        head_size = 64
        config = Phi3Config(
            architectures=["Phi3ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=8192,
            original_max_position_embeddings=4096,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_scaling={"type": "longrope", "short_factor": [1.0] * (head_size // 2), "long_factor": [1.0] * (head_size // 2)},
            vocab_size=32064,
        )

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = Phi3ForCausalLM(config)
        model.eval()
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=MODEL_NAME,
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
