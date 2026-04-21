# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import itertools
import os
import unittest

import numpy as np
from model_builder_test_case import ModelBuilderTestCase, hide_stdout, requires_cuda, run_session_or_io_binding

MISTRAL_NEMO_MODEL_NAME = "mistralai/Mistral-Nemo-Instruct-2407"


class TestMistralNeMo(ModelBuilderTestCase):
    def common_fast_mistral_nemo_random_weights(self, precision, provider):
        import torch
        from models.builder import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, MistralConfig, PreTrainedTokenizerFast

        num_hidden_layers = 1
        config = MistralConfig(
            architectures=["MistralNeMoForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            model_type="mistral",
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_theta=1000000.0,
            sliding_window=None,
            vocab_size=32000,
        )

        basename = f"test_discrepancies_mistral_nemo_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MISTRAL_NEMO_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        log_data = dict(
            precision=precision,
            model_id=MISTRAL_NEMO_MODEL_NAME,
            experiment="forward",
            provider=provider,
            test=basename,
            input_type="text",
            kind="random",
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self.check_ort(onnx_path)

        batch_size = 1
        seq_len = 5
        head_size = config.hidden_size // config.num_attention_heads

        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(provider)
        onnx_input_names = [i.name for i in sess.get_inputs()]

        with self.subTest(step="prefill"):
            prefill_feed = {
                "input_ids": input_ids.cpu().numpy().astype(np.int64),
                "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
            }
            for i in range(num_hidden_layers):
                prefill_feed[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
                prefill_feed[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
            prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}

            prefill_results, ort_logits_np = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=prefill_feed,
                sess=sess,
                vocab_size=config.vocab_size,
            )

            with torch.no_grad():
                pt_prefill = model(input_ids)

            np_prefill = pt_prefill.logits.detach().cpu().numpy()
            disc = self.get_numpy_discrepancy(np_prefill[:, :1, :], ort_logits_np[:, :1, :])
            self.log_results({"step": "prefill", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}
            self.assertEqual(np_prefill.shape, ort_logits_np.shape)
            # Verify first-token logits are numerically close; subsequent positions
            # can diverge slightly in FP32 due to GQA kernel differences.
            np.testing.assert_allclose(np_prefill[:, :1, :], ort_logits_np[:, :1, :], atol=atol[precision], rtol=1e-3)

        with self.subTest(step="decode"):
            next_token = int(np.argmax(prefill_results["logits"][0, -1, :]))

            decode_feed = {
                "input_ids": np.array([[next_token]], dtype=np.int64),
                "attention_mask": np.ones((batch_size, seq_len + 1), dtype=np.int64),
                "position_ids": np.array([[seq_len]], dtype=np.int64),
            }
            for i in range(num_hidden_layers):
                decode_feed[f"past_key_values.{i}.key"] = prefill_results[f"present.{i}.key"]
                decode_feed[f"past_key_values.{i}.value"] = prefill_results[f"present.{i}.value"]
            decode_feed = {k: v for k, v in decode_feed.items() if k in onnx_input_names}

            prefill_results, onnx_decode_logits = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=decode_feed,
                sess=sess,
                vocab_size=config.vocab_size,
                results=prefill_results,
            )

            with torch.no_grad():
                pt_past_kv = pt_prefill.past_key_values
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(provider)
                pt_decode = model(next_token_tensor, past_key_values=pt_past_kv)
                pt_decode_logits = pt_decode.logits.detach().cpu().numpy()

            disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
            self.log_results({"step": "decode", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}
            rtol = {"fp16": 10, "bf16": 1e-2, "fp32": 1e-3, "int4": 10000}
            np.testing.assert_allclose(pt_decode_logits, onnx_decode_logits, atol=atol[precision], rtol=rtol[precision])

    def common_mistral_nemo_greedy_generation(self, precision, provider):
        import torch
        from models.builder import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, MistralConfig, PreTrainedTokenizerFast

        num_hidden_layers = 1
        config = MistralConfig(
            architectures=["MistralNeMoForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            model_type="mistral",
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_theta=1000000.0,
            sliding_window=None,
            vocab_size=32000,
        )

        basename = f"test_generation_mistral_nemo_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MISTRAL_NEMO_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")

        input_names = {inp.name for inp in sess.get_inputs()}

        batch_size = 1
        head_size = config.hidden_size // config.num_attention_heads
        max_new_tokens = 10

        torch.manual_seed(0)
        prompt_ids = torch.randint(3, config.vocab_size, (batch_size, 5)).to(provider)

        with torch.no_grad():
            pt_output = model.generate(
                prompt_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=config.eos_token_id
            )
        pt_tokens = pt_output[0].tolist()

        current_ids = prompt_ids.detach().cpu().numpy().astype(np.int64)

        past_kv = {}
        for i in range(num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
            )
            past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
            )

        onnx_tokens = current_ids[0].tolist()
        results = None
        for _ in range(max_new_tokens):
            past_len = past_kv["past_key_values.0.key"].shape[2]
            cur_len = current_ids.shape[1]

            feed = {
                "input_ids": current_ids,
                "attention_mask": np.ones((batch_size, past_len + cur_len), dtype=np.int64),
                "position_ids": np.arange(past_len, past_len + cur_len, dtype=np.int64).reshape(batch_size, cur_len),
            }
            for i in range(num_hidden_layers):
                feed[f"past_key_values.{i}.key"] = past_kv[f"past_key_values.{i}.key"]
                feed[f"past_key_values.{i}.value"] = past_kv[f"past_key_values.{i}.value"]
            feed = {k: v for k, v in feed.items() if k in input_names}

            results, _ = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=feed,
                sess=sess,
                vocab_size=config.vocab_size,
                results=results,
            )

            next_token = int(np.argmax(results["logits"][0, -1, :]))
            onnx_tokens.append(next_token)

            for i in range(num_hidden_layers):
                past_kv[f"past_key_values.{i}.key"] = results[f"present.{i}.key"]
                past_kv[f"past_key_values.{i}.value"] = results[f"present.{i}.value"]

            current_ids = np.array([[next_token]], dtype=np.int64)

            if next_token == config.eos_token_id:
                break

        diff = self.first_token_diff(pt_tokens, onnx_tokens)
        diff.update(
            dict(
                precision=precision,
                model_id=MISTRAL_NEMO_MODEL_NAME,
                experiment="generate",
                provider=provider,
                test=basename,
                input_type="text",
                kind="fast",
            )
        )
        self.log_results(diff)
        if precision in ("fp16", "bf16"):
            pt_tokens = pt_tokens[:-5]
            onnx_tokens = onnx_tokens[:-5]
        self.assertEqual(pt_tokens, onnx_tokens)

    @hide_stdout()
    def test_mistral_nemo_fp32_cpu_greedy_generation(self):
        self.common_mistral_nemo_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_mistral_nemo_fp16_cpu_greedy_generation(self):
        self.common_mistral_nemo_greedy_generation("fp16", "cpu")

    @unittest.skip("fails due to incorrect model")
    @hide_stdout()
    def test_mistral_nemo_fp32_cuda_greedy_generation(self):
        self.common_mistral_nemo_greedy_generation("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_mistral_nemo_fp16_cuda_greedy_generation(self):
        self.common_mistral_nemo_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_mistral_nemo_bf16_cuda_greedy_generation(self):
        self.common_mistral_nemo_greedy_generation("bf16", "cuda")

    @hide_stdout()
    def test_fast_discrepancy_mistral_nemo_fp32_cpu(self):
        self.common_fast_mistral_nemo_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_mistral_nemo_fp16_cpu(self):
        self.common_fast_mistral_nemo_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_mistral_nemo_int4_cpu(self):
        self.common_fast_mistral_nemo_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_mistral_nemo_fp16_cuda(self):
        self.common_fast_mistral_nemo_random_weights("fp16", "cuda")

    @hide_stdout()
    def test_mistral_nemo_torch_onnx_export(self):
        """
        Verify that a randomly-initialised MistralNeMoForCausalLM can be
        exported to ONNX using ``torch.onnx.export`` with ``attention_mask``
        and a dynamic KV-cache (present/past key-value tensors).

        The model is wrapped so that ``torch.onnx.export`` sees plain tensor
        inputs and outputs instead of the ``DynamicCache`` objects that
        transformers uses internally.  The wrapper:

        * accepts ``input_ids``, ``attention_mask``, and one pair of
          ``past_key`` / ``past_value`` tensors per hidden layer;
        * reconstructs a ``DynamicCache`` from those tensors before calling
          the underlying model with ``use_cache=True``; and
        * unpacks the updated KV-cache back into plain tensors for the
          outputs.

        Using random weights and a single hidden layer avoids downloading any
        files from Hugging Face, keeping the test completely offline.

        The test verifies that:
        * ``torch.onnx.export`` completes without error.
        * The exported ONNX file can be loaded by ``onnxruntime``.
        * The ONNX logits closely match those of the original PyTorch model
          for both a **prefill** step (empty KV-cache) and a **decode** step
          (non-empty KV-cache from the prefill).
        """
        import torch
        from transformers import AutoModelForCausalLM, MistralConfig
        from transformers.cache_utils import DynamicCache

        num_hidden_layers = 1
        config = MistralConfig(
            architectures=["MistralNeMoForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            model_type="mistral",
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_theta=1000000.0,
            sliding_window=None,
            vocab_size=32000,
        )

        output_dir, _ = self.get_dirs("test_mistral_nemo_torch_onnx_export")
        onnx_path = os.path.join(output_dir, "model.onnx")

        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()

        batch_size = 1
        seq_len = 5
        head_size = config.hidden_size // config.num_attention_heads
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        # Empty past KV-cache tensors for the prefill step.
        past_key = torch.zeros(batch_size, config.num_key_value_heads, 0, head_size)
        past_value = torch.zeros(batch_size, config.num_key_value_heads, 0, head_size)

        # Wrapper that presents plain tensors to torch.onnx.export while
        # using DynamicCache internally (transformers >=5 requirement).
        class MistralNeMoWithKVCache(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask, past_key, past_value):
                cache = DynamicCache()
                cache.update(past_key, past_value, layer_idx=0)
                out = self.m(input_ids=input_ids, attention_mask=attention_mask, past_key_values=cache, use_cache=True)
                layer = out.past_key_values.layers[0]
                return out.logits, layer.keys, layer.values

        wrapper = MistralNeMoWithKVCache(model)
        wrapper.eval()

        # Capture PyTorch reference outputs for the prefill step.
        with torch.no_grad():
            pt_logits, pt_present_key, pt_present_value = wrapper(input_ids, attention_mask, past_key, past_value)

        # Export to ONNX using torch.onnx.export.
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (input_ids, attention_mask, past_key, past_value),
                onnx_path,
                input_names=["input_ids", "attention_mask", "past_key_values.0.key", "past_key_values.0.value"],
                output_names=["logits", "present.0.key", "present.0.value"],
                dynamic_shapes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
                    "past_key": {0: "batch_size", 2: "past_sequence_length"},
                    "past_value": {0: "batch_size", 2: "past_sequence_length"},
                },
                opset_version=22,
            )

        self.assertExists(onnx_path)
        sess = self.check_ort(onnx_path)

        # ------------------------------------------------------------------
        # Step 1: prefill — empty KV-cache
        # ------------------------------------------------------------------
        prefill_out = sess.run(
            None,
            {
                "input_ids": input_ids.numpy().astype(np.int64),
                "attention_mask": attention_mask.numpy().astype(np.int64),
                "past_key_values.0.key": past_key.numpy(),
                "past_key_values.0.value": past_value.numpy(),
            },
        )
        onnx_logits, onnx_present_key, onnx_present_value = prefill_out

        disc = self.get_numpy_discrepancy(pt_logits.numpy(), onnx_logits)
        self.log_results({"step": "prefill", **disc})
        self.assertLess(disc["max_abs_err"], 1e-3)

        # ------------------------------------------------------------------
        # Step 2: decode — use KV-cache produced by the prefill step
        # ------------------------------------------------------------------
        decode_ids = torch.randint(0, config.vocab_size, (batch_size, 1))
        decode_mask = torch.ones(batch_size, seq_len + 1, dtype=torch.long)

        with torch.no_grad():
            pt_dec_logits, _, _ = wrapper(decode_ids, decode_mask, pt_present_key, pt_present_value)

        dec_out = sess.run(
            None,
            {
                "input_ids": decode_ids.numpy().astype(np.int64),
                "attention_mask": decode_mask.numpy().astype(np.int64),
                "past_key_values.0.key": onnx_present_key,
                "past_key_values.0.value": onnx_present_value,
            },
        )
        disc = self.get_numpy_discrepancy(pt_dec_logits.numpy(), dec_out[0])
        self.log_results({"step": "decode", **disc})
        self.assertLess(disc["max_abs_err"], 1e-3)

    @hide_stdout()
    @unittest.skip("https://github.com/pytorch/pytorch/issues/179555")
    def test_mistral_nemo_torch_onnx_export_custom_flatten(self):
        """
        Verify that a randomly-initialised MistralNeMoForCausalLM can be
        exported to ONNX using ``torch.onnx.export`` with ``attention_mask``
        and a dynamic KV-cache (present/past key-value tensors).

        The model is wrapped so that ``torch.onnx.export`` sees plain tensor
        inputs and outputs instead of the ``DynamicCache`` objects that
        transformers uses internally.  The wrapper:

        * accepts ``input_ids``, ``attention_mask``, and one pair of
          ``past_key`` / ``past_value`` tensors per hidden layer;
        * reconstructs a ``DynamicCache`` from those tensors before calling
          the underlying model with ``use_cache=True``; and
        * unpacks the updated KV-cache back into plain tensors for the
          outputs.

        Using random weights and a single hidden layer avoids downloading any
        files from Hugging Face, keeping the test completely offline.

        The test verifies that:
        * ``torch.onnx.export`` completes without error.
        * The exported ONNX file can be loaded by ``onnxruntime``.
        * The ONNX logits closely match those of the original PyTorch model
          for both a **prefill** step (empty KV-cache) and a **decode** step
          (non-empty KV-cache from the prefill).
        """
        import torch
        import transformers
        from models.helpers.cache_helper import registers_dynamic_cache
        from transformers import AutoModelForCausalLM, MistralConfig

        num_hidden_layers = 1
        config = MistralConfig(
            architectures=["MistralNeMoForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            model_type="mistral",
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_theta=1000000.0,
            sliding_window=None,
            vocab_size=32000,
        )

        output_dir, _ = self.get_dirs("test_mistral_nemo_torch_onnx_export_custom_flatten")
        onnx_path = os.path.join(output_dir, "model.onnx")

        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        device = "cpu"

        batch_size = 2
        seq_len = 3
        past_len = 30
        head_size = config.hidden_size // config.num_attention_heads
        num_heads = config.num_key_value_heads
        dc = transformers.cache_utils.DynamicCache()
        dc.update(
            torch.rand((batch_size, num_heads, past_len, head_size)).to(device),
            torch.rand((batch_size, num_heads, past_len, head_size)).to(device),
            layer_idx=0,
        )
        inputs = dict(
            input_ids=torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.int64).to(device),
            attention_mask=torch.randint(0, 1, (batch_size, seq_len + past_len), dtype=torch.int64).to(device),
            past_key_values=dc,
        )
        dynamic_shapes = {
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "past_len+seq_len"},
            "past_key_values": [{0: "batch", 2: "past_len"}, {0: "batch", 2: "past_len"}],
        }

        # Capture PyTorch reference outputs for the prefill step.
        with torch.no_grad():
            expected = model(**inputs)

        # Export to ONNX using torch.onnx.export.
        registers_dynamic_cache()
        with torch.no_grad():
            torch.onnx.export(model, (), onnx_path, kwargs=inputs, dynamic_shapes=dynamic_shapes, opset_version=22)

        self.assertExists(onnx_path)
        sess = self.check_ort(onnx_path)

        # ------------------------------------------------------------------
        # Step 1: prefill — empty KV-cache
        # ------------------------------------------------------------------
        feeds = dict(
            zip(
                [i.name for i in sess.get_inputs()],
                [
                    inputs["input_ids"].numpy().astype(np.int64),
                    inputs["attention_mask"].numpy().astype(np.int64),
                    inputs["past_key_values"].layers[0].keys.numpy(),
                    inputs["past_key_values"].layers[0].values.numpy(),
                ],
            )
        )
        prefill_out = sess.run(None, feeds)
        onnx_logits, onnx_present_key, onnx_present_value = prefill_out

        np.testing.assert_allclose(expected.last_hidden_state.numpy(), onnx_logits, atol=1e-4, rtol=1e-4)

        # ------------------------------------------------------------------
        # Step 2: decode — use KV-cache produced by the prefill step
        # ------------------------------------------------------------------
        decode_ids = torch.randint(0, config.vocab_size, (batch_size, 1))
        decode_mask = torch.ones(batch_size, seq_len + 1, dtype=torch.long)
        dc = transformers.cache_utils.DynamicCache()
        dc.update(torch.from_numpy(onnx_present_key), torch.from_numpy(onnx_present_value), layer_idx=0)

        with torch.no_grad():
            expected = model(input_ids=decode_ids, attention_mask=decode_mask, past_key_values=dc)

        feeds = dict(
            zip(
                [i.name for i in sess.get_inputs()],
                [
                    decode_ids.numpy().astype(np.int64),
                    decode_mask.numpy().astype(np.int64),
                    onnx_present_key,
                    onnx_present_value,
                ],
            )
        )

        dec_out = sess.run(None, feeds)
        np.testing.assert_allclose(expected.last_hidden_state.numpy(), dec_out[0], atol=1e-4, rtol=1e-4)

    @hide_stdout()
    @unittest.skip("https://github.com/pytorch/pytorch/issues/179555")
    def test_mistral_nemo_torch_exoprt_export_issue(self):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, MistralConfig

        num_hidden_layers = 1
        config = MistralConfig(
            architectures=["MistralNeMoForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            model_type="mistral",
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_theta=1000000.0,
            sliding_window=None,
            vocab_size=32000,
        )

        output_dir, _ = self.get_dirs("test_mistral_nemo_torch_exoprt_export_issue")
        ep_path = os.path.join(output_dir, "ep.txt")
        ep_flatten_path = os.path.join(output_dir, "ep_flatten.txt")

        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        device = "cpu"

        batch_size = 2
        seq_len = 3
        past_len = 30
        head_size = config.hidden_size // config.num_attention_heads
        num_heads = config.num_key_value_heads
        dc = transformers.cache_utils.DynamicCache()
        dc.update(
            torch.rand((batch_size, num_heads, past_len, head_size)).to(device),
            torch.rand((batch_size, num_heads, past_len, head_size)).to(device),
            layer_idx=0,
        )
        inputs = dict(
            input_ids=torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.int64).to(device),
            attention_mask=torch.randint(0, 1, (batch_size, seq_len + past_len), dtype=torch.int64).to(device),
            past_key_values=dc,
        )

        ############
        # First wrap
        ############

        class MistralNeMoWithKVCache(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask, past_key, past_value):
                cache = transformers.cache_utils.DynamicCache()
                cache.update(past_key, past_value, layer_idx=0)
                out = self.m(input_ids=input_ids, attention_mask=attention_mask, past_key_values=cache, use_cache=True)
                layer = out.past_key_values.layers[0]
                return out.logits, layer.keys, layer.values

        wrapper = MistralNeMoWithKVCache(model)
        wrapper.eval()

        DYN = torch.export.Dim.DYNAMIC
        ep = torch.export.export(
            wrapper,
            (),
            kwargs=dict(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                past_key=inputs["past_key_values"].layers[0].keys,
                past_value=inputs["past_key_values"].layers[0].values,
            ),
            dynamic_shapes=dict(
                input_ids={0: DYN, 1: DYN},
                attention_mask={0: DYN, 1: DYN},
                past_key={0: DYN, 2: DYN},
                past_value={0: DYN, 2: DYN},
            ),
        )
        with open(ep_path, "w") as f:
            f.write(str(ep))
        ep.run_decompositions()

        ############
        # Flattening
        ############

        def _flatten_key_value_cache(cache):
            keys = [lay.keys for lay in cache.layers]
            values = [lay.values for lay in cache.layers]
            flat = list(itertools.chain.from_iterable(zip(keys, values)))
            unique = set(type(lay) for lay in cache.layers)
            assert unique == {transformers.cache_utils.DynamicLayer}, f"Not implemented for layers type {unique}"
            keys = list(itertools.chain.from_iterable((f"key_{i}", f"value_{i}") for i in range(len(cache.layers))))
            return flat, keys

        def _flatten_with_keys_cache(cache):
            values, context = _flatten_key_value_cache(cache)
            return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context

        def _unflatten_cache(values, context, output_type=None):
            expected = list(itertools.chain.from_iterable((f"key_{i}", f"value_{i}") for i in range(len(values) // 2)))
            assert expected == context, f"Does not seem to be a dynamic cache {expected} != {context}"
            res = transformers.cache_utils.DynamicCache()
            for i in range(len(values) // 2):
                res.update(values[i * 2], values[i * 2 + 1], layer_idx=i)
            assert output_type is None or isinstance(res, output_type), (
                f"Type mismatch between {output_type} (expected) and {type(res)}"
            )
            return res

        def registers_dynamic_cache():
            cls = transformers.cache_utils.DynamicCache
            torch.utils._pytree.register_pytree_node(
                cls,
                _flatten_key_value_cache,
                _unflatten_cache,
                serialized_type_name=f"{cls.__module__}.{cls.__name__}",
                flatten_with_keys_fn=_flatten_with_keys_cache,
            )

        registers_dynamic_cache()

        # checks that flattening is ok
        flat_dc, spec = torch.utils._pytree.tree_flatten(inputs["past_key_values"])
        assert isinstance(flat_dc, list)
        assert len(flat_dc) == 2
        restored = torch.utils._pytree.tree_unflatten(flat_dc, spec)
        assert len(restored.layers) == 1
        torch.testing.assert_close(inputs["past_key_values"].layers[0].keys, restored.layers[0].keys)
        torch.testing.assert_close(inputs["past_key_values"].layers[0].values, restored.layers[0].values)

        ep = torch.export.export(
            model,
            (),
            kwargs=inputs,
            dynamic_shapes=dict(
                input_ids={0: DYN, 1: DYN},
                attention_mask={0: DYN, 1: DYN},
                past_key_values=[{0: DYN, 2: DYN}, {0: DYN, 2: DYN}],
            ),
        )
        with open(ep_flatten_path, "w") as f:
            f.write(str(ep))
        ep.run_decompositions()  # fails here
        # File "torch/export/exported_program.py", line 1530, in run_decompositions
        #     return _decompose_exported_program(
        #         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # File "torch/export/exported_program.py", line 1017, in _decompose_exported_program
        #     new_module_call_graph = _get_updated_module_call_graph(
        #                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # File "torch/export/exported_program.py", line 926, in _get_updated_module_call_graph
        #     old_name = old_user_input_names[user_input_counter]
        #             ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
        # IndexError: list index out of range

    @hide_stdout()
    def test_fast_discrepancy_mistral_nemo_fp32_cpu_with_explicit_head_dim(self):
        """
        Verify that MistralNeMoForCausalLM can be converted to ONNX when the
        model config has an explicit ``head_dim`` that differs from
        ``hidden_size // num_attention_heads``.

        The real Mistral-Nemo-Instruct-2407 model uses ``head_dim=128`` while
        ``hidden_size / num_attention_heads = 5120 / 32 = 160``.  This test
        mirrors that scenario at a small scale (``head_dim=48`` vs the naive
        ``512 // 8 = 64``) to ensure:

        * ``create_model`` produces an ONNX file whose KV-cache tensors use the
          explicit ``head_dim`` (48) and **not** the naive ratio (64).
        * The ONNX logits on the prefill step are close to those of the
          original PyTorch model.
        """
        import torch
        from models.builder import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, MistralConfig, PreTrainedTokenizerFast

        head_dim = 48  # intentionally != hidden_size // num_attention_heads (64)
        num_hidden_layers = 1
        config = MistralConfig(
            architectures=["MistralNeMoForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            head_dim=head_dim,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            model_type="mistral",
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_theta=1000000.0,
            sliding_window=None,
            vocab_size=32000,
        )

        basename = "test_discrepancies_mistral_nemo_fp32_cpu_with_explicit_head_dim"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MISTRAL_NEMO_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self.check_ort(onnx_path)

        # Verify that the ONNX KV-cache tensors use head_dim=48, not 64.
        kv_shapes = {inp.name: inp.shape for inp in sess.get_inputs() if inp.name.startswith("past_key_values")}
        for name, shape in kv_shapes.items():
            self.assertEqual(shape[3], head_dim, f"{name} has KV head size {shape[3]}, expected {head_dim}")

        batch_size = 1
        seq_len = 5
        onnx_input_names = [i.name for i in sess.get_inputs()]

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        prefill_feed = {
            "input_ids": input_ids.numpy().astype(np.int64),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
        }
        for i in range(num_hidden_layers):
            prefill_feed[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_dim), dtype=np.float32
            )
            prefill_feed[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_dim), dtype=np.float32
            )
        prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}

        onnx_out = sess.run(None, prefill_feed)
        ort_logits = onnx_out[0]

        with torch.no_grad():
            pt_out = model(input_ids)
        np_logits = pt_out.logits.detach().cpu().numpy()

        self.assertEqual(np_logits.shape, ort_logits.shape)
        np.testing.assert_allclose(np_logits[:, :1, :], ort_logits[:, :1, :], atol=1e-3, rtol=1e-3)

    def test_dequantize_fp8_weights_no_op_when_no_fp8(self):
        """_dequantize_fp8_weights leaves normal float32 weights unchanged."""
        import torch
        from models.builders.mistral import Ministral3TextModel

        linear = torch.nn.Linear(8, 4, bias=False)
        original_data = linear.weight.data.clone()
        Ministral3TextModel._dequantize_fp8_weights(linear)
        self.assertTrue(torch.allclose(linear.weight.data, original_data))

    def test_dequantize_fp8_weights_applies_scale(self):
        """_dequantize_fp8_weights dequantizes float8_e4m3fn weights correctly."""
        import torch

        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            self.skipTest("float8_e4m3fn not available in this PyTorch build")

        from models.builders.mistral import Ministral3TextModel

        linear = torch.nn.Linear(8, 4, bias=False)
        # Simulate FP8 quantization: store weight as float8_e4m3fn.
        fp8_weight = linear.weight.data.to(fp8_dtype)
        linear.weight = torch.nn.Parameter(fp8_weight, requires_grad=False)
        # Use a scale > 1 to ensure the multiplication is exercised distinctly.
        scale_inv = torch.tensor([2.0])
        linear.register_buffer("weight_scale_inv", scale_inv)

        Ministral3TextModel._dequantize_fp8_weights(linear)

        self.assertEqual(linear.weight.dtype, torch.float32)
        expected = fp8_weight.float() * scale_inv.float()
        self.assertTrue(torch.allclose(linear.weight.data, expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
