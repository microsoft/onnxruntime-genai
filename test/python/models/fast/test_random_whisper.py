# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np
from ext_test_case import ExtTestCase, hide_stdout, requires_cuda

WHISPER_MODEL_NAME = "openai/whisper-tiny"


class TestWhisperModel(ExtTestCase):
    def common_fast_whisper_random_weights(self, precision, provider):
        import torch
        from models.builder import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast, WhisperConfig, WhisperForConditionalGeneration

        # Tiny Whisper config: 1 encoder layer, 1 decoder layer, small d_model.
        # num_mel_bins and max_source_positions are kept at standard values
        # because the preprocessing Conv layers expect exactly 3000 input frames
        # and max_source_positions = 3000 // 2 = 1500.
        num_encoder_layers = 1
        num_decoder_layers = 1
        config = WhisperConfig(
            architectures=["WhisperForConditionalGeneration"],
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            d_model=128,
            encoder_layers=num_encoder_layers,
            encoder_attention_heads=4,
            decoder_layers=num_decoder_layers,
            decoder_attention_heads=4,
            encoder_ffn_dim=256,
            decoder_ffn_dim=256,
            num_mel_bins=80,
            max_source_positions=1500,
            max_target_positions=32,
            vocab_size=1000,
        )

        basename = f"test_whisper_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(0)
        model = WhisperForConditionalGeneration(config)
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
            model_name=WHISPER_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
        )

        log_data = dict(
            precision=precision,
            model_id=WHISPER_MODEL_NAME,
            experiment="forward",
            provider=provider,
            test=basename,
            input_type="audio",
            kind="fast",
        )

        encoder_path = os.path.join(output_dir, "encoder.onnx")
        decoder_path = os.path.join(output_dir, "decoder.onnx")
        self.assertExists(encoder_path)
        self.assertExists(decoder_path)

        enc_sess = self._check_with_ort(encoder_path, cpu=provider == "cpu")
        dec_sess = self._check_with_ort(decoder_path, cpu=provider == "cpu")

        np_dtype = self.get_input_np_dtype(precision)
        batch_size = 1
        num_attn_heads = config.decoder_attention_heads
        head_size = config.d_model // config.decoder_attention_heads
        seq_len = 3

        enc_input_names = {i.name for i in enc_sess.get_inputs()}
        enc_output_names = [i.name for i in enc_sess.get_outputs()]
        dec_input_names = {i.name for i in dec_sess.get_inputs()}
        dec_output_names = [i.name for i in dec_sess.get_outputs()]

        # ------------------------------------------------------------------
        # Step 1: Run encoder with random audio features
        # ------------------------------------------------------------------
        rng = np.random.default_rng(42)
        audio_np = rng.standard_normal((batch_size, config.num_mel_bins, 3000)).astype(np_dtype)

        enc_feed = {"audio_features": audio_np}
        enc_feed = {k: v for k, v in enc_feed.items() if k in enc_input_names}
        enc_outputs = enc_sess.run(None, enc_feed)
        enc_results = dict(zip(enc_output_names, enc_outputs))

        # Compare encoder hidden_states with PyTorch
        with torch.no_grad():
            pt_audio = torch.from_numpy(audio_np.astype(np.float32)).to(provider)
            pt_enc_out = model.model.encoder(pt_audio)
        pt_hidden = pt_enc_out.last_hidden_state.detach().cpu().numpy()

        enc_disc = self.get_numpy_discrepancy(pt_hidden.astype(np_dtype), enc_results["hidden_states"])
        self.log_results({"step": "encoder", **enc_disc, **log_data})

        atol_enc = {"fp32": 2e-4, "fp16": 5e-2, "int4": 0.5}
        np.testing.assert_allclose(
            pt_hidden.astype(np_dtype), enc_results["hidden_states"], atol=atol_enc[precision], rtol=1e-3
        )

        # ------------------------------------------------------------------
        # Step 2: Run decoder prefill with cross-attention KV from encoder
        # ------------------------------------------------------------------
        input_ids = np.array([[1, 2, 3]], dtype=np.int32)

        dec_feed = {"input_ids": input_ids}
        # Empty self-attention KV caches (no prior decode steps)
        for i in range(num_decoder_layers):
            dec_feed[f"past_key_self_{i}"] = np.zeros((batch_size, num_attn_heads, 0, head_size), dtype=np_dtype)
            dec_feed[f"past_value_self_{i}"] = np.zeros((batch_size, num_attn_heads, 0, head_size), dtype=np_dtype)
        # Cross-attention KV caches from the encoder pass
        for i in range(num_encoder_layers):
            dec_feed[f"past_key_cross_{i}"] = enc_results[f"present_key_cross_{i}"]
            dec_feed[f"past_value_cross_{i}"] = enc_results[f"present_value_cross_{i}"]

        dec_feed = {k: v for k, v in dec_feed.items() if k in dec_input_names}
        dec_outputs = dec_sess.run(None, dec_feed)
        dec_results = dict(zip(dec_output_names, dec_outputs))

        # Compare decoder logits with PyTorch
        with torch.no_grad():
            pt_input_ids = torch.from_numpy(input_ids.astype(np.int64)).to(provider)
            pt_dec_out = model.model.decoder(input_ids=pt_input_ids, encoder_hidden_states=pt_enc_out.last_hidden_state)
        pt_logits = model.proj_out(pt_dec_out.last_hidden_state).detach().cpu().numpy()

        dec_disc = self.get_numpy_discrepancy(pt_logits.astype(np_dtype), dec_results["logits"])
        self.log_results({"step": "decoder", **dec_disc, **log_data})

        atol_dec = {"fp32": 1e-3, "fp16": 5e-2, "int4": 0.5}
        np.testing.assert_allclose(
            pt_logits.astype(np_dtype), dec_results["logits"], atol=atol_dec[precision], rtol=1e-3
        )

        # Verify KV cache shapes from decoder
        for i in range(num_decoder_layers):
            expected_kv_shape = (batch_size, num_attn_heads, seq_len, head_size)
            self.assertEqual(dec_results[f"present_key_self_{i}"].shape, expected_kv_shape)
            self.assertEqual(dec_results[f"present_value_self_{i}"].shape, expected_kv_shape)

    @hide_stdout()
    def test_fast_whisper_random_weights_fp32_cpu(self):
        self.common_fast_whisper_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_whisper_random_weights_fp16_cpu(self):
        self.common_fast_whisper_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_whisper_random_weights_int4_cpu(self):
        self.common_fast_whisper_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_whisper_random_weights_fp32_cuda(self):
        self.common_fast_whisper_random_weights("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_whisper_random_weights_fp16_cuda(self):
        self.common_fast_whisper_random_weights("fp16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
