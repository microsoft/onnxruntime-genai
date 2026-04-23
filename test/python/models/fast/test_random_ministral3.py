# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np
from model_builder_test_case import (
    ModelBuilderTestCase,
    hide_stdout,
    requires_cuda,
    requires_transformers,
)

MINISTRAL3_MODEL_NAME = "mistralai/Ministral-3-3B-Instruct-2512"


@requires_transformers("5")
class TestMinistral3(ModelBuilderTestCase):
    def common_fast_ministral3_random_weights(self, precision, provider):
        from transformers import AutoModelForCausalLM, Ministral3Config

        num_hidden_layers = 1
        config = Ministral3Config(
            architectures=["Ministral3ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-05,
            sliding_window=None,
            vocab_size=32000,
        )

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MINISTRAL3_MODEL_NAME,
            basename=f"test_discrepancies_ministral3_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
        )

    def common_ministral3_greedy_generation(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM, Ministral3Config

        num_hidden_layers = 1
        config = Ministral3Config(
            architectures=["Ministral3ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-05,
            sliding_window=None,
            vocab_size=32000,
        )

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_greedy_generation_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MINISTRAL3_MODEL_NAME,
            basename=f"test_generation_ministral3_{precision}_{provider}",
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
    def test_ministral3_fp32_cpu_greedy_generation(self):
        self.common_ministral3_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_ministral3_fp16_cpu_greedy_generation(self):
        self.common_ministral3_greedy_generation("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_ministral3_fp32_cuda_greedy_generation(self):
        self.common_ministral3_greedy_generation("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_ministral3_fp16_cuda_greedy_generation(self):
        self.common_ministral3_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_ministral3_bf16_cuda_greedy_generation(self):
        self.common_ministral3_greedy_generation("bf16", "cuda")

    @hide_stdout()
    def test_fast_discrepancy_ministral3_fp32_cpu(self):
        self.common_fast_ministral3_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_ministral3_fp16_cpu(self):
        self.common_fast_ministral3_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_ministral3_int4_cpu(self):
        self.common_fast_ministral3_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_ministral3_fp16_cuda(self):
        self.common_fast_ministral3_random_weights("fp16", "cuda")

    @hide_stdout()
    def test_ministral3_fp32_cpu_genai_generate(self):
        import torch
        from transformers import AutoModelForCausalLM, Ministral3Config

        from models.builder import create_model

        prefix = "test_ministral3_fp32_cpu_genai_generate"
        num_hidden_layers = 1
        config = Ministral3Config(
            architectures=["Ministral3ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-05,
            sliding_window=None,
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
            model_name=MINISTRAL3_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        self.run_genai_generation_test(output_dir, model, config.vocab_size, config.eos_token_id)

    @hide_stdout()
    def test_ministral3_conditional_generation_fp32_cpu_random_weights(self):
        """
        Convert a randomly-initialised Mistral3ForConditionalGeneration model to
        fp32 ONNX models targeting the CPU execution provider.

        Mistral3ForConditionalGeneration is the multimodal model class for the
        Ministral 3 family.  The builder now exports **two** artifacts:

        1. ``vision_encoder.onnx`` – the Pixtral vision encoder (patch
           convolution + transformer) together with the multimodal projector.
           Input: ``pixel_values`` [1, 3, image_size, image_size].
           Output: ``image_features`` [num_merged_patches, text_hidden_size].

        2. ``model.onnx`` – the Mistral text decoder with
           ``exclude_embeds=True`` so that it accepts ``inputs_embeds``
           (produced by the vision encoder or from plain embed_tokens) rather
           than raw ``input_ids``.

        The test verifies that:
        * ``create_model`` completes without error when given a local model directory.
        * Both ``vision_encoder.onnx`` and ``model.onnx`` are written to the
          output directory.
        * Both ONNX files can be loaded by ``onnxruntime``.
        * The vision encoder produces output of the correct shape.
        * The text decoder produces output when fed ``inputs_embeds``.
        """
        import torch
        from models.builder import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import (
            Ministral3Config,
            Mistral3Config,
            Mistral3ForConditionalGeneration,
            PixtralVisionConfig,
            PreTrainedTokenizerFast,
        )

        num_hidden_layers = 1
        # Use a small image size (56×56) so the test stays fast; patch_size=14
        # gives 4×4=16 patches, then spatial_merge_size=2 → 4 merged patches.
        image_size = 56
        patch_size = 14
        spatial_merge_size = 2

        vision_config = PixtralVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            head_dim=16,
            image_size=image_size,
            patch_size=patch_size,
        )
        text_config = Ministral3Config(
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-05,
            sliding_window=None,
            vocab_size=32000,
        )
        config = Mistral3Config(
            text_config=text_config, vision_config=vision_config, spatial_merge_size=spatial_merge_size
        )
        config.architectures = ["Mistral3ForConditionalGeneration"]

        model_dir = self.get_model_dir("test_ministral3_conditional_generation_fp32_cpu_random_weights")
        output_dir, cache_dir = self.get_dirs("test_ministral3_conditional_generation_fp32_cpu_random_weights")

        model = Mistral3ForConditionalGeneration(config)
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
            model_name=MINISTRAL3_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        # --- Verify both ONNX files exist and load correctly ---
        vision_onnx_path = os.path.join(output_dir, "vision_encoder.onnx")
        self.assertExists(vision_onnx_path)

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        # --- Verify genai_config.json has vision + embedding sections ---
        import json

        genai_config_path = os.path.join(output_dir, "genai_config.json")
        self.assertExists(genai_config_path)
        with open(genai_config_path) as f:
            genai_config = json.load(f)
        self.assertEqual(genai_config["model"]["type"], "phi3v")
        self.assertIn("vision", genai_config["model"])
        ve_cfg = genai_config["model"]["vision"]
        self.assertEqual(ve_cfg["filename"], "vision_encoder.onnx")
        self.assertEqual(ve_cfg["spatial_merge_size"], spatial_merge_size)
        self.assertIn("embedding", genai_config["model"])
        em_cfg = genai_config["model"]["embedding"]
        self.assertEqual(em_cfg["filename"], "embedding.onnx")

        # --- Run vision encoder forward pass ---
        num_patches_per_side = image_size // patch_size
        expected_merged_patches = (num_patches_per_side**2) // (spatial_merge_size**2)

        vision_sess = self.check_ort(vision_onnx_path)
        pixel_values = np.zeros((1, vision_config.num_channels, image_size, image_size), dtype=np.float32)
        vision_outputs = vision_sess.run(None, {"pixel_values": pixel_values})
        self.assertIsNotNone(vision_outputs[0])
        self.assertEqual(vision_outputs[0].shape[0], expected_merged_patches)
        self.assertEqual(vision_outputs[0].shape[1], text_config.hidden_size)

        # --- Run text decoder forward pass ---
        # The ONNX model was built with exclude_embeds=True, so it expects
        # `inputs_embeds` (shape [batch, seq, hidden_size]) rather than
        # `input_ids`.  Compute the embeddings using the saved model weights.
        batch_size = 1
        seq_len = 5
        input_ids = torch.randint(0, text_config.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            inputs_embeds = model.model.language_model.embed_tokens(input_ids).numpy().astype(np.float32)

        text_sess = self.check_ort(text_onnx_path)
        onnx_input_names = {inp.name for inp in text_sess.get_inputs()}

        head_size = text_config.head_dim
        onnx_feed = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
        }
        for i in range(num_hidden_layers):
            onnx_feed[f"past_key_values.{i}.key"] = np.zeros((batch_size, text_config.num_key_value_heads, 0, head_size), dtype=np.float32)
            onnx_feed[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, text_config.num_key_value_heads, 0, head_size), dtype=np.float32
            )
        onnx_feed = {k: v for k, v in onnx_feed.items() if k in onnx_input_names}

        onnx_outputs = text_sess.run(None, onnx_feed)
        self.assertIsNotNone(onnx_outputs[0])

    @hide_stdout()
    def test_ministral3_two_images_and_text_fp32_cpu_random_weights(self):
        """
        Build a randomly-initialised Mistral3ForConditionalGeneration model,
        export it to fp32 ONNX on CPU, then run an end-to-end forward pass
        that interleaves two image encodings with text token embeddings.

        The flow mirrors how ``onnxruntime-genai`` processes a multimodal
        prompt that contains two inline images:

        1. ``vision_encoder.onnx`` is invoked **once per image** to produce
           ``image_features`` of shape ``[num_merged_patches, text_hidden_size]``.
        2. The two image-feature tensors plus regular text-token embeddings are
           concatenated along the sequence dimension to form ``inputs_embeds``
           of shape ``[1, 2*num_merged_patches + text_seq_len, text_hidden_size]``.
        3. ``model.onnx`` (text decoder, built with ``exclude_embeds=True``) is
           invoked with these combined ``inputs_embeds``.

        Assertions:
        * Both ONNX files exist and load correctly.
        * Vision encoder produces the expected shape for each of the two images.
        * Text decoder produces output (non-None logits) when fed the combined
          ``inputs_embeds`` that mixes two image encodings with text embeddings.
        """
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import (
            Ministral3Config,
            Mistral3Config,
            Mistral3ForConditionalGeneration,
            PixtralVisionConfig,
            PreTrainedTokenizerFast,
        )

        from models.builder import create_model

        num_hidden_layers = 1
        # 56×56 image with patch_size=14 → 4×4=16 patches;
        # spatial_merge_size=2 → 4 merged patches per image.
        image_size = 56
        patch_size = 14
        spatial_merge_size = 2

        vision_config = PixtralVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            head_dim=16,
            image_size=image_size,
            patch_size=patch_size,
        )
        text_config = Ministral3Config(
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-05,
            sliding_window=None,
            vocab_size=32000,
        )
        config = Mistral3Config(text_config=text_config, vision_config=vision_config, spatial_merge_size=spatial_merge_size)
        config.architectures = ["Mistral3ForConditionalGeneration"]

        basename = "test_ministral3_two_images_and_text_fp32_cpu_random_weights"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        model = Mistral3ForConditionalGeneration(config)
        model.eval()
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MINISTRAL3_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        # --- Verify both ONNX files exist and load correctly ---
        vision_onnx_path = os.path.join(output_dir, "vision_encoder.onnx")
        self.assertExists(vision_onnx_path)
        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        num_patches_per_side = image_size // patch_size
        num_merged_patches = (num_patches_per_side**2) // (spatial_merge_size**2)

        vision_sess = self.check_ort(vision_onnx_path)

        # --- Encode two images independently with the vision encoder ---
        image1 = np.zeros((1, vision_config.num_channels, image_size, image_size), dtype=np.float32)
        image2 = np.ones((1, vision_config.num_channels, image_size, image_size), dtype=np.float32) * 0.5

        image1_features = vision_sess.run(None, {"pixel_values": image1})[0]
        image2_features = vision_sess.run(None, {"pixel_values": image2})[0]

        # Each image produces [num_merged_patches, text_hidden_size]
        self.assertEqual(image1_features.shape, (num_merged_patches, text_config.hidden_size))
        self.assertEqual(image2_features.shape, (num_merged_patches, text_config.hidden_size))

        # --- Build combined inputs_embeds: [img1 | text | img2] ---
        text_seq_len = 3
        text_token_ids = torch.randint(0, text_config.vocab_size, (1, text_seq_len))
        with torch.no_grad():
            text_embeds = model.model.language_model.embed_tokens(text_token_ids).numpy().astype(np.float32)
        # text_embeds: [1, text_seq_len, hidden_size]; squeeze batch for cat
        text_embeds_2d = text_embeds[0]  # [text_seq_len, hidden_size]

        # Concatenate along sequence: [num_merged_patches + text_seq_len + num_merged_patches, hidden_size]
        combined_2d = np.concatenate([image1_features, text_embeds_2d, image2_features], axis=0)
        total_seq_len = combined_2d.shape[0]
        inputs_embeds = combined_2d[np.newaxis]  # [1, total_seq_len, hidden_size]

        self.assertEqual(inputs_embeds.shape, (1, 2 * num_merged_patches + text_seq_len, text_config.hidden_size))

        # --- Run text decoder on the combined inputs_embeds ---
        text_sess = self.check_ort(text_onnx_path)
        onnx_input_names = {inp.name for inp in text_sess.get_inputs()}

        head_size = text_config.head_dim
        batch_size = 1
        onnx_feed = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
            "position_ids": np.arange(total_seq_len, dtype=np.int64).reshape(batch_size, total_seq_len),
        }
        for i in range(num_hidden_layers):
            onnx_feed[f"past_key_values.{i}.key"] = np.zeros((batch_size, text_config.num_key_value_heads, 0, head_size), dtype=np.float32)
            onnx_feed[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, text_config.num_key_value_heads, 0, head_size), dtype=np.float32
            )
        onnx_feed = {k: v for k, v in onnx_feed.items() if k in onnx_input_names}

        onnx_outputs = text_sess.run(None, onnx_feed)
        self.assertIsNotNone(onnx_outputs[0])
        # Logits shape: [batch_size, total_seq_len, vocab_size]
        self.assertEqual(onnx_outputs[0].shape, (batch_size, total_seq_len, text_config.vocab_size))

    @hide_stdout()
    def test_ministral3_two_images_and_text_fp32_cpu_genai(self):
        self.common_ministral3_two_images_and_text_cpu_genai("fp32")

    @hide_stdout()
    def test_ministral3_two_images_and_text_int4_cpu_genai(self):
        self.common_ministral3_two_images_and_text_cpu_genai("int4")

    def common_ministral3_two_images_and_text_cpu_genai(self, precision):
        """
        Draw a dummy cross image, run ``model.generate()`` from HuggingFace
        ``Mistral3ForConditionalGeneration`` and ``onnxruntime-genai``, then
        compare the full generated token sequences.

        The comparison pipeline is:

        1. Build a tiny randomly-initialised ``Mistral3ForConditionalGeneration``
           and export both ``vision_encoder.onnx`` and ``model.onnx``.
        2. Draw a white-cross-on-black-background image as the visual prompt.
        3. **HF reference**: construct a prompt that places
           ``config.image_token_id`` placeholder tokens where image features
           will be merged, followed by plain text token IDs; call
           ``model.generate(input_ids=..., pixel_values=..., image_sizes=...)``
           to obtain ``pt_tokens`` (prompt + ``max_new_tokens`` generated
           tokens); extract the newly-generated part as ``pt_generated``.
        4. **genai**: load the ONNX models with ``og.Model``; provide the same
           ``pixel_values`` via ``og.NamedTensors`` / ``set_inputs``; provide
           the same text token IDs (without image placeholders) via
           ``append_tokens``; iterate ``generate_next_token`` until done and
           collect ``og_generated``.
        5. Log results with :meth:`log_results` (like ``run_genai_generation``).
        6. Assert ``pt_generated == og_generated``.

        The test name retains ``two_images`` for continuity with earlier
        iterations of the PR (the ONNX sibling test
        ``test_ministral3_two_images_and_text_fp32_cpu_random_weights``
        does exercise two images).

        Requires ``onnxruntime-genai`` with ``vision_encoder`` support in
        ``genai_config.json``; run with ``LONGTEST=1``.
        """
        import torch
        import onnxruntime_genai as og

        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import (
            Ministral3Config,
            Mistral3Config,
            Mistral3ForConditionalGeneration,
            PixtralVisionConfig,
            PreTrainedTokenizerFast,
        )

        from models.builder import create_model

        # --- Tiny model configuration (same as the ONNX-only sibling test) ---
        num_hidden_layers = 1
        image_size = 56
        patch_size = 14
        spatial_merge_size = 2

        vision_config = PixtralVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            head_dim=16,
            image_size=image_size,
            patch_size=patch_size,
        )
        text_config = Ministral3Config(
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-05,
            sliding_window=None,
            vocab_size=32000,
        )
        config = Mistral3Config(text_config=text_config, vision_config=vision_config, spatial_merge_size=spatial_merge_size)
        config.architectures = ["Mistral3ForConditionalGeneration"]

        basename = f"test_ministral3_two_images_and_text_{precision}_cpu_genai"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(0)
        model = Mistral3ForConditionalGeneration(config)
        model.eval()
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MINISTRAL3_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        # --- Draw a dummy cross image (white cross on black background) ---
        # 56×56 image: horizontal + vertical bar each 1/3-wide, centred.
        img_plane = np.zeros((image_size, image_size), dtype=np.float32)
        bar = slice(image_size // 3, 2 * image_size // 3)
        img_plane[bar, :] = 1.0  # horizontal bar
        img_plane[:, bar] = 1.0  # vertical bar
        # shape: [1, num_channels, H, W]
        cross_image = np.stack([img_plane] * vision_config.num_channels, axis=0)[np.newaxis]

        # 56×56 image / (patch_size=14 * spatial_merge_size=2) → 2×2 = 4 merged patches
        n_merged_patches = (image_size // patch_size) ** 2 // spatial_merge_size**2

        # --- HF reference: generate with a proper multimodal prompt ---
        # Build prompt with image placeholder tokens followed by text tokens.
        # HF merge logic replaces each image_token_id with the corresponding
        # image feature vector; the count must equal n_merged_patches.
        image_token_id = config.image_token_id  # 10 for Mistral3
        text_ids = [100, 200]  # two in-vocab text tokens appended after the image
        hf_prompt = torch.tensor([[image_token_id] * n_merged_patches + text_ids])
        image_sizes = torch.tensor([[image_size, image_size]])
        cross_pixel_values = torch.tensor(cross_image)

        max_new_tokens = 5
        with torch.no_grad():
            pt_output = model.generate(
                input_ids=hf_prompt,
                pixel_values=cross_pixel_values,
                image_sizes=image_sizes,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=config.text_config.eos_token_id,
            )
        pt_tokens = pt_output[0].tolist()
        # pt_tokens = [img_tok*N, text_ids..., gen1, gen2, ...]
        pt_generated = pt_tokens[hf_prompt.shape[1] :]

        # --- genai: same pixel_values + full prompt → multiple generation steps ---
        # phi3v-style flow: the full prompt (image placeholder tokens + text tokens)
        # is passed via append_tokens; set_inputs provides pixel_values and
        # num_image_tokens so the runtime calls the vision encoder.
        # The embedding model replaces the image placeholder positions with actual
        # image features from the vision encoder, matching the HF merged sequence.
        # The vision encoder dtype follows the model precision: fp16 models
        # expect float16 pixel_values; fp32 and int4 models expect float32.
        genai_pixel_dtype = np.float16 if precision == "fp16" else np.float32
        og_model = og.Model(output_dir)
        full_prompt_ids = np.array([image_token_id] * n_merged_patches + text_ids, dtype=np.int64)
        params = og.GeneratorParams(og_model)
        params.set_search_options(do_sample=False, max_length=n_merged_patches + len(text_ids) + max_new_tokens, temperature=1.0, top_k=1)
        generator = og.Generator(og_model, params)
        named_tensors = og.NamedTensors()
        named_tensors["pixel_values"] = cross_image.astype(genai_pixel_dtype)
        named_tensors["num_image_tokens"] = np.array([n_merged_patches], dtype=np.int64)
        generator.set_inputs(named_tensors)
        generator.append_tokens(full_prompt_ids)
        og_generated = []
        while not generator.is_done():
            generator.generate_next_token()
            og_generated.append(int(generator.get_next_tokens()[0]))

        log_data = dict(
            precision=precision,
            model_id=MINISTRAL3_MODEL_NAME,
            experiment="genai_vision_generate",
            provider="cpu",
            test=basename,
            input_type="vision+text",
            kind="fast",
        )
        diff = self.first_token_diff(pt_generated, og_generated)
        self.log_results({**log_data, **diff})
        # For fp32, ORT and HuggingFace produce identical tokens.
        # For lossy precisions (int4, …) the embedding table and linear layers
        # are quantised, so token mismatches are expected; just verify that the
        # correct number of tokens was generated without error.
        if precision == "fp32":
            self.assertEqual(pt_generated, og_generated)
        else:
            self.assertEqual(len(og_generated), max_new_tokens)

    @hide_stdout()
    def test_ministral3_apply_chat_template_fp32_cpu_genai(self):
        self.common_ministral3_apply_chat_template_cpu_genai("fp32")

    @hide_stdout()
    def test_ministral3_apply_chat_template_fp16_cpu_genai(self):
        self.common_ministral3_apply_chat_template_cpu_genai("fp16")

    @hide_stdout()
    def test_ministral3_apply_chat_template_int4_cpu_genai(self):
        self.common_ministral3_apply_chat_template_cpu_genai("int4")

    def common_ministral3_apply_chat_template_cpu_genai(self, precision):
        """
        Test that ``apply_chat_template`` with text and images produces the
        same generation from HuggingFace and ``onnxruntime-genai``.

        Flow:

        1. Build a tiny randomly-initialised ``Mistral3ForConditionalGeneration``
           and export both ``vision_encoder.onnx`` and ``model.onnx``.
        2. Create a ``PreTrainedTokenizerFast`` whose vocabulary maps ``[IMG]``
           to ``config.image_token_id`` (10).  Set a minimal Jinja2
           ``chat_template`` that places ``[IMG]`` for each image content item
           and the raw text for text items.
        3. Apply the chat template to a conversation with one image and a
           two-token text phrase:
           ``tokenizer.apply_chat_template(messages, tokenize=False)``
           → ``"[IMG] hello world"``.
        4. Tokenize the template output (``add_special_tokens=False``) to get
           compact token IDs, then **expand** every ``[IMG]`` (ID 10) into
           ``n_merged_patches`` copies, mirroring what the pixtral processor
           would do.
        5. Run HF ``model.generate()`` with the expanded prompt,
           ``pixel_values``, and ``image_sizes``.
        6. Run ``onnxruntime-genai`` with the same inputs.
        7. Assert that both backends produce identical tokens (fp32) or the
           correct number of tokens (lossy precisions such as int4).

        Requires ``onnxruntime-genai`` with ``vision_encoder`` support in
        ``genai_config.json``; run with ``LONGTEST=1``.
        """
        import torch
        import onnxruntime_genai as og

        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import (
            Ministral3Config,
            Mistral3Config,
            Mistral3ForConditionalGeneration,
            PixtralVisionConfig,
            PreTrainedTokenizerFast,
        )

        from modelbuilder.builder import create_model

        # --- Tiny model configuration (same geometry as sibling tests) ---
        num_hidden_layers = 1
        image_size = 56
        patch_size = 14
        spatial_merge_size = 2

        vision_config = PixtralVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            head_dim=16,
            image_size=image_size,
            patch_size=patch_size,
        )
        text_config = Ministral3Config(
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-05,
            sliding_window=None,
            vocab_size=32000,
        )
        config = Mistral3Config(text_config=text_config, vision_config=vision_config, spatial_merge_size=spatial_merge_size)
        config.architectures = ["Mistral3ForConditionalGeneration"]

        basename = f"test_ministral3_apply_chat_template_{precision}_cpu_genai"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(0)
        model = Mistral3ForConditionalGeneration(config)
        model.eval()
        model.save_pretrained(model_dir)

        # Build a tokenizer whose vocabulary maps [IMG] → config.image_token_id
        # (10) so that apply_chat_template → tokenize → expand mirrors the
        # pixtral-processor flow used in production.
        # A WhitespaceSplit pre-tokenizer is required so that the WordLevel model
        # splits "[IMG] hello world" into ["[IMG]", "hello", "world"] before
        # doing vocabulary lookups; without it the entire string is treated as
        # a single unknown token (ID 0).  WhitespaceSplit (not Whitespace) must
        # be used because Whitespace also splits on punctuation, which would
        # break "[IMG]" into ["[", "IMG", "]"], none of which map to the image
        # token ID.
        image_token_id = config.image_token_id  # 10 for Mistral3
        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "[IMG]": image_token_id, "hello": 100, "world": 101}
        tokenizer_object = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer_object.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_object, bos_token="<s>", eos_token="</s>", unk_token="<unk>")

        # Minimal Jinja2 chat template: emits "[IMG] " for each image content
        # item and the raw text for text items.  Space-separated tokens allow
        # the WordLevel tokenizer to split correctly on whitespace.
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{% for item in message['content'] %}"
            "{% if item['type'] == 'image' %}[IMG] "
            "{% elif item['type'] == 'text' %}{{ item['text'] }}"
            "{% endif %}"
            "{% endfor %}"
            "{% endif %}"
            "{% endfor %}"
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MINISTRAL3_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        # --- Draw a dummy cross image (white cross on black background) ---
        img_plane = np.zeros((image_size, image_size), dtype=np.float32)
        bar = slice(image_size // 3, 2 * image_size // 3)
        img_plane[bar, :] = 1.0  # horizontal bar
        img_plane[:, bar] = 1.0  # vertical bar
        # shape: [1, num_channels, H, W]
        cross_image = np.stack([img_plane] * vision_config.num_channels, axis=0)[np.newaxis]

        # 56×56 / (patch_size=14 * spatial_merge_size=2)² → 4 merged patches
        n_merged_patches = (image_size // patch_size) ** 2 // spatial_merge_size**2

        # --- apply_chat_template: conversation with one image + two text tokens ---
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hello world"}]}]
        # tokenize=False → "[IMG] hello world" (single [IMG] marker + text)
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False)

        # Tokenize without adding BOS/EOS (the template owns the structure).
        # WordLevel splits on whitespace: ["[IMG]", "hello", "world"] → [10, 100, 101]
        compact_ids = tokenizer.encode(chat_text, add_special_tokens=False)

        # Expand every [IMG] token (ID == image_token_id) to n_merged_patches
        # copies, mirroring the pixtral processor expansion that the runtime
        # expects before calling vision_encoder.
        expanded_ids = []
        for tid in compact_ids:
            if tid == image_token_id:
                expanded_ids.extend([image_token_id] * n_merged_patches)
            else:
                expanded_ids.append(tid)

        # --- HF reference: generate with expanded prompt + pixel_values ---
        hf_prompt = torch.tensor([expanded_ids])
        image_sizes = torch.tensor([[image_size, image_size]])
        cross_pixel_values = torch.tensor(cross_image)

        max_new_tokens = 5
        with torch.no_grad():
            pt_output = model.generate(
                input_ids=hf_prompt,
                pixel_values=cross_pixel_values,
                image_sizes=image_sizes,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=config.text_config.eos_token_id,
            )
        pt_generated = pt_output[0].tolist()[hf_prompt.shape[1] :]

        # --- genai: same pixel_values + expanded prompt ---
        # The vision encoder dtype follows the model precision: fp16 models
        # expect float16 pixel_values; fp32 and int4 models expect float32.
        genai_pixel_dtype = np.float16 if precision == "fp16" else np.float32
        genai_pixel_values = cross_image.astype(genai_pixel_dtype)

        og_model = og.Model(output_dir)
        full_prompt_ids = np.array(expanded_ids, dtype=np.int64)
        params = og.GeneratorParams(og_model)
        params.set_search_options(do_sample=False, max_length=len(expanded_ids) + max_new_tokens, temperature=1.0, top_k=1)
        generator = og.Generator(og_model, params)
        named_tensors = og.NamedTensors()
        named_tensors["pixel_values"] = genai_pixel_values
        named_tensors["num_image_tokens"] = np.array([n_merged_patches], dtype=np.int64)
        generator.set_inputs(named_tensors)
        generator.append_tokens(full_prompt_ids)
        og_generated = []
        while not generator.is_done():
            generator.generate_next_token()
            og_generated.append(int(generator.get_next_tokens()[0]))

        log_data = dict(
            precision=precision,
            model_id=MINISTRAL3_MODEL_NAME,
            experiment="genai_vision_generate_chat_template",
            provider="cpu",
            test=basename,
            input_type="vision+text",
            kind="fast",
        )
        diff = self.first_token_diff(pt_generated, og_generated)
        self.log_results({**log_data, **diff})
        # For fp32 the computation is exact and both backends must agree.
        # For lossy precisions (int4, …) token values may differ; verify only
        # that the correct number of tokens was generated without error.
        if precision == "fp32":
            self.assertEqual(pt_generated, og_generated)
        else:
            self.assertEqual(len(og_generated), max_new_tokens)

    @hide_stdout()
    def test_ministral3_vision_encoder_int4_cpu_random_weights(self):
        """
        Build a randomly-initialised Mistral3ForConditionalGeneration model,
        export it with int4 precision, and verify that the vision encoder ONNX
        model is correctly quantised.

        Specifically the test checks:

        1. ``vision_encoder.onnx`` is written to the output directory.
        2. The vision encoder ONNX graph contains at least one ``MatMulNBits``
           node, confirming that int4 weight quantisation was applied to the
           linear layers inside the Pixtral vision tower and projector.
        3. An ORT forward pass on the quantised model produces
           ``image_features`` with the expected shape
           ``[num_merged_patches, text_hidden_size]``.
        """
        import onnx
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import (
            Ministral3Config,
            Mistral3Config,
            Mistral3ForConditionalGeneration,
            PixtralVisionConfig,
            PreTrainedTokenizerFast,
        )

        from models.builder import create_model

        num_hidden_layers = 1
        # Same tiny geometry as the fp32 sibling test:
        # 56×56 / patch_size=14 → 4×4=16 patches;
        # spatial_merge_size=2 → 4 merged patches.
        image_size = 56
        patch_size = 14
        spatial_merge_size = 2

        vision_config = PixtralVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            head_dim=16,
            image_size=image_size,
            patch_size=patch_size,
        )
        text_config = Ministral3Config(
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-05,
            sliding_window=None,
            vocab_size=32000,
        )
        config = Mistral3Config(text_config=text_config, vision_config=vision_config, spatial_merge_size=spatial_merge_size)
        config.architectures = ["Mistral3ForConditionalGeneration"]

        basename = "test_ministral3_vision_encoder_int4_cpu_random_weights"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = Mistral3ForConditionalGeneration(config)
        model.eval()
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MINISTRAL3_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="int4",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        # --- Verify vision encoder ONNX exists ---
        vision_onnx_path = os.path.join(output_dir, "vision_encoder.onnx")
        self.assertExists(vision_onnx_path)

        # --- Verify MatMulNBits nodes are present (int4 quantisation applied) ---
        # Load graph structure only (skip external weight data) to check op types.
        vision_proto = onnx.load(vision_onnx_path, load_external_data=False)
        op_types = {node.op_type for node in vision_proto.graph.node}
        self.assertIn("MatMulNBits", op_types, "Vision encoder ONNX should contain MatMulNBits nodes after int4 quantisation")

        # --- Run forward pass and verify output shape ---
        num_patches_per_side = image_size // patch_size
        expected_merged_patches = (num_patches_per_side**2) // (spatial_merge_size**2)

        vision_sess = self.check_ort(vision_onnx_path)
        pixel_values = np.zeros((1, vision_config.num_channels, image_size, image_size), dtype=np.float32)
        vision_outputs = vision_sess.run(None, {"pixel_values": pixel_values})
        self.assertIsNotNone(vision_outputs[0])
        self.assertEqual(vision_outputs[0].shape[0], expected_merged_patches)
        self.assertEqual(vision_outputs[0].shape[1], text_config.hidden_size)

    @hide_stdout()
    def test_ministral3_vision_encoder_output_matches_pytorch(self):
        """Vision encoder ONNX output should match the PyTorch reference numerically.

        Creates a tiny randomly-initialised ``Mistral3ForConditionalGeneration``
        with a fixed seed, exports only the vision encoder, then runs both the
        PyTorch vision tower + projector and the ONNX model on the same pixel
        values.  Asserts that the outputs agree to within fp32 tolerance
        (``atol=1e-4``).

        This test specifically guards the ``_build_patch_embedding`` and
        ``_build_projector`` implementations against regressions in the
        reshape / transpose ordering.
        """
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import (
            Ministral3Config,
            Mistral3Config,
            Mistral3ForConditionalGeneration,
            PixtralVisionConfig,
            PreTrainedTokenizerFast,
        )

        from models.builder import create_model

        num_hidden_layers = 1
        image_size = 56
        patch_size = 14
        spatial_merge_size = 2

        vision_config = PixtralVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            head_dim=16,
            image_size=image_size,
            patch_size=patch_size,
        )
        text_config = Ministral3Config(
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-05,
            sliding_window=None,
            vocab_size=32000,
        )
        config = Mistral3Config(text_config=text_config, vision_config=vision_config, spatial_merge_size=spatial_merge_size)
        config.architectures = ["Mistral3ForConditionalGeneration"]

        basename = "test_ministral3_vision_encoder_output_matches_pytorch"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(0)
        model = Mistral3ForConditionalGeneration(config)
        model.eval()
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MINISTRAL3_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        vision_onnx_path = os.path.join(output_dir, "vision_encoder.onnx")
        self.assertExists(vision_onnx_path)

        # --- Reference: PyTorch vision tower + projector ---
        # Use model.model.get_image_features(), which is the same call path used
        # by Mistral3ForConditionalGeneration.forward().  It squeezes the batch
        # dimension before the projector, matching the ONNX encoder's behaviour.
        torch.manual_seed(1)
        pixel_values = torch.randn(1, vision_config.num_channels, image_size, image_size)
        image_sizes = torch.tensor([[image_size, image_size]])

        with torch.no_grad():
            hf_out = model.model.get_image_features(pixel_values, image_sizes, vision_feature_layer=config.vision_feature_layer)
        # get_image_features returns a BaseModelOutputWithPooling whose
        # pooler_output is a tuple of per-image feature tensors.
        pt_image_features = hf_out.pooler_output[0]  # [n_merged, text_hidden]
        pt_np = pt_image_features.numpy().astype(np.float32)

        # --- ONNX vision encoder ---
        vision_sess = self.check_ort(vision_onnx_path)
        ort_out = vision_sess.run(None, {"pixel_values": pixel_values.numpy().astype(np.float32)})[0]

        # Shapes should match
        self.assertEqual(ort_out.shape, pt_np.shape)

        # Numerical values should match to within fp32 tolerance
        np.testing.assert_allclose(
            ort_out, pt_np, atol=1e-4, rtol=1e-3, err_msg="Vision encoder ONNX output does not match PyTorch reference"
        )

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

    @hide_stdout()
    def test_ministral3_projector_linear1_bias_adds_bias_node(self):
        """Exercises the bias-addition branch in _build_projector (line 528 of
        mistral.py) by injecting a non-zero bias into the projector's linear_1.

        The standard HuggingFace ``Mistral3MultiModalProjector.linear_1`` is
        always created with ``bias=False``, so the branch

            if lin1_bias is not None and torch.count_nonzero(lin1_bias) > 0:

        is never reached during normal save/load cycles.  This test replaces
        ``linear_1`` with an ``nn.Linear`` that carries a non-zero bias, then
        patches ``Ministral3VisionEncoderModel._load_hf_model`` so the builder
        receives the in-memory model directly (avoiding the round-trip through
        ``save_pretrained`` / ``from_pretrained`` which would silently drop the
        bias).  The resulting ``vision_encoder.onnx`` must contain a
        ``/vision/projector/linear_1/MatMul/BiasAdd`` node, confirming that
        line 528 was executed.
        """
        import onnx
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from unittest.mock import patch
        from transformers import (
            Ministral3Config,
            Mistral3Config,
            Mistral3ForConditionalGeneration,
            PixtralVisionConfig,
            PreTrainedTokenizerFast,
        )

        from models.builder import create_model
        from models.builders.mistral import Ministral3VisionEncoderModel

        num_hidden_layers = 1
        image_size = 56
        patch_size = 14
        spatial_merge_size = 2

        vision_config = PixtralVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            head_dim=16,
            image_size=image_size,
            patch_size=patch_size,
        )
        text_config = Ministral3Config(
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-05,
            sliding_window=None,
            vocab_size=32000,
        )
        config = Mistral3Config(text_config=text_config, vision_config=vision_config, spatial_merge_size=spatial_merge_size)
        config.architectures = ["Mistral3ForConditionalGeneration"]

        basename = "test_ministral3_projector_linear1_bias_adds_bias_node"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = Mistral3ForConditionalGeneration(config)
        model.eval()

        # Inject a non-zero bias into linear_1 so that the bias-addition
        # branch (mistral.py line 528) is reached during ONNX construction.
        proj = model.model.multi_modal_projector
        in_features = proj.linear_1.weight.shape[1]
        out_features = proj.linear_1.weight.shape[0]
        new_linear_1 = torch.nn.Linear(in_features, out_features, bias=True)
        with torch.no_grad():
            new_linear_1.weight.copy_(proj.linear_1.weight)
            new_linear_1.bias.fill_(0.1)
        proj.linear_1 = new_linear_1

        # Save model and tokenizer so that the embedding and text sub-models
        # can load normally from the directory.  The bias weights are written
        # to the safetensors file but silently ignored on reload because the
        # default architecture uses bias=False; only the vision encoder (whose
        # _load_hf_model is patched below) sees the modified projector.
        model.save_pretrained(model_dir)
        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
        )
        tokenizer.save_pretrained(model_dir)

        # Patch _load_hf_model on Ministral3VisionEncoderModel so the builder
        # receives our in-memory model (with non-zero bias) directly.
        def _load_hf_model_with_bias(self_inner, input_path):
            return model

        with patch.object(Ministral3VisionEncoderModel, "_load_hf_model", _load_hf_model_with_bias):
            create_model(
                model_name=MINISTRAL3_MODEL_NAME,
                input_path=model_dir,
                output_dir=output_dir,
                precision="fp32",
                execution_provider="cpu",
                cache_dir=cache_dir,
                num_hidden_layers=num_hidden_layers,
            )

        # The vision encoder ONNX must exist.
        vision_onnx_path = os.path.join(output_dir, "vision_encoder.onnx")
        self.assertExists(vision_onnx_path)

        # A BiasAdd node for linear_1 must be present, confirming that line 528
        # in _build_projector was executed (non-zero bias branch).
        vision_proto = onnx.load(vision_onnx_path, load_external_data=False)
        node_names = {node.name for node in vision_proto.graph.node}
        self.assertIn(
            "/vision/projector/linear_1/MatMul/BiasAdd",
            node_names,
            "BiasAdd node for linear_1 should be present when projector.linear_1 has a non-zero bias",
        )

        # Verify the forward pass still produces the correct output shape.
        vision_sess = self.check_ort(vision_onnx_path)
        pixel_values = np.zeros((1, vision_config.num_channels, image_size, image_size), dtype=np.float32)
        vision_outputs = vision_sess.run(None, {"pixel_values": pixel_values})
        self.assertIsNotNone(vision_outputs[0])
        num_patches_per_side = image_size // patch_size
        expected_merged_patches = (num_patches_per_side**2) // (spatial_merge_size**2)
        self.assertEqual(vision_outputs[0].shape[0], expected_merged_patches)
        self.assertEqual(vision_outputs[0].shape[1], text_config.hidden_size)


if __name__ == "__main__":
    unittest.main(verbosity=2)
