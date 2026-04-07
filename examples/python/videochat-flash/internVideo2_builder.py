import torch
import torch.nn as nn
import gc
import os
import argparse

model_id = "OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B"

parser = argparse.ArgumentParser()
parser.add_argument("--video", action="store_true", help="Export in video mode (compress=True, T=local_num_frames)")
parser.add_argument("--embed", action="store_true", help="Also export the text embedding layer (embed_tokens)")
parser.add_argument("--embed-only", action="store_true", help="Export ONLY the embedding layer (skip vision export)")
args = parser.parse_args()


IMAGE_PAD_TOKEN_ID = 151655  # <|image_pad|>


class EmbeddingWithMerge(nn.Module):
    """Embedding lookup + visual feature injection at <|image_pad|> positions.

    Inputs:  input_ids [batch, seq_len], image_features [1, num_visual_tokens, hidden_size]
    Output:  inputs_embeds [batch, seq_len, hidden_size]

    At positions where input_ids == image_pad_id, the text embedding is
    replaced by the corresponding visual feature from image_features.
    For text-only prompts (no image_pad tokens), image_features can be
    empty [1, 0, hidden_size] and the output is pure text embeddings.
    """

    def __init__(self, embed_weight, image_pad_id=IMAGE_PAD_TOKEN_ID):
        super().__init__()
        vocab_size, embed_dim = embed_weight.shape
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.embed_tokens.weight = nn.Parameter(embed_weight)
        self.image_pad_id = image_pad_id

    def forward(self, input_ids, image_features):
        text_embeds = self.embed_tokens(input_ids)
        hidden_size = text_embeds.shape[-1]

        mask = (input_ids == self.image_pad_id)

        # Map each image_pad position to its 0-based index in image_features
        indices = mask.long().cumsum(dim=-1) - 1
        indices = indices.clamp(min=0)

        # Flatten visual features; append a dummy zero row so indexing is
        # always safe (text-only case: image_features has 0 tokens)
        flat_features = image_features.reshape(-1, hidden_size)
        safe_features = torch.cat([
            flat_features,
            torch.zeros(1, hidden_size, dtype=flat_features.dtype, device=flat_features.device)
        ], dim=0)

        visual_at_positions = torch.nn.functional.embedding(indices, safe_features)

        mask_3d = mask.unsqueeze(-1).expand_as(text_embeds)
        inputs_embeds = torch.where(mask_3d, visual_at_positions, text_embeds)
        return inputs_embeds


def export_embedding():
    """Export embedding+merge model as fp32 ONNX.
    Loads only the embedding weight from safetensors — no full model needed (~2GB RAM).
    """
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    print("[1/3] Loading embedding weight from safetensors (fp32)...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    vocab_size = config.vocab_size
    embed_dim = config.hidden_size

    model_dir = snapshot_download(model_id)

    embed_key = "model.embed_tokens.weight"
    embed_weight = None
    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith(".safetensors"):
            continue
        shard_path = os.path.join(model_dir, fname)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            if embed_key in f.keys():
                embed_weight = f.get_tensor(embed_key).float()
                print(f"  Loaded {embed_key} from {fname}: {embed_weight.shape} → fp32")
                break

    if embed_weight is None:
        print(f"  ERROR: Could not find {embed_key} in safetensors shards")
        return

    model = EmbeddingWithMerge(embed_weight, image_pad_id=IMAGE_PAD_TOKEN_ID)
    model.eval()
    del embed_weight
    print(f"  embed_tokens: vocab_size={vocab_size:,}, dim={embed_dim}")
    print(f"  image_pad_id: {IMAGE_PAD_TOKEN_ID} (<|image_pad|>)")

    # Test: simulate a prompt with 64 image_pad tokens
    NUM_VISUAL_TOKENS = 64
    print(f"\n[2/3] Running test forward pass...")
    dummy_ids = torch.ones(1, 10 + NUM_VISUAL_TOKENS, dtype=torch.long) * 100
    dummy_ids[0, 5:5 + NUM_VISUAL_TOKENS] = IMAGE_PAD_TOKEN_ID
    dummy_features = torch.randn(1, NUM_VISUAL_TOKENS, embed_dim)

    with torch.no_grad():
        test_out = model(dummy_ids, dummy_features)
        print(f"  input_ids:      {dummy_ids.shape}  (with {NUM_VISUAL_TOKENS} image_pad tokens)")
        print(f"  image_features: {dummy_features.shape}")
        print(f"  inputs_embeds:  {test_out.shape}  (dtype={test_out.dtype})")

        # Verify merge: image_pad positions should have visual features, not text embeds
        text_only = model.embed_tokens(dummy_ids)
        merged_at_pad = test_out[0, 5]
        text_at_pad = text_only[0, 5]
        visual_expected = dummy_features[0, 0]
        assert torch.allclose(merged_at_pad, visual_expected), "Merge verification failed!"
        assert not torch.allclose(merged_at_pad, text_at_pad), "Merge did not replace text embed!"
        print("  Merge verification: PASSED")

    print(f"\n[3/3] Exporting to ONNX...")
    embed_onnx = "vcf-embed.onnx"
    embed_data = "vcf-embed.onnx.data"

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_ids, dummy_features),
            embed_onnx,
            input_names=["input_ids", "image_features"],
            output_names=["inputs_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "image_features": {0: "num_images", 1: "num_image_tokens"},
                "inputs_embeds": {0: "batch", 1: "seq_len"},
            },
            opset_version=18,
            dynamo=False,
        )

    del model
    gc.collect()

    embed_proto = onnx.load(embed_onnx, load_external_data=True)

    for f_name in os.listdir("."):
        if f_name.endswith((".onnx", ".onnx.data", ".py", ".json")):
            continue
        if os.path.isfile(f_name) and not f_name.startswith("."):
            _, ext = os.path.splitext(f_name)
            if ext == "":
                os.remove(f_name)

    convert_model_to_external_data(
        embed_proto,
        all_tensors_to_one_file=True,
        location=embed_data,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.save_model(embed_proto, embed_onnx)
    print(f"\nExported {embed_onnx} + {embed_data} successfully")
    print(f"  Inputs:  input_ids [B, seq_len] + image_features [1, N, {embed_dim}]")
    print(f"  Output:  inputs_embeds [B, seq_len, {embed_dim}] (fp32)")
    print(f"  Merges visual tokens at <|image_pad|> (id={IMAGE_PAD_TOKEN_ID}) positions")


def export_vision():
    """Export vision tower + mm_projector as an ONNX model."""
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data
    from transformers import AutoModel, AutoConfig

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    LOCAL_NUM_FRAMES = getattr(config, "mm_local_num_frames", 4)

    class VisionWithProjectorImage(nn.Module):
        def __init__(self, vision_tower, mm_projector):
            super().__init__()
            self.vision_tower = vision_tower
            self.mm_projector = mm_projector

        def forward(self, images):
            visual_features = self.vision_tower(images)
            projected = self.mm_projector(visual_features, compress=False)
            return projected

    class VisionWithProjectorVideo(nn.Module):
        def __init__(self, vision_tower, mm_projector, local_num_frames):
            super().__init__()
            self.vision_tower = vision_tower
            self.mm_projector = mm_projector
            self.local_num_frames = local_num_frames

        def forward(self, images):
            T = self.local_num_frames
            visual_features = self.vision_tower(images)
            B = visual_features.shape[0]
            visual_features = visual_features.reshape(B * T, -1, visual_features.shape[-1])
            projected = self.mm_projector(visual_features, compress=True, local_num_frames=T)
            return projected

    print("[1/5] Loading model in float16...")
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
    )

    vision_tower = model.get_vision_tower()
    mm_projector = model.model.mm_projector

    del model.model.layers, model.model.embed_tokens, model.lm_head
    del model
    gc.collect()

    meta_params = {n: p for n, p in vision_tower.named_parameters() if p.device.type == "meta"}
    if meta_params:
        print(f"[2/5] Fixing {len(meta_params)} meta-device params (gamma→weight name mismatch)...")

        from safetensors import safe_open
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(model_id)

        vt_prefix = "model.vision_tower."
        ckpt_lookup = {}
        for fname in sorted(os.listdir(model_dir)):
            if not fname.endswith(".safetensors"):
                continue
            shard_path = os.path.join(model_dir, fname)
            with safe_open(shard_path, framework="pt") as f:
                for key in f.keys():
                    if key.startswith(vt_prefix):
                        model_key = key[len(vt_prefix):]
                        ckpt_lookup[model_key] = (shard_path, key)

        print(f"  Found {len(ckpt_lookup)} vision tower keys in safetensors")

        needed = {}
        for param_name in meta_params:
            for candidate in [param_name, param_name.replace(".weight", ".gamma")]:
                if candidate in ckpt_lookup:
                    shard_path, ckpt_key = ckpt_lookup[candidate]
                    needed.setdefault(shard_path, []).append((param_name, ckpt_key))
                    break

        fixed = 0
        for shard_path, items in needed.items():
            print(f"  Loading {len(items)} tensors from {os.path.basename(shard_path)}...")
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for param_name, ckpt_key in items:
                    tensor = f.get_tensor(ckpt_key)
                    parts = param_name.rsplit(".", 1)
                    parent = vision_tower
                    for part in parts[0].split("."):
                        parent = getattr(parent, part)
                    setattr(parent, parts[1], nn.Parameter(tensor.to(torch.float16)))
                    fixed += 1

        if fixed < len(meta_params):
            unmatched = [n for n in meta_params if n not in {p for items in needed.values() for p, _ in items}]
            print(f"  UNMATCHED ({len(unmatched)}): {unmatched[:5]}")

        print(f"  Fixed {fixed}/{len(meta_params)} params")
    else:
        print("[2/5] All vision tower parameters on CPU - OK")

    if args.video:
        combined = VisionWithProjectorVideo(vision_tower, mm_projector, LOCAL_NUM_FRAMES)
        num_frames = LOCAL_NUM_FRAMES
        mode_str = f"video (compress=True, T={LOCAL_NUM_FRAMES}, 16*T={16*LOCAL_NUM_FRAMES} tokens/segment)"
    else:
        combined = VisionWithProjectorImage(vision_tower, mm_projector)
        num_frames = 1
        mode_str = "image (compress=False, 64 tokens/image)"
    combined.float().eval()
    print(f"  Mode: {mode_str}")

    proj_params = sum(p.numel() for p in mm_projector.parameters())
    print(f"  mm_projector: {proj_params:,} params, MLP {mm_projector.mm_hidden_size} → {mm_projector.mlp[0].out_features}")

    dummy_images = torch.randn(1, num_frames, 3, 224, 224)

    print("[3/5] Running a test forward pass...")
    with torch.no_grad():
        test_out = combined(dummy_images)
        print(f"  Input:  {dummy_images.shape}")
        print(f"  Output: {test_out.shape}")

    print("[4/5] Exporting to ONNX...")
    onnx_path = "vcf-vision-video.onnx" if args.video else "vcf-vision.onnx"
    data_file = onnx_path.replace(".onnx", ".onnx.data")
    with torch.no_grad():
        torch.onnx.export(
            combined,
            (dummy_images,),
            onnx_path,
            input_names=["images"],
            output_names=["visual_tokens"],
            dynamic_axes={
                "images": {0: "batch", 1: "num_frames"},
                "visual_tokens": {0: "batch", 1: "num_visual_tokens"},
            },
            opset_version=18,
            dynamo=False,
        )

    print("[5/5] Consolidating weights...")
    model_proto = onnx.load(onnx_path, load_external_data=True)

    for f in os.listdir("."):
        if f == onnx_path or f.endswith((".onnx.data", ".py", ".json")):
            continue
        if os.path.isfile(f) and not f.startswith("."):
            _, ext = os.path.splitext(f)
            if ext == "" or f.startswith("vision_tower") or f.startswith("mm_projector"):
                os.remove(f)

    convert_model_to_external_data(
        model_proto,
        all_tensors_to_one_file=True,
        location=data_file,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.save_model(model_proto, onnx_path)

    print(f"\nExported {onnx_path} + {data_file} successfully")
    if args.video:
        print(f"  Pipeline: [{num_frames} frames] → InternVideo2 → reshape → ToMe(compress) → MLP → visual_tokens")
        print(f"  Fixed at T={LOCAL_NUM_FRAMES} frames (mm_local_num_frames from config)")
    else:
        print("  Pipeline: [1 image] → InternVideo2 → ToMe → MLP → visual_tokens")

    del combined, vision_tower, mm_projector, model_proto
    gc.collect()


# ── Main ──
if args.embed_only:
    export_embedding()
elif args.embed:
    export_vision()
    export_embedding()
else:
    export_vision()
