# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import os
import sys
import time
import numpy as np
import onnxruntime_genai as og
from common import get_config

# Maps short language codes / locale tags to (lang_id, human-readable name).
# Mirrors the upstream NeMo multilingual Nemotron prompt schema.
LANG_TO_ID = {
    "en":    (0,   "English (default / US)"),
    "en-US": (0,   "English (United States)"),
    "en-GB": (1,   "English (United Kingdom)"),
    "es-ES": (2,   "Spanish (Spain)"),
    "es":    (3,   "Spanish (default / Latin America)"),
    "es-US": (3,   "Spanish (US Latin American)"),
    "zh-CN": (4,   "Chinese (Simplified, Mainland)"),
    "zh-TW": (5,   "Chinese (Traditional, Taiwan)"),
    "hi":    (6,   "Hindi"),
    "ar":    (7,   "Arabic"),
    "fr":    (8,   "French (default / France)"),
    "fr-FR": (8,   "French (France)"),
    "fr-CA": (100, "French (Canada)"),
    "de":    (9,   "German"),
    "de-DE": (9,   "German (Germany)"),
    "it":    (10,  "Italian"),
    "ru":    (11,  "Russian"),
    "pt-BR": (12,  "Portuguese (Brazil)"),
    "pt":    (13,  "Portuguese (default / Portugal)"),
    "pt-PT": (13,  "Portuguese (Portugal)"),
    "ja":    (14,  "Japanese"),
    "ko":    (15,  "Korean"),
    "nl":    (16,  "Dutch"),
    "pl":    (17,  "Polish"),
    "tr":    (18,  "Turkish"),
    "uk":    (19,  "Ukrainian"),
    "ro":    (20,  "Romanian"),
    "el":    (21,  "Greek"),
    "cs":    (22,  "Czech"),
    "hu":    (23,  "Hungarian"),
    "sv":    (24,  "Swedish"),
    "da":    (25,  "Danish"),
    "fi":    (26,  "Finnish"),
    "no":    (27,  "Norwegian"),
    "sk":    (28,  "Slovak"),
    "hr":    (29,  "Croatian"),
    "bg":    (30,  "Bulgarian"),
    "auto":  (101, "Auto-detect"),
}


def load_config(model_path):
    """Read sample_rate and chunk_samples from genai_config.json."""
    config_path = os.path.join(model_path, "genai_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    sample_rate = config["model"]["sample_rate"]
    chunk_samples = config["model"]["chunk_samples"]
    return sample_rate, chunk_samples


def load_audio(audio_path, sample_rate):
    import soundfile as sf
    audio, sr = sf.read(audio_path, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        import scipy.signal
        num_samples = int(len(audio) * sample_rate / sr)
        audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
    return audio


def decode_tokens(generator, tokenizer_stream):
    """Decode all available tokens from the generator, returning the text."""
    text = ""
    while not generator.is_done():
        generator.generate_next_token()
        tokens = generator.get_next_tokens()
        if len(tokens) > 0:
            token_text = tokenizer_stream.decode(tokens[0])
            if token_text:
                print(token_text, end="", flush=True)
                text += token_text
    return text


def simulate_microphone(model_path, audio_path, execution_provider, use_vad=None, language=None):
    """Stream audio through Generator + StreamingProcessor API."""
    sample_rate, chunk_samples = load_config(model_path)
    audio = load_audio(audio_path, sample_rate)
    duration = len(audio) / sample_rate

    config = get_config(model_path, execution_provider)
    selected_lang = None
    if language is not None:
        if language not in LANG_TO_ID:
            raise ValueError(f"Unknown language '{language}'. Known: {sorted(LANG_TO_ID)}")
        selected_lang = LANG_TO_ID[language]
        lang_id, lang_name = selected_lang
        print(f"  Language: {language} -> {lang_name} (lang_id={lang_id})")
    model = og.Model(config)
    processor = og.StreamingProcessor(model)

    # VAD is off by default. Use --use_vad true to enable (requires "vad" section in genai_config.json).
    processor.set_option("use_vad", "false")
    if use_vad:
        try:
            processor.set_option("use_vad", "true")
        except Exception as e:
            print(f"  VAD: disabled (no VAD config in genai_config.json: {e})")
    vad_status = processor.get_option("use_vad")
    print(f"  Use VAD: {vad_status}")
    if vad_status == "true":
        print(f"  VAD threshold: {processor.get_option('vad_threshold')}")

    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    params = og.GeneratorParams(model)
    generator = og.Generator(model, params)
    # Per-generator language selection
    if selected_lang is not None:
        generator.set_runtime_option("lang_id", str(int(selected_lang[0])))

    print("-" * 60)
    stream_start = time.perf_counter()
    full_transcript = ""
    vad_enabled = vad_status == "true"
    chunks_total = 0
    chunks_processed = 0
    chunks_skipped = 0

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples].astype(np.float32)
        chunks_total += 1
        inputs = processor.process(chunk)
        if inputs is not None:
            chunks_processed += 1
            generator.set_inputs(inputs)
            full_transcript += decode_tokens(generator, tokenizer_stream)
        else:
            chunks_skipped += 1

    # Flush remaining audio
    inputs = processor.flush()
    if inputs is not None:
        generator.set_inputs(inputs)
        full_transcript += decode_tokens(generator, tokenizer_stream)

    total_wall = time.perf_counter() - stream_start

    print(f"\n{'=' * 60}")
    print(f"  {full_transcript.strip()}")
    print(f"{'=' * 60}")
    print(f"  Audio: {duration:.2f}s | Wall: {total_wall:.2f}s | RTF: {duration/total_wall:.2f}x")
    if vad_enabled:
        pct_saved = chunks_skipped / max(chunks_total, 1) * 100
        print(f"  VAD Metrics: {chunks_total} total chunks, {chunks_processed} processed, "
              f"{chunks_skipped} skipped ({pct_saved:.1f}% compute saved)")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--audio_file", type=str, required=True)
    parser.add_argument("--use_vad", type=str, choices=["true", "false"], default=None,
                        help="Override VAD setting from genai_config.json (true/false).")
    lang_help = "Language / locale code for the multilingual encoder. " \
                "Overrides model.default_lang_id from genai_config.json. " \
                "Use a bare ISO 639-1 code (e.g. 'de', 'fr', 'pt') for the default locale, " \
                "or a BCP-47 locale tag for region-specific variants. " \
                "Pass 'auto' to let the model detect the language. Supported codes:\n" \
                + "\n".join(f"  {code:<7} {name}" for code, (_, name) in sorted(LANG_TO_ID.items()))
    parser.add_argument("--language", "-l", type=str, default=None, help=lang_help)
    parser.add_argument("-e", "--execution_provider", type=str, required=False, default="follow_config",
                        choices=["cpu", "cuda", "dml", "follow_config"],
                        help="Execution provider to run with. Defaults to follow_config.")
    args = parser.parse_args()
    if not os.path.exists(args.audio_file):
        print(f"Error: {args.audio_file} not found")
        sys.exit(1)
    use_vad_override = None
    if args.use_vad is not None:
        use_vad_override = args.use_vad == "true"
    simulate_microphone(args.model_path, args.audio_file, args.execution_provider,
                        use_vad=use_vad_override, language=args.language)


if __name__ == "__main__":
    main()