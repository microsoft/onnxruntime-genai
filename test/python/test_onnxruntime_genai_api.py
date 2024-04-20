# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import os
import sys
import sysconfig
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest

# TODO: CUDA pipelines use python3.6 and do not have a way to download models since downloading models
# requires pytorch and hf transformers. This test should be re-enabled once the pipeline is updated.
@pytest.mark.skipif(
    sysconfig.get_platform().endswith("arm64") or sys.version_info.minor < 8,
    reason="Python 3.8 is required for downloading models.",
)
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if og.is_cuda_available() else ["cpu"]
)
def test_batching(device, phi2_for):
    model = og.Model(phi2_for(device))
    tokenizer = og.Tokenizer(model)

    prompts = [
        "This is a test.",
        "Rats are awesome pets!",
        "The quick brown fox jumps over the lazy dog.",
    ]

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=20)  # To run faster
    params.input_ids = tokenizer.encode_batch(prompts)

    output_sequences = model.generate(params)
    print(tokenizer.decode_batch(output_sequences))
