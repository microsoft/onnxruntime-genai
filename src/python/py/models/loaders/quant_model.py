# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Dispatch supported quantized checkpoint formats to their loaders."""

from .awq import AWQModel
from .gptq import GPTQModel
from .modelopt import ModeloptModel
from .olive import OliveModel
from .quark import QuarkModel


class QuantModel:
    @staticmethod
    def from_pretrained(quant_type, **kwargs):
        """
        Unpack quantized weights in PyTorch models, store them in a standard format, and repack them
        into ONNX Runtime's format. Also performs any pre-processing and post-processing when unpacking
        the quantized weights.
        """
        if quant_type == "awq":
            model = AWQModel(quant_type, **kwargs)
        elif quant_type == "gptq":
            model = GPTQModel(quant_type, **kwargs)
        elif quant_type == "olive":
            model = OliveModel(quant_type, **kwargs)
        elif quant_type == "quark":
            model = QuarkModel(quant_type, **kwargs)
        elif quant_type == "modelopt":
            model = ModeloptModel(quant_type, **kwargs)
        else:
            raise NotImplementedError(f"The {quant_type} quantized model is not currently supported.")

        return model