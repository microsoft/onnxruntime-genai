# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import onnxruntime_genai as og

def save_lora_params_to_flatbuffers(npz_file_path, fb_file_path):
    '''The function converts lora parameters in npz to flatbuffers file
    '''
    with np.load(npz_file_path) as data:
        to_save = {}
        for k, v in data.items():
            to_save[k] = v

        og.save_lora_parameters_to_flatbuffers(fb_file_path, to_save)
