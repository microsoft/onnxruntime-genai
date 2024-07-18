# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import numpy as np
import onnxruntime_genai as og

def save_lora_params_to_flatbuffers(npz_file_path, fb_file_path):
    '''The function converts lora parameters in npz to flatbuffers file
    '''
    with np.load(npz_file_path) as data:
        to_save = {}
        for k, v in data.items():
            to_save[k] = v

        og.save_lora_parameters_to_flatbuffers(str(fb_file_path), to_save)


def add_adapters_to_genai_config(json_file_path : str,
                               modified_json_file_path : str,
                               adapters : dict):
    '''The function reads genai config json file, adds adapters section
       Expects a dictionary of adapters with weights (name to adapter dictionary)
       Supported format:
        adapters = {
            "guanaco": {
                "weights": "models/exported/guanaco_qlora.fb",
            },
            "tiny-codes": {
                "weights": "models/tiny-codes-qlora/qlora-conversion-transformers_optimization-extract-metadata/gpu-cuda_model/adapter_weights.fb",
            }
        }       
    '''

    genai_config = None
    with open(json_file_path, 'r') as f:
        genai_config = json.load(f)
        # Add an array adapters to root of json
        if 'adapters' not in genai_config:
            genai_config['adapters'] = {}
        
    for adapter_name, adapter in adapters.items():
        genai_config['adapters'][adapter_name] = adapter

    with open(modified_json_file_path, 'w') as f:
        json.dump(genai_config, f)
        