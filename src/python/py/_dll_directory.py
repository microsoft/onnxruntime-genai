# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import os
import sys

def _is_windows():
    return sys.platform.startswith("win")


def add_onnxruntime_dependency():
    """Add the onnxruntime DLL directory to the DLL search path.
    
    This function is a no-op on non-Windows platforms.
    """
    if _is_windows():
        import importlib.util
        ort_package = importlib.util.find_spec("onnxruntime")
        if not ort_package:
            raise ImportError("Could not find the onnxruntime package.")
        ort_package_path = ort_package.submodule_search_locations[0]
        os.add_dll_directory(os.path.join(ort_package_path[0], "capi"))


def add_cuda_dependency():
    """Add the CUDA DLL directory to the DLL search path.
    
    This function is a no-op on non-Windows platforms.
    """
    if _is_windows():
        cuda_path = os.getenv("CUDA_PATH", None)
        if cuda_path:
            os.add_dll_directory(os.path.join(cuda_path, "bin"))
        else:
            raise ImportError("Could not find the CUDA libraries. Please set the CUDA_PATH environment variable.")
