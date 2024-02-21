# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import os
import sys

def _is_windows():
    return sys.platform.startswith("win")


def add_dll_directory():
    """Add the CUDA DLL directory to the DLL search path.
    
    This function is a no-op on non-Windows platforms.
    """
    if _is_windows():
        cuda_path = os.getenv("CUDA_PATH", None)
        if cuda_path:
            os.add_dll_directory(os.path.join(cuda_path, "bin"))
        else:
            raise ImportError("Could not find the CUDA libraries. Please set the CUDA_PATH environment variable.")
