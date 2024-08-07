# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import os
import sys

def _is_windows():
    return sys.platform.startswith("win")


def _is_linux():
    return sys.platform.startswith("linux")


def add_onnxruntime_dependency(package_id: str):
    """Add the onnxruntime shared library dependency.
    
    On Windows, this function adds the onnxruntime DLL directory to the DLL search path.
    On Linux, this function loads the onnxruntime shared library and its dependencies
    so that they can be found by the dynamic linker.
    """
    if _is_windows():
        import importlib.util
        ort_package = importlib.util.find_spec("onnxruntime")
        if not ort_package:
            raise ImportError("Could not find the onnxruntime package.")
        ort_package_path = ort_package.submodule_search_locations[0]
        os.add_dll_directory(os.path.join(ort_package_path, "capi"))

        if package_id == "onnxruntime-genai-directml":
            # Load the DirectML.dll library to avoid loading it again in the native code.
            # This avoids needing to know the exact path of the shared library from native code.
            dml_path = os.path.join(ort_package_path, "capi", "DirectML.dll")
            if not os.path.exists(dml_path):
                raise ImportError("Could not find the DirectML.dll library. "
                                  "Please check if the onnxruntime directml package is installed.")

            import ctypes
            _ = ctypes.CDLL(dml_path)

    elif _is_linux():
        import importlib.util
        import ctypes
        import glob

        ort_package = importlib.util.find_spec("onnxruntime")
        if not ort_package:
            raise ImportError("Could not find the onnxruntime package.")
        
        # Load the onnxruntime shared library here since we can find the path in python with ease.
        # This avoids needing to know the exact path of the shared library from native code.
        ort_package_path = ort_package.submodule_search_locations[0]
        ort_lib_path = glob.glob(os.path.join(ort_package_path, "capi", "libonnxruntime.so*"))
        if not ort_lib_path:
            raise ImportError("Could not find the onnxruntime shared library.")

        _ = ctypes.CDLL(ort_lib_path[0])


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
