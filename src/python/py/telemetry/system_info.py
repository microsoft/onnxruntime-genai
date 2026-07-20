# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""System information collector for telemetry heartbeat events.

Collects HW/OS/GPU metadata modeled on Foundry Local's DeviceIdEvent
and ORT's LogProcessInfo for MAD/DAD tracking.
"""

import os
import platform
import subprocess
import sys
from functools import lru_cache
from typing import Any, Optional


@lru_cache(maxsize=1)
def get_system_info() -> dict[str, Any]:
    """Collect system information once and cache it."""
    info: dict[str, Any] = {}

    # OS information
    info["os"] = platform.system()
    info["os_version"] = platform.version()
    info["os_release"] = platform.release()
    info["os_arch"] = platform.machine()
    info["os_process_arch"] = platform.architecture()[0]

    # CPU information
    info["processor_count"] = os.cpu_count() or 0
    info["cpu_model"] = _get_cpu_model()

    # Memory
    info["total_memory_mb"] = _get_total_memory_mb()

    # GPU
    gpu_info = _get_gpu_info()
    info.update(gpu_info)

    # Device identification
    info["device_manufacturer"] = _get_device_manufacturer()
    info["device_model"] = _get_device_model()

    # Python / runtime
    info["python_version"] = sys.version.split()[0]

    # ORT version
    info["ort_version"] = _get_ort_version()

    # Process info
    info["process_name"] = os.path.basename(sys.argv[0]) if sys.argv else ""

    return info


def _get_cpu_model() -> str:
    """Get CPU model string."""
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Windows":
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        pass
    return platform.processor() or ""


def _get_total_memory_mb() -> int:
    """Get total system memory in MB using only the standard library."""
    try:
        system = platform.system()
        if system == "Windows":
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys / (1024 * 1024))
        elif system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // 1024
        elif system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) // (1024 * 1024)
    except Exception:
        pass
    return 0


def _get_gpu_info() -> dict[str, Any]:
    """Get GPU information via nvidia-smi."""
    info: dict[str, Any] = {
        "gpu_name": "",
        "gpu_driver_version": "",
        "gpu_memory_mb": 0,
        "gpu_count": 0,
    }

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) >= 3:
                info["gpu_name"] = parts[0]
                info["gpu_driver_version"] = parts[1]
                info["gpu_memory_mb"] = int(float(parts[2]))
                info["gpu_count"] = len(lines)
    except Exception:
        pass

    # Try DirectML / Windows WMI if nvidia-smi failed
    if not info["gpu_name"] and platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "Name,DriverVersion,AdapterRAM", "/format:csv"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
                if len(lines) > 1:
                    parts = lines[1].split(",")
                    if len(parts) >= 4:
                        info["gpu_memory_mb"] = int(parts[1]) // (1024 * 1024) if parts[1].isdigit() else 0
                        info["gpu_driver_version"] = parts[2]
                        info["gpu_name"] = parts[3]
        except Exception:
            pass

    return info


def _get_device_manufacturer() -> str:
    """Get device manufacturer."""
    try:
        if platform.system() == "Darwin":
            return "Apple"
        elif platform.system() == "Linux":
            with open("/sys/class/dmi/id/sys_vendor") as f:
                return f.read().strip()
        elif platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "computersystem", "get", "manufacturer", "/format:value"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("Manufacturer="):
                        return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return ""


def _get_device_model() -> str:
    """Get device model."""
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.model"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        elif platform.system() == "Linux":
            with open("/sys/class/dmi/id/product_name") as f:
                return f.read().strip()
        elif platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "computersystem", "get", "model", "/format:value"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("Model="):
                        return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return ""


def _get_ort_version() -> str:
    """Get ONNX Runtime version if installed."""
    try:
        import onnxruntime
        return onnxruntime.__version__
    except ImportError:
        pass
    return ""


def get_execution_provider_info() -> dict[str, Any]:
    """Get information about available execution providers."""
    info: dict[str, Any] = {
        "available_providers": [],
    }
    try:
        import onnxruntime
        info["available_providers"] = onnxruntime.get_available_providers()
    except ImportError:
        pass
    return info
