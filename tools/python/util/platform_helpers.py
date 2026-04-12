# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import platform
import sys


def is_windows():
    return sys.platform.startswith("win")


def is_mac():
    return sys.platform.startswith("darwin")


def is_linux():
    return sys.platform.startswith("linux")


def is_linux_arm():
    return is_linux() and "arm" in platform.machine().lower()


def is_aix():
    return sys.platform.startswith("aix")


def is_windows_arm():
    return is_windows() and "arm" in platform.machine().lower()
