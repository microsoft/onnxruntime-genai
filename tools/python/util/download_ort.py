# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import requests
from os import PathLike
import sys
from pathlib import Path
from .logger import get_logger
from .platform_helpers import is_linux, is_windows
import tempfile
import tarfile

_log = get_logger("util.download_ort")


def _extract(archive_name: str, archive_path: str | os.PathLike, root: str | os.PathLike):
    archive_include_dir = f"{archive_name[:-4]}/include/" # Remove `.tgz`
    archive_lib_dir = f"{archive_name[:-4]}/lib/" # Remove `.tgz`
    l = len(archive_name[:-3]) # Remove package_name/
    members = []
    with tarfile.open(archive_path, 'r') as tar:
        for member in tar.getmembers():
            print(member.path)
            if member.path.startswith(archive_include_dir):
                member.path = member.path[l:]
                members.append(member)
            elif member.path.startswith(archive_lib_dir):
                member.path = member.path[l:]
                members.append(member)

        tar.extractall(members=members, path=root)


def _download(url: str, package_name: str, root: str | os.PathLike):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with tempfile.TemporaryDirectory() as dir_name:
            package_path = Path(dir_name) / package_name
            with open(package_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=16*1024): 
                    f.write(chunk)
            
            _extract(package_name, package_path, root)


def download_latest(use_cuda: bool, ort_home: str | os.PathLike):
    url = requests.head("https://github.com/microsoft/onnxruntime/releases/latest").headers["Location"]
    version = url.split('/')[-1][1:]
    package_name = None
    package_suffix = "-gpu" if use_cuda else ""
    if is_linux():
        package_name = f"onnxruntime-linux-x64{package_suffix}-{version}.tgz"
    elif is_windows():
        package_name = f"onnxruntime-win-x64{package_suffix}-{version}.zip"
    else:
        raise OSError(f"Downloading onnxruntime for {sys.platform} is not supported.")
    package_url = f"https://github.com/microsoft/onnxruntime/releases/download/v{version}/{package_name}"
    _download(package_url, package_name, ort_home)


def download_ort(use_cuda: bool, ort_home: str | os.PathLike):
    download_latest(use_cuda, ort_home)
