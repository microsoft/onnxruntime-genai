# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations  # for '|' shorthand in type hints

import shlex
import subprocess
from os import PathLike
from pathlib import Path

from .logger import get_logger

_log = get_logger("util.run")


def run(
    *args,
    cwd: PathLike | str | bytes | None = None,
    input=None,
    capture_stdout: bool = False,
    capture_stderr: bool = False,
    shell: bool = False,
    env: dict[str, str] | None = None,
    check: bool = True,
    quiet: bool = False,
):
    """Runs a subprocess.

    Args:
        *args: The subprocess arguments. May be passed as a single argument containing a list of strings.
        cwd: The working directory. If None, specifies the current directory.
        input: The optional input byte sequence.
        capture_stdout: Whether to capture stdout.
        capture_stderr: Whether to capture stderr.
        shell: Whether to run using the shell.
        env: The environment variables as a dict. If None, inherits the current
            environment.
        check: Whether to raise an error if the return code is not zero.
        quiet: If true, do not print output from the subprocess.

    Returns:
        A subprocess.CompletedProcess instance.
    """
    if len(args) == 1 and isinstance(args[0], list):
        cmd = args[0]
    else:
        cmd = [*args]

    if not cwd:
        cwd = Path.cwd()

    _log.info(f"Running subprocess in '{cwd}'\n  {' '.join([shlex.quote(arg) for arg in cmd])}")

    def output(is_stream_captured):
        return subprocess.PIPE if is_stream_captured else (subprocess.DEVNULL if quiet else None)

    completed_process = subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        input=input,
        stdout=output(capture_stdout),
        stderr=output(capture_stderr),
        env=env,
        shell=shell,
    )

    _log.debug(f"Subprocess completed. Return code: {completed_process.returncode}")

    return completed_process
