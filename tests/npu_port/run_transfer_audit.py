#!/usr/bin/env python3
"""Exec an unchanged command with the test-only ACL transfer auditor preloaded.

Usage:
  python -B tests/npu_port/run_transfer_audit.py --audit /path/to/auditor.so \
      --log /path/to/fresh-transfer-log.jsonl -- python existing_case.py

/dev/shm is mounted noexec on some hosts. An anonymous executable file descriptor
holds only the compiled auditor, so no mount changes, copied production libraries,
Python source injection or executable output outside the requested build area is
needed. The descriptor survives exec and must remain open in the target process.
The target must use normal dynamic linking; secure-execution mode ignores preload.
For ctypes/RTLD_LOCAL targets, use --runtime-library /path/to/libascendcl.so to
place the real ACL library after the auditor in the global loader scope, allowing
RTLD_NEXT forwarding without changing any Python source. This loads the existing
runtime normally, never rebuilds it or initializes an ACL device in this launcher.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def executable_copy(path: Path) -> int:
    """Return an inheritable anonymous executable descriptor for a compiled file."""
    fd = os.memfd_create("xgb-transfer-audit", flags=0)
    try:
        with path.open("rb") as source, os.fdopen(os.dup(fd), "wb") as output:
            while block := source.read(1024 * 1024):
                output.write(block)
        os.fchmod(fd, 0o500)
        os.lseek(fd, 0, os.SEEK_SET)
        return fd
    except BaseException:
        os.close(fd)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--runtime-library", action="append", type=Path, default=[],
                        help="existing ACL library to preload globally after the auditor")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("an unchanged target command is required after --")
    try:
        fd = executable_copy(args.audit)
        env = os.environ.copy()
        preload = f"/proc/self/fd/{fd}"
        for library in args.runtime_library:
            # Preserve /proc/self/fd paths and reject LD_PRELOAD delimiters.
            path = str(library.absolute())
            if any(character.isspace() or character == ":" for character in path):
                parser.error("runtime library path contains an LD_PRELOAD delimiter")
            if not library.is_file():
                parser.error("runtime library must be an existing regular file")
            preload += ":" + path
        if env.get("LD_PRELOAD"):
            preload += ":" + env["LD_PRELOAD"]
        env["LD_PRELOAD"] = preload
        env["XGB_NPU_TRANSFER_LOG"] = str(args.log.resolve())
        os.execvpe(command[0], command, env)
    except OSError as error:
        # Avoid printing environment, payloads or potentially sensitive paths.
        print(f"XGB_NPU_TRANSFER_AUDIT: launcher failed (errno={error.errno})", file=sys.stderr)
        raise SystemExit(125) from None


if __name__ == "__main__":
    main()