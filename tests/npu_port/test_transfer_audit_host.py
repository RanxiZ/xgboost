#!/usr/bin/env python3
"""Compile and validate the opt-in ACL transfer auditor without a device.

Run: python -B tests/npu_port/test_transfer_audit_host.py
All output is kept in a fresh directory under /dev/shm/xgb-transfer-audit-build.
Only a host mock library is executed; no CANN library, training or device tool runs.
The printed auditor path is usable with run_transfer_audit.py, which handles a
noexec /dev/shm using an anonymous descriptor and execs an unchanged target command.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from run_transfer_audit import executable_copy


def main() -> None:
    source = Path(__file__).resolve().parent
    headers = Path("/usr/local/Ascend/cann-9.1.0/x86_64-linux/include")
    root = Path("/dev/shm/xgb-transfer-audit-build")
    root.mkdir(mode=0o700, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix="host-", dir=root))
    audit = build / "libxgb_npu_transfer_audit.so"
    mock = build / "libxgb_transfer_host_mock.so"
    probe = build / "transfer_audit_host_probe"
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    env.pop("XGB_NPU_TRANSFER_LOG", None)
    env.pop("XGB_TRANSFER_MOCK_CALLED", None)
    flags = ["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread",
             f"-I{headers}"]
    commands = [
        flags + ["-fPIC", "-shared", str(source / "transfer_audit.cc"), "-ldl", "-o", str(audit)],
        flags + ["-fPIC", "-shared", "-DXGB_TRANSFER_MOCK_LIBRARY",
                 str(source / "transfer_audit_host_fixture.cc"),
                 "-Wl,-soname,libxgb_transfer_host_mock.so", "-o", str(mock)],
        flags + [str(source / "transfer_audit_host_fixture.cc"), f"-L{build}",
                 "-lxgb_transfer_host_mock", f"-Wl,-rpath,{build}", "-o", str(probe)],
    ]
    for command in commands:
        subprocess.run(command, env=env, check=True)
    # Noexec shared memory still permits compilation. Execute anonymous copies;
    # the mock's SONAME satisfies the probe dependency without loading real ACL.
    fds = tuple(executable_copy(path) for path in (audit, mock, probe))
    env["LD_PRELOAD"] = ":".join(f"/proc/self/fd/{fd}" for fd in fds[:2])
    probe_command = [f"/proc/self/fd/{fds[2]}"]

    def run_probe(*args: str, **kwargs):
        return subprocess.run(probe_command + list(args), env=env, pass_fds=fds, **kwargs)

    # Disabled auditor still forwards and preserves return codes, data and errno.
    run_probe(check=True)
    assert not list(build.glob("*.jsonl"))
    log = build / "transfers.jsonl"
    with log.open("x", encoding="utf-8") as stream:
        stream.write('{"stage":"external-host-marker"}\n')
    env["XGB_NPU_TRANSFER_LOG"] = str(log)
    for _ in range(2):
        run_probe(check=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[0] == {"stage": "external-host-marker"}, "existing log was overwritten"
    events = rows[1:]
    assert len(events) == 2 * 811, len(events)
    assert {r["api"] for r in events} == {
        "aclrtMemcpy", "aclrtMemcpyAsync", "aclrtMalloc", "aclrtMallocHost",
        "aclrtFree", "aclrtFreeHost"}
    pids = {r["pid"] for r in events}
    assert len(pids) == 2
    for pid in pids:
        group = [r for r in events if r["pid"] == pid]
        assert [r["seq"] for r in group] == list(range(1, 812))
        assert all(a["mono_ns"] <= b["mono_ns"] for a, b in zip(group, group[1:]))
        assert len({r["tid"] for r in group}) == 9
        assert {r["kind"] for r in group if r["api"] == "aclrtMemcpy"} == {0, 1, 2, 3}
        assert [(r["result"], r["success"]) for r in group if not r["success"]] == [
            (123, False), (124, False)]
        assert all(r["bytes"] == 32 and r["dst_max"] == 32 for r in group
                   if r["api"] == "aclrtMemcpy" and r["success"])
    # Invalid destinations must fail loudly, but only after the real mock call ran.
    called = build / "forwarded-before-log-error"
    env["XGB_TRANSFER_MOCK_CALLED"] = str(called)
    for bad in (build / "missing-parent" / "log", build):
        env["XGB_NPU_TRANSFER_LOG"] = str(bad)
        result = run_probe("single", capture_output=True, text=True)
        assert result.returncode == 125, result
        assert "requested log cannot be opened as a regular file" in result.stderr
    assert called.read_bytes() == b"xx"
    # A symlink must not overwrite or append to an unrelated evidence file.
    link = build / "symlink-log"
    link.symlink_to(log)
    before = log.read_bytes()
    env["XGB_NPU_TRANSFER_LOG"] = str(link)
    result = run_probe("single", capture_output=True, text=True)
    assert result.returncode == 125 and log.read_bytes() == before
    # Force a post-open write failure without touching a device or filling a disk.
    # RLIMIT_FSIZE=0 + ignored SIGXFSZ makes write return EFBIG.
    import resource
    import signal

    def write_guard() -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)

    env["XGB_NPU_TRANSFER_LOG"] = str(build / "write-failure.jsonl")
    result = run_probe("single", capture_output=True, text=True, preexec_fn=write_guard)
    assert result.returncode == 125 and "log write failed" in result.stderr, result
    assert called.read_bytes() == b"xxx", "file-size guard should block mock marker write"
    launcher_env = env.copy()
    launcher_env.pop("LD_PRELOAD", None)
    launcher_env.pop("XGB_TRANSFER_MOCK_CALLED", None)
    launcher_log = build / "launcher.jsonl"
    subprocess.run([sys.executable, "-B", str(source / "run_transfer_audit.py"),
                    "--audit", str(audit), "--log", str(launcher_log),
                    "--runtime-library", f"/proc/self/fd/{fds[1]}", "--",
                    probe_command[0], "single"], env=launcher_env, pass_fds=fds, check=True)
    launcher_events = [json.loads(line) for line in launcher_log.read_text().splitlines()]
    assert len(launcher_events) == 1 and launcher_events[0]["success"]
    assert launcher_events[0]["kind"] == 2
    for fd in fds:
        os.close(fd)
    print("PASS: 1,622 complete records; forwarding, 8-thread logging, append, disabled mode,")
    print("      return/errno preservation, failed allocation, bad path/symlink/write guards,")
    print("      plus one event through the noexec-safe unchanged-command launcher.")
    print(f"AUDITOR={audit}")
    print(f"HOST_EVIDENCE={build}")


if __name__ == "__main__":
    main()