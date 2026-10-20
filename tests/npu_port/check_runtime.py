#!/usr/bin/env python3
"""Build and validate the isolated NPU runtime without initializing any device.

The injected lifecycle tests link no device runtime. The real CANN adapter is
compiled and linked separately, then checked under a fail-before-call preload
library. Device/int64 operator execution is deliberately a separate opt-in step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-root", type=Path, required=True)
    parser.add_argument("--cann-root", type=Path, default=Path("/usr/local/Ascend/cann-9.1.0"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    require(root == args.expected_source_root.resolve(), "unexpected source checkout")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cann = args.cann_root.resolve(strict=True)
    report = {
        "status": "failed", "validation_kind": "host_only_npu_runtime_validation",
        "source_root": str(root), "cann_root": str(cann),
        "npu_execution": "not_run", "cuda_execution": "not_run",
        "training_backend": "not_implemented", "commands": [],
    }
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    env["LD_LIBRARY_PATH"] = ""

    def run(command: list[str], log: str, *, run_env: dict | None = None,
            expected: int = 0) -> str:
        report["commands"].append({"argv": command, "log": log, "expected_exit_code": expected})
        with (out / log).open("w") as stream:
            result = subprocess.run(command, cwd=root, env=run_env or env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=180)
        text = (out / log).read_text()
        require(result.returncode == expected,
                f"exit {result.returncode}, expected {expected}; see {out / log}")
        return text

    try:
        branch = subprocess.check_output(["git", "-C", str(root), "branch", "--show-current"]).decode().strip()
        require(branch == "xgb-npu-port", "unexpected integration branch")
        report["branch"] = branch
        compiler = os.environ.get("CXX", "g++")
        report["compiler"] = subprocess.check_output([compiler, "--version"], env=env).decode()
        source = root / "tests/npu_port"
        base = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic"]
        host_binary = out / "runtime_host_test"
        run([*base, "-O2", str(source / "runtime_host_test.cc"), "-o", str(host_binary)],
            "host-compile.log")
        libraries = run(["ldd", str(host_binary)], "host-linked-libraries.txt")
        require(not any(x in libraries.lower() for x in
                        ("ascend", "libopapi", "libcuda", "libcudart", "not found")),
                "injected runtime tests must not link a device runtime")
        text = run([str(host_binary)], "host-tests.log")
        tests = [line[5:] for line in text.splitlines() if line.startswith("PASS ")]
        require(len(tests) == len(set(tests)) == 41, "unexpected/duplicate host test count")
        sentinel = f"RUNTIME_HOST_PASS checks={len(tests)} device_execution=not_run"
        require(sentinel in text, "missing host sentinel")
        report["host_tests"] = tests
        report["host_test_count"] = len(tests)

        sanitized = out / "runtime_host_test_sanitized"
        run([*base, "-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=address,undefined",
             str(source / "runtime_host_test.cc"), "-o", str(sanitized)], "sanitizer-compile.log")
        san_env = {**env, "ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1",
                   "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"}
        text = run([str(sanitized)], "sanitizer-tests.log", run_env=san_env)
        require(sentinel in text, "missing sanitizer sentinel")
        require([line[5:] for line in text.splitlines() if line.startswith("PASS ")] == tests,
                "sanitizer host test list differs")
        require("ERROR:" not in text and "runtime error:" not in text, "sanitizer diagnostic")
        report["sanitizers"] = {"address": "passed", "undefined_behavior": "passed",
                                "leak_detection": "passed"}

        # Compile each production header independently to catch accidental include
        # dependencies. Compilation does not execute an ACL function.
        for header in ("runtime.h", "acl_api.h"):
            unit = out / (header + ".cc")
            unit.write_text(f'#include "{root / "src/common/npu" / header}"\nint main() {{}}\n')
            run([*base, "-isystem", str(cann / "include"), "-fsyntax-only", str(unit)],
                header + "-self-contained.log")

        cann_env = {**env, "LD_LIBRARY_PATH": ":".join(map(str, [
            cann / "lib64", cann / "lib64/plugin/opskernel", cann / "lib64/plugin/nnengine",
            cann / "opp/built-in/op_impl/ai_core/tbe/op_tiling",
            Path("/usr/local/Ascend/driver/lib64"), Path("/usr/local/Ascend/driver/lib64/common"),
            Path("/usr/local/Ascend/driver/lib64/driver")]))}
        cann_env.update(ASCEND_HOME_PATH=str(cann), ASCEND_TOOLKIT_HOME=str(cann),
                        ASCEND_TOOLKIT_LATEST_HOME=str(cann), ASCEND_OPP_PATH=str(cann / "opp"),
                        ASCEND_AICPU_PATH=str(cann))
        report["device_environment"] = {key: value for key, value in cann_env.items()
                                        if key.startswith("ASCEND_") or key == "LD_LIBRARY_PATH"}
        native = out / "runtime_int64_probe"
        run([*base, "-O2", "-isystem", str(cann / "include"), str(source / "runtime_int64_probe.cc"),
             f"-L{cann / 'lib64'}", f"-Wl,-rpath,{cann / 'lib64'}", "-lopapi", "-lnnopbase",
             "-lascendcl", "-ldl", "-o", str(native)], "adapter-compile.log", run_env=cann_env)
        linked = run(["ldd", str(native)], "adapter-linked-libraries.txt", run_env=cann_env)
        require("not found" not in linked, "unresolved adapter library")
        for line in linked.splitlines():
            for word in line.split():
                if word.startswith("/usr/local/Ascend/"):
                    path = Path(word).resolve()
                    require(path.is_relative_to(cann) or path.is_relative_to("/usr/local/Ascend/driver"),
                            f"mixed CANN library: {path}")
        require("libxgboost" not in linked, "standalone runtime probe must not link XGBoost")
        guard = out / "runtime_no_device_guard.so"
        run([*base, "-O2", "-fPIC", "-shared", "-isystem", str(cann / "include"),
             str(source / "runtime_no_device_guard.cc"), "-o", str(guard)], "guard-compile.log")
        guard_env = {**cann_env, "LD_PRELOAD": str(guard)}
        # Positive control: fully acknowledged device syntax MUST hit our guard
        # at aclInit and exit before the real ACL API executes. This never grants
        # permission to use the real device.
        text = run([str(native), "--device-test", "0", "--confirm-device-idle"],
                   "guard-positive-control.log", run_env=guard_env, expected=86)
        require("FORBIDDEN_ACL_CALL aclInit" in text, "preload guard was not reached")
        text = run([str(native), "--host-api-only"], "adapter-host-tests.log", run_env=guard_env)
        require("FORBIDDEN_ACL_CALL" not in text and "RUNTIME_ADAPTER_HOST_PASS" in text,
                "host mode invoked a forbidden ACL operation")
        fixtures = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
        require(len(fixtures) == 4 and all(test["status"] == "passed" for test in fixtures),
                "unexpected integer host fixtures")
        report["integer_host_fixtures"] = fixtures
        guards = [([], 2), (["--device-test", "0"], 2),
                  (["--device-test", "0", "--wrong-acknowledgement"], 2),
                  (["--device-test", "-1", "--confirm-device-idle"], 1),
                  (["--device-test", "0junk", "--confirm-device-idle"], 1),
                  (["--device-test", "9999999999999999999999", "--confirm-device-idle"], 1)]
        for i, (argv, status) in enumerate(guards):
            text = run([str(native), *argv], f"argument-guard-{i}.log", run_env=guard_env, expected=status)
            require("FORBIDDEN_ACL_CALL" not in text, "invalid arguments reached ACL")
        report["device_opt_in_guards"] = {"argument_cases": len(guards), "status": "passed",
                                           "preload_positive_control": "passed"}
        # This file can be sourced by a separately coordinated device test. It
        # contains toolkit-specific paths only, never LD_PRELOAD or global edits.
        import shlex
        exports = [f"export {key}={shlex.quote(value)}" for key, value in report["device_environment"].items()]
        (out / "device-env.sh").write_text("unset LD_PRELOAD\n" + "\n".join(exports) + "\n")
        paths = [root / "src/common/npu/runtime.h", root / "src/common/npu/acl_api.h",
                 source / "runtime_host_test.cc", source / "runtime_int64_probe.cc",
                 source / "runtime_no_device_guard.cc", Path(__file__), host_binary, native, guard]
        report["artifact_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        report["device_probe_binary"] = str(native)
        report["status"] = "passed"
        print(f"NPU_RUNTIME_HOST_VALIDATION_PASS: {len(tests)} lifecycle checks, sanitizers, "
              "4 integer host fixtures, 6 argument guards; device execution not run")
        return 0
    except Exception as error:
        report["error"] = str(error)
        print(f"NPU_RUNTIME_HOST_VALIDATION_FAIL: {error}", file=sys.stderr)
        return 1
    finally:
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
