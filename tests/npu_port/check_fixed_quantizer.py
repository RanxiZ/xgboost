#!/usr/bin/env python3
"""Validate supplied-scale NPU quantization; device execution is explicit opt-in.

This is not a fitted quantizer or training-backend test. Device bounds reduction
is pending. A test-only binary64 oracle supplies scales and expected integers.
"""
from __future__ import annotations
import argparse
import collections
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def tests_in(text: str) -> list[dict]:
    tests = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    require(len({item["test"] for item in tests}) == len(tests), "duplicate test identifiers")
    require(all(item["status"] == "passed" for item in tests), "failed test record")
    return tests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-root", type=Path, required=True)
    parser.add_argument("--device-test", type=int)
    parser.add_argument("--confirm-device-idle", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    require((args.device_test is None) == (not args.confirm_device_idle), "device test needs both opt-in arguments")
    require(args.device_test in (None, 0), "this checkpoint validates single device ordinal 0")
    require(not args.profile or args.device_test is not None, "profiling requires device opt-in")
    root = Path(__file__).resolve().parents[2]
    require(root == args.expected_source_root.resolve(), "unexpected source checkout")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cann = Path("/usr/local/Ascend/cann-9.1.0").resolve(strict=True)
    report = {"status": "failed", "scope": "supplied_scale_float32_quantization_and_integer_root",
              "source_root": str(root), "cann_root": str(cann), "commands": [],
              "npu_execution": "not_run", "cuda_execution": "not_run",
              "dynamic_device_bounds": "not_implemented", "training_backend": "not_implemented"}
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", LD_LIBRARY_PATH="")

    def run(command: list[str], name: str, *, e: dict | None = None,
            expected: int = 0, timeout: int = 180) -> str:
        report["commands"].append({"argv": command, "log": name, "expected_exit": expected})
        with (out / name).open("w") as log:
            result = subprocess.run(command, cwd=root, env=e or env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout)
        text = (out / name).read_text()
        require(result.returncode == expected, f"exit {result.returncode}, expected {expected}; see {out / name}")
        return text

    def idle(name: str) -> None:
        text = run(["npu-smi", "info"], name, e=native_env)
        require("No running processes found in NPU 0" in text, "NPU occupied; device test deferred")

    def device_tests(text: str) -> list[dict]:
        require("FIXED_QUANTIZER_NPU_PASS checks=27" in text, "missing NPU completion sentinel")
        require("cleanup failure" not in text and "cleanup status=" not in text, "resource cleanup diagnostic")
        tests = tests_in(text)
        require(len(tests) == 27, "unexpected NPU test count")
        for test in tests:
            if "component_control_download_bytes" in test and "reject_" not in test["test"]:
                require(test["component_control_download_bytes"] == 48, "non-summary component transfer")
        for line in text.splitlines():
            if line.startswith("mapping "):
                path = Path(line.split()[-1]).resolve()
                require(path.is_relative_to(cann) or path.is_relative_to("/usr/local/Ascend/driver"),
                        f"mixed device library: {path}")
        return tests

    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root).decode().strip()
        require(branch == "xgb-npu-port", "unexpected branch")
        report["branch"] = branch
        cxx = os.environ.get("CXX", "g++")
        source = root / "tests/npu_port"
        flags = [cxx, "-std=c++17", "-fopenmp", "-Wall", "-Wextra", "-Werror", "-pedantic",
                 "-fno-fast-math", "-ffp-contract=off", f"-I{root / 'include'}", f"-I{root / 'dmlc-core/include'}"]
        host = out / "fixed_quantizer_host_test"
        run([*flags, "-O2", str(source / "fixed_quantizer_host_test.cc"), "-o", str(host)], "host-compile.log")
        linked = run(["ldd", str(host)], "host-libraries.txt")
        require(not any(token in linked.lower() for token in ("ascend", "libopapi", "libcuda", "not found")),
                "host oracle linked a device library")
        text = run([str(host)], "host-tests.log")
        require("FIXED_QUANTIZER_HOST_PASS checks=32" in text, "missing host sentinel")
        report["host_tests"] = tests_in(text)
        require(len(report["host_tests"]) == 32, "unexpected host test count")
        sanitized = out / "fixed_quantizer_host_sanitized"
        run([*flags, "-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
             str(source / "fixed_quantizer_host_test.cc"), "-o", str(sanitized)], "sanitizer-compile.log")
        sanenv = {**env, "ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1", "UBSAN_OPTIONS": "halt_on_error=1"}
        text = run([str(sanitized)], "sanitizer-tests.log", e=sanenv)
        require("FIXED_QUANTIZER_HOST_PASS checks=32" in text, "sanitizer test sentinel missing")
        report["host_sanitizers"] = "passed"
        native_env = {**env, "LD_LIBRARY_PATH": ":".join(str(p) for p in [
            cann / "lib64", cann / "lib64/plugin/opskernel", cann / "lib64/plugin/nnengine",
            cann / "opp/built-in/op_impl/ai_core/tbe/op_tiling", Path("/usr/local/Ascend/driver/lib64"),
            Path("/usr/local/Ascend/driver/lib64/common"), Path("/usr/local/Ascend/driver/lib64/driver")])}
        native_env.update(ASCEND_HOME_PATH=str(cann), ASCEND_TOOLKIT_HOME=str(cann),
                          ASCEND_TOOLKIT_LATEST_HOME=str(cann), ASCEND_OPP_PATH=str(cann / "opp"), ASCEND_AICPU_PATH=str(cann))
        report["device_environment"] = {key: value for key, value in native_env.items()
                                        if key.startswith("ASCEND_") or key == "LD_LIBRARY_PATH"}
        binary = out / "fixed_quantizer_test"
        run([*flags, "-O2", "-isystem", str(cann / "include"), str(source / "fixed_quantizer_test.cc"),
             f"-L{cann / 'lib64'}", f"-Wl,-rpath,{cann / 'lib64'}", "-lopapi", "-lnnopbase",
             "-lascendcl", "-ldl", "-o", str(binary)], "device-compile.log", e=native_env)
        linked = run(["ldd", str(binary)], "device-libraries.txt", e=native_env)
        require("not found" not in linked and "libxgboost" not in linked, "invalid standalone component linkage")
        for line in linked.splitlines():
            for word in line.split():
                if word.startswith("/usr/local/Ascend/"):
                    p = Path(word).resolve()
                    require(p.is_relative_to(cann) or p.is_relative_to("/usr/local/Ascend/driver"), "mixed CANN linkage")
        guard = out / "no_device_guard.so"
        run([*flags, "-O2", "-fPIC", "-shared", "-isystem", str(cann / "include"),
             str(source / "runtime_no_device_guard.cc"), "-o", str(guard)], "guard-compile.log")
        guarded = {**native_env, "LD_PRELOAD": str(guard)}
        text = run([str(binary), "--host-api-only"], "host-api-guard.log", e=guarded)
        require("FIXED_QUANTIZER_API_HOST_PASS checks=2" in text and "FORBIDDEN_ACL_CALL" not in text,
                "host API touched ACL or exception-path tests did not complete")
        report["host_executor_tests"] = tests_in(text)
        require(len(report["host_executor_tests"]) == 2, "unexpected host executor test count")
        text = run([str(binary), "--device-test", "0", "--confirm-device-idle"], "guard-positive.log", e=guarded, expected=86)
        require("FORBIDDEN_ACL_CALL aclInit" in text, "guard positive control failed")
        for i, argv in enumerate([[], ["--device-test", "0"], ["--device-test", "0", "--wrong-ack"]]):
            text = run([str(binary), *argv], f"guard-argument-{i}.log", e=guarded, expected=2)
            require("FORBIDDEN_ACL_CALL" not in text, "invalid arguments touched ACL")
        report["device_call_guards"] = "passed"
        if args.device_test is not None:
            with (out.parent / "npu0-test.lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                idle("device-before.txt"); time.sleep(1); idle("device-recheck.txt")
                command = [str(binary), "--device-test", "0", "--confirm-device-idle"]
                text = run(command, "device-tests.log", e=native_env)
                report["device_tests"] = device_tests(text)
                idle("device-after.txt")
                report["npu_execution"] = "passed"
                if args.profile:
                    profile = out / "profile"
                    require(not profile.exists(), "profiling output already exists; use a new output directory")
                    idle("profile-before.txt")
                    text = run([str(cann / "tools/profiler/bin/msprof"), f"--output={profile}", "--runtime-api=on",
                                "--task-time=on", "--ai-core=on", "--aicpu=on", "--sys-lp=off", *command],
                               "profile-run.log", e=native_env)
                    require(device_tests(text) == report["device_tests"], "profile replay differs from normal run")
                    idle("profile-after.txt")
                    summaries = list(profile.rglob("op_summary_*.csv"))
                    require(len(summaries) == 1, "profiler operator summary missing or ambiguous")
                    with summaries[0].open() as stream:
                        rows = list(csv.DictReader(stream))
                    require(rows and all(row["Device_id"] == "0" for row in rows), "no device task evidence")
                    require(all(row["Task Type"] in ("MIX_AIV", "AI_VECTOR_CORE") for row in rows), "unexpected non-vector task")
                    names = {row["OP Type"] for row in rows}
                    require(any("Shift" in name for name in names) and any("Sum" in name for name in names),
                            "missing shift/root execution evidence")
                    report["execution_evidence"] = {"operator_summary": str(summaries[0]), "task_count": len(rows),
                        "op_counts": dict(collections.Counter(row["OP Type"] for row in rows)),
                        "task_types": dict(collections.Counter(row["Task Type"] for row in rows)),
                        "performance_acceptance": "not_run; diagnostic component tests only"}
        files = [root / "src/common/npu" / name for name in
                 ("runtime.h", "acl_api.h", "acl_ops.h", "quantizer_scale.h", "fixed_quantizer.h")]
        files += [source / name for name in ("quantizer_cases.h", "fixed_quantizer_host_test.cc", "fixed_quantizer_test.cc")]
        files += [Path(__file__), binary, host]
        report["artifact_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        report["status"] = "passed"
        print(f"FIXED_QUANTIZER_VALIDATION_PASS host_checks=32 npu_execution={report['npu_execution']} "
              "dynamic_device_bounds=not_implemented")
        return 0
    except Exception as error:
        report["error"] = str(error)
        print(f"FIXED_QUANTIZER_VALIDATION_FAIL: {error}", file=sys.stderr)
        return 1
    finally:
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
