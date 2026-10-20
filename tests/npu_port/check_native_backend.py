#!/usr/bin/env python3
"""Opt-in native XGBoost/NPU end-to-end acceptance harness.

The default invocation is host-only and performs no device call.  A device run
requires both ``--run-npu`` and ``--confirm-device-idle`` plus an expected
source root and a fresh output directory.  This checker never treats a device
parameter as evidence: the child must load ``lib-npu/libxgboost.so``, report a
native updater, execute the real learner cases, and (for acceptance) produce
an operator profile with native kernel and ACLNN task names.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CASES = Path(__file__).with_name("native_backend_cases.py")
SCOPE = "native_xgboost_end_to_end_v1"
CANN = Path("/usr/local/Ascend/cann-9.1.0")
DRIVER = Path("/usr/local/Ascend/driver")
NPU_SMI = Path("/usr/local/bin/npu-smi")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_expected_identity():
    """Import Qlib's identity helper without running its CLI or acceptance flow."""
    candidates = [
        Path("/workspace/qlib-xgb-npu/examples/benchmarks/XGBoost/npu_port/run_final_acceptance.py"),
        ROOT.parent / "qlib-xgb-npu/examples/benchmarks/XGBoost/npu_port/run_final_acceptance.py",
    ]
    source = next((p for p in candidates if p.is_file()), None)
    require(source is not None, "final Qlib acceptance helper is not available")
    spec = importlib.util.spec_from_file_location("final_qlib_acceptance_identity", source)
    require(spec is not None and spec.loader is not None, "cannot import final acceptance helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "expected_backend_identity", None)
    require(callable(function), "final helper has no expected_backend_identity function")
    return function, source


def validate_root_and_identity(expected_source_root: Path) -> tuple[dict[str, Any], Path]:
    root = ROOT.resolve()
    require(root == expected_source_root.resolve(), f"unexpected source root: {root}")
    require(CANN.is_dir() and CANN.resolve() == CANN, "CANN 9.1.0 is absent or redirected")
    identity_function, helper = import_expected_identity()
    identity = identity_function()
    library = Path(identity["library_path"]).resolve(strict=True)
    kernel = Path(identity["kernel_library_path"]).resolve(strict=True)
    private = ROOT / "lib-npu"
    require(library.is_relative_to(private.resolve()), "expected library escapes private lib-npu")
    require(kernel.is_relative_to(private.resolve()), "expected kernel library escapes private lib-npu")
    require(identity["cann_root"] == str(CANN), "identity is not pinned to CANN 9.1.0")
    return identity, helper


def idle_check(output: Path, label: str, env: dict[str, str]) -> str:
    """Check NPU 0 with the same isolated environment used by the child."""
    path = output / f"{label}.log"
    require(NPU_SMI.is_file(), f"npu-smi unavailable: {NPU_SMI}")
    result = subprocess.run([str(NPU_SMI), "info"], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=30)
    path.write_text(result.stdout)
    require(result.returncode == 0, f"npu-smi failed for {label}: {result.returncode}")
    require(re.search(r"\|\s*0\s*\|", result.stdout) is not None,
            f"NPU 0 is not visible in {label} device table")
    require("No running processes found in NPU 0" in result.stdout,
            f"NPU 0 is occupied during {label}; refusing execution")
    return str(path)


def private_env(output: Path, identity: dict[str, Any]) -> dict[str, str]:
    bootstrap = output / "python-bootstrap"
    bootstrap.mkdir(mode=0o700, exist_ok=True)
    # Pin the source library before xgboost is imported.  The helper is kept in
    # the fresh run directory so the child cannot use a user's sitecustomize.
    (bootstrap / "sitecustomize.py").write_text(
        "import importlib.util, json, os, pathlib, sys\n"
        "try:\n"
        " expected=json.loads(pathlib.Path(os.environ['XGB_NPU_EXPECTED']).read_text())\n"
        " p=pathlib.Path(expected['package_root'])/'libpath.py'\n"
        " s=importlib.util.spec_from_file_location('xgboost.libpath',p)\n"
        " m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
        " m.find_lib_path=lambda:[expected['library_path']]\n"
        " sys.modules['xgboost.libpath']=m\n"
        "except BaseException:\n"
        " import traceback; traceback.print_exc(); os._exit(121)\n"
    )
    env = {key: os.environ[key] for key in ("HOME", "LANG", "LC_ALL", "TZ") if key in os.environ}
    env.update({
        "PATH": f"/workspace/xgb-npu-venv/bin:{CANN}/bin:{CANN}/tools/profiler/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": f"{bootstrap}:{ROOT / 'python-package'}",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "XGB_NPU_EXPECTED": str(output / "backend_expected.json"),
        "ASCEND_HOME_PATH": str(CANN), "ASCEND_TOOLKIT_HOME": str(CANN),
        "ASCEND_TOOLKIT_LATEST_HOME": str(CANN), "ASCEND_OPP_PATH": str(CANN / "opp"),
        "ASCEND_AICPU_PATH": str(CANN), "TMPDIR": str(output / "tmp"),
        "XDG_CACHE_HOME": str(output / "cache"), "VIRTUAL_ENV": "/workspace/xgb-npu-venv",
        "LD_LIBRARY_PATH": ":".join(str(p) for p in (
            ROOT / "lib-npu", CANN / "lib64", CANN / "lib64/plugin/opskernel",
            CANN / "lib64/plugin/nnengine", CANN / "opp/built-in/op_impl/ai_core/tbe/op_tiling",
            DRIVER / "lib64", DRIVER / "lib64/common", DRIVER / "lib64/driver")),
    })
    for name in ("tmp", "cache"):
        (output / name).mkdir(mode=0o700, exist_ok=True)
    return env


def run_logged(command: list[str], cwd: Path, env: dict[str, str], log: Path,
               timeout: int = 300) -> subprocess.CompletedProcess[str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, text=True, timeout=timeout)
    require(result.returncode == 0, f"command failed ({result.returncode}); see {log}")
    return result


def parse_case_summary(path: Path) -> dict[str, Any]:
    summary = path / "summary.json"
    require(summary.is_file(), "native case summary is missing")
    report = json.loads(summary.read_text())
    require(report.get("status") == "passed" and report.get("device") == "npu:0",
            "native case execution did not pass on npu:0")
    require(report.get("check_count", 0) >= 10, "native case coverage is unexpectedly small")
    return report


def production_kernel_sources() -> tuple[set[str], list[Path]]:
    """Read kernel definitions from the production translation unit's ASC includes."""
    directory = (ROOT / "src/common/npu").resolve()
    pending = [directory / "kernel/production_kernels.asc"]
    sources: set[Path] = set()
    kernels: set[str] = set()
    while pending:
        source = pending.pop().resolve()
        require(source.is_relative_to(directory), f"production ASC include escapes source tree: {source}")
        require(source.is_file(), f"production ASC source missing: {source}")
        if source in sources:
            continue
        sources.add(source)
        # Comments must not supply either includes or kernel definitions.
        text = re.sub(r"/\*.*?\*/|//[^\n]*", "", source.read_text(), flags=re.DOTALL)
        kernels.update(re.findall(
            r"\b__global__\b[^;{}]*?\bvoid\s+(\w+)\s*\([^;{}]*\)\s*\{", text))
        for include in re.findall(r'^\s*#\s*include\s*"([^"\n]+\.asc)"', text, re.MULTILINE):
            pending.append(source.parent / include)
    return kernels, sorted(sources)


def profile_evidence(profile: Path) -> dict[str, Any]:
    """Require positive-duration device compute records for every essential path."""
    tables = sorted(profile.rglob("op_summary_*.csv"))
    require(tables, "msprof produced no operator summary")
    rows: list[dict[str, str]] = []
    task_types: set[str] = set()
    compute_types = {"AI_CORE", "AI_VECTOR_CORE", "AICORE", "AIC", "AIV", "MIX_AIC", "MIX_AIV"}
    for table in tables:
        with table.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            headers = [name.strip() for name in (reader.fieldnames or [])]
            fields = set(headers)
            require(len(headers) == len(fields) and "Device_id" in fields
                    and bool(fields & {"Op Name", "OP Type"})
                    and bool(fields & {"Task Type", "Core Type"})
                    and bool(fields & {"Task Duration(us)", "Total Time(us)"}),
                    f"unknown profiler columns in {table}")
            reader.fieldnames = headers
            table_rows = 0
            for row in reader:
                location = f"{table}:{reader.line_num}"
                require(None not in row and all(value is not None for value in row.values()),
                        f"malformed profiler row at {location}")
                row = {key: value.strip() for key, value in row.items()}
                require(row["Device_id"] == "0", f"profile task is not on NPU 0 at {location}")
                types = {row.get(key, "").upper() for key in ("Task Type", "Core Type")} - {""}
                require(all("AICPU" not in re.sub(r"[^A-Z0-9]", "", task) for task in types),
                        f"AI_CPU task cannot establish native acceptance at {location}")
                require(types and types <= compute_types, f"unknown/non-compute profiler task at {location}: {types}")
                duration = row.get("Task Duration(us)") or row.get("Total Time(us)")
                try:
                    elapsed = float(duration) if duration else float("nan")
                except ValueError:
                    elapsed = float("nan")
                require(math.isfinite(elapsed) and elapsed > 0, f"invalid profiler duration at {location}")
                require(row.get("Op Name") or row.get("OP Type"), f"unnamed profiler task at {location}")
                rows.append(row)
                task_types.update(types)
                table_rows += 1
            require(table_rows, f"operator summary is empty: {table}")
    source_kernels, sources = production_kernel_sources()
    required = {
        "gradient": ("NativeSquareErrorKernel",),
        "bin": ("NativeBinRowsKernel",),
        "sample": ("NativeSampleRowsKernel",),
        "predict": ("NativePredictKernel",),
        "histogram": ("IntegerHistogramBuildKernel",),
        # Production evaluates every sampled feature in four node-batch stages.
        # Standalone component kernels cannot substitute for this chain.
        "scoring": ("BatchedSplitPrefixKernel", "BatchedSplitScoreKernel",
                    "BatchedSplitSelectKernel", "BatchedSplitFinishKernel"),
    }
    # Keep names separate: substrings and text in unrelated CSV columns cannot
    # supply evidence. ACLNN API names usually occur in Op Name, not OP Type.
    names = {row.get(key, "").lower() for row in rows for key in ("Op Name", "OP Type")} - {""}
    found = {}
    for category, tokens in required.items():
        require(all(token in source_kernels for token in tokens),
                f"required {category} kernel is absent from production ASC sources")
        found[category] = [token for token in tokens if token.lower() in names]
        missing = sorted(set(tokens) - set(found[category]))
        require(not missing, f"profile lacks native {category} kernels: {', '.join(missing)}")
    # Match the ACLNN API component, not a generic Gather/Reduce OP Type or a
    # coincidental substring. CANN exports IndexSelect and Index_Select forms.
    aclnn = {}
    for name in names:
        match = re.match(r"^aclnn_?(scatter_?add|reduce_?sum|cum_?sum|index_?select|s_?where)(?:_|$)", name)
        if match:
            aclnn[name] = match.group(1).replace("_", "")
    hist = sorted(name for name, api in aclnn.items() if api in {"scatteradd", "reducesum"})
    partition = sorted(name for name, api in aclnn.items() if api in {"cumsum", "indexselect", "swhere"})
    required_apis = {"histogram": {"scatteradd", "reducesum"},
                     "partition": {"cumsum", "indexselect", "swhere"}}
    for category, apis in required_apis.items():
        missing = sorted(apis - set(aclnn.values()))
        require(not missing, f"profile lacks ACLNN {category} operators: {', '.join(missing)}")
    return {"status": "passed", "operator_tables": [str(p) for p in tables],
            "operator_table_sha256": {str(p): sha256(p) for p in tables},
            "task_count": len(rows), "native_kernels": found, "source_kernels": sorted(source_kernels),
            "production_asc_sha256": {str(p): sha256(p) for p in sources},
            "aclnn_hist_partition": sorted(aclnn),
            "aclnn_histogram": hist, "aclnn_partition": partition,
            "task_types": sorted(task_types)}


def artifact_hashes(output: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"summary.json"}:
            result[str(path.relative_to(output))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return result


def host_preflight(output: Path) -> dict[str, Any]:
    """Import-only smoke test proving the default path has no device execution."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONNOUSERSITE": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "python-package")}
    log = output / "host-preflight.log"
    python = Path("/workspace/xgb-npu-venv/bin/python")
    require(python.is_file(), f"isolated Python is missing: {python}")
    run_logged([str(python), str(CASES), "--help"], ROOT, env, log)
    text = log.read_text()
    require("--library" in text and "--output-dir" in text, "case module help did not run")
    return {"status": "passed", "device_execution": "not_run", "log": str(log)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-npu", action="store_true", help="opt in to NPU execution")
    parser.add_argument("--confirm-device-idle", action="store_true", help="confirm exclusive NPU 0 use")
    parser.add_argument("--profile", action="store_true", help="run msprof and require parsed compute evidence")
    parser.add_argument("--expected-source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(args.confirm_device_idle == args.run_npu,
            "--run-npu and --confirm-device-idle must be supplied together")
    require(not args.profile or args.run_npu, "--profile requires --run-npu")
    output = args.output_dir.absolute()
    require(output == output.resolve(), "output directory must be canonical")
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "output directory must be fresh")
    identity, helper = validate_root_and_identity(args.expected_source_root)
    (output / "backend_expected.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    report: dict[str, Any] = {"status": "unaccepted", "scope": SCOPE,
        "source_root": str(ROOT), "identity": identity, "identity_helper": str(helper),
        "commands": [], "limitations": ["training: 32 rows x 4 features; unseen prediction up to 4097 rows", "NPU ordinal 0 only",
        "cross-backend training trees intentionally not compared", "large training matrices are future coverage"]}
    host = host_preflight(output); report["host_preflight"] = host
    if not args.run_npu:
        report["reason"] = "device execution requires explicit --run-npu --confirm-device-idle; no backend acceptance"
        (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("NATIVE_BACKEND_PREFLIGHT_PASS device_execution=not_run acceptance=unaccepted")
        return 0

    env = private_env(output, identity)
    case_output = output / "case-output"
    lock_path = output.parent / "npu0-native-backend.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        before = idle_check(output, "device-before", env)
        command = [str(Path(env["PATH"].split(":")[0]) / "python"), str(CASES),
                   "--library", identity["library_path"], "--output-dir", str(case_output)]
        case_log = output / "native-case.log"
        report["commands"].append({"argv": command, "log": str(case_log)})
        run_logged(command, ROOT, env, case_log)
        case_report = parse_case_summary(case_output)
        after = idle_check(output, "device-after", env)
        report["checks"] = {
            "native_training": {"status": "passed", "device": "npu:0",
                                 "evidence": [{"path": str(case_output / "case_data.json"),
                                               "sha256": sha256(case_output / "case_data.json")}]},
            "native_predict": {"status": "passed", "device": "npu:0",
                                "evidence": [{"path": str(case_output / "prediction_native.npy"),
                                              "sha256": sha256(case_output / "prediction_native.npy")}]},
            "native_save_load": {"status": "passed", "device": "npu:0",
                                  "evidence": [{"path": str(case_output / "model.ubj"),
                                                "sha256": sha256(case_output / "model.ubj")},
                                               {"path": str(case_output / "model.json"),
                                                "sha256": sha256(case_output / "model.json")},
                                               {"path": str(case_output / "model.raw"),
                                                "sha256": sha256(case_output / "model.raw")}]},
        }
        report["case_report"] = case_report
        report["idle_checks"] = [before, after]
        if args.profile:
            profile = output / "profile"
            profiler = Path(CANN / "tools/profiler/bin/msprof")
            require(profiler.is_file(), f"msprof unavailable: {profiler}")
            report["idle_checks"].append(idle_check(output, "profile-before", env))
            profile_case_output = output / "profile-case-output"
            require(not profile_case_output.exists(), "profile case output must be fresh")
            # Rerunning the cases must never overwrite the unprofiled evidence.
            profile_cases = [*command[:-1], str(profile_case_output)]
            profile_command = [str(profiler), f"--output={profile}", "--runtime-api=on",
                               "--task-time=on", "--ai-core=on", "--aicpu=on", "--sys-lp=off", *profile_cases]
            profile_log = output / "profile.log"
            report["commands"].append({"argv": profile_command, "log": str(profile_log)})
            run_logged(profile_command, ROOT, env, profile_log, timeout=600)
            report["profile_case_report"] = parse_case_summary(profile_case_output)
            report["profile_case_output"] = str(profile_case_output)
            report["profile"] = profile_evidence(profile)
            report["idle_checks"].append(idle_check(output, "profile-after", env))
            report["status"] = "passed"
        else:
            report["reason"] = "native execution passed, but acceptance remains unaccepted until --profile is parsed"
    report["artifacts"] = artifact_hashes(output)
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "passed":
        print("NATIVE_BACKEND_EXECUTION_PASS acceptance=unaccepted profiler_required")
        return 2
    print("NATIVE_BACKEND_ACCEPTANCE_PASS scope=" + SCOPE)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Keep a machine-readable failure record even when preflight, identity,
        # idle, execution, or profile validation rejects the run.
        try:
            argv = sys.argv
            if "--output-dir" in argv:
                output = Path(argv[argv.index("--output-dir") + 1]).absolute()
                output.mkdir(parents=True, exist_ok=True)
                (output / "summary.json").write_text(json.dumps({
                    "status": "failed", "scope": SCOPE, "error": str(error),
                    "device_execution": "not_run" if not "--run-npu" in argv else "failed",
                }, indent=2, sort_keys=True) + "\n")
        except Exception:
            pass
        print(f"NATIVE_BACKEND_ACCEPTANCE_FAIL: {error}", file=sys.stderr)
        raise