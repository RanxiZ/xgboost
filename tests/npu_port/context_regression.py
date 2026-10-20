#!/usr/bin/env python3
"""Host-only context regression checks against this checkout's rebuilt CPU library.

The Emscripten cases are native-host macro simulations of the parser branch,
not WebAssembly toolchain tests. The NPU macro tests verify host-only ordinal
parsing and fail-closed generic dispatch, not device availability or execution.
No device runtime is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def require(valid, message):
    if not valid:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-library-root", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status="failed", native_backend="not_assessed_by_host_parser", npu_execution="not_run",
                  source_root=str(root), command=sys.argv, host_cases=[], python_cases=[])
    try:
        import numpy as np
        import xgboost as xgb

        library = Path(xgb.core._LIB._name).resolve()
        require(library.is_relative_to(args.expected_library_root.resolve()), str(library))
        require(library.is_relative_to(root / "lib"), "library is not from this source checkout")
        require(xgb.build_info()["USE_CUDA"] is False, "expected a CPU-only library")
        require(library.stat().st_mtime >= (root / "src/context.cc").stat().st_mtime,
                "library predates context source; rebuild build-cpu first")
        require(library.stat().st_mtime >= (root / "include/xgboost/context.h").stat().st_mtime,
                "library predates context header; rebuild build-cpu first")
        report.update(library_path=str(library), build_info=xgb.build_info(),
                      library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
        env = os.environ.copy()
        env.pop("LD_PRELOAD", None)
        env["LD_LIBRARY_PATH"] = str(root / "lib")
        env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        compiler = os.environ.get("CXX", "g++")
        common = [compiler, "-std=c++17", "-O1", "-fopenmp", "-DDMLC_LOG_CUSTOMIZE=1",
                  f"-I{root / 'include'}", f"-I{root / 'dmlc-core/include'}",
                  f"-I{root / 'build-cpu/dmlc-core/include'}"]
        test = root / "tests/npu_port/context_host_test.cc"
        variants = [("rebuilt-library", [], False),
                    ("emscripten-host-parser", ["-D__EMSCRIPTEN__"], True),
                    ("npu-guard-scaffold", ["-DXGBOOST_USE_NPU"], True),
                    ("emscripten-npu-guard-scaffold",
                     ["-D__EMSCRIPTEN__", "-DXGBOOST_USE_NPU"], True)]
        for name, defines, source_override in variants:
            binary = output / name
            sources = [str(test)]
            if source_override:
                sources.append(str(root / "src/context.cc"))
            command = common + defines + sources + [f"-L{root / 'lib'}",
                      f"-Wl,-rpath,{root / 'lib'}", "-lxgboost", "-o", str(binary)]
            with (output / f"{name}-compile.log").open("w") as log:
                subprocess.run(command, cwd=root, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            linked = subprocess.check_output(["ldd", str(binary)], env=env).decode()
            (output / f"{name}-linked-libraries.txt").write_text(linked)
            require("not found" not in linked, linked)
            require(not any(item in linked.lower() for item in
                            ("libcuda", "libcudart", "libascend", "libopapi", "/ascend/")), linked)
            with (output / f"{name}-tests.log").open("w") as log:
                subprocess.run([str(binary)], cwd=root, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            result = (output / f"{name}-tests.log").read_text()
            require("CONTEXT_HOST_PASS" in result, result)
            report["host_cases"].append(dict(name=name, status="passed", compile_command=command,
                                             source_override=source_override, result=result.strip()))

        features = np.arange(128, dtype=np.float32).reshape(32, 4) / 16
        labels = (features[:, 0] > 4).astype(np.float32)
        data = xgb.DMatrix(features, label=labels, nthread=2)
        base = dict(tree_method="hist", objective="reg:squarederror", nthread=2,
                    max_depth=2, seed=20261017)
        default = xgb.train(base, data, num_boost_round=3)
        cpu = xgb.train(base | dict(device="cpu"), data, num_boost_round=3)
        np.testing.assert_array_equal(default.predict(data), cpu.predict(data))
        require(cpu.num_boosted_rounds() == 3 and cpu.get_score(), "CPU trees did not split")
        require(json.loads(cpu.save_config())["learner"]["generic_param"]["device"] == "cpu",
                "CPU context changed")
        restored = xgb.Booster(model_file=cpu.save_raw())
        np.testing.assert_array_equal(cpu.predict(data), restored.predict(data))
        report["python_cases"].append(dict(name="cpu_training_default_explicit_roundtrip", status="passed"))

        def reject(call, name, expected):
            try:
                call()
            except xgb.core.XGBoostError as error:
                require(expected in str(error), str(error))
                report["python_cases"].append(dict(name=name, status="passed"))
            else:
                raise AssertionError(f"unexpectedly accepted {name}")

        for device in ("npu", "npu:0", "npu:7", "npu:32767"):
            booster = xgb.Booster(params=base | dict(device=device), cache=[data])
            reject(booster.save_config, f"reject_before_training_{device}",
                   "not compiled with NPU support (XGBOOST_USE_NPU)")
        for extra in ({}, {"booster": "dart"}, {"booster": "gblinear"},
                      {"updater": "grow_quantile_histmaker"}, {"tree_method": "exact"}):
            reject(lambda extra=extra: xgb.train(base | extra | dict(device="npu:0"), data, 1),
                   "reject_training_" + json.dumps(extra, sort_keys=True),
                   "CPU fallback is disabled")
        for device in ("npu:", "npu:-1", "npu:+0", "npu:0:1", "NPU:0", "npu:2147483648"):
            booster = xgb.Booster(params=base | dict(device=device), cache=[data])
            reject(booster.save_config, f"reject_malformed_{device}", "Invalid argument for `device`")
        config = json.loads(cpu.save_config())
        config["learner"]["generic_param"]["device"] = "npu:0"
        loaded = xgb.Booster(model_file=cpu.save_raw())
        reject(lambda: (loaded.load_config(json.dumps(config)), loaded.save_config()),
               "reject_npu_loaded_config", "not compiled with NPU support")
        cpu.set_param({"device": "npu:0"})
        reject(cpu.save_config, "reject_cpu_model_device_switch", "CPU fallback is disabled")
        cpu.set_param({"device": "cpu"})
        np.testing.assert_array_equal(default.predict(data), cpu.predict(data))
        report["python_cases"].append(dict(name="cpu_recovery_after_rejected_switch", status="passed"))
        report["status"] = "passed"
        print(f"CONTEXT_REGRESSION_PASS: {len(report['host_cases'])} host variants, "
              f"{len(report['python_cases'])} rebuilt-library Python cases; no devices used")
        return 0
    except Exception:
        report["failure"] = traceback.format_exc()
        print(report["failure"], file=sys.stderr)
        return 1
    finally:
        (output / "context-summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())