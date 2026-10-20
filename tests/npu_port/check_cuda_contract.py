#!/usr/bin/env python3
"""Validate CUDA-source contracts on the host. Never imports a device runtime.

The official reference layer reads only immutable objects from the pinned git
commit. The worktree layer is separate and remains strict by default.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OFFICIAL_REFERENCE_SHA256 = "40e09b4b365b7e7f1eff34ed8b9110d7f2c411506b161c1c904d2c3d468396f2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def referenced_sources(contract: dict) -> dict[str, str | None]:
    sources: dict[str, str | None] = {}
    for component in contract["components"]:
        sources[component["source"]] = component["anchor"]
        for path in component.get("additional_sources", []):
            sources.setdefault(path, None)
    return sources


def inspect_official_reference(contract: dict, root: Path, git_runner=git,
                               expected_hash: str = OFFICIAL_REFERENCE_SHA256) -> dict:
    """Validate tag, commit objects, official anchors and aggregate content identity."""
    reference = contract["reference"]
    commit = git_runner(root, "rev-parse", reference["tag"] + "^{commit}").decode().strip()
    require(commit == reference["commit"], "reference tag does not resolve to pinned commit")
    file_records = {}
    digest = hashlib.sha256()
    for path, anchor in sorted(referenced_sources(contract).items()):
        try:
            data = git_runner(root, "show", f"{commit}:{path}")
        except (subprocess.CalledProcessError, OSError) as error:
            raise RuntimeError(f"official reference object unavailable: {path}") from error
        require(anchor is None or anchor.encode() in data,
                f"official reference anchor missing: {path}")
        file_hash = hashlib.sha256(data).hexdigest()
        digest.update(path.encode() + b"\0" + bytes.fromhex(file_hash))
        file_records[path] = {"sha256": file_hash, "anchor": "passed" if anchor else "not_required"}
    actual_hash = digest.hexdigest()
    require(actual_hash == expected_hash,
            f"official reference manifest hash mismatch: expected {expected_hash}, got {actual_hash}")
    return {"commit": commit, "reference_hash_sha256": actual_hash,
            "reference_file_count": len(file_records), "reference_files": file_records,
            "status": "official_reference_verified"}


def inspect_worktree(contract: dict, root: Path, official: dict) -> dict:
    statuses = {}
    changed = []
    for path in referenced_sources(contract):
        data = (root / path).read_bytes()
        worktree_hash = hashlib.sha256(data).hexdigest()
        official_hash = official["reference_files"][path]["sha256"]
        status = "matches_official_reference" if worktree_hash == official_hash else "changed_unaccepted"
        statuses[path] = {"status": status, "sha256": worktree_hash,
                          "official_reference_sha256": official_hash}
        if status != "matches_official_reference":
            changed.append(path)
    return {"status": "accepted" if not changed else "changed_unaccepted",
            "worktree_file_count": len(statuses), "worktree_files": statuses,
            "changed_unaccepted_files": changed}


def check_fixture(fixture: dict) -> dict:
    scale = int(fixture["fixed_scale_each_channel"])
    bins = fixture["bins"]
    gradients = fixture["gradients"]
    hessians = fixture["hessians"]
    require(len(bins) == len(gradients) == len(hessians), "fixture row count mismatch")
    pairs = [[int(g * scale), int(h * scale)] for g, h in zip(gradients, hessians)]
    root = [sum(pair[i] for pair in pairs) for i in (0, 1)]
    hist = [[0, 0] for _ in range(fixture["feature_bins"])]
    for index, pair in zip(bins, pairs):
        if index == fixture["missing_sentinel"]:
            continue
        require(0 <= index < len(hist), "fixture bin out of range")
        for channel in (0, 1):
            hist[index][channel] += pair[channel]
    prefixes = []
    total = [0, 0]
    for pair in hist:
        prefixes.append(total.copy())
        total = [total[i] + pair[i] for i in (0, 1)]
    missing = [root[i] - total[i] for i in (0, 1)]
    expected_root = list(map(int, fixture["expected_root"]))
    expected_hist = [list(map(int, pair)) for pair in fixture["expected_histogram"]]
    expected_prefix = [list(map(int, pair)) for pair in fixture["expected_exclusive_prefix"]]
    require(root == expected_root, "fixture root mismatch")
    require(hist == expected_hist, "fixture histogram mismatch")
    require(prefixes == expected_prefix, "fixture exclusive scan mismatch")
    require(missing == list(map(int, fixture["expected_missing"])), "fixture missing mismatch")
    for pair in pairs + hist + prefixes + [root, missing]:
        require(all(-(1 << 63) <= v < (1 << 63) for v in pair), "fixture int64 overflow")
    return {"name": fixture["name"], "status": "passed", "comparison": "exact integers"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-root", type=Path, required=True)
    parser.add_argument("--reference-only", action="store_true",
                        help="validate pinned git objects and fixture only; skip compilation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    require(root == args.expected_source_root.resolve(), "unexpected XGBoost source checkout")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_file = root / "tests/npu_port/cuda_contract.json"
    report = {"status": "failed", "validation_kind": "host_only_cuda_source_contract",
              "cuda_execution": "not_run", "npu_execution": "not_run",
              "native_backend": "not_assessed_by_host_contract", "source_root": str(root),
              "command": sys.argv, "integration_source_identity": "unaccepted_until_reviewed"}
    try:
        contract = json.loads(contract_file.read_text())
        official = inspect_official_reference(contract, root)
        report["official_reference"] = official
        report["reference_commit"] = official["commit"]
        report["fixture"] = check_fixture(contract["fixed_fixture"])
        if args.reference_only:
            report["worktree_identity"] = inspect_worktree(contract, root, official)
            report["status"] = "reference_verified_worktree_unaccepted"
            report["mode"] = "reference_only_no_compile_no_device"
            print(f"CUDA_SOURCE_REFERENCE_PASS: {official['reference_file_count']} official objects; "
                  f"hash {official['reference_hash_sha256']}; worktree identity unaccepted")
            return 0

        branch = git(root, "branch", "--show-current").decode().strip()
        require(branch == "xgb-npu-port", "unexpected integration branch")
        worktree = inspect_worktree(contract, root, official)
        report["worktree_identity"] = worktree
        require(worktree["status"] == "accepted",
                "source changed; contract review required: " + ", ".join(worktree["changed_unaccepted_files"]))
        guard_tests = []
        for field in ("expected_root", "expected_histogram", "expected_missing", "expected_exclusive_prefix"):
            damaged = copy.deepcopy(contract["fixed_fixture"])
            if isinstance(damaged[field][0], list):
                damaged[field][0][0] = str(int(damaged[field][0][0]) + 1)
            else:
                damaged[field][0] = str(int(damaged[field][0]) + 1)
            try:
                check_fixture(damaged)
            except RuntimeError:
                guard_tests.append({"field": field, "corruption_rejected": True})
            else:
                raise RuntimeError(f"fixture checker accepted corrupt {field}")
        report["fixture_guard_tests"] = guard_tests
        env = os.environ.copy()
        env.pop("LD_PRELOAD", None)
        env["LD_LIBRARY_PATH"] = str(root / "lib")
        env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        compiler = os.environ.get("CXX", "g++")
        binary = output / "cuda_semantics_test"
        command = [compiler, "-std=c++17", "-O2", "-fopenmp", "-Wall", "-Wextra", "-Werror",
                   "-fno-fast-math", "-ffp-contract=off", f"-I{root / 'include'}",
                   f"-I{root / 'dmlc-core/include'}", str(root / "tests/npu_port/cuda_semantics_test.cc"),
                   f"-L{root / 'lib'}", f"-Wl,-rpath,{root / 'lib'}", "-lxgboost", "-ldl",
                   "-o", str(binary)]
        report["build_command"] = command
        report["compiler"] = subprocess.check_output([compiler, "--version"], env=env).decode()
        with (output / "compile.log").open("w") as log:
            subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        libraries = subprocess.check_output(["ldd", str(binary)], env=env).decode()
        (output / "linked-libraries.txt").write_text(libraries)
        require("not found" not in libraries, "unresolved host test library")
        require(not any(token in libraries.lower() for token in
                        ("libcuda", "libcudart", "/ascend/", "libascend", "libopapi")),
                "host contract binary must not link device runtime libraries")
        report["device_runtime_libraries_linked"] = False
        with (output / "host-tests.log").open("w") as log:
            subprocess.run([str(binary)], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        lines = (output / "host-tests.log").read_text().splitlines()
        require(lines and lines[-1].startswith("CUDA_SOURCE_CONTRACT_PASS"), "missing test sentinel")
        tests = [json.loads(line) for line in lines if line.startswith("{")]
        require(len(tests) == 18, "unexpected host contract test count")
        require(len({test["test"] for test in tests}) == len(tests), "duplicate test identifiers")
        require(all(test["status"] == "passed" for test in tests), "host contract test failed")
        report["tests"] = tests
        report["host_test_count"] = len(tests)
        report["artifact_sha256"] = {
            str(path.relative_to(root)) if path.is_relative_to(root) else str(path):
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (contract_file, Path(__file__), root / "tests/npu_port/cuda_semantics_test.cc", binary)
        }
        report["status"] = "passed"
        print("CUDA_SOURCE_CONTRACT_PASS: 18 host checks, exact fixture, official and worktree identity; no devices used")
        return 0
    except Exception as error:
        report["error"] = str(error)
        print(f"CUDA_SOURCE_CONTRACT_FAIL: {error}", file=sys.stderr)
        return 1
    finally:
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())