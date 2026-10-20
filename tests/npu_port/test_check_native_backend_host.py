#!/usr/bin/env python3
"""Host-only regression tests; all child processes are forbidden or mocked.

Run with: python -B tests/npu_port/test_check_native_backend_host.py
No xgboost, CANN library, training or device API is imported or executed.
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import call, patch

SPEC = importlib.util.spec_from_file_location(
    "check_native_backend", Path(__file__).with_name("check_native_backend.py"))
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)

KERNELS = (
    "NativeSquareErrorKernel", "NativeBinRowsKernel", "NativeSampleRowsKernel",
    "NativePredictKernel", "BatchedSplitPrefixKernel", "BatchedSplitScoreKernel",
    "BatchedSplitSelectKernel", "BatchedSplitFinishKernel", "IntegerHistogramBuildKernel",
)
APIS = (
    "aclnnScatterAdd_ScatterElementsV2AiCore_ScatterElementsV2",
    "aclnnReduceSum_ReduceSumOpAiCore_ReduceSum",
    "aclnnCumsum_CumsumAiCore_Cumsum",
    "aclnnIndexSelect_GatherV2AiCore_GatherV2",
    "aclnnSWhere_SelectV2AiCore_SelectV2",
)
FIELDS = ("Device_id", "Op Name", "OP Type", "Task Type", "Task Duration(us)")


class ProfileEvidenceHostTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        # An unexpected subprocess, including npu-smi, fails before execution.
        guard = patch.object(subprocess, "run", side_effect=AssertionError("forbidden child process"))
        guard.start()
        self.addCleanup(guard.stop)

    def rows(self):
        rows = [{"Device_id": "0", "Op Name": name, "OP Type": "custom",
                 "Task Type": "MIX_AIC", "Task Duration(us)": "1.25"} for name in KERNELS]
        rows.extend({"Device_id": "0", "Op Name": name, "OP Type": name.rsplit("_", 1)[-1],
                     "Task Type": "AI_VECTOR_CORE", "Task Duration(us)": "2.5"} for name in APIS)
        return rows

    def write_table(self, rows, fields=FIELDS, name="op_summary_fixture.csv"):
        path = self.directory / name
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def validate(self, rows, fields=FIELDS):
        self.write_table(rows, fields)
        return checker.profile_evidence(self.directory)

    def test_accept_op_name_evidence_and_hash_all_included_sources(self):
        report = self.validate(self.rows())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["task_count"], len(KERNELS) + len(APIS))
        self.assertEqual(set(sum(report["native_kernels"].values(), [])), set(KERNELS))
        self.assertEqual(report["native_kernels"]["histogram"], ["IntegerHistogramBuildKernel"])
        self.assertIn("IntegerHistogramBuildKernel", report["source_kernels"])
        self.assertEqual(len(report["production_asc_sha256"]), 8)
        self.assertTrue(any(path.endswith("precise_scoring.asc")
                            for path in report["production_asc_sha256"]))

    def test_accept_op_type_names_core_type_and_total_time(self):
        rows = []
        for row in self.rows():
            rows.append({"Device_id": row["Device_id"], "OP Type": row["Op Name"],
                         "Core Type": row["Task Type"], "Total Time(us)": row["Task Duration(us)"]})
        self.assertEqual(self.validate(rows, ("Device_id", "OP Type", "Core Type", "Total Time(us)"))
                         ["status"], "passed")

    def test_accept_index_select_spelling_variants(self):
        for spelling in ("aclnnIndexSelect", "aclnnIndex_Select", "aclnn_Index_Select"):
            with self.subTest(spelling=spelling):
                rows = self.rows()
                rows[-2]["Op Name"] = spelling + "_GatherV2AiCore_GatherV2"
                self.assertEqual(self.validate(rows)["status"], "passed")

    def test_reject_every_missing_native_kernel_including_histogram_and_each_scoring_kernel(self):
        for kernel in KERNELS:
            with self.subTest(kernel=kernel):
                rows = [row for row in self.rows() if row["Op Name"] != kernel]
                with self.assertRaisesRegex(RuntimeError, "profile lacks native"):
                    self.validate(rows)

    def test_reject_histogram_kernel_absent_from_production_sources(self):
        kernels, sources = checker.production_kernel_sources()
        with patch.object(checker, "production_kernel_sources",
                          return_value=(kernels - {"IntegerHistogramBuildKernel"}, sources)):
            with self.assertRaisesRegex(RuntimeError, "required histogram kernel is absent from production"):
                self.validate(self.rows())

    def test_reject_histogram_kernel_substrings(self):
        for name in ("FakeIntegerHistogramBuildKernel", "IntegerHistogramBuildKernelFake"):
            with self.subTest(name=name):
                rows = self.rows()
                for row in rows:
                    if row["Op Name"] == "IntegerHistogramBuildKernel":
                        row["Op Name"] = name
                with self.assertRaisesRegex(RuntimeError, "profile lacks native histogram"):
                    self.validate(rows)

    def test_standalone_component_kernels_cannot_replace_production_chain(self):
        rows = [row for row in self.rows() if row["Op Name"] not in
                {"BatchedSplitPrefixKernel", "BatchedSplitScoreKernel",
                 "BatchedSplitSelectKernel", "BatchedSplitFinishKernel"}]
        rows.extend({"Device_id": "0", "Op Name": name, "OP Type": "custom",
                     "Task Type": "MIX_AIC", "Task Duration(us)": "1.25"}
                    for name in ("PreciseScoreKernel", "ChooseDirectionKernel",
                                 "NodeQuantitiesKernel", "SubtractParentKernel", "SplitStatisticsKernel",
                                 "ScoreColumnsKernel", "SelectCandidatesKernel", "SplitInitializeKernel",
                                 "SplitMergeKernel", "SplitFinishKernel"))
        with self.assertRaisesRegex(RuntimeError, "profile lacks native scoring"):
            self.validate(rows)

    def test_reject_every_missing_histogram_or_partition_api(self):
        for api in APIS:
            with self.subTest(api=api):
                with self.assertRaisesRegex(RuntimeError, "profile lacks ACLNN"):
                    self.validate([row for row in self.rows() if row["Op Name"] != api])

    def test_reject_generic_operator_types_without_aclnn_api_names(self):
        rows = self.rows()
        for row in rows[len(KERNELS):]:
            row["Op Name"] = row["OP Type"]
        with self.assertRaisesRegex(RuntimeError, "profile lacks ACLNN"):
            self.validate(rows)

    def test_reject_kernel_substrings_and_names_in_unrelated_columns(self):
        for name in ("FakeNativeSquareErrorKernel", "NativeSquareErrorKernelFake", "unrelated"):
            with self.subTest(name=name):
                rows = self.rows()
                rows[0]["Op Name"] = name
                for row in rows:
                    row["Input Shapes"] = "NativeSquareErrorKernel"
                with self.assertRaisesRegex(RuntimeError, "profile lacks native gradient"):
                    self.validate(rows, (*FIELDS, "Input Shapes"))

    def test_reject_aicpu_in_either_type_column_even_on_unrelated_task(self):
        for key in ("Task Type", "Core Type"):
            for task in ("AI_CPU", "AICPU", "AI-CPU"):
                with self.subTest(key=key, task=task):
                    rows = self.rows()
                    rows.append({"Device_id": "0", "Op Name": "unrelated", "OP Type": "unrelated",
                                 "Task Type": "AI_CORE", "Core Type": "AI_CORE",
                                 "Task Duration(us)": "1"})
                    rows[-1][key] = task
                    with self.assertRaisesRegex(RuntimeError, "AI_CPU"):
                        self.validate(rows, (*FIELDS, "Core Type"))

    def test_reject_invalid_task_device_or_duration(self):
        cases = [("Device_id", "1"), ("Task Type", ""), ("Task Type", "HOST_CPU"),
                 ("Task Type", "MEMCPY"), ("Task Duration(us)", "0"),
                 ("Task Duration(us)", "-1"), ("Task Duration(us)", "NaN"),
                 ("Task Duration(us)", "inf"), ("Task Duration(us)", "bad"),
                 ("Task Duration(us)", "")]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                rows = self.rows()
                rows[0][key] = value
                with self.assertRaises(RuntimeError):
                    self.validate(rows)

    def test_reject_no_empty_unknown_or_malformed_tables(self):
        with self.assertRaisesRegex(RuntimeError, "no operator summary"):
            checker.profile_evidence(self.directory)
        with self.assertRaisesRegex(RuntimeError, "empty"):
            self.validate([])
        self.write_table([], ("Device_id", "OP Type"))
        with self.assertRaisesRegex(RuntimeError, "columns"):
            checker.profile_evidence(self.directory)
        path = self.write_table(self.rows())
        with path.open("a") as stream:
            stream.write("0,missing-fields\n")
        with self.assertRaisesRegex(RuntimeError, "malformed"):
            checker.profile_evidence(self.directory)

    def test_reject_bad_additional_table_even_with_complete_good_evidence(self):
        self.write_table(self.rows())
        self.write_table([], name="op_summary_empty.csv")
        with self.assertRaisesRegex(RuntimeError, "empty"):
            checker.profile_evidence(self.directory)

    def test_follow_recursive_local_asc_includes_and_ignore_comments(self):
        root = self.directory / "src/common/npu"
        (root / "kernel").mkdir(parents=True)
        (root / "kernel/production_kernels.asc").write_text('#include "../first.asc"\n')
        (root / "first.asc").write_text(
            '#include "second.asc"\n#include "kernel_operator.h"\n'
            '// __global__ void CommentKernel() {}\n'
            '/* #include "absent.asc"\n__global__ void AlsoCommentKernel() {} */\n'
            '__global__\n__aicore__ void RealKernel(int n) {}\n')
        (root / "second.asc").write_text(
            '#include "first.asc"\n__global__ void DeclarationOnly();\n')
        with patch.object(checker, "ROOT", self.directory):
            kernels, sources = checker.production_kernel_sources()
            self.assertEqual(kernels, {"RealKernel"})
            self.assertEqual(len(sources), 3)
            (root / "second.asc").write_text('#include "missing.asc"\n')
            with self.assertRaisesRegex(RuntimeError, "source missing"):
                checker.production_kernel_sources()
            (root / "second.asc").write_text('#include "../../../escape.asc"\n')
            with self.assertRaisesRegex(RuntimeError, "escapes source tree"):
                checker.production_kernel_sources()

    def test_existing_stage7_profile_is_not_end_to_end_evidence(self):
        profile = Path("/workspace/xgb-npu-runs/stage7-precise-scoring-device-v3/profile")
        if not profile.is_dir():
            self.skipTest("optional existing stage7 profiler capture is unavailable")
        with self.assertRaisesRegex(RuntimeError, "profile lacks native gradient"):
            checker.profile_evidence(profile)

    def test_idle_check_passes_explicit_executable_and_private_environment(self):
        smi = self.directory / "npu-smi-placeholder"
        smi.touch()
        env = {"PATH": "/private/bin", "LD_LIBRARY_PATH": "/private/lib"}
        result = subprocess.CompletedProcess([], 0, "| 0 | Ascend950PR | OK |\nNo running processes found in NPU 0\n")
        with patch.object(checker, "NPU_SMI", smi), patch.object(subprocess, "run", return_value=result) as run:
            checker.idle_check(self.directory, "idle", env)
        self.assertEqual(run.call_args.args[0], [str(smi), "info"])
        self.assertIs(run.call_args.kwargs["env"], env)
        self.assertEqual((self.directory / "idle.log").read_text(), result.stdout)

    def test_idle_check_distinguishes_occupied_from_missing_device(self):
        smi = self.directory / "npu-smi-placeholder"
        smi.touch()
        cases = (
            ("| 0 | Ascend950PR | OK |\n| 0 | 1188844 | python | 45 |\n", "occupied"),
            ("| 1 | Ascend950PR | OK |\n", "not visible"),
            ("No running processes found in NPU 0\n", "not visible"),
        )
        for text, message in cases:
            with self.subTest(message=message, text=text):
                result = subprocess.CompletedProcess([], 0, text)
                with patch.object(checker, "NPU_SMI", smi), \
                        patch.object(subprocess, "run", return_value=result), \
                        self.assertRaisesRegex(RuntimeError, message):
                    checker.idle_check(self.directory, "idle", {})

    def test_profile_rerun_preserves_case_outputs_and_validates_both_summaries(self):
        output = self.directory / "run"
        cann = self.directory / "cann"
        profiler = cann / "tools/profiler/bin/msprof"
        profiler.parent.mkdir(parents=True)
        profiler.touch()  # Placeholder only: run_logged is mocked below.
        env = {"PATH": "/private/bin"}
        identity = {"library_path": "/private/libxgboost.so"}
        with ExitStack() as stack:
            stack.enter_context(patch.object(checker, "CANN", cann))
            stack.enter_context(patch.object(checker, "validate_root_and_identity", return_value=(identity, "helper")))
            stack.enter_context(patch.object(checker, "host_preflight", return_value={"status": "passed"}))
            stack.enter_context(patch.object(checker, "private_env", return_value=env))
            idle = stack.enter_context(patch.object(checker, "idle_check", return_value="idle.log"))
            run = stack.enter_context(patch.object(checker, "run_logged"))
            summaries = stack.enter_context(patch.object(checker, "parse_case_summary", return_value={"status": "passed"}))
            stack.enter_context(patch.object(checker, "sha256", return_value="digest"))
            stack.enter_context(patch.object(checker, "artifact_hashes", return_value={}))
            stack.enter_context(patch.object(checker, "profile_evidence", return_value={"status": "passed"}))
            stack.enter_context(patch.object(checker.sys, "argv", ["checker", "--run-npu", "--confirm-device-idle",
                                "--profile", "--expected-source-root", str(checker.ROOT), "--output-dir", str(output)]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(checker.main(), 0)
        commands = [entry.args[0] for entry in run.call_args_list]
        self.assertEqual(commands[0][-1], str(output / "case-output"))
        self.assertEqual(commands[1][-1], str(output / "profile-case-output"))
        self.assertEqual(summaries.call_args_list,
                         [call(output / "case-output"), call(output / "profile-case-output")])
        self.assertEqual([entry.args[2] for entry in idle.call_args_list], [env] * 4)
        report = json.loads((output / "summary.json").read_text())
        self.assertEqual(len(report["idle_checks"]), 4)
        self.assertIn("profile_case_report", report)


if __name__ == "__main__":
    unittest.main()