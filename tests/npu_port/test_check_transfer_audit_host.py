#!/usr/bin/env python3
"""Host-only parser tests: python -B tests/npu_port/test_check_transfer_audit_host.py.

Synthetic records only. No device tools, runtime imports, builds or training.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import check_transfer_audit as checker


def fixture(exports=1):
    events = []

    def event(time, api="aclrtMemcpy", dst="0x1000", src="0x9000", size=516, kind=1):
        events.append({"audit": 1, "pid": 7, "tid": 7, "seq": len(events) + 1,
                       "mono_ns": time, "api": api, "dst": dst, "src": src,
                       "bytes": size, "dst_max": size, "kind": kind,
                       "result": 0, "success": True})
    for time, ptr in ((110, "0x1000"), (120, "0x2000"), (130, "0x3000")):
        event(time, "aclrtMalloc", ptr, "(nil)", kind=-1)
        event(time + 1, dst=ptr)
    # An allocation of the same size without H2D has no inferred gradient identity.
    event(140, "aclrtMalloc", "0x4000", "(nil)", kind=-1)
    for time, size in zip(range(150, 154), (8, 96, 112, 4096)):
        event(time, dst="0x9000", src="0x8000", size=size, kind=2)
    event(180, dst="0x1000")  # first-round prediction reset, not a later-round upload
    event(230, dst="0x9000", src="0x8000", size=8, kind=2)
    for index in range(exports):
        event(350 + index, dst="0x9000", src="0x1000", kind=2)
    for time, ptr in ((420, "0x1000"), (430, "0x2000"), (440, "0x3000"), (450, "0x4000")):
        event(time, "aclrtFree", ptr, "(nil)", size=0, kind=-1)
    markers = []
    for kind, start, end, iteration in (("train", 100, 200, 0), ("train", 210, 300, 1),
                                        ("predict", 310, 400, None), ("cleanup", 410, 500, None)):
        for edge, timestamp in (("begin", start), ("end", end)):
            marker = {"stage": f"{kind}_{edge}", "mono_ns": timestamp, "rows": 129}
            if iteration is not None:
                marker["iteration"] = iteration
            markers.append(marker)
    return events, markers


def resequence(events):
    events.sort(key=lambda event: event["mono_ns"])
    for seq, event in enumerate(events, 1):
        event["seq"] = seq


class TransferAuditHostTest(unittest.TestCase):
    def result(self, events=None, markers=None, **kwargs):
        original_events, original_markers = fixture()
        return checker.analyze(original_events if events is None else events,
                               original_markers if markers is None else markers, **kwargs)

    def codes(self, report):
        return {item["code"] for item in report["errors"]}

    def test_pass_and_exact_allowlist_and_pointer_uploads(self):
        report = self.result()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(set(report["training_d2h_allowlist"]), {"8", "96", "112", "4096"})
        row = report["row_sized_uploads"][0]
        self.assertEqual(row["first_train_row_sized_allocation_count"], 4)
        self.assertEqual(row["first_train_row_h2d_count"], 4)
        self.assertEqual(row["first_train_uploaded_allocation_count"], 3)
        self.assertEqual(row["later_train_row_h2d_count"], 0)
        self.assertEqual([item["upload_count"] for item in row["initial_uploaded_buffers"]], [2, 1, 1])
        self.assertTrue(all(not item["role_identity_proven"] for item in row["initial_uploaded_buffers"]))
        self.assertEqual(report["allocations"]["live_at_end_of_capture"], 0)
        self.assertEqual(report["allocations"]["cleanup_checkpoints"][0]["live_count"], 0)
        self.assertTrue(report["acl"]["sequences_contiguous_from_one"])

    def test_v1_duplicate_public_export_rejected_by_default(self):
        events, markers = fixture(exports=2)
        report = self.result(events, markers)
        self.assertEqual(self.codes(report), {"public_prediction_export_count"})
        self.assertEqual(report["public_prediction"][0]["public_export_count"], 2)
        self.assertEqual(self.result(events, markers, expected_public_exports=2)["status"], "passed")

    def test_training_unlisted_vector_and_async_export_rejected(self):
        events, markers = fixture()
        target = next(e for e in events if e["kind"] == 2)
        target.update(api="aclrtMemcpyAsync", bytes=1032, dst_max=1032)
        self.assertIn("unexpected_training_d2h_size", self.codes(self.result(events, markers)))

    def test_async_prediction_export_counted(self):
        events, _ = fixture()
        next(e for e in events if e["mono_ns"] == 350)["api"] = "aclrtMemcpyAsync"
        self.assertEqual(self.result(events)["status"], "passed")

    def test_failed_or_inconsistent_acl_event_rejected(self):
        for result, success in ((1, False), (0, False), (1, True)):
            with self.subTest(result=result, success=success):
                events, _ = fixture()
                events[1].update(result=result, success=success)
                self.assertIn("acl_call_failed_or_inconsistent", self.codes(self.result(events)))

    def test_missing_start_sequence_gap_duplicate_and_reversed_time(self):
        for mutation in (lambda e: e[0].update(seq=2), lambda e: e[4].update(seq=99),
                         lambda e: e[4].update(seq=e[3]["seq"]),
                         lambda e: e[4].update(mono_ns=e[3]["mono_ns"] - 1)):
            events, _ = fixture()
            mutation(events)
            report = self.result(events)
            self.assertTrue(self.codes(report) & {"sequence_gap_or_duplicate", "nonmonotonic_event_timestamp"})

    def test_stage_matching_and_required_completion(self):
        for mutation in (lambda s: s[1].update(rows=130), lambda s: s.pop(),
                         lambda s: s[2].update(iteration=2),
                         lambda s: s[1].update(stage="predict_begin"),
                         lambda s: s[1].update(mono_ns=s[0]["mono_ns"])):
            _, stages = fixture()
            mutation(stages)
            with self.assertRaises(checker.AuditFormatError):
                self.result(markers=stages)

    def test_stage_half_open_boundary_export_outside_rejected(self):
        events, _ = fixture()
        next(e for e in events if e["mono_ns"] == 350)["mono_ns"] = 400
        resequence(events)
        self.assertIn("d2h_outside_train_predict_windows", self.codes(self.result(events)))

    def test_unexpected_prediction_transfer_rejected(self):
        events, _ = fixture()
        extra = deepcopy(next(e for e in events if e["mono_ns"] == 350))
        extra.update(mono_ns=360, bytes=1000, dst_max=1000)
        events.append(extra)
        resequence(events)
        self.assertIn("unexpected_prediction_d2h_size", self.codes(self.result(events)))

    def test_duplicate_allocation_missing_free_and_wrong_family_rejected(self):
        events, _ = fixture()
        events[2]["dst"] = events[0]["dst"]
        self.assertIn("duplicate_live_allocation", self.codes(self.result(events)))
        events, _ = fixture()
        events.pop()
        report = self.result(events)
        self.assertIn("live_allocations_at_cleanup", self.codes(report))
        self.assertIn("live_allocations_at_end_of_capture", self.codes(report))
        events, _ = fixture()
        events[-1]["api"] = "aclrtFreeHost"
        self.assertIn("allocation_free_family_mismatch", self.codes(self.result(events)))

    def test_unknown_free_reported_and_only_explicitly_allowed(self):
        events, _ = fixture()
        extra = deepcopy(events[-1])
        extra.update(seq=extra["seq"] + 1, mono_ns=460, dst="0xeeee")
        events.append(extra)
        self.assertIn("free_without_tracked_allocation", self.codes(self.result(events)))
        report = self.result(events, allow_external_frees=True)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["allocations"]["unknown_free_count"], 1)
        self.assertTrue(report["warnings"])

    def test_reupload_report_and_pointer_lifetime_reuse(self):
        events, _ = fixture()
        extra = deepcopy(events[3])
        extra.update(mono_ns=240)
        events.append(extra)
        resequence(events)
        row = self.result(events)["row_sized_uploads"][0]
        self.assertEqual(row["later_train_row_h2d_count"], 1)
        self.assertTrue(row["initial_uploaded_buffers"][1]["reuploaded_after_first_train"])
        # Same address, new allocation lifetime: not a reupload of the initial buffer.
        events, _ = fixture()
        free = deepcopy(events[-3])
        free.update(mono_ns=220)
        alloc = deepcopy(events[2])
        alloc.update(mono_ns=221)
        upload = deepcopy(events[3])
        upload.update(mono_ns=240)
        events += [free, alloc, upload]
        resequence(events)
        report = self.result(events)
        self.assertEqual(report["status"], "passed")
        row = report["row_sized_uploads"][0]
        self.assertFalse(row["initial_uploaded_buffers"][1]["reuploaded_after_first_train"])
        self.assertEqual(row["later_train_row_h2d_count"], 1)

    def test_multiple_pids_require_explicit_stage_pid(self):
        events, markers = fixture()
        second = deepcopy(events)
        for event in second:
            event["pid"] = event["tid"] = 8
        with self.assertRaises(checker.AuditFormatError):
            self.result(events + second, markers)
        first_stages = [{**stage, "pid": 7} for stage in markers]
        second_stages = [{**stage, "pid": 8} for stage in markers]
        self.assertEqual(self.result(events + second, first_stages + second_stages)["status"], "passed")

    def test_host_allocations_and_numeric_pointer_normalization(self):
        events, _ = fixture()
        events[0]["api"] = "aclrtMallocHost"
        events[-4]["api"] = "aclrtFreeHost"
        events[-4]["dst"] = "0x00001000"
        self.assertEqual(self.result(events)["status"], "passed")

    def test_missing_fields_invalid_schema_and_truncated_jsonl(self):
        for field in ("seq", "pid", "success", "result", "src", "bytes"):
            events, _ = fixture()
            events[0].pop(field)
            with self.assertRaises(checker.AuditFormatError):
                self.result(events)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.jsonl"
            for raw in (b'{"audit":1}', b"\n", b"{bad}\n"):
                path.write_bytes(raw)
                with self.assertRaises(checker.AuditFormatError):
                    checker.read_jsonl(path)

    def test_cli_preserves_inputs_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transfers, stages = root / "transfers.jsonl", root / "stages.jsonl"
            events, markers = fixture(exports=2)
            for path, records in ((transfers, events), (stages, markers)):
                with path.open("x") as stream:
                    for record in records:
                        stream.write(json.dumps(record) + "\n")
            before = (transfers.read_bytes(), stages.read_bytes())
            args = ["check_transfer_audit", "--transfers", str(transfers), "--stages", str(stages),
                    "--output-dir", str(root / "report")]
            with patch("sys.argv", args), redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main(), 1)
                with self.assertRaises(FileExistsError):
                    checker.main()
            report = json.loads((root / "report" / "report.json").read_text())
            self.assertEqual(report["errors"][0]["code"], "public_prediction_export_count")
            self.assertEqual((transfers.read_bytes(), stages.read_bytes()), before)
            self.assertIn("sha256", report["inputs"]["transfers"])


if __name__ == "__main__":
    unittest.main()