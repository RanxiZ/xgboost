#!/usr/bin/env python3
"""Offline, test-only checker for transfer_audit.cc and native_scale_cases.py.

Example:
  python -B tests/npu_port/check_transfer_audit.py --transfers TRACE.jsonl \
      --stages cases/stages.jsonl --output-dir /dev/shm/FRESH-report

No device APIs, XGBoost imports, compilation or subprocesses are used. Input files
are read-only and the output directory must not already exist. Exit 0 means this
bounded, visible-ACL policy passed; exit 1 means rejection, not a device failure.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

COPY_APIS = {"aclrtMemcpy", "aclrtMemcpyAsync"}
ALLOC_APIS = {"aclrtMalloc": "device", "aclrtMallocHost": "host"}
FREE_APIS = {"aclrtFree": "device", "aclrtFreeHost": "host"}
APIS = COPY_APIS | ALLOC_APIS.keys() | FREE_APIS.keys()
# Reviewed source interpretation, not identification of a payload from its size.
CONTROL_DOWNLOADS = {
    8: {"purpose": "int64 control scalar (counts/reductions)",
        "source": "src/common/npu/acl_ops.h: Ops::SumControl"},
    96: {"purpose": "native prediction/square-error Status",
         "source": "src/common/npu/native_prediction_abi.h: sizeof(Status) == 96; "
                   "src/common/npu/native_prediction.cc: status->Download"},
    112: {"purpose": "fixed-size numerical split SplitSummary",
          "source": "src/common/npu/numerical_split_evaluator_abi.h: sizeof(SplitSummary) == 112; "
                    "src/common/npu/numerical_split_evaluator.h: summary.Download"},
    4096: {"purpose": "512 int64 sign/exponent buckets for scalar scale certification",
           "source": "src/common/npu/certified_scale.h: SignificandBuckets; "
                     "src/common/npu/dynamic_quantizer.h: ReduceBounds/DownloadControl; "
                     "src/common/npu/acl_ops.h: DownloadControl"},
}
LIMITATIONS = [
    "This checks only captured, interposable ACL memcpy/allocation/free APIs; hidden/internal "
    "runtime paths, other transfer APIs and other processes without a capture are not proved absent.",
    "No buffer contents or call stacks were captured. Byte sizes do not establish gradient, "
    "Hessian, histogram, row-ID, label, weight or prediction buffer identity.",
    "The size-to-purpose mapping is a reviewed interpretation of the current source, not proof "
    "that every matching transfer has that payload or that the checkout equals the captured binary.",
    "Initial-upload role hints use current source ordering, not byte-size identity. Allocation "
    "lifetimes, rather than reused addresses alone, determine reuploads.",
    "Monotonic timestamps describe API return/log time, not entry time. Async success means "
    "enqueue success, not independently observed completion; windows need caller synchronization.",
    "A contiguous sequence starting at 1 detects internal gaps, not a missing suffix or events "
    "never intercepted. Process exit and capture completeness need external acceptance evidence.",
    "Zero live tracked allocations at cleanup/end of capture proves tracked lifetime balance, "
    "not absence of untracked allocation leaks or proof of all process teardown operations.",
    "This trace cannot prove absence of all host computation. Host control summaries, model "
    "management and scalar scale certification are expected.",
    "A fixed-size allowlist cannot distinguish a full vector that happens to have an allowed "
    "byte length, nor exclude chunked payload exports masquerading as allowed-size calls.",
]


class AuditFormatError(ValueError):
    """An input lacks enough structure for an unambiguous audit."""


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise AuditFormatError(f"{name} must be an integer >= {minimum}")
    return value


def pointer(value: Any) -> str:
    if not isinstance(value, str):
        raise AuditFormatError("pointer fields must be strings")
    if value in {"(nil)", "0x0", "0", "nullptr"}:
        return "0x0"
    try:
        if not value.startswith("0x") or int(value, 16) < 0:
            raise ValueError
        return hex(int(value, 16))
    except ValueError as error:
        raise AuditFormatError("invalid hexadecimal pointer") from error


def validate_events(events: list[dict]) -> set[int]:
    if not events:
        raise AuditFormatError("transfer log contains no ACL events")
    pids = set()
    for number, event in enumerate(events, 1):
        if not isinstance(event, dict) or event.get("audit") != 1 or event.get("api") not in APIS:
            raise AuditFormatError(f"transfer record {number}: unsupported event/schema")
        for field in ("pid", "tid", "seq", "mono_ns"):
            integer(event.get(field), f"transfer record {number}.{field}", 1)
        for field in ("bytes", "dst_max"):
            integer(event.get(field), f"transfer record {number}.{field}")
        for field in ("kind", "result"):
            if type(event.get(field)) is not int:
                raise AuditFormatError(f"transfer record {number}.{field} must be an integer")
        if type(event.get("success")) is not bool:
            raise AuditFormatError(f"transfer record {number}.success must be boolean")
        pointer(event.get("dst"))
        pointer(event.get("src"))
        if event["api"] in COPY_APIS and event["kind"] not in range(8):
            raise AuditFormatError(f"transfer record {number}: unknown memcpy kind")
        pids.add(event["pid"])
    return pids


def stage_windows(markers: list[dict], pids: set[int]) -> list[dict]:
    """Require native_scale's complete train(s), predict, cleanup protocol per case."""
    if not markers:
        raise AuditFormatError("no stage markers")
    grouped = defaultdict(list)
    for number, marker in enumerate(markers, 1):
        if not isinstance(marker, dict):
            raise AuditFormatError(f"stage record {number} must be an object")
        pid = marker.get("pid")
        if pid is None:
            if len(pids) != 1:
                raise AuditFormatError("stage markers without pid are ambiguous for multiple processes")
            pid = next(iter(pids))
        integer(pid, "stage pid", 1)
        if pid not in pids:
            raise AuditFormatError("stage pid has no captured ACL events")
        integer(marker.get("mono_ns"), "stage mono_ns", 1)
        integer(marker.get("rows"), "stage rows", 1)
        grouped[pid].append(marker)
    if set(grouped) != pids:
        raise AuditFormatError("every captured pid requires its own stage markers")
    windows = []
    for pid, stages in grouped.items():
        previous = 0
        active = None
        current_rows = None
        next_iteration = 0
        phase = "train"
        seen_rows = set()
        for marker in stages:
            timestamp = marker["mono_ns"]
            if timestamp <= previous:
                raise AuditFormatError("stage timestamps must be strictly increasing per pid")
            previous = timestamp
            name = marker.get("stage", "")
            if name not in {f"{kind}_{edge}" for kind in ("train", "predict", "cleanup")
                            for edge in ("begin", "end")}:
                raise AuditFormatError("unknown stage marker")
            kind, edge = name.rsplit("_", 1)
            rows = marker["rows"]
            iteration = marker.get("iteration") if kind == "train" else None
            if kind == "train":
                integer(iteration, "training iteration")
            if edge == "begin":
                if active is not None:
                    raise AuditFormatError("overlapping or nested stage windows")
                if current_rows is None:
                    if kind != "train" or iteration != 0 or rows in seen_rows:
                        raise AuditFormatError("each unique row-count case must begin at training iteration 0")
                    current_rows = rows
                    seen_rows.add(rows)
                if rows != current_rows:
                    raise AuditFormatError("case row count changed before cleanup")
                if kind == "train":
                    if phase != "train" or iteration != next_iteration:
                        raise AuditFormatError("training iterations must be contiguous before prediction")
                elif kind == "predict":
                    if phase != "train" or next_iteration == 0:
                        raise AuditFormatError("prediction must follow completed training")
                    phase = "predict"
                elif phase != "predict":
                    raise AuditFormatError("cleanup must follow prediction")
                else:
                    phase = "cleanup"
                active = {"pid": pid, "kind": kind, "rows": rows, "iteration": iteration,
                          "start_ns": timestamp}
            else:
                if active is None or (kind, rows, iteration) != (
                        active["kind"], active["rows"], active["iteration"]):
                    raise AuditFormatError("stage end does not match its begin")
                windows.append({**active, "end_ns": timestamp})
                active = None
                if kind == "train":
                    next_iteration += 1
                elif kind == "cleanup":
                    current_rows = None
                    next_iteration = 0
                    phase = "train"
        if active is not None or current_rows is not None:
            raise AuditFormatError("incomplete stage/case: missing end, prediction or cleanup")
    return sorted(windows, key=lambda window: (window["pid"], window["start_ns"]))


def ref(event: dict) -> dict:
    return {key: event[key] for key in ("pid", "seq", "mono_ns", "api", "bytes", "dst", "src")}


def counts(events: list[dict]) -> dict[str, int]:
    return {str(size): count for size, count in sorted(Counter(e["bytes"] for e in events).items())}


def analyze(events: list[dict], markers: list[dict], expected_public_exports: int = 1,
            allow_external_frees: bool = False) -> dict:
    integer(expected_public_exports, "expected_public_exports", 1)
    pids = validate_events(events)
    windows = stage_windows(markers, pids)
    errors, warnings = [], []

    def reject(code: str, **details) -> None:
        errors.append({"code": code, **details})

    per_pid_windows = {pid: [w for w in windows if w["pid"] == pid] for pid in pids}
    starts = {pid: [w["start_ns"] for w in group] for pid, group in per_pid_windows.items()}

    def locate(event: dict) -> dict | None:
        index = bisect_right(starts[event["pid"]], event["mono_ns"]) - 1
        if index >= 0:
            window = per_pid_windows[event["pid"]][index]
            if event["mono_ns"] < window["end_ns"]:
                return window
        return None

    # Keep input order: sorting would conceal broken sequence/timestamp evidence.
    last_seq, last_time = defaultdict(int), defaultdict(int)
    live = {}
    lifetimes = {}
    event_allocation = {}
    unknown_frees = []
    peak_count = peak_bytes = live_bytes = 0
    alloc_counts, free_counts = Counter(), Counter()
    checkpoints = []
    cleanup_by_pid = {pid: [w for w in group if w["kind"] == "cleanup"]
                      for pid, group in per_pid_windows.items()}
    cleanup_index = defaultdict(int)

    def checkpoint(pid: int, window: dict) -> None:
        outstanding = [a for (owner, _), a in live.items() if owner == pid]
        item = {"pid": pid, "rows": window["rows"], "mono_ns": window["end_ns"],
                "live_count": len(outstanding), "live_bytes": sum(a["bytes"] for a in outstanding)}
        checkpoints.append(item)
        if outstanding:
            reject("live_allocations_at_cleanup", **item,
                   allocations=[{k: a[k] for k in ("allocation_id", "ptr", "bytes", "api")}
                                for a in outstanding])

    for event in events:
        pid, seq, timestamp = event["pid"], event["seq"], event["mono_ns"]
        if seq != last_seq[pid] + 1:
            reject("sequence_gap_or_duplicate", pid=pid, expected=last_seq[pid] + 1, actual=seq)
        if timestamp < last_time[pid]:
            reject("nonmonotonic_event_timestamp", pid=pid, seq=seq)
        last_seq[pid], last_time[pid] = seq, timestamp
        group = cleanup_by_pid[pid]
        while cleanup_index[pid] < len(group) and group[cleanup_index[pid]]["end_ns"] <= timestamp:
            checkpoint(pid, group[cleanup_index[pid]])
            cleanup_index[pid] += 1
        if event["result"] != 0 or event["success"] is not True:
            reject("acl_call_failed_or_inconsistent", pid=pid, seq=seq, api=event["api"],
                   result=event["result"], success=event["success"])
            continue
        api = event["api"]
        ptr = pointer(event["dst"])
        key = (pid, ptr)
        if api in ALLOC_APIS:
            alloc_counts[api] += 1
            if ptr == "0x0":
                reject("successful_allocation_has_null_pointer", pid=pid, seq=seq)
                continue
            if key in live:
                reject("duplicate_live_allocation", pid=pid, seq=seq, ptr=ptr,
                       previous_allocation_id=live[key]["allocation_id"])
                continue
            allocation_id = f"{pid}:{seq}"
            allocation = {"allocation_id": allocation_id, "pid": pid, "ptr": ptr,
                          "bytes": event["bytes"], "api": api, "seq": seq,
                          "mono_ns": timestamp, "free_seq": None}
            live[key] = lifetimes[allocation_id] = allocation
            live_bytes += event["bytes"]
            peak_count = max(peak_count, len(live))
            peak_bytes = max(peak_bytes, live_bytes)
        elif api in FREE_APIS:
            free_counts[api] += 1
            if key not in live:
                item = {"pid": pid, "seq": seq, "ptr": ptr, "api": api}
                unknown_frees.append(item)
                if not allow_external_frees:
                    reject("free_without_tracked_allocation", **item)
            else:
                allocation = live[key]
                if ALLOC_APIS[allocation["api"]] != FREE_APIS[api]:
                    reject("allocation_free_family_mismatch", pid=pid, seq=seq, ptr=ptr)
                    continue
                del live[key]
                allocation["free_seq"] = seq
                live_bytes -= allocation["bytes"]
        elif api in COPY_APIS:
            device_ptr = event["src"] if event["kind"] == 2 else event["dst"]
            allocation = live.get((pid, pointer(device_ptr)))
            if allocation is not None:
                event_allocation[(pid, seq)] = allocation["allocation_id"]
            if event["bytes"] > event["dst_max"]:
                reject("successful_copy_exceeds_dst_max", pid=pid, seq=seq)
    for pid, group in cleanup_by_pid.items():
        while cleanup_index[pid] < len(group):
            checkpoint(pid, group[cleanup_index[pid]])
            cleanup_index[pid] += 1
    if live:
        reject("live_allocations_at_end_of_capture", count=len(live), bytes=live_bytes,
               allocations=[{k: a[k] for k in ("allocation_id", "ptr", "bytes", "api")}
                            for a in live.values()])
    if unknown_frees and allow_external_frees:
        warnings.append("Unknown external frees were explicitly allowed but not reconciled; "
                        "only tracked-allocation balance is asserted.")

    event_windows = {(e["pid"], e["seq"]): locate(e) for e in events}
    window_events = defaultdict(list)
    outside_d2h = []
    for event in events:
        window = event_windows[(event["pid"], event["seq"])]
        if window is not None:
            window_events[(window["pid"], window["start_ns"])].append(event)
        if event["api"] in COPY_APIS and event["kind"] == 2 and (
                window is None or window["kind"] == "cleanup"):
            outside_d2h.append(ref(event))
    if outside_d2h:
        reject("d2h_outside_train_predict_windows", events=outside_d2h)

    training, predictions = [], []
    for window in windows:
        group = window_events[(window["pid"], window["start_ns"])]
        d2h = [e for e in group if e["api"] in COPY_APIS and e["kind"] == 2]
        item = {**window, "acl_event_count": len(group), "d2h_sizes": counts(d2h),
                "d2h_count": len(d2h), "d2h_bytes": sum(e["bytes"] for e in d2h)}
        if window["kind"] == "train":
            if not group:
                reject("training_window_has_no_acl_evidence", pid=window["pid"],
                       rows=window["rows"], iteration=window["iteration"])
            unexpected = [ref(e) for e in d2h if e["bytes"] not in CONTROL_DOWNLOADS]
            item["unexpected_d2h"] = unexpected
            if unexpected:
                reject("unexpected_training_d2h_size", pid=window["pid"], rows=window["rows"],
                       iteration=window["iteration"], events=unexpected)
            training.append(item)
        elif window["kind"] == "predict":
            exports = [e for e in d2h if e["bytes"] == window["rows"] * 4]
            item.update(public_export_bytes=window["rows"] * 4,
                        public_export_count=len(exports), expected_public_exports=expected_public_exports,
                        public_exports=[ref(e) for e in exports])
            if len(exports) != expected_public_exports:
                reject("public_prediction_export_count", pid=window["pid"], rows=window["rows"],
                       expected=expected_public_exports, actual=len(exports))
            unexpected = [ref(e) for e in d2h if e["bytes"] not in CONTROL_DOWNLOADS
                          and e["bytes"] != window["rows"] * 4]
            if unexpected:
                reject("unexpected_prediction_d2h_size", pid=window["pid"], rows=window["rows"],
                       events=unexpected)
            predictions.append(item)

    uploads = []
    for first in (w for w in windows if w["kind"] == "train" and w["iteration"] == 0):
        pid, rows = first["pid"], first["rows"]
        case_events = [e for e in events if (w := event_windows[(e["pid"], e["seq"])])
                       is not None and w["pid"] == pid and w["rows"] == rows]
        row_uploads = [e for e in case_events if e["api"] in COPY_APIS and e["kind"] == 1
                       and e["bytes"] == rows * 4]
        initial = [e for e in row_uploads if first["start_ns"] <= e["mono_ns"] < first["end_ns"]]
        ids = list(dict.fromkeys(event_allocation.get((pid, e["seq"]),
                                                     f"untracked:{pid}:{e['seq']}") for e in initial))
        candidates = []
        hints = ["prediction initialization/reset (source-order hint)",
                 "labels (source-order hint)", "weights (source-order hint)"]
        for index, allocation_id in enumerate(ids):
            matching = [e for e in row_uploads if event_allocation.get((pid, e["seq"])) == allocation_id]
            if allocation_id.startswith("untracked:"):
                matching = [e for e in initial if f"untracked:{pid}:{e['seq']}" == allocation_id]
                warnings.append(f"row-sized H2D has no exact-base tracked allocation: {allocation_id}")
            allocation = lifetimes.get(allocation_id)
            later = [e for e in matching if e["mono_ns"] >= first["end_ns"]]
            candidates.append({"allocation_id": allocation_id,
                               "ptr": pointer(matching[0]["dst"]),
                               "allocation_bytes": allocation["bytes"] if allocation else None,
                               "free_seq": allocation["free_seq"] if allocation else None,
                               "role_hint": hints[index] if len(ids) == 3 else "unidentified row-sized upload",
                               "role_identity_proven": False, "upload_count": len(matching),
                               "first_train_upload_count": len(matching) - len(later),
                               "reuploaded_after_first_train": bool(later),
                               "later_upload_count": len(later),
                               "uploads": [{**ref(e), "stage": event_windows[(pid, e["seq"])]["kind"],
                                            "iteration": event_windows[(pid, e["seq"])]["iteration"]}
                                           for e in matching]})
        later_training = [e for e in row_uploads if (w := event_windows[(pid, e["seq"])])["kind"]
                          == "train" and w["iteration"] > 0]
        first_allocations = [a for a in lifetimes.values() if a["pid"] == pid
                             and a["bytes"] == rows * 4
                             and first["start_ns"] <= a["mono_ns"] < first["end_ns"]]
        uploads.append({"pid": pid, "rows": rows, "row_bytes": rows * 4,
                        "first_train_row_sized_allocation_count": len(first_allocations),
                        "first_train_row_h2d_count": len(initial),
                        "first_train_uploaded_allocation_count": len(ids),
                        "later_train_row_h2d_count": len(later_training),
                        "row_sized_h2d_count_all_case_windows": len(row_uploads),
                        "initial_uploaded_buffers": candidates,
                        "later_train_uploads": [ref(e) for e in later_training]})

    return {"schema": 1, "status": "passed" if not errors else "failed",
            "expected_public_exports": expected_public_exports, "errors": errors, "warnings": warnings,
            "acl": {"event_count": len(events), "api_counts": dict(sorted(Counter(e["api"] for e in events).items())),
                    "all_calls_successful": all(e["result"] == 0 and e["success"] for e in events),
                    "per_pid": [{"pid": pid, "first_seq": next(e["seq"] for e in events if e["pid"] == pid),
                                 "last_seq": last_seq[pid],
                                 "event_count": sum(e["pid"] == pid for e in events)} for pid in sorted(pids)],
                    "sequences_contiguous_from_one": not any(e["code"] == "sequence_gap_or_duplicate" for e in errors)},
            "training_d2h_allowlist": {str(k): v for k, v in CONTROL_DOWNLOADS.items()},
            "training": training, "public_prediction": predictions, "row_sized_uploads": uploads,
            "allocations": {"alloc_counts": dict(alloc_counts), "free_counts": dict(free_counts),
                            "peak_live_count": peak_count, "peak_live_bytes": peak_bytes,
                            "live_at_end_of_capture": len(live), "live_bytes_at_end_of_capture": live_bytes,
                            "unknown_free_count": len(unknown_frees), "unknown_frees": unknown_frees,
                            "unknown_external_frees_allowed": allow_external_frees,
                            "cleanup_checkpoints": checkpoints,
                            "tracked_buffers_released_at_all_cleanup_markers_and_capture_end":
                            not live and all(c["live_count"] == 0 for c in checkpoints)},
            "d2h_outside_train_predict": outside_d2h, "limitations": LIMITATIONS}


def read_jsonl(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    records = []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            raise AuditFormatError(f"blank JSONL record at line {number}")
        try:
            records.append(json.loads(line))
        except (ValueError, UnicodeError) as error:
            raise AuditFormatError(f"invalid JSONL record at line {number}") from error
    if raw and not raw.endswith(b"\n"):
        raise AuditFormatError("JSONL lacks a final newline; possible truncated capture")
    return records, hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfers", required=True, type=Path)
    parser.add_argument("--stages", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-public-exports", type=int, default=1)
    parser.add_argument("--allow-external-frees", action="store_true",
                        help="report unknown frees without treating them as reconciled allocations")
    args = parser.parse_args()
    # Fail before writing anything if the destination exists, including input directories.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        events, event_hash = read_jsonl(args.transfers)
        markers, stage_hash = read_jsonl(args.stages)
        report = analyze(events, markers, args.expected_public_exports, args.allow_external_frees)
        report["inputs"] = {"transfers": {"path": str(args.transfers.resolve()), "sha256": event_hash},
                            "stages": {"path": str(args.stages.resolve()), "sha256": stage_hash}}
    except (AuditFormatError, OSError) as error:
        report = {"schema": 1, "status": "failed", "errors": [{"code": "invalid_input", "reason": str(error)}],
                  "limitations": LIMITATIONS}
    with (args.output_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    codes = Counter(error["code"] for error in report["errors"])
    print(json.dumps({"status": report["status"], "error_counts": dict(codes),
                      "report": str((args.output_dir / "report.json").resolve())}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())