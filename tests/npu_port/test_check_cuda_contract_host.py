from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("check_cuda_contract.py")
spec = importlib.util.spec_from_file_location("check_cuda_contract", SCRIPT)
checker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checker
spec.loader.exec_module(checker)


class ReferenceLayerTests(unittest.TestCase):
    def fixture_contract(self):
        return {"reference": {"tag": "v3.4.1", "commit": "abc123"},
                "components": [{"source": "a.cu", "anchor": "OFFICIAL", "additional_sources": ["b.cuh"]}]}

    def git_stub(self, mapping):
        def run(_root, *args):
            command = " ".join(args)
            if command == "rev-parse v3.4.1^{commit}":
                return b"abc123\n"
            if command.startswith("show abc123:"):
                path = command.split(":", 1)[1]
                if path not in mapping:
                    raise subprocess.CalledProcessError(128, command)
                return mapping[path]
            raise AssertionError(command)
        return run

    def expected_hash(self, mapping):
        digest = hashlib.sha256()
        for path in sorted(mapping):
            digest.update(path.encode() + b"\0" + hashlib.sha256(mapping[path]).digest())
        return digest.hexdigest()

    def test_reference_hash_validated(self):
        contract = self.fixture_contract()
        mapping = {"a.cu": b"OFFICIAL source", "b.cuh": b"header"}
        verified = checker.inspect_official_reference(
            contract, Path("."), self.git_stub(mapping), self.expected_hash(mapping))
        self.assertEqual(verified["reference_file_count"], 2)
        with self.assertRaisesRegex(RuntimeError, "manifest hash mismatch"):
            checker.inspect_official_reference(contract, Path("."), self.git_stub(mapping), "0" * 64)

    def test_bad_pinned_commit_rejected(self):
        contract = self.fixture_contract()
        contract["reference"]["commit"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "tag does not resolve"):
            checker.inspect_official_reference(
                contract, Path("."), self.git_stub({"a.cu": b"OFFICIAL source", "b.cuh": b"header"}), "")

    def test_bad_official_anchor_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "anchor missing"):
            checker.inspect_official_reference(self.fixture_contract(), Path("."),
                                               self.git_stub({"a.cu": b"changed", "b.cuh": b"header"}), "")

    def test_missing_official_object_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "object unavailable"):
            checker.inspect_official_reference(self.fixture_contract(), Path("."),
                                               self.git_stub({"a.cu": b"OFFICIAL source"}), "")

    def test_real_reference_only_and_strict_default(self):
        root = SCRIPT.parents[2]
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            out = Path(temporary) / "reference"
            base = [sys.executable, str(SCRIPT), "--output-dir", str(out),
                    "--expected-source-root", str(root)]
            reference = subprocess.run(base + ["--reference-only"], capture_output=True, text=True)
            self.assertEqual(reference.returncode, 0, reference.stderr)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["official_reference"]["reference_file_count"], 33)
            self.assertEqual(summary["official_reference"]["reference_hash_sha256"],
                             checker.OFFICIAL_REFERENCE_SHA256)
            self.assertEqual(summary["worktree_identity"]["status"], "changed_unaccepted")
            self.assertEqual(summary["integration_source_identity"], "unaccepted_until_reviewed")
            self.assertEqual(summary["mode"], "reference_only_no_compile_no_device")
            self.assertFalse((out / "compile.log").exists())
            strict = subprocess.run(base, capture_output=True, text=True)
            self.assertNotEqual(strict.returncode, 0)
            strict_summary = json.loads((out / "summary.json").read_text())
            self.assertIn("source changed; contract review required", strict_summary["error"])
            self.assertEqual(strict_summary["worktree_identity"]["status"], "changed_unaccepted")
            self.assertFalse((out / "compile.log").exists())


if __name__ == "__main__":
    unittest.main()