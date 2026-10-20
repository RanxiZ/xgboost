#!/usr/bin/env python3
"""Host-only argument tests for the opt-in native scale driver."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import native_scale_cases as scale


class ScaleArgumentTests(unittest.TestCase):
    def test_invalid_bounds_and_duplicate_rows_stop_before_backend_or_output(self):
        invalid = [
            ['--rows', '0'], ['--rows', '65538'], ['--rows', '129', '129'],
            ['--features', '2'], ['--features', '159'],
            ['--max-depth', '0'], ['--max-depth', '9'],
            ['--max-bin', '1'], ['--max-bin', '257'],
            ['--rounds', '0'], ['--rounds', '65'],
            ['--max-leaves', '-1'], ['--max-leaves', '257'],
            ['--grow-policy', 'unsupported'],
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, arguments in enumerate(invalid):
                output = Path(directory) / str(index)
                with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                    with patch.object(scale, '_private_xgboost') as backend:
                        with self.assertRaises(SystemExit) as result:
                            scale.main(['--library', '/unused.so', '--output-dir', str(output),
                                        *arguments])
                        self.assertEqual(result.exception.code, 2)
                        backend.assert_not_called()
                        self.assertFalse(output.exists())

    def test_long_round_and_shape_boundaries_are_accepted_without_device_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            for rounds in (1, 3, 16, 64):
                output = Path(directory) / str(rounds)
                stop = RuntimeError('host-only stop before backend import')
                with self.subTest(rounds=rounds), patch.object(
                        scale, '_private_xgboost', side_effect=stop) as backend:
                    with self.assertRaisesRegex(RuntimeError, 'host-only stop'):
                        scale.main(['--library', '/unused.so', '--output-dir', str(output),
                                    '--rows', '1', '65537', '--features', '158',
                                    '--max-depth', '8', '--max-bin', '256',
                                    '--rounds', str(rounds)])
                    backend.assert_called_once_with(Path('/unused.so'))

    def test_baseline_comparison_requires_identical_model_and_float_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'reference'
            reference.mkdir()
            model = root / 'model-129.ubj'
            model.write_bytes(b'fixed-model')
            (reference / model.name).write_bytes(model.read_bytes())
            prediction = scale.np.array([0.0, 1.0], dtype=scale.np.float32)
            scale.np.save(reference / 'prediction-129-npu.npy', prediction)
            baseline = {'library_sha256': 'fixed-library-hash'}
            report = scale._compare_baseline(reference, baseline, model, prediction, 129)
            self.assertEqual(report['status'], 'bit_identical')
            (reference / model.name).write_bytes(b'changed-model')
            with self.assertRaisesRegex(AssertionError, 'training model'):
                scale._compare_baseline(reference, baseline, model, prediction, 129)
            (reference / model.name).write_bytes(model.read_bytes())
            for changed in (prediction.astype(scale.np.float64), prediction.reshape(1, 2),
                            scale.np.array([-0.0, 1.0], dtype=scale.np.float32),
                            scale.np.array([0.0, 1.000001], dtype=scale.np.float32)):
                with self.subTest(changed=changed), self.assertRaisesRegex(AssertionError, 'prediction'):
                    scale._compare_baseline(reference, baseline, model, changed, 129)

    def test_baseline_status_and_parameter_guards_precede_backend(self):
        case = dict(rows=129, features=8, rounds=3, max_depth=2, max_bin=16,
                    grow_policy='depthwise', max_leaves=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'reference'
            reference.mkdir()
            mismatches = [{'status': 'failed', 'cases': [case]},
                          {'status': 'passed', 'cases': []}]
            for key in case:
                changed = case | {key: 'different'}
                mismatches.append({'status': 'passed', 'cases': [changed]})
            for index, report in enumerate(mismatches):
                (reference / 'summary.json').write_text(json.dumps(report))
                output = root / str(index)
                with self.subTest(report=report), patch.object(scale, '_private_xgboost') as backend:
                    with self.assertRaises(ValueError):
                        scale.main(['--library', '/unused.so', '--output-dir', str(output),
                                    '--rows', '129', '--baseline-output-dir', str(reference)])
                    backend.assert_not_called()
                    self.assertFalse(output.exists())
            (reference / 'summary.json').write_text(json.dumps({'status': 'passed', 'cases': [case]}))
            with patch.object(scale, '_private_xgboost', side_effect=RuntimeError('host-only stop')):
                with self.assertRaisesRegex(RuntimeError, 'host-only stop'):
                    scale.main(['--library', '/unused.so', '--output-dir', str(root / 'valid'),
                                '--rows', '129', '--baseline-output-dir', str(reference)])


if __name__ == '__main__':
    unittest.main()