#!/usr/bin/env python3
"""Device-free negative tests for standalone histogram profiler validation."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('histogram_checker', Path(__file__).with_name('check_integer_histogram.py'))
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class HistogramProfileTests(unittest.TestCase):
    def rows(self):
        return [dict(Device_id='0', **{'OP Type': name, 'Task Type': task,
                                       'Task Duration(us)': '1.25'})
                for name, task in [('IntegerHistogramBuildKernel', 'MIX_AIC'),
                                   ('ReduceSum', 'AI_VECTOR_CORE'), ('Gather', 'MIX_AIV')]]

    def test_exact_mixed_fused_kernel_and_vector_stages(self):
        checker.validate_profile_rows(self.rows())

    def test_missing_fused_kernel(self):
        with self.assertRaisesRegex(RuntimeError, 'missing fused'):
            checker.validate_profile_rows(self.rows()[1:])

    def test_impostor_mixed_kernel(self):
        for name in ('OtherKernel', 'PrefixIntegerHistogramBuildKernelSuffix'):
            rows = self.rows(); rows[0]['OP Type'] = name
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'unexpected compute'):
                checker.validate_profile_rows(rows)

    def test_existing_stage_cannot_become_mixed_aic(self):
        rows = self.rows(); rows[1]['Task Type'] = 'MIX_AIC'
        with self.assertRaisesRegex(RuntimeError, 'unexpected compute'):
            checker.validate_profile_rows(rows)

    def test_cpu_unknown_and_wrong_device_rejected(self):
        for field, value in [('Task Type', 'AI_CPU'), ('Task Type', 'UNKNOWN'), ('Device_id', '1')]:
            rows = self.rows(); rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                checker.validate_profile_rows(rows)

    def test_invalid_durations_rejected_for_every_stage(self):
        for index in range(3):
            for value in ('0', '-1', 'nan', 'inf', '', 'malformed', None):
                rows = self.rows(); rows[index]['Task Duration(us)'] = value
                with self.subTest(index=index, value=value), self.assertRaises(RuntimeError):
                    checker.validate_profile_rows(rows)

    def test_empty_profile(self):
        with self.assertRaisesRegex(RuntimeError, 'missing device-task'):
            checker.validate_profile_rows([])


if __name__ == '__main__':
    unittest.main()