#!/usr/bin/env python3
"""Opt-in bounded native scale cases, with stage markers for ACL transfer audit.

Run only in the private acceptance environment after checking device idle. This
is not full Qlib acceptance or proof of hidden runtime-internal transfers.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np

from native_backend_cases import _private_xgboost, _sha256


def _compare_baseline(reference: Path, baseline: dict, model_path: Path,
                      prediction: np.ndarray, rows: int) -> dict:
    if model_path.read_bytes() != (reference / model_path.name).read_bytes():
        raise AssertionError('training model differs bitwise from the baseline build')
    reference_prediction = np.load(reference / f'prediction-{rows}-npu.npy', allow_pickle=False)
    if (prediction.dtype != reference_prediction.dtype or
            prediction.shape != reference_prediction.shape or
            prediction.tobytes() != reference_prediction.tobytes()):
        raise AssertionError('prediction differs bitwise from the baseline build')
    return {'status': 'bit_identical', 'output_dir': str(reference.resolve()),
            'library_sha256': baseline['library_sha256']}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rows', type=int, nargs='+', default=[129, 2049, 4097])
    parser.add_argument('--features', type=int, default=8)
    parser.add_argument('--max-depth', type=int, default=2)
    parser.add_argument('--max-bin', type=int, default=16)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--grow-policy', choices=['depthwise', 'lossguide'], default='depthwise')
    parser.add_argument('--max-leaves', type=int, default=0)
    parser.add_argument('--baseline-output-dir', type=Path,
                        help='require bit-identical models and predictions from a previous build')
    args = parser.parse_args(argv)
    if not 0 <= args.max_leaves <= 256:
        parser.error('bounded cases require max_leaves within 0..256')
    if not (3 <= args.features <= 158 and 1 <= args.max_depth <= 8 and
            2 <= args.max_bin <= 256 and 1 <= args.rounds <= 64):
        parser.error('bounded cases require features 3..158, depth 1..8, bins 2..256, rounds 1..64')
    if any(rows < 1 or rows > 65537 for rows in args.rows):
        parser.error('bounded scale case rows must be within 1..65537')
    if len(set(args.rows)) != len(args.rows):
        parser.error('row counts must be unique for unambiguous transfer stage markers')
    baseline = None
    if args.baseline_output_dir is not None:
        baseline = json.loads((args.baseline_output_dir / 'summary.json').read_text())
        if baseline.get('status') != 'passed':
            raise ValueError('baseline must be a completed successful diagnostic')
        expected_cases = [dict(rows=rows, features=args.features, rounds=args.rounds,
                               max_depth=args.max_depth, max_bin=args.max_bin,
                               grow_policy=args.grow_policy, max_leaves=args.max_leaves)
                          for rows in args.rows]
        actual_cases = [{key: case.get(key) for key in expected_cases[0]}
                        for case in baseline.get('cases', [])]
        if actual_cases != expected_cases:
            raise ValueError('baseline data shapes or training parameters differ')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    xgb = _private_xgboost(args.library)
    if xgb.build_info().get('USE_NPU') is not True:
        raise RuntimeError('scale cases require the private native NPU build')
    rng = np.random.default_rng(20261017)
    reports = []
    with (output / 'stages.jsonl').open('w', buffering=1) as stages:
        def mark(stage, **details):
            stages.write(json.dumps({'stage': stage, 'mono_ns': time.monotonic_ns(), **details}) + '\n')

        for rows in args.rows:
            features = rng.normal(size=(rows, args.features)).astype(np.float32)
            features[::7, 0] = np.nan
            features[:, -1] = np.nan
            labels = (np.nan_to_num(features[:, 0]) * 0.75 - features[:, 2] * 0.4).astype(np.float32)
            weights = (np.arange(rows, dtype=np.float32) % 5) / 4
            data = xgb.DMatrix(features, label=labels, weight=weights, nthread=2)
            params = dict(device='npu:0', tree_method='hist', objective='reg:squarederror',
                          max_depth=args.max_depth, max_bin=args.max_bin, min_child_weight=0, reg_alpha=0.25,
                          grow_policy=args.grow_policy, max_leaves=args.max_leaves,
                          reg_lambda=2, subsample=0.75, colsample_bytree=0.75,
                          colsample_bylevel=0.8, colsample_bynode=0.8,
                          seed=20261017, nthread=2)
            model = xgb.Booster(params=params, cache=[data])
            started = time.perf_counter()
            per_round_seconds = []
            for iteration in range(args.rounds):
                mark('train_begin', rows=rows, iteration=iteration)
                round_started = time.perf_counter()
                model.update(data, iteration)
                elapsed = time.perf_counter() - round_started
                per_round_seconds.append(elapsed)
                mark('train_end', rows=rows, iteration=iteration, elapsed_seconds=elapsed)
                print(json.dumps({'event': 'round_completed', 'rows': rows,
                                  'iteration': iteration, 'elapsed_seconds': elapsed}), flush=True)
            if model.num_boosted_rounds() != args.rounds:
                raise AssertionError('native model did not retain every requested training round')
            mark('predict_begin', rows=rows)
            prediction = model.predict(data)
            mark('predict_end', rows=rows)
            if not np.isfinite(prediction).all():
                raise AssertionError('nonfinite native prediction')
            path = output / f'model-{rows}.ubj'
            model.save_model(path)
            cpu = xgb.Booster(model_file=path)
            cpu.set_param({'device': 'cpu', 'nthread': 2})
            expected = cpu.predict(data)
            np.testing.assert_allclose(prediction, expected, rtol=1e-6, atol=1e-6)
            np.save(output / f'prediction-{rows}-npu.npy', prediction)
            np.save(output / f'prediction-{rows}-cpu.npy', expected)
            baseline_check = None
            if baseline is not None:
                baseline_check = _compare_baseline(args.baseline_output_dir, baseline, path,
                                                   prediction, rows)
            reports.append({'rows': rows, 'features': args.features, 'rounds': args.rounds,
                            'max_depth': args.max_depth, 'max_bin': args.max_bin,
                            'grow_policy': args.grow_policy, 'max_leaves': args.max_leaves,
                            'per_round_seconds': per_round_seconds,
                            'max_abs_diff': float(np.max(np.abs(prediction - expected))),
                            'elapsed_seconds': time.perf_counter() - started,
                            'baseline_comparison': baseline_check,
                            'model_sha256': _sha256(path), 'status': 'passed'})
            mark('cleanup_begin', rows=rows)
            del model, cpu, data
            gc.collect()
            mark('cleanup_end', rows=rows)
    report = {'status': 'passed', 'device': 'npu:0', 'cases': reports,
              'library_path': str(args.library.resolve()), 'library_sha256': _sha256(args.library),
              'driver_sha256': _sha256(Path(__file__)),
              'limitations': ['bounded synthetic matrices; not full Qlib or original parameter acceptance',
                              'CPU/NPU training trees not compared',
                              'timings include transfer-audit overhead and are not performance results']}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())