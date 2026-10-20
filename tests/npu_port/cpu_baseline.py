#!/usr/bin/env python3
"""Create deterministic CPU fixtures before developing the NPU backend.

Run with the source package on PYTHONPATH; artifacts are never overwritten.
This is a correctness baseline, not a performance benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SEED = 20261016
BASE = dict(objective="reg:squarederror", eval_metric="rmse", device="cpu",
            tree_method="hist", max_depth=3, max_bin=32, eta=0.15,
            nthread=2, seed=SEED)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-library-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="failed", command=sys.argv, cases=[], seed=SEED,
                  versions=dict(python=platform.python_version(), numpy=np.__version__))
    started = time.perf_counter()
    try:
        import xgboost as xgb

        lib = Path(xgb.core._LIB._name).resolve()
        assert lib.is_relative_to(args.expected_library_root.resolve()), lib
        assert xgb.build_info()["USE_CUDA"] is False
        report.update(package_path=xgb.__file__, library_path=str(lib),
                      library_sha256=sha256(lib), build_info=xgb.build_info())
        report["versions"]["xgboost"] = xgb.__version__
        rng = np.random.default_rng(SEED)
        dense = rng.normal(size=(384, 8)).astype(np.float32)
        labels = (2 * dense[:, 0] - dense[:, 1] + 0.3 * dense[:, 2] ** 2).astype(np.float32)
        missing = dense.copy()
        missing[::3, 1] = np.nan
        missing[:, 6] = 3.0
        missing[:, 7] = np.nan
        weights = rng.uniform(0.1, 2.0, len(labels)).astype(np.float32)
        weights[::5] = 0.0
        cases = [
            ("dense", dense, labels, None, {}),
            ("nan_constant_all_missing", missing, labels, None, {}),
            ("weighted_zero_rows", missing, labels, weights, {}),
            ("regularized", missing, labels, weights,
             dict(reg_alpha=0.2, reg_lambda=2.0, gamma=0.1, min_child_weight=2.0, max_delta_step=0.7)),
            ("sampling", missing, labels, weights,
             dict(subsample=0.75, colsample_bytree=0.8, colsample_bylevel=0.75, colsample_bynode=0.75)),
            ("single_row", dense[:1], labels[:1], None, {}),
        ]
        for name, features, y, weight, overrides in cases:
            item = dict(name=name, status="failed", parameters=BASE | overrides)
            report["cases"].append(item)
            case_start = time.perf_counter()
            folder = output / name
            folder.mkdir()
            split = 288 if len(y) > 1 else 1
            train = xgb.DMatrix(features[:split], label=y[:split],
                                weight=None if weight is None else weight[:split], nthread=2)
            valid = xgb.DMatrix(features[split:] if len(y) > 1 else features,
                                label=y[split:] if len(y) > 1 else y, nthread=2)
            history = {}
            model = xgb.train(item["parameters"], train, num_boost_round=16,
                              evals=[(train, "train"), (valid, "valid")],
                              evals_result=history, verbose_eval=False)
            pred = model.predict(valid)
            assert np.isfinite(pred).all()
            repeat = xgb.train(item["parameters"], train, num_boost_round=16)
            np.testing.assert_array_equal(pred, repeat.predict(valid))
            for extension in ("json", "ubj"):
                path = folder / ("model." + extension)
                model.save_model(path)
                loaded = xgb.Booster(model_file=path)
                np.testing.assert_array_equal(pred, loaded.predict(valid))
                assert loaded.get_score(importance_type="gain") == model.get_score(importance_type="gain")
            if name == "dense":
                assert model.get_score(), "Dense fixture must exercise real splits"
            assert len(history["valid"]["rmse"]) == 16
            fixtures = dict(features=features, labels=y)
            if weight is not None:
                fixtures["weights"] = weight
            np.savez_compressed(folder / "fixture.npz", **fixtures)
            np.save(folder / "prediction.npy", pred)
            (folder / "eval_history.json").write_text(json.dumps(history, indent=2))
            item.update(status="passed", train_rows=split, prediction_rows=len(pred),
                        actual_rounds=model.num_boosted_rounds(),
                        fixture_sha256=sha256(folder / "fixture.npz"),
                        elapsed_seconds=time.perf_counter() - case_start)
        # eta=0 guarantees a flat validation curve: stopping must really trigger.
        history = {}
        stopped = xgb.train(BASE | dict(eta=0.0),
                            xgb.DMatrix(dense[:288], label=labels[:288], nthread=2),
                            num_boost_round=25,
                            evals=[(xgb.DMatrix(dense[288:], label=labels[288:], nthread=2), "valid")],
                            early_stopping_rounds=3, evals_result=history, verbose_eval=False)
        assert stopped.best_iteration == 0
        assert len(history["valid"]["rmse"]) == 4
        report["cases"].append(dict(name="actual_early_stopping", status="passed",
                                     best_iteration=stopped.best_iteration,
                                     actual_rounds=stopped.num_boosted_rounds(), history=history))
        try:
            xgb.train(BASE | dict(device="npu:0"),
                      xgb.DMatrix(dense[:4], label=labels[:4], nthread=2), num_boost_round=1)
        except xgb.core.XGBoostError as exc:
            assert "not compiled with NPU support" in str(exc), str(exc)
            assert "CPU fallback is disabled" in str(exc), str(exc)
            report["cases"].append(dict(name="cpu_only_npu_rejected", status="passed"))
        else:
            raise AssertionError("CPU-only reference unexpectedly accepted npu:0")
        report["status"] = "passed"
    except Exception:
        report["failure"] = traceback.format_exc()
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())