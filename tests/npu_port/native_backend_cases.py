#!/usr/bin/env python3
"""Small, real XGBoost learner cases used by the native NPU acceptance check.

This module is intentionally a library of cases: importing it does not import
XGBoost, inspect a device, or execute a test.  The checker supplies the private
native library and output directory before calling :func:`run_cases`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import importlib.util
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _private_xgboost(library: Path):
    """Import the checkout package while pinning its library locator.

    This is deliberately process-local.  It does not install a wheel or alter
    the checkout, and prevents an installed XGBoost from satisfying the test.
    """
    library = library.resolve(strict=True)
    package = ROOT / "python-package"
    sys.path.insert(0, str(package))
    locator_path = package / "xgboost" / "libpath.py"
    spec = importlib.util.spec_from_file_location("xgboost.libpath", locator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source library locator: {locator_path}")
    locator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(locator)
    locator.find_lib_path = lambda: [str(library)]
    sys.modules["xgboost.libpath"] = locator
    import xgboost as xgb
    actual = Path(xgb.core._LIB._name).resolve()
    if actual != library:
        raise RuntimeError(f"wrong XGBoost library: {actual}, expected {library}")
    return xgb


def _assert_close(actual, expected, message: str, *, atol: float = 1e-6) -> None:
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=atol, err_msg=message)


def _reject(call, xgb, label: str) -> None:
    try:
        call()
    except (xgb.core.XGBoostError, ValueError, TypeError):
        return
    raise AssertionError(f"unsupported configuration was accepted: {label}")


def run_cases(output_dir: Path, library: Path, *, device: str = "npu:0") -> dict[str, Any]:
    """Run the complete bounded native acceptance set and write evidence files."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    xgb = _private_xgboost(library)
    if device != "npu:0":
        raise ValueError("this acceptance harness only executes the explicitly selected npu:0")
    if xgb.build_info().get("USE_NPU") is not True:
        raise RuntimeError(f"library is not an NPU build: {xgb.build_info()}")

    rng = np.random.default_rng(20261017)
    rows = 32
    X = (rng.normal(0, 1, (rows, 4))).astype(np.float32)
    X[::7, 0] = np.nan
    X[::11, 1] = 999.0  # explicit missing sentinel below
    X[:, 3] = np.nan  # all-missing feature
    y = (0.75 * np.nan_to_num(X[:, 0], nan=0.0) - 0.4 * X[:, 2] + 0.2).astype(np.float32)
    weights = (np.arange(rows, dtype=np.float32) % 5).astype(np.float32)
    d = xgb.DMatrix(X, label=y, weight=weights, missing=999.0, nthread=1)
    d_all_missing = xgb.DMatrix(np.full((rows, 4), np.nan, dtype=np.float32), label=y,
                                weight=np.where(weights == 0, 1.0, weights), nthread=1)

    base = dict(device=device, tree_method="hist", objective="reg:squarederror",
                nthread=1, seed=20261017, max_bin=8, max_depth=2,
                min_child_weight=0, reg_alpha=0.0, reg_lambda=1.0,
                max_delta_step=0.0)
    checks: list[str] = []
    model = xgb.train(base, d, num_boost_round=1)
    config = json.loads(model.save_config())
    config_text = json.dumps(config)
    if config["learner"]["generic_param"].get("device") != device or "grow_npu_hist" not in config_text:
        raise AssertionError("training did not select native NPU hist updater")
    checks.append("dmatrix_one_tree_missing_weights")
    pred = model.predict(d)
    if not np.isfinite(pred).all():
        raise AssertionError("one-tree prediction contains non-finite values")

    # The weighted intercept is an exact public-objective invariant.
    intercept = xgb.train(base | {"eta": 0.0}, d, num_boost_round=1)
    expected = np.average(y.astype(np.float64), weights=weights.astype(np.float64))
    _assert_close(intercept.predict(d), expected, "weighted initial prediction")
    checks.append("weighted_intercept")

    multi = xgb.train(base | {"reg_alpha": 0.25, "reg_lambda": 2.0, "max_delta_step": 0.5},
                      d, num_boost_round=3)
    pred_multi = multi.predict(d)
    if not np.isfinite(pred_multi).all():
        raise AssertionError("regularized prediction contains non-finite values")
    checks.append("multi_round_l1_l2_delta")

    for depth, leaves in ((1, 2), (2, 4)):
        leaf_model = xgb.train(base | {"max_depth": depth, "max_leaves": leaves}, d, 1)
        if not np.isfinite(leaf_model.predict(d)).all():
            raise AssertionError("depth/leaves prediction contains non-finite values")
    checks.append("depth_and_leaves")

    sampled_params = base | {"subsample": 0.75, "colsample_bytree": 0.75}
    sampled = xgb.train(sampled_params, d, num_boost_round=3)
    if not np.isfinite(sampled.predict(d)).all():
        raise AssertionError("sampled prediction contains non-finite values")
    # Same seed and same data must be deterministic on one backend.
    sampled_again = xgb.train(sampled_params, d, num_boost_round=3)
    _assert_close(sampled_again.predict(d), sampled.predict(d), "same-seed sampled training")
    checks.append("sampling_and_same_seed")

    all_missing = xgb.train(base, d_all_missing, num_boost_round=1)
    if not np.isfinite(all_missing.predict(d_all_missing)).all():
        raise AssertionError("all-missing feature prediction contains non-finite values")
    checks.append("all_missing_feature")

    early_history: dict[str, Any] = {}
    early = xgb.train(base | {"eta": 0.0}, d, num_boost_round=10, evals=[(d, "train")],
                      evals_result=early_history, early_stopping_rounds=1, verbose_eval=False)
    if early.best_iteration != 0 or early.num_boosted_rounds() != 2 or \
            len(early_history.get("train", {}).get("rmse", [])) != 2:
        raise AssertionError("flat validation metric must actually trigger early stopping")
    checks.append("actual_early_stopping")

    # Iteration range and a full-cache prediction are compared to the same
    # model.  This catches cache misuse without assuming CPU/NPU tree identity.
    full = xgb.train(base, d, num_boost_round=3)
    full_pred = full.predict(d)
    range_pred = full.predict(d, iteration_range=(0, 3))
    _assert_close(range_pred, full_pred, "full iteration range")
    cached = xgb.Booster(params=base, cache=[d])
    cached.load_model(full.save_raw())
    _assert_close(cached.predict(d), full_pred, "full prediction cache")
    checks.append("iteration_range_and_cache")

    first = xgb.train(base, d, num_boost_round=1)
    continued = xgb.train(base, d, num_boost_round=2, xgb_model=first)
    direct = xgb.train(base, d, num_boost_round=3)
    _assert_close(continued.predict(d), direct.predict(d), "same-seed continuation")
    checks.append("continuation")

    # The same serialized model is the cross-backend comparison; training trees
    # themselves are intentionally not compared across CPU and NPU.
    model_paths: list[Path] = []
    for suffix in ("ubj", "json"):
        path = output_dir / f"model.{suffix}"
        full.save_model(path)
        loaded = xgb.Booster(model_file=path)
        _assert_close(loaded.predict(d), full_pred, f"{suffix} file roundtrip")
        model_paths.append(path)
    raw_path = output_dir / "model.raw"
    raw_path.write_bytes(full.save_raw())
    raw_loaded = xgb.Booster(model_file=raw_path)
    _assert_close(raw_loaded.predict(d), full_pred, "raw roundtrip")
    model_paths.append(raw_path)
    cpu_loaded = xgb.Booster(model_file=model_paths[0])
    cpu_loaded.set_param({"device": "cpu", "nthread": 1})
    _assert_close(cpu_loaded.predict(d), full_pred, "CPU inference from same model")
    checks.append("ubj_json_raw_and_cpu_inference")

    # Exercise more than one 128-thread block and the 16-block grid-stride
    # boundary. Separate DMatrix objects also test input binding/lifetime.
    for count in (129, 2049, 4097):
        unseen = rng.normal(size=(count, 4)).astype(np.float32)
        unseen[::5, 0] = np.nan
        unseen[::11, 1] = 999.0
        unseen[:, 3] = np.nan
        margin = np.linspace(-0.25, 0.25, count, dtype=np.float32)
        np.savez_compressed(output_dir / f"unseen_{count}.npz", features=unseen, base_margin=margin)
        other = xgb.DMatrix(unseen, missing=999.0, base_margin=margin, nthread=1)
        expected_other = cpu_loaded.predict(other)
        actual_other = full.predict(other)
        np.save(output_dir / f"prediction_unseen_{count}_cpu.npy", expected_other)
        np.save(output_dir / f"prediction_unseen_{count}_npu.npy", actual_other)
        _assert_close(actual_other, expected_other, f"unseen {count}-row prediction")
        _assert_close(full.predict(other, training=True), expected_other,
                      f"public training prediction export at {count} rows")
        for first_round, last_round in ((0, 1), (1, 3), (0, 3)):
            _assert_close(full.predict(other, iteration_range=(first_round, last_round)),
                          cpu_loaded.predict(other, iteration_range=(first_round, last_round)),
                          f"unseen {count}-row range [{first_round},{last_round})")
        checks.append(f"unseen_base_margin_ranges_{count}_rows")

    # Explicit rejection checks are run through the native parameter validator;
    # no unsupported route is silently treated as a successful CPU fallback.
    for label, override in (
        ("binary objective", {"objective": "binary:logistic"}),
        ("gblinear", {"booster": "gblinear"}),
        ("dart", {"booster": "dart"}),
        ("multi output", {"num_target": 2}),
        ("scale_pos_weight", {"scale_pos_weight": 2}),
        ("approx tree method", {"tree_method": "approx"}),
    ):
        _reject(lambda override=override: xgb.train(base | override, d, 1), xgb, label)
    _reject(lambda: full.predict(d, pred_leaf=True), xgb, "leaf prediction")
    _reject(lambda: xgb.train(base, xgb.DMatrix(X, label=y, weight=np.zeros_like(weights)), 1),
            xgb, "all zero weights")
    checks.append("unsupported_rejections")

    np.save(output_dir / "prediction_native.npy", full_pred)
    np.save(output_dir / "prediction_cpu_same_model.npy", cpu_loaded.predict(d))
    (output_dir / "case_data.json").write_text(json.dumps({
        "rows": rows, "features": 4, "max_bin": 8, "seed": 20261017,
        "device": device, "training_backend": "native_npu_only",
        "cross_backend_training_comparison": "not_asserted; same-model CPU inference asserted at 1e-6",
    }, indent=2) + "\n")
    artifacts = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name not in {"summary.json"}:
            artifacts[path.name] = {"path": str(path), "sha256": _sha256(path),
                                    "bytes": path.stat().st_size}
    return {"status": "passed", "device": device, "checks": checks,
            "check_count": len(checks), "artifacts": artifacts,
            "limitations": ["training matrix: 32 rows x 4 features; unseen prediction up to 4097 rows",
                            "one NPU ordinal (0)", "cross-backend training trees are not compared",
                            "large training matrices are future coverage"]}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_cases(args.output_dir, args.library)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())